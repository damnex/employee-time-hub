#!/usr/bin/env python3
"""Train a face classifier from folder-based images and export attendance profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import face_recognition
import numpy as np
from sklearn import neighbors, svm

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FaceSample:
    label: str
    image_path: Path
    encoding: np.ndarray
    quality: float


@dataclass
class SkippedImage:
    label: str
    image_path: Path
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Python face model from dataset folders. "
            "Each subfolder under --dataset represents one employee/person."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Root folder containing one subfolder per employee.")
    parser.add_argument("--output", required=True, type=Path, help="Where the model and exported profiles should be written.")
    parser.add_argument("--metadata", type=Path, help="Optional CSV keyed by folder_name for employee import payloads.")
    parser.add_argument("--model-type", choices=("knn", "svm"), default="knn", help="Classifier type to train.")
    parser.add_argument("--n-neighbors", type=int, help="KNN neighbors. Defaults to sqrt(sample_count).")
    parser.add_argument("--knn-algorithm", default="ball_tree", help="scikit-learn KNN algorithm.")
    parser.add_argument("--distance-threshold", type=float, default=0.55, help="Prediction threshold for unknown faces.")
    parser.add_argument("--min-samples", type=int, default=3, help="Minimum valid images required per person.")
    parser.add_argument("--max-anchors", type=int, default=5, help="Max profile anchors exported per person.")
    parser.add_argument("--upsample", type=int, default=1, help="Face detection upsample count.")
    parser.add_argument("--jitters", type=int, default=1, help="Encoding jitter count.")
    parser.add_argument("--detection-model", choices=("hog", "cnn"), default="hog", help="Face detector backend.")
    parser.add_argument(
        "--landmark-model",
        choices=("small", "large"),
        default="small",
        help="Landmark model used by face_recognition.face_encodings.",
    )
    return parser.parse_args()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def to_descriptor(values: np.ndarray) -> list[float]:
    return [round_float(value) for value in values.tolist()]


def calculate_legacy_match_confidence(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.size == 0 or v1.shape != v2.shape:
        return 0.0

    centered_v1 = v1 - np.mean(v1)
    centered_v2 = v2 - np.mean(v2)
    magnitude_v1 = float(np.linalg.norm(centered_v1))
    magnitude_v2 = float(np.linalg.norm(centered_v2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    score = float(np.dot(centered_v1 / magnitude_v1, centered_v2 / magnitude_v2))
    return round_float((score + 1.0) / 2.0, 4)


def image_quality(image: np.ndarray, face_location: tuple[int, int, int, int]) -> float:
    top, right, bottom, left = face_location
    height = max(1, bottom - top)
    width = max(1, right - left)
    frame_height, frame_width = image.shape[:2]
    face_ratio = (height * width) / max(1, frame_height * frame_width)
    face_score = clamp(face_ratio / 0.12, 0.0, 1.0)

    crop = image[top:bottom, left:right]
    if crop.size == 0:
      return 0.0

    gray = crop.mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
    brightness = float(np.mean(gray) / 255.0)
    brightness_score = 1.0 - clamp(abs(brightness - 0.55) / 0.45, 0.0, 1.0)

    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    sharpness = float(np.mean(dx * dx) + np.mean(dy * dy))
    sharpness_score = clamp(sharpness / 1800.0, 0.0, 1.0)

    aspect_ratio = height / max(1, width)
    aspect_score = 1.0 - clamp(abs(aspect_ratio - 1.15) / 0.8, 0.0, 1.0)

    quality = (
        face_score * 0.45
        + sharpness_score * 0.25
        + brightness_score * 0.2
        + aspect_score * 0.1
    )
    return round_float(clamp(quality, 0.0, 1.0), 4)


def iter_person_directories(dataset_root: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def iter_image_files(person_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in person_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_samples(args: argparse.Namespace) -> tuple[dict[str, list[FaceSample]], list[SkippedImage], list[dict[str, Any]]]:
    people: dict[str, list[FaceSample]] = {}
    skipped_images: list[SkippedImage] = []
    skipped_people: list[dict[str, Any]] = []

    for person_dir in iter_person_directories(args.dataset):
        label = person_dir.name
        samples: list[FaceSample] = []

        for image_path in iter_image_files(person_dir):
            try:
                image = face_recognition.load_image_file(str(image_path))
                locations = face_recognition.face_locations(
                    image,
                    number_of_times_to_upsample=args.upsample,
                    model=args.detection_model,
                )
            except Exception as exc:  # noqa: BLE001
                skipped_images.append(SkippedImage(label, image_path, f"load-error: {exc}"))
                continue

            if len(locations) == 0:
                skipped_images.append(SkippedImage(label, image_path, "no-face-detected"))
                continue
            if len(locations) > 1:
                skipped_images.append(SkippedImage(label, image_path, "multiple-faces-detected"))
                continue

            encodings = face_recognition.face_encodings(
                image,
                known_face_locations=locations,
                num_jitters=args.jitters,
                model=args.landmark_model,
            )
            if not encodings:
                skipped_images.append(SkippedImage(label, image_path, "encoding-failed"))
                continue

            samples.append(
                FaceSample(
                    label=label,
                    image_path=image_path,
                    encoding=np.asarray(encodings[0], dtype=np.float32),
                    quality=image_quality(image, locations[0]),
                )
            )

        if len(samples) < args.min_samples:
            skipped_people.append(
                {
                    "label": label,
                    "reason": "not-enough-valid-images",
                    "validSamples": len(samples),
                    "requiredSamples": args.min_samples,
                }
            )
            continue

        people[label] = samples

    return people, skipped_images, skipped_people


def average_descriptor(encodings: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(encodings)
    return np.mean(stacked, axis=0)


def select_anchor_descriptors(samples: list[FaceSample], max_anchors: int) -> list[list[float]]:
    if len(samples) <= max_anchors:
        ordered = sorted(samples, key=lambda item: item.quality, reverse=True)
        return [to_descriptor(sample.encoding) for sample in ordered]

    encodings = np.vstack([sample.encoding for sample in samples])
    qualities = [sample.quality for sample in samples]
    chosen_indices = [int(np.argmax(np.asarray(qualities)))]

    while len(chosen_indices) < max_anchors:
        best_index = None
        best_distance = -1.0
        for index, encoding in enumerate(encodings):
            if index in chosen_indices:
                continue
            distance = min(
                float(np.linalg.norm(encoding - encodings[chosen_index]))
                for chosen_index in chosen_indices
            )
            if distance > best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            break
        chosen_indices.append(best_index)

    chosen = [samples[index].encoding for index in chosen_indices]
    return [to_descriptor(encoding) for encoding in chosen]


def build_profile(samples: list[FaceSample], max_anchors: int) -> dict[str, Any]:
    encodings = [sample.encoding for sample in samples]
    primary = average_descriptor(encodings)
    consistency = (
        1.0
        if len(encodings) == 1
        else float(np.mean([calculate_legacy_match_confidence(encoding, primary) for encoding in encodings]))
    )
    average_quality = float(np.mean([sample.quality for sample in samples]))

    return {
        "version": 2,
        "captureMode": "fallback",
        "primaryDescriptor": to_descriptor(primary),
        "anchorDescriptors": select_anchor_descriptors(samples, max_anchors),
        "averageQuality": round_float(average_quality, 4),
        "sampleCount": len(samples),
        "consistency": round_float(consistency, 4),
    }


def load_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "folder_name" not in (reader.fieldnames or []):
            raise ValueError("metadata CSV must include a folder_name column")

        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            key = (row.get("folder_name") or "").strip()
            if not key:
                continue
            rows[key] = {column: (value or "").strip() for column, value in row.items()}
        return rows


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def build_profile_exports(
    people: dict[str, list[FaceSample]],
    metadata_rows: dict[str, dict[str, str]],
    max_anchors: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    profiles: list[dict[str, Any]] = []
    employee_import: list[dict[str, Any]] = []
    missing_metadata: list[str] = []

    for label, samples in sorted(people.items()):
        profile = build_profile(samples, max_anchors)
        meta = metadata_rows.get(label, {})
        display_name = meta.get("name") or label.replace("_", " ").replace("-", " ").title()
        relative_files = [sample.image_path.name for sample in samples]

        profiles.append(
            {
                "folderName": label,
                "displayName": display_name,
                "sampleCount": profile["sampleCount"],
                "averageQuality": profile["averageQuality"],
                "consistency": profile["consistency"],
                "faceDescriptor": profile,
                "images": relative_files,
            }
        )

        required = ("employeeCode", "name", "department", "rfidUid")
        if not metadata_rows:
            continue
        if not all(meta.get(field) for field in required):
            missing_metadata.append(label)
            continue

        employee_import.append(
            {
                "employeeCode": meta["employeeCode"],
                "name": meta["name"],
                "department": meta["department"],
                "phone": meta.get("phone") or None,
                "email": meta.get("email") or None,
                "rfidUid": meta["rfidUid"].upper(),
                "isActive": parse_bool(meta.get("isActive") or "true"),
                "faceDescriptor": profile,
            }
        )

    return profiles, employee_import, missing_metadata


def train_classifier(
    people: dict[str, list[FaceSample]],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    encodings: list[np.ndarray] = []
    labels: list[str] = []
    for label, samples in sorted(people.items()):
        for sample in samples:
            encodings.append(sample.encoding)
            labels.append(label)

    if not encodings:
        raise ValueError("No valid training samples were found.")

    x_train = np.vstack(encodings)
    if args.model_type == "knn":
        n_neighbors = args.n_neighbors or max(1, int(round(math.sqrt(len(encodings)))))
        classifier = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=args.knn_algorithm,
            weights="distance",
        )
        classifier.fit(x_train, labels)
        details = {
            "modelType": "knn",
            "nNeighbors": n_neighbors,
            "algorithm": args.knn_algorithm,
        }
        return classifier, details

    classifier = svm.SVC(kernel="linear", probability=True)
    classifier.fit(x_train, labels)
    return classifier, {"modelType": "svm", "kernel": "linear"}


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_report(
    args: argparse.Namespace,
    people: dict[str, list[FaceSample]],
    profiles: list[dict[str, Any]],
    skipped_images: list[SkippedImage],
    skipped_people: list[dict[str, Any]],
    model_details: dict[str, Any],
    missing_metadata: list[str],
) -> dict[str, Any]:
    return {
        "generatedAt": datetime.now(UTC).isoformat(),
        "datasetRoot": str(args.dataset.resolve()),
        "outputRoot": str(args.output.resolve()),
        "profilesGenerated": len(profiles),
        "trainedLabels": sorted(people.keys()),
        "trainedSampleCount": sum(len(samples) for samples in people.values()),
        "skippedPeople": skipped_people,
        "missingMetadata": sorted(missing_metadata),
        "skippedImages": [
            {
                "label": item.label,
                "imagePath": str(item.image_path),
                "reason": item.reason,
            }
            for item in skipped_images
        ],
        "model": model_details,
    }


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()

    if not args.dataset.exists() or not args.dataset.is_dir():
        print(f"Dataset folder not found: {args.dataset}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    people, skipped_images, skipped_people = collect_samples(args)
    if not people:
        print(
            "No people met the minimum training sample requirement. "
            "Check the dataset folders and image quality.",
            file=sys.stderr,
        )
        return 1

    metadata_rows = load_metadata(args.metadata.resolve() if args.metadata else None)
    profiles, employee_import, missing_metadata = build_profile_exports(
        people,
        metadata_rows,
        args.max_anchors,
    )
    classifier, model_details = train_classifier(people, args)

    model_artifact = {
        "classifier": classifier,
        "modelType": model_details["modelType"],
        "distanceThreshold": args.distance_threshold,
        "createdAt": datetime.now(UTC).isoformat(),
        "labels": sorted(people.keys()),
    }

    with (args.output / "face_classifier.pkl").open("wb") as handle:
        pickle.dump(model_artifact, handle)

    json_dump(args.output / "profiles.json", profiles)
    if employee_import:
        json_dump(args.output / "employee-import.json", employee_import)
    report = build_report(
        args,
        people,
        profiles,
        skipped_images,
        skipped_people,
        model_details,
        missing_metadata,
    )
    json_dump(args.output / "training-report.json", report)

    print(f"Trained {len(people)} labels from {sum(len(samples) for samples in people.values())} images.")
    print(f"Model: {args.output / 'face_classifier.pkl'}")
    print(f"Profiles: {args.output / 'profiles.json'}")
    if employee_import:
        print(f"Employee import payload: {args.output / 'employee-import.json'}")
    if missing_metadata:
        print("Metadata missing for:", ", ".join(sorted(missing_metadata)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
