#!/usr/bin/env python3
"""Train an OpenCV face model from dataset folders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore
from opencv_face_backend import (
    GRAY_NN_MODEL_TYPE,
    LBPH_MODEL_TYPE,
    create_lbph_recognizer,
    estimate_gray_nn_threshold,
    face_to_feature_vector,
    has_lbph_support,
    save_gray_nn_model,
)
from opencv_face_service import (
    create_eye_detector,
    detect_faces as detect_runtime_faces,
    normalize_lighting,
    prepare_face_crop as prepare_runtime_face_crop,
    resize_frame as resize_runtime_frame,
    scale_box as scale_runtime_box,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class TrainingSample:
    label_id: int
    folder_name: str
    image_path: Path
    face_image: np.ndarray


@dataclass
class SkippedImage:
    folder_name: str
    image_path: Path
    reason: str


@dataclass
class BackendSummary:
    model_type: str
    threshold: float
    training_backend: str
    score_margin_threshold: float | None = None
    centroid_margin_threshold: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an OpenCV face recognizer using one folder per employee. "
            "Prefers LBPH when available and falls back to a grayscale nearest-neighbor model."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Root folder containing one subfolder per employee.")
    parser.add_argument("--output", required=True, type=Path, help="Where the trained model and label map should be written.")
    parser.add_argument("--metadata", type=Path, help="Optional CSV keyed by folder_name.")
    parser.add_argument("--min-samples", type=int, default=5, help="Minimum valid face images required per person.")
    parser.add_argument("--image-size", type=int, default=200, help="Square face crop size used for training.")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Haar cascade scale factor.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Haar cascade minNeighbors.")
    parser.add_argument("--min-face-size", type=int, default=40, help="Minimum face size in pixels.")
    parser.add_argument(
        "--lbph-radius",
        type=int,
        default=1,
        help="LBPH radius parameter.",
    )
    parser.add_argument("--lbph-neighbors", type=int, default=8, help="LBPH neighbors parameter.")
    parser.add_argument("--lbph-grid-x", type=int, default=8, help="LBPH grid_x parameter.")
    parser.add_argument("--lbph-grid-y", type=int, default=8, help="LBPH grid_y parameter.")
    parser.add_argument(
        "--lbph-threshold",
        type=float,
        default=65.0,
        help="Suggested recognition threshold. Lower prediction distances are better.",
    )
    return parser.parse_args()


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


def build_person_sources(
    dataset_root: Path,
    metadata_rows: dict[str, dict[str, str]],
) -> list[tuple[str, list[Path]]]:
    dataset_dirs = iter_person_directories(dataset_root)
    if not metadata_rows:
        return [(person_dir.name, [person_dir]) for person_dir in dataset_dirs]

    dirs_by_lower_name = {person_dir.name.lower(): person_dir for person_dir in dataset_dirs}
    people: list[tuple[str, list[Path]]] = []
    for folder_name, row in sorted(metadata_rows.items()):
        candidate_names = [
            folder_name,
            row.get("employeeCode") or "",
            row.get("name") or "",
        ]
        resolved_dirs: list[Path] = []
        seen_lower_names: set[str] = set()
        for candidate_name in candidate_names:
            normalized = candidate_name.strip().lower()
            if not normalized:
                continue
            person_dir = dirs_by_lower_name.get(normalized)
            if person_dir is None or person_dir.name.lower() in seen_lower_names:
                continue
            seen_lower_names.add(person_dir.name.lower())
            resolved_dirs.append(person_dir)
        people.append((folder_name, resolved_dirs))

    return people


def iter_image_files_for_sources(person_dirs: list[Path]) -> list[Path]:
    seen_paths: set[Path] = set()
    image_files: list[Path] = []
    for person_dir in person_dirs:
        for image_path in iter_image_files(person_dir):
            resolved = image_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            image_files.append(image_path)
    return sorted(image_files)


def build_runtime_detection_args() -> argparse.Namespace:
    return argparse.Namespace(
        scale_factor=1.04,
        min_neighbors=6,
        min_face_size=24,
        resize_width=480,
    )


def create_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")
    return detector


def detect_single_face(
    gray_image: np.ndarray,
    detector: cv2.CascadeClassifier,
    args: argparse.Namespace,
) -> tuple[int, int, int, int] | None:
    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=args.scale_factor,
        minNeighbors=args.min_neighbors,
        minSize=(args.min_face_size, args.min_face_size),
    )
    if len(faces) == 0:
        return None

    if len(faces) == 1:
        x, y, w, h = faces[0]
        return int(x), int(y), int(w), int(h)

    image_height, image_width = gray_image.shape[:2]
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0
    image_diagonal = max(1.0, float((image_width**2 + image_height**2) ** 0.5))

    ranked_faces: list[tuple[float, float, float, tuple[int, int, int, int]]] = []
    for raw_face in faces:
        x, y, w, h = [int(value) for value in raw_face]
        area = float(w * h)
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        center_distance = float(
            (((center_x - image_center_x) ** 2) + ((center_y - image_center_y) ** 2)) ** 0.5
        ) / image_diagonal
        # Favor large faces near the middle while still rejecting highly ambiguous scenes.
        score = area * max(0.15, 1.0 - (center_distance * 1.25))
        ranked_faces.append((score, area, center_distance, (x, y, w, h)))

    ranked_faces.sort(key=lambda item: item[0], reverse=True)
    best_score, best_area, best_center_distance, best_face = ranked_faces[0]
    if len(ranked_faces) == 1:
        return best_face

    second_score, second_area, second_center_distance, _ = ranked_faces[1]
    area_ratio = best_area / max(1.0, second_area)
    score_ratio = best_score / max(1.0, second_score)
    center_gap = second_center_distance - best_center_distance
    if area_ratio < 1.18 and score_ratio < 1.12 and center_gap < 0.06:
        return None

    return best_face


def extract_face(gray_image: np.ndarray, face_box: tuple[int, int, int, int], image_size: int) -> np.ndarray:
    x, y, w, h = face_box
    padding_x = int(w * 0.15)
    padding_y = int(h * 0.15)
    start_x = max(0, x - padding_x)
    start_y = max(0, y - padding_y)
    end_x = min(gray_image.shape[1], x + w + padding_x)
    end_y = min(gray_image.shape[0], y + h + padding_y)
    crop = gray_image[start_y:end_y, start_x:end_x]
    resized = cv2.resize(crop, (image_size, image_size))
    return cv2.equalizeHist(resized)


def extract_runtime_validation_face(
    image: np.ndarray,
    detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
    runtime_detection_args: argparse.Namespace,
    image_size: int,
) -> np.ndarray | None:
    normalized = normalize_lighting(image)
    working_frame, scale_factor = resize_runtime_frame(normalized, runtime_detection_args.resize_width)
    gray_working = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
    detected_faces = detect_runtime_faces(gray_working, detector, eye_detector, runtime_detection_args)
    if not detected_faces:
        return None

    gray_full = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    primary_box = scale_runtime_box(detected_faces[0], scale_factor)
    prepared_face, _, _, _ = prepare_runtime_face_crop(gray_full, primary_box, image_size)
    return prepared_face


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args: Any = parse_args()
    args.dataset = args.dataset.resolve()  # type: ignore
    args.output = args.output.resolve()  # type: ignore

    if not args.dataset.exists() or not args.dataset.is_dir():
        print(f"Dataset folder not found: {args.dataset}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    detector = create_detector()
    eye_detector = create_eye_detector()
    runtime_detection_args = build_runtime_detection_args()
    metadata_rows = load_metadata(args.metadata.resolve() if args.metadata else None)

    label_names: list[str] = []
    training_faces: list[np.ndarray] = []
    training_ids: list[int] = []
    validation_faces: list[np.ndarray] = []
    validation_ids: list[int] = []
    skipped_images: list[SkippedImage] = []
    skipped_people: list[dict[str, Any]] = []
    label_samples: dict[str, int] = {}

    person_sources = build_person_sources(args.dataset, metadata_rows)
    label_source_names: dict[str, list[str]] = {}
    for label_id, (folder_name, source_dirs) in enumerate(person_sources):
        valid_count = 0
        label_names.append(folder_name)
        label_source_names[folder_name] = [person_dir.name for person_dir in source_dirs]

        for image_path in iter_image_files_for_sources(source_dirs):
            image = cv2.imread(str(image_path))
            if image is None:
                skipped_images.append(SkippedImage(folder_name, image_path, "load-error"))
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_box = detect_single_face(gray, detector, args)
            if face_box is None:
                skipped_images.append(SkippedImage(folder_name, image_path, "expected-exactly-one-face"))
                continue

            face = extract_face(gray, face_box, args.image_size)
            validation_face = extract_runtime_validation_face(
                image,
                detector,
                eye_detector,
                runtime_detection_args,
                args.image_size,
            )
            training_faces.append(face)
            training_ids.append(label_id)
            validation_faces.append(validation_face if validation_face is not None else face)
            validation_ids.append(label_id)
            valid_count += 1

            # Data Augmentation: Flip horizontally
            flipped = cv2.flip(face, 1)
            training_faces.append(flipped)
            training_ids.append(label_id)

            # Data Augmentation: Darker
            darker = cv2.convertScaleAbs(face, alpha=0.8, beta=0)
            training_faces.append(darker)
            training_ids.append(label_id)

            # Data Augmentation: Brighter
            brighter = cv2.convertScaleAbs(face, alpha=1.2, beta=0)
            training_faces.append(brighter)
            training_ids.append(label_id)

            # Data Augmentation: Small rotations
            for angle in (-12, 12):
                matrix = cv2.getRotationMatrix2D((args.image_size / 2, args.image_size / 2), angle, 1.0)
                rotated = cv2.warpAffine(face, matrix, (args.image_size, args.image_size), borderMode=cv2.BORDER_REPLICATE)
                training_faces.append(rotated)
                training_ids.append(label_id)
        label_samples[folder_name] = valid_count

    valid_labels = {
        index
        for index, folder_name in enumerate(label_names)
        if label_samples.get(folder_name, 0) >= args.min_samples
    }
    if not valid_labels:
        print(
            "No employee folders met the minimum sample requirement for LBPH training.",
            file=sys.stderr,
        )
        return 1

    filtered_faces: list[np.ndarray] = []
    filtered_ids: list[int] = []
    for face_image, label_id in zip(training_faces, training_ids):
        if label_id in valid_labels:
            filtered_faces.append(face_image)
            filtered_ids.append(label_id)  # type: ignore

    filtered_validation_faces: list[np.ndarray] = []
    filtered_validation_ids: list[int] = []
    for face_image, label_id in zip(validation_faces, validation_ids):
        if label_id in valid_labels:
            filtered_validation_faces.append(face_image)
            filtered_validation_ids.append(label_id)

    for index, folder_name in enumerate(label_names):
        if index not in valid_labels:
            skipped_people.append(
                {
                    "folderName": folder_name,
                    "reason": "not-enough-valid-images",
                    "validSamples": label_samples.get(folder_name, 0),
                    "requiredSamples": args.min_samples,
                }
            )

    model_path = args.output / "lbph-model.yml"
    backend_summary: BackendSummary
    filtered_id_array = np.array(filtered_ids, dtype=np.int32)
    if has_lbph_support():
        recognizer = create_lbph_recognizer(
            radius=args.lbph_radius,
            neighbors=args.lbph_neighbors,
            grid_x=args.lbph_grid_x,
            grid_y=args.lbph_grid_y,
            threshold=args.lbph_threshold,
        )
        recognizer.train(filtered_faces, filtered_id_array)
        recognizer.save(str(model_path))
        backend_summary = BackendSummary(
            model_type=LBPH_MODEL_TYPE,
            threshold=float(args.lbph_threshold),
            training_backend="lbph",
        )
    else:
        feature_rows = np.vstack([face_to_feature_vector(face_image) for face_image in filtered_faces])
        validation_feature_rows = np.vstack([face_to_feature_vector(face_image) for face_image in filtered_validation_faces])
        threshold_estimate = estimate_gray_nn_threshold(
            feature_rows,
            filtered_id_array,
            validation_features=validation_feature_rows,
            validation_label_ids=np.array(filtered_validation_ids, dtype=np.int32),
        )
        save_gray_nn_model(
            model_path,
            features=feature_rows,
            label_ids=filtered_id_array,
            threshold=threshold_estimate.threshold,
            image_size=int(args.image_size),
        )
        backend_summary = BackendSummary(
            model_type=GRAY_NN_MODEL_TYPE,
            threshold=threshold_estimate.threshold,
            training_backend="gray-nearest-neighbor",
            score_margin_threshold=threshold_estimate.score_margin_threshold,
            centroid_margin_threshold=threshold_estimate.centroid_margin_threshold,
        )

    labels_payload = {
        "createdAt": datetime.now(UTC).isoformat(),
        "modelType": backend_summary.model_type,
        "trainingBackend": backend_summary.training_backend,
        "imageSize": args.image_size,
        "threshold": backend_summary.threshold,
        "scoreMarginThreshold": backend_summary.score_margin_threshold,
        "centroidMarginThreshold": backend_summary.centroid_margin_threshold,
        "labels": [
            {
                "id": index,
                "folderName": folder_name,
                "displayName": (metadata_rows.get(folder_name, {}).get("name") or folder_name),
                "employeeCode": metadata_rows.get(folder_name, {}).get("employeeCode") or None,
                "department": metadata_rows.get(folder_name, {}).get("department") or None,
                "rfidUid": metadata_rows.get(folder_name, {}).get("rfidUid") or None,
                "email": metadata_rows.get(folder_name, {}).get("email") or None,
                "phone": metadata_rows.get(folder_name, {}).get("phone") or None,
                "isActive": parse_bool(metadata_rows.get(folder_name, {}).get("isActive") or "true"),
                "sourceFolders": label_source_names.get(folder_name, [folder_name]),
                "sampleCount": label_samples.get(folder_name, 0),
                "includedInTraining": index in valid_labels,
            }
            for index, folder_name in enumerate(label_names)
        ],
    }
    json_dump(args.output / "lbph-labels.json", labels_payload)
    report_payload = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "datasetRoot": str(args.dataset),
        "outputRoot": str(args.output),
        "trainedLabels": [folder_name for index, folder_name in enumerate(label_names) if index in valid_labels],
        "sourceFolders": label_source_names,
        "trainedImageCount": len(filtered_faces),
        "skippedPeople": skipped_people,
        "skippedImages": [
            {
                "folderName": item.folder_name,
                "imagePath": str(item.image_path),
                "reason": item.reason,
            }
            for item in skipped_images
        ],
        "modelType": backend_summary.model_type,
        "trainingBackend": backend_summary.training_backend,
        "threshold": backend_summary.threshold,
        "scoreMarginThreshold": backend_summary.score_margin_threshold,
        "centroidMarginThreshold": backend_summary.centroid_margin_threshold,
        "imageSize": args.image_size,
    }
    json_dump(args.output / "lbph-training-report.json", report_payload)

    print(f"LBPH model: {model_path}")
    print(f"Label map: {args.output / 'lbph-labels.json'}")
    print(f"Training backend: {backend_summary.training_backend}")
    trained_count = len([label for label in labels_payload["labels"] if isinstance(label, dict) and label.get("includedInTraining")])
    print(f"Trained employees: {trained_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

