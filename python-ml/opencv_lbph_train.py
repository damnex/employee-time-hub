#!/usr/bin/env python3
"""Train an OpenCV LBPH face model from dataset folders."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an OpenCV LBPH face recognizer using one folder per employee. "
            "This path is laptop-friendly and avoids dlib."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Root folder containing one subfolder per employee.")
    parser.add_argument("--output", required=True, type=Path, help="Where the LBPH model and label map should be written.")
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
    if len(faces) != 1:
        return None
    x, y, w, h = faces[0]
    return int(x), int(y), int(w), int(h)


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


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()

    if not args.dataset.exists() or not args.dataset.is_dir():
        print(f"Dataset folder not found: {args.dataset}", file=sys.stderr)
        return 1
    if not hasattr(cv2, "face"):
        print(
            "OpenCV face module is unavailable. Install requirements-webcam.txt with opencv-contrib-python.",
            file=sys.stderr,
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    detector = create_detector()
    metadata_rows = load_metadata(args.metadata.resolve() if args.metadata else None)

    label_names: list[str] = []
    training_faces: list[np.ndarray] = []
    training_ids: list[int] = []
    skipped_images: list[SkippedImage] = []
    skipped_people: list[dict[str, Any]] = []
    label_samples: dict[str, int] = {}

    for label_id, person_dir in enumerate(iter_person_directories(args.dataset)):
        folder_name = person_dir.name
        valid_count = 0
        label_names.append(folder_name)

        for image_path in iter_image_files(person_dir):
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
            training_faces.append(face)
            training_ids.append(label_id)
            valid_count += 1

            # Data Augmentation: Flip horizontally
            flipped = cv2.flip(face, 1)
            training_faces.append(flipped)
            training_ids.append(label_id)
            valid_count += 1

            # Data Augmentation: Darker
            darker = cv2.convertScaleAbs(face, alpha=0.8, beta=0)
            training_faces.append(darker)
            training_ids.append(label_id)
            valid_count += 1

            # Data Augmentation: Brighter
            brighter = cv2.convertScaleAbs(face, alpha=1.2, beta=0)
            training_faces.append(brighter)
            training_ids.append(label_id)
            valid_count += 1

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
            filtered_ids.append(label_id)

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

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=args.lbph_radius,
        neighbors=args.lbph_neighbors,
        grid_x=args.lbph_grid_x,
        grid_y=args.lbph_grid_y,
        threshold=args.lbph_threshold,
    )
    recognizer.train(filtered_faces, np.array(filtered_ids, dtype=np.int32))
    model_path = args.output / "lbph-model.yml"
    recognizer.save(str(model_path))

    labels_payload = {
        "createdAt": datetime.now(UTC).isoformat(),
        "modelType": "opencv-lbph",
        "imageSize": args.image_size,
        "threshold": args.lbph_threshold,
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
        "threshold": args.lbph_threshold,
        "imageSize": args.image_size,
    }
    json_dump(args.output / "lbph-training-report.json", report_payload)

    print(f"LBPH model: {model_path}")
    print(f"Label map: {args.output / 'lbph-labels.json'}")
    print(f"Trained employees: {len([label for label in labels_payload['labels'] if label['includedInTraining']])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
