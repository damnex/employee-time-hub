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
    DEFAULT_SFACE_DETECTOR_PATH,
    DEFAULT_SFACE_RECOGNIZER_PATH,
    GRAY_NN_MODEL_TYPE,
    LBPH_MODEL_TYPE,
    SFACE_MODEL_TYPE,
    create_sface_detector,
    create_sface_recognizer,
    create_lbph_recognizer,
    detect_sface_faces,
    estimate_gray_nn_threshold,
    estimate_sface_threshold,
    extract_sface_embedding,
    face_to_feature_vector,
    has_lbph_support,
    has_sface_support,
    save_sface_model,
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
class SFaceTrainingSample:
    label_id: int
    folder_name: str
    image_path: Path
    embedding: np.ndarray


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
            "Prefers SFace embeddings when the OpenCV Zoo models are available, "
            "then falls back to LBPH or a grayscale nearest-neighbor model."
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
    parser.add_argument(
        "--sface-detector",
        type=Path,
        default=DEFAULT_SFACE_DETECTOR_PATH,
        help="Path to the OpenCV Zoo YuNet face detector ONNX model.",
    )
    parser.add_argument(
        "--sface-recognizer",
        type=Path,
        default=DEFAULT_SFACE_RECOGNIZER_PATH,
        help="Path to the OpenCV Zoo SFace recognition ONNX model.",
    )
    parser.add_argument(
        "--sface-score-threshold",
        type=float,
        default=0.78,
        help="Minimum YuNet face detection confidence used for SFace training.",
    )
    parser.add_argument(
        "--disable-sface",
        action="store_true",
        help="Skip SFace training even when the OpenCV Zoo models are available.",
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


def create_sface_training_runtime(args: argparse.Namespace) -> tuple[Any, Any] | None:
    if args.disable_sface:
        return None

    detector_path = args.sface_detector.resolve()
    recognizer_path = args.sface_recognizer.resolve()
    if not has_sface_support(detector_path=detector_path, recognizer_path=recognizer_path):
        return None

    try:
        detector = create_sface_detector(
            detector_path,
            score_threshold=float(args.sface_score_threshold),
        )
        recognizer = create_sface_recognizer(recognizer_path)
        return detector, recognizer
    except Exception as error:  # noqa: BLE001
        print(f"SFace training disabled: {error}", file=sys.stderr)
        return None


def extract_sface_training_embedding(
    image: np.ndarray,
    detector: Any,
    recognizer: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, str | None]:
    normalized = normalize_lighting(image)
    detected_faces = detect_sface_faces(
        normalized,
        detector,
        min_face_size=max(18, int(args.min_face_size * 0.6)),
        max_faces=8,
    )
    if not detected_faces:
        return None, "sface-no-face"

    if len(detected_faces) > 1:
        best_area = float(detected_faces[0][2] * detected_faces[0][3])
        second_area = float(detected_faces[1][2] * detected_faces[1][3])
        if second_area / max(best_area, 1.0) >= 0.35:
            return None, "sface-ambiguous-multiple-faces"

    try:
        return extract_sface_embedding(normalized, detected_faces[0], recognizer), None
    except Exception as error:  # noqa: BLE001
        return None, f"sface-embedding-error:{error}"


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


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    if not np.isfinite(norm) or norm <= 1e-6:
        return np.zeros_like(flattened, dtype=np.float32)
    return (flattened / norm).astype(np.float32)


def build_embedding_centroids(features: np.ndarray, label_ids: np.ndarray) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for label_id in np.unique(label_ids):
        matching_features = features[label_ids == label_id]
        centroids[int(label_id)] = normalize_embedding(np.mean(matching_features, axis=0))
    return centroids


def prune_sface_samples(
    samples: list[SFaceTrainingSample],
    *,
    min_samples: int,
) -> tuple[list[SFaceTrainingSample], list[SkippedImage]]:
    if not samples:
        return [], []

    features = np.vstack([sample.embedding for sample in samples]).astype(np.float32)
    features = np.vstack([normalize_embedding(row) for row in features])
    label_ids = np.array([sample.label_id for sample in samples], dtype=np.int32)
    keep_mask = np.ones(len(samples), dtype=bool)

    for _ in range(3):
        kept_features = features[keep_mask]
        kept_label_ids = label_ids[keep_mask]
        if len(np.unique(kept_label_ids)) <= 1:
            break

        centroids = build_embedding_centroids(kept_features, kept_label_ids)
        removed_this_round = 0
        for index, sample in enumerate(samples):
            if not keep_mask[index]:
                continue

            label_id = int(sample.label_id)
            label_keep_count = int(np.sum(keep_mask & (label_ids == label_id)))
            if label_keep_count <= min_samples:
                continue

            own_centroid = centroids.get(label_id)
            if own_centroid is None:
                continue

            feature = features[index]
            own_distance = float(1.0 - np.clip(own_centroid @ feature, -1.0, 1.0))
            other_distances = [
                float(1.0 - np.clip(centroid @ feature, -1.0, 1.0))
                for other_label_id, centroid in centroids.items()
                if int(other_label_id) != label_id
            ]
            best_other_distance = min(other_distances) if other_distances else 2.0

            label_distances = np.array(
                [
                    float(1.0 - np.clip(own_centroid @ features[candidate_index], -1.0, 1.0))
                    for candidate_index in range(len(samples))
                    if keep_mask[candidate_index] and int(label_ids[candidate_index]) == label_id
                ],
                dtype=np.float32,
            )
            if label_distances.size >= 8:
                q1 = float(np.percentile(label_distances, 25))
                q3 = float(np.percentile(label_distances, 75))
                outlier_cap = max(0.42, min(0.56, q3 + 1.5 * (q3 - q1)))
            else:
                outlier_cap = 0.56

            if own_distance > outlier_cap or best_other_distance <= own_distance + 0.03:
                keep_mask[index] = False
                removed_this_round += 1

        if removed_this_round == 0:
            break

    kept_samples = [sample for index, sample in enumerate(samples) if keep_mask[index]]
    removed_images = [
        SkippedImage(sample.folder_name, sample.image_path, "sface-identity-outlier")
        for index, sample in enumerate(samples)
        if not keep_mask[index]
    ]
    return kept_samples, removed_images


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
    sface_runtime = create_sface_training_runtime(args)
    runtime_detection_args = build_runtime_detection_args()
    metadata_rows = load_metadata(args.metadata.resolve() if args.metadata else None)

    label_names: list[str] = []
    training_faces: list[np.ndarray] = []
    training_ids: list[int] = []
    validation_faces: list[np.ndarray] = []
    validation_ids: list[int] = []
    sface_samples: list[SFaceTrainingSample] = []
    sface_skipped_images: list[SkippedImage] = []
    skipped_images: list[SkippedImage] = []
    skipped_people: list[dict[str, Any]] = []
    label_samples: dict[str, int] = {}
    sface_label_samples: dict[str, int] = {}

    person_sources = build_person_sources(args.dataset, metadata_rows)
    label_source_names: dict[str, list[str]] = {}
    for label_id, (folder_name, source_dirs) in enumerate(person_sources):
        valid_count = 0
        sface_valid_count = 0
        label_names.append(folder_name)
        label_source_names[folder_name] = [person_dir.name for person_dir in source_dirs]

        for image_path in iter_image_files_for_sources(source_dirs):
            image = cv2.imread(str(image_path))
            if image is None:
                skipped_images.append(SkippedImage(folder_name, image_path, "load-error"))
                sface_skipped_images.append(SkippedImage(folder_name, image_path, "load-error"))
                continue

            if sface_runtime is not None:
                sface_detector, sface_recognizer = sface_runtime
                sface_embedding, sface_skip_reason = extract_sface_training_embedding(
                    image,
                    sface_detector,
                    sface_recognizer,
                    args,
                )
                if sface_embedding is not None:
                    sface_samples.append(
                        SFaceTrainingSample(
                            label_id=label_id,
                            folder_name=folder_name,
                            image_path=image_path,
                            embedding=sface_embedding,
                        )
                    )
                    sface_valid_count += 1
                elif sface_skip_reason:
                    sface_skipped_images.append(SkippedImage(folder_name, image_path, sface_skip_reason))

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
        sface_label_samples[folder_name] = sface_valid_count

    model_path = args.output / "lbph-model.yml"
    backend_summary: BackendSummary
    backend_valid_labels: set[int]
    backend_label_samples: dict[str, int]
    backend_skipped_images: list[SkippedImage]
    trained_image_count = 0

    pruned_sface_samples, sface_outlier_images = prune_sface_samples(
        sface_samples,
        min_samples=args.min_samples,
    )
    sface_skipped_images.extend(sface_outlier_images)
    kept_sface_counts: dict[int, int] = {}
    for sample in pruned_sface_samples:
        kept_sface_counts[sample.label_id] = kept_sface_counts.get(sample.label_id, 0) + 1
    sface_valid_labels = {
        label_id
        for label_id, count in kept_sface_counts.items()
        if count >= args.min_samples
    }

    if sface_runtime is not None and sface_valid_labels:
        filtered_sface_samples = [
            sample
            for sample in pruned_sface_samples
            if sample.label_id in sface_valid_labels
        ]
        feature_rows = np.vstack([sample.embedding for sample in filtered_sface_samples]).astype(np.float32)
        filtered_id_array = np.array([sample.label_id for sample in filtered_sface_samples], dtype=np.int32)
        threshold_estimate = estimate_sface_threshold(feature_rows, filtered_id_array)
        save_sface_model(
            model_path,
            features=feature_rows,
            label_ids=filtered_id_array,
            threshold=threshold_estimate.threshold,
        )
        backend_summary = BackendSummary(
            model_type=SFACE_MODEL_TYPE,
            threshold=threshold_estimate.threshold,
            training_backend="opencv-sface",
            score_margin_threshold=threshold_estimate.score_margin_threshold,
            centroid_margin_threshold=threshold_estimate.centroid_margin_threshold,
        )
        backend_valid_labels = sface_valid_labels
        backend_label_samples = {
            folder_name: kept_sface_counts.get(index, 0)
            for index, folder_name in enumerate(label_names)
        }
        backend_skipped_images = sface_skipped_images
        trained_image_count = len(filtered_sface_samples)
    else:
        valid_labels = {
            index
            for index, folder_name in enumerate(label_names)
            if label_samples.get(folder_name, 0) >= args.min_samples
        }
        if not valid_labels:
            print(
                "No employee folders met the minimum sample requirement for face training.",
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
        backend_valid_labels = valid_labels
        backend_label_samples = label_samples
        backend_skipped_images = skipped_images
        trained_image_count = len(filtered_faces)

    for index, folder_name in enumerate(label_names):
        if index not in backend_valid_labels:
            skipped_people.append(
                {
                    "folderName": folder_name,
                    "reason": "not-enough-valid-images",
                    "validSamples": backend_label_samples.get(folder_name, 0),
                    "requiredSamples": args.min_samples,
                }
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
                "sampleCount": backend_label_samples.get(folder_name, 0),
                "includedInTraining": index in backend_valid_labels,
            }
            for index, folder_name in enumerate(label_names)
        ],
    }
    json_dump(args.output / "lbph-labels.json", labels_payload)
    report_payload = {
        "generatedAt": datetime.now(UTC).isoformat(),
        "datasetRoot": str(args.dataset),
        "outputRoot": str(args.output),
        "trainedLabels": [folder_name for index, folder_name in enumerate(label_names) if index in backend_valid_labels],
        "sourceFolders": label_source_names,
        "trainedImageCount": trained_image_count,
        "skippedPeople": skipped_people,
        "skippedImages": [
            {
                "folderName": item.folder_name,
                "imagePath": str(item.image_path),
                "reason": item.reason,
            }
            for item in backend_skipped_images
        ],
        "modelType": backend_summary.model_type,
        "trainingBackend": backend_summary.training_backend,
        "threshold": backend_summary.threshold,
        "scoreMarginThreshold": backend_summary.score_margin_threshold,
        "centroidMarginThreshold": backend_summary.centroid_margin_threshold,
        "imageSize": args.image_size,
    }
    json_dump(args.output / "lbph-training-report.json", report_payload)

    print(f"Face model: {model_path}")
    print(f"Label map: {args.output / 'lbph-labels.json'}")
    print(f"Training backend: {backend_summary.training_backend}")
    trained_count = len([label for label in labels_payload["labels"] if isinstance(label, dict) and label.get("includedInTraining")])
    print(f"Trained employees: {trained_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

