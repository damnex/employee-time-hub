#!/usr/bin/env python3
"""Shared OpenCV face-recognition backends for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

LBPH_MODEL_TYPE = "opencv-lbph"
GRAY_NN_MODEL_TYPE = "opencv-gray-nn"
SFACE_MODEL_TYPE = "opencv-sface"
DEFAULT_SFACE_DETECTOR_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE_RECOGNIZER_PATH = Path(__file__).resolve().parent / "models" / "face_recognition_sface_2021dec.onnx"


@dataclass(frozen=True)
class GrayPredictionDiagnostics:
    best_label_id: int
    best_score: float
    best_sample_distance: float
    best_centroid_distance: float
    second_label_id: int | None
    second_score: float | None
    second_sample_distance: float | None
    second_centroid_distance: float | None
    score_margin: float
    centroid_margin: float


@dataclass(frozen=True)
class GrayThresholdEstimate:
    threshold: float
    score_margin_threshold: float
    centroid_margin_threshold: float


@dataclass(frozen=True)
class SFacePredictionDiagnostics:
    best_label_id: int
    best_score: float
    best_sample_distance: float
    best_centroid_distance: float
    second_label_id: int | None
    second_score: float | None
    second_sample_distance: float | None
    second_centroid_distance: float | None
    score_margin: float
    centroid_margin: float


@dataclass(frozen=True)
class SFaceThresholdEstimate:
    threshold: float
    score_margin_threshold: float
    centroid_margin_threshold: float


def resolve_prediction(
    recognizer: object,
    face_image: np.ndarray,
    distance_threshold: float,
    score_margin_threshold: float | None = None,
    centroid_margin_threshold: float | None = None,
) -> tuple[int, float, bool]:
    if hasattr(recognizer, "predict_details") and hasattr(recognizer, "is_prediction_confident"):
        diagnostics = recognizer.predict_details(face_image)
        accepted = recognizer.is_prediction_confident(
            diagnostics,
            distance_threshold,
            score_margin_threshold,
            centroid_margin_threshold,
        )
        return (
            int(diagnostics.best_label_id),
            float(diagnostics.best_score),
            bool(accepted),
        )

    label_id, distance = recognizer.predict(face_image)
    return int(label_id), float(distance), float(distance) <= distance_threshold


def has_lbph_support() -> bool:
    face_module = getattr(cv2, "face", None)
    return bool(face_module) and hasattr(face_module, "LBPHFaceRecognizer_create")


def has_sface_support(
    detector_path: Path = DEFAULT_SFACE_DETECTOR_PATH,
    recognizer_path: Path = DEFAULT_SFACE_RECOGNIZER_PATH,
) -> bool:
    if not detector_path.exists() or not recognizer_path.exists():
        return False
    return hasattr(cv2, "FaceDetectorYN_create") and hasattr(cv2, "FaceRecognizerSF_create")


def create_sface_detector(
    detector_path: Path = DEFAULT_SFACE_DETECTOR_PATH,
    *,
    input_size: tuple[int, int] = (320, 320),
    score_threshold: float = 0.7,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
):
    if not has_sface_support(detector_path=detector_path):
        raise RuntimeError(
            f"OpenCV YuNet face detector is unavailable or missing: {detector_path}"
        )
    return cv2.FaceDetectorYN_create(
        str(detector_path),
        "",
        input_size,
        float(score_threshold),
        float(nms_threshold),
        int(top_k),
    )


def create_sface_recognizer(
    recognizer_path: Path = DEFAULT_SFACE_RECOGNIZER_PATH,
):
    if not has_sface_support(recognizer_path=recognizer_path):
        raise RuntimeError(
            f"OpenCV SFace recognizer is unavailable or missing: {recognizer_path}"
        )
    return cv2.FaceRecognizerSF_create(str(recognizer_path), "")


def create_lbph_recognizer(
    *,
    radius: int = 1,
    neighbors: int = 8,
    grid_x: int = 8,
    grid_y: int = 8,
    threshold: float = 65.0,
):
    if not has_lbph_support():
        raise RuntimeError(
            "OpenCV LBPH support is unavailable in this Python environment. "
            "Install an opencv-contrib build that exposes cv2.face.LBPHFaceRecognizer_create()."
        )

    return cv2.face.LBPHFaceRecognizer_create(  # type: ignore[attr-defined]
        radius=radius,
        neighbors=neighbors,
        grid_x=grid_x,
        grid_y=grid_y,
        threshold=threshold,
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    flattened = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flattened))
    if not np.isfinite(norm) or norm <= 1e-6:
        return np.zeros_like(flattened, dtype=np.float32)
    return (flattened / norm).astype(np.float32)


def face_to_feature_vector(face_image: np.ndarray) -> np.ndarray:
    face = np.asarray(face_image, dtype=np.float32)
    if face.ndim != 2:
        raise ValueError("Expected a single-channel grayscale face crop.")

    vector = face.reshape(-1) / 255.0
    vector -= float(np.mean(vector))
    return _normalize_vector(vector)


def sface_feature_to_vector(feature: np.ndarray) -> np.ndarray:
    return _normalize_vector(feature)


def detect_sface_faces(
    image: np.ndarray,
    detector: object,
    *,
    min_face_size: int = 24,
    max_faces: int = 50,
) -> list[np.ndarray]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return []

    detector.setInputSize((int(width), int(height)))
    _, raw_faces = detector.detect(image)
    if raw_faces is None or len(raw_faces) == 0:
        return []

    frame_area = max(1.0, float(width * height))
    candidates: list[np.ndarray] = []
    for raw_face in raw_faces:
        face = np.asarray(raw_face, dtype=np.float32).reshape(-1)
        if face.size < 15:
            continue
        x, y, face_width, face_height = [float(value) for value in face[:4]]
        confidence = float(face[-1])
        if confidence < 0.7:
            continue
        if face_width < min_face_size or face_height < min_face_size:
            continue
        area_ratio = (face_width * face_height) / frame_area
        if area_ratio < 0.002 or area_ratio > 0.65:
            continue
        candidates.append(face)

    candidates.sort(
        key=lambda face: float(face[2] * face[3]) * float(face[-1]),
        reverse=True,
    )
    return candidates[: max(1, int(max_faces))]


def sface_box_to_tlbr(
    face: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    frame_height, frame_width = image_shape[:2]
    x, y, width, height = [float(value) for value in face[:4]]
    left = int(max(0, round(x)))
    top = int(max(0, round(y)))
    right = int(min(frame_width, round(x + width)))
    bottom = int(min(frame_height, round(y + height)))
    return top, right, bottom, left


def extract_sface_embedding(
    image: np.ndarray,
    face: np.ndarray,
    face_recognizer: object,
) -> np.ndarray:
    aligned_face = face_recognizer.alignCrop(image, face)
    feature = face_recognizer.feature(aligned_face)
    return sface_feature_to_vector(feature)


class GrayNearestNeighborRecognizer:
    def __init__(self, features: np.ndarray, label_ids: np.ndarray) -> None:
        feature_matrix = np.asarray(features, dtype=np.float32)
        labels = np.asarray(label_ids, dtype=np.int32).reshape(-1)
        if feature_matrix.ndim != 2 or feature_matrix.shape[0] <= 0:
            raise ValueError("Gray NN recognizer requires a non-empty 2D feature matrix.")
        if feature_matrix.shape[0] != labels.shape[0]:
            raise ValueError("Feature and label counts must match.")

        self._features = np.ascontiguousarray(feature_matrix, dtype=np.float32)
        self._label_ids = np.ascontiguousarray(labels, dtype=np.int32)
        self._unique_label_ids = np.unique(self._label_ids)
        self._label_centroids = _build_label_centroids(self._features, self._label_ids)

    def _compute_diagnostics_for_vector(self, feature_vector: np.ndarray) -> GrayPredictionDiagnostics:
        label_scores: list[tuple[int, float, float, float]] = []
        for label_id in self._unique_label_ids:
            label_features = self._features[self._label_ids == label_id]
            sample_distances = 1.0 - np.clip(label_features @ feature_vector, -1.0, 1.0)
            sample_distance = float(np.min(sample_distances))
            centroid_distance = float(
                1.0 - np.clip(self._label_centroids[int(label_id)] @ feature_vector, -1.0, 1.0)
            )
            combined_score = float(sample_distance * 0.35 + centroid_distance * 0.65)
            label_scores.append((int(label_id), combined_score, sample_distance, centroid_distance))

        label_scores.sort(key=lambda item: item[1])
        best = label_scores[0]
        second = label_scores[1] if len(label_scores) > 1 else None
        return GrayPredictionDiagnostics(
            best_label_id=best[0],
            best_score=best[1],
            best_sample_distance=best[2],
            best_centroid_distance=best[3],
            second_label_id=second[0] if second else None,
            second_score=second[1] if second else None,
            second_sample_distance=second[2] if second else None,
            second_centroid_distance=second[3] if second else None,
            score_margin=float((second[1] - best[1]) if second else 1.0),
            centroid_margin=float((second[3] - best[3]) if second else 1.0),
        )

    def predict_details(self, face_image: np.ndarray) -> GrayPredictionDiagnostics:
        feature_vector = face_to_feature_vector(face_image)
        return self._compute_diagnostics_for_vector(feature_vector)

    def is_prediction_confident(
        self,
        diagnostics: GrayPredictionDiagnostics,
        distance_threshold: float,
        score_margin_threshold: float | None = None,
        centroid_margin_threshold: float | None = None,
    ) -> bool:
        score_margin_floor = (
            float(score_margin_threshold)
            if score_margin_threshold is not None
            else max(0.045, distance_threshold * 0.22)
        )
        centroid_margin_floor = (
            float(centroid_margin_threshold)
            if centroid_margin_threshold is not None
            else max(0.055, distance_threshold * 0.26)
        )
        sample_distance_cap = min(0.28, distance_threshold * 1.65 + 0.02)
        return (
            diagnostics.best_score <= distance_threshold
            and diagnostics.best_sample_distance <= sample_distance_cap
            and diagnostics.score_margin >= score_margin_floor
            and diagnostics.centroid_margin >= centroid_margin_floor
        )

    def predict(self, face_image: np.ndarray) -> tuple[int, float]:
        diagnostics = self.predict_details(face_image)
        return int(diagnostics.best_label_id), float(diagnostics.best_score)


def _build_label_centroids(features: np.ndarray, label_ids: np.ndarray) -> dict[int, np.ndarray]:
    centroids: dict[int, np.ndarray] = {}
    for label_id in np.unique(label_ids):
        matching_features = features[label_ids == label_id]
        centroid = _normalize_vector(np.mean(matching_features, axis=0))
        centroids[int(label_id)] = centroid
    return centroids


class SFaceEmbeddingRecognizer:
    def __init__(self, features: np.ndarray, label_ids: np.ndarray) -> None:
        feature_matrix = np.asarray(features, dtype=np.float32)
        labels = np.asarray(label_ids, dtype=np.int32).reshape(-1)
        if feature_matrix.ndim != 2 or feature_matrix.shape[0] <= 0:
            raise ValueError("SFace recognizer requires a non-empty 2D feature matrix.")
        if feature_matrix.shape[0] != labels.shape[0]:
            raise ValueError("SFace feature and label counts must match.")

        normalized_features = np.vstack([_normalize_vector(row) for row in feature_matrix])
        self._features = np.ascontiguousarray(normalized_features, dtype=np.float32)
        self._label_ids = np.ascontiguousarray(labels, dtype=np.int32)
        self._unique_label_ids = np.unique(self._label_ids)
        self._label_centroids = _build_label_centroids(self._features, self._label_ids)

    def _compute_diagnostics_for_vector(self, feature_vector: np.ndarray) -> SFacePredictionDiagnostics:
        query = _normalize_vector(feature_vector)
        label_scores: list[tuple[int, float, float, float]] = []
        for label_id in self._unique_label_ids:
            label_features = self._features[self._label_ids == label_id]
            sample_distances = 1.0 - np.clip(label_features @ query, -1.0, 1.0)
            sample_distance = float(np.min(sample_distances))
            centroid_distance = float(
                1.0 - np.clip(self._label_centroids[int(label_id)] @ query, -1.0, 1.0)
            )
            combined_score = float(sample_distance * 0.35 + centroid_distance * 0.65)
            label_scores.append((int(label_id), combined_score, sample_distance, centroid_distance))

        label_scores.sort(key=lambda item: item[1])
        best = label_scores[0]
        second = label_scores[1] if len(label_scores) > 1 else None
        return SFacePredictionDiagnostics(
            best_label_id=best[0],
            best_score=best[1],
            best_sample_distance=best[2],
            best_centroid_distance=best[3],
            second_label_id=second[0] if second else None,
            second_score=second[1] if second else None,
            second_sample_distance=second[2] if second else None,
            second_centroid_distance=second[3] if second else None,
            score_margin=float((second[1] - best[1]) if second else 1.0),
            centroid_margin=float((second[3] - best[3]) if second else 1.0),
        )

    def predict_embedding_details(self, feature_vector: np.ndarray) -> SFacePredictionDiagnostics:
        return self._compute_diagnostics_for_vector(feature_vector)

    def is_prediction_confident(
        self,
        diagnostics: SFacePredictionDiagnostics,
        distance_threshold: float,
        score_margin_threshold: float | None = None,
        centroid_margin_threshold: float | None = None,
    ) -> bool:
        score_margin_floor = (
            float(score_margin_threshold)
            if score_margin_threshold is not None
            else max(0.08, distance_threshold * 0.24)
        )
        centroid_margin_floor = (
            float(centroid_margin_threshold)
            if centroid_margin_threshold is not None
            else max(0.08, distance_threshold * 0.22)
        )
        sample_distance_cap = min(0.58, distance_threshold + 0.12)
        return (
            diagnostics.best_score <= distance_threshold
            and diagnostics.best_sample_distance <= sample_distance_cap
            and diagnostics.score_margin >= score_margin_floor
            and diagnostics.centroid_margin >= centroid_margin_floor
        )

    def confidence_for_diagnostics(
        self,
        diagnostics: SFacePredictionDiagnostics,
        distance_threshold: float,
        score_margin_threshold: float | None = None,
        centroid_margin_threshold: float | None = None,
    ) -> float:
        threshold = max(float(distance_threshold), 1e-6)
        score_margin_floor = (
            float(score_margin_threshold)
            if score_margin_threshold is not None
            else max(0.08, threshold * 0.24)
        )
        centroid_margin_floor = (
            float(centroid_margin_threshold)
            if centroid_margin_threshold is not None
            else max(0.08, threshold * 0.22)
        )
        distance_component = 1.0 - min(1.0, diagnostics.best_score / threshold) * 0.28
        score_margin_component = min(1.0, diagnostics.score_margin / max(score_margin_floor * 3.0, 0.22))
        centroid_margin_component = min(1.0, diagnostics.centroid_margin / max(centroid_margin_floor * 3.0, 0.22))
        confidence = (
            distance_component * 0.78
            + score_margin_component * 0.14
            + centroid_margin_component * 0.08
        )
        if not self.is_prediction_confident(
            diagnostics,
            distance_threshold,
            score_margin_threshold,
            centroid_margin_threshold,
        ):
            confidence = min(confidence, 0.82)
        return float(np.clip(confidence, 0.0, 1.0))


def estimate_gray_nn_threshold(
    features: np.ndarray,
    label_ids: np.ndarray,
    validation_features: np.ndarray | None = None,
    validation_label_ids: np.ndarray | None = None,
) -> GrayThresholdEstimate:
    feature_matrix = np.asarray(features, dtype=np.float32)
    labels = np.asarray(label_ids, dtype=np.int32).reshape(-1)
    if feature_matrix.shape[0] != labels.shape[0] or feature_matrix.shape[0] <= 0:
        raise ValueError("Threshold estimation requires matching feature and label counts.")

    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        return GrayThresholdEstimate(
            threshold=0.14,
            score_margin_threshold=0.05,
            centroid_margin_threshold=0.06,
        )

    validation_matrix = (
        np.asarray(validation_features, dtype=np.float32)
        if validation_features is not None
        else feature_matrix
    )
    validation_labels = (
        np.asarray(validation_label_ids, dtype=np.int32).reshape(-1)
        if validation_label_ids is not None
        else labels
    )
    if validation_matrix.shape[0] != validation_labels.shape[0] or validation_matrix.shape[0] <= 0:
        raise ValueError("Validation features must match validation labels.")

    recognizer = GrayNearestNeighborRecognizer(feature_matrix, labels)
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    score_margins: list[float] = []
    centroid_margins: list[float] = []

    for feature_vector, label_id in zip(validation_matrix, validation_labels):
        diagnostics = recognizer._compute_diagnostics_for_vector(_normalize_vector(feature_vector))

        label_scores: list[tuple[float, float, float]] = []
        for candidate_label_id in recognizer._unique_label_ids:
            label_features = recognizer._features[recognizer._label_ids == candidate_label_id]
            sample_distances = 1.0 - np.clip(label_features @ feature_vector, -1.0, 1.0)
            sample_distance = float(np.min(sample_distances))
            centroid_distance = float(
                1.0 - np.clip(recognizer._label_centroids[int(candidate_label_id)] @ feature_vector, -1.0, 1.0)
            )
            combined_score = float(sample_distance * 0.35 + centroid_distance * 0.65)
            label_scores.append((float(candidate_label_id), combined_score, centroid_distance))

        own_entry = next((item for item in label_scores if int(item[0]) == int(label_id)), None)
        other_entries = [item for item in label_scores if int(item[0]) != int(label_id)]
        if own_entry is None or not other_entries:
            continue

        best_other = min(other_entries, key=lambda item: item[1])
        best_other_centroid = min(other_entries, key=lambda item: item[2])
        positive_scores.append(float(own_entry[1]))
        negative_scores.append(float(best_other[1]))
        score_margins.append(float(best_other[1] - own_entry[1]))
        centroid_margins.append(float(best_other_centroid[2] - own_entry[2]))

    positive_ceiling = float(np.percentile(positive_scores, 95))
    negative_floor = float(np.percentile(negative_scores, 5)) if negative_scores else positive_ceiling + 0.06
    if negative_floor > positive_ceiling:
        threshold = (positive_ceiling + negative_floor) / 2.0
    else:
        threshold = positive_ceiling + 0.025

    score_margin_floor = max(0.045, float(np.percentile(score_margins, 5)) * 0.8) if score_margins else 0.05
    centroid_margin_floor = max(0.055, float(np.percentile(centroid_margins, 5)) * 0.8) if centroid_margins else 0.06

    return GrayThresholdEstimate(
        threshold=float(np.clip(threshold, 0.1, 0.28)),
        score_margin_threshold=float(np.clip(score_margin_floor, 0.04, 0.18)),
        centroid_margin_threshold=float(np.clip(centroid_margin_floor, 0.05, 0.22)),
    )


def estimate_sface_threshold(
    features: np.ndarray,
    label_ids: np.ndarray,
) -> SFaceThresholdEstimate:
    feature_matrix = np.asarray(features, dtype=np.float32)
    labels = np.asarray(label_ids, dtype=np.int32).reshape(-1)
    if feature_matrix.shape[0] != labels.shape[0] or feature_matrix.shape[0] <= 0:
        raise ValueError("SFace threshold estimation requires matching feature and label counts.")

    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        return SFaceThresholdEstimate(
            threshold=0.3,
            score_margin_threshold=0.12,
            centroid_margin_threshold=0.12,
        )

    normalized_features = np.vstack([_normalize_vector(row) for row in feature_matrix])
    centroids = _build_label_centroids(normalized_features, labels)
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    score_margins: list[float] = []
    centroid_margins: list[float] = []

    for index, feature_vector in enumerate(normalized_features):
        own_label_id = int(labels[index])
        label_scores: list[tuple[int, float, float, float]] = []
        for candidate_label_id in unique_labels:
            candidate_label_id = int(candidate_label_id)
            label_features = normalized_features[labels == candidate_label_id]
            sample_distances = 1.0 - np.clip(label_features @ feature_vector, -1.0, 1.0)
            if candidate_label_id == own_label_id and len(sample_distances) > 1:
                sample_distances = sample_distances[sample_distances > 1e-6]
            sample_distance = float(np.min(sample_distances)) if len(sample_distances) else 0.0
            centroid_distance = float(
                1.0 - np.clip(centroids[candidate_label_id] @ feature_vector, -1.0, 1.0)
            )
            combined_score = float(sample_distance * 0.35 + centroid_distance * 0.65)
            label_scores.append((candidate_label_id, combined_score, sample_distance, centroid_distance))

        own_entry = next((item for item in label_scores if item[0] == own_label_id), None)
        other_entries = [item for item in label_scores if item[0] != own_label_id]
        if own_entry is None or not other_entries:
            continue

        best_other = min(other_entries, key=lambda item: item[1])
        best_other_centroid = min(other_entries, key=lambda item: item[3])
        positive_scores.append(float(own_entry[1]))
        negative_scores.append(float(best_other[1]))
        score_margins.append(float(best_other[1] - own_entry[1]))
        centroid_margins.append(float(best_other_centroid[3] - own_entry[3]))

    if not positive_scores:
        return SFaceThresholdEstimate(
            threshold=0.3,
            score_margin_threshold=0.12,
            centroid_margin_threshold=0.12,
        )

    positive_ceiling = float(np.percentile(positive_scores, 98))
    negative_floor = float(np.percentile(negative_scores, 2)) if negative_scores else positive_ceiling + 0.12
    if negative_floor > positive_ceiling:
        threshold = (positive_ceiling + negative_floor) / 2.0
    else:
        threshold = positive_ceiling + 0.035

    score_margin_floor = max(0.075, float(np.percentile(score_margins, 5)) * 0.72) if score_margins else 0.09
    centroid_margin_floor = max(0.075, float(np.percentile(centroid_margins, 5)) * 0.72) if centroid_margins else 0.09

    return SFaceThresholdEstimate(
        threshold=float(np.clip(threshold, 0.34, 0.54)),
        score_margin_threshold=float(np.clip(score_margin_floor, 0.075, 0.22)),
        centroid_margin_threshold=float(np.clip(centroid_margin_floor, 0.075, 0.22)),
    )


def save_gray_nn_model(
    path: Path,
    features: np.ndarray,
    label_ids: np.ndarray,
    threshold: float,
    image_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=np.asarray(features, dtype=np.float32),
            label_ids=np.asarray(label_ids, dtype=np.int32),
            threshold=np.asarray([threshold], dtype=np.float32),
            image_size=np.asarray([image_size], dtype=np.int32),
        )


def load_gray_nn_model(path: Path) -> tuple[GrayNearestNeighborRecognizer, float, int]:
    with path.open("rb") as handle:
        payload = np.load(handle)
        features = np.asarray(payload["features"], dtype=np.float32)
        label_ids = np.asarray(payload["label_ids"], dtype=np.int32)
        threshold = float(np.asarray(payload["threshold"], dtype=np.float32).reshape(-1)[0])
        image_size = int(np.asarray(payload["image_size"], dtype=np.int32).reshape(-1)[0])

    recognizer = GrayNearestNeighborRecognizer(features, label_ids)
    return recognizer, threshold, image_size


def save_sface_model(
    path: Path,
    features: np.ndarray,
    label_ids: np.ndarray,
    threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=np.asarray(features, dtype=np.float32),
            label_ids=np.asarray(label_ids, dtype=np.int32),
            threshold=np.asarray([threshold], dtype=np.float32),
        )


def load_sface_model(path: Path) -> tuple[SFaceEmbeddingRecognizer, float]:
    with path.open("rb") as handle:
        payload = np.load(handle)
        features = np.asarray(payload["features"], dtype=np.float32)
        label_ids = np.asarray(payload["label_ids"], dtype=np.int32)
        threshold = float(np.asarray(payload["threshold"], dtype=np.float32).reshape(-1)[0])

    recognizer = SFaceEmbeddingRecognizer(features, label_ids)
    return recognizer, threshold
