from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import InsightFaceModel


@dataclass(slots=True)
class FaceDetectorConfig:
    crop_margin_ratio: float = 0.12
    min_face_box_size: int = 40
    min_face_score: float = 0.45


@dataclass(slots=True)
class TrackedPersonInput:
    track_id: int
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class DetectedFace:
    track_id: int
    person_bbox: tuple[int, int, int, int]
    face_bbox: tuple[int, int, int, int]
    face_score: float
    embedding: list[float]


class FaceCropDetector:
    def __init__(self, model: InsightFaceModel, config: FaceDetectorConfig) -> None:
        self._model = model
        self._config = config

    def detect_faces_for_tracks(
        self,
        frame: np.ndarray,
        tracks: list[TrackedPersonInput],
    ) -> list[DetectedFace]:
        detections: list[DetectedFace] = []
        for track in tracks:
            cropped, offset_x, offset_y = self._crop_person(frame, track.bbox)
            if cropped.size == 0:
                continue

            faces = self._model.detect(cropped)
            best_face = self._choose_best_face(faces)
            if best_face is None:
                continue

            bbox = np.asarray(best_face.bbox, dtype=np.float32).tolist()
            x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
            face_bbox = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            detections.append(
                DetectedFace(
                    track_id=track.track_id,
                    person_bbox=track.bbox,
                    face_bbox=face_bbox,
                    face_score=float(getattr(best_face, "det_score", 0.0)),
                    embedding=self._model.normalize_embedding(best_face),
                )
            )
        return detections

    def detect_faces_in_image(self, image: np.ndarray) -> list[DetectedFace]:
        faces = self._model.detect(image)
        detected: list[DetectedFace] = []
        for index, face in enumerate(faces, start=1):
            if not self._passes_threshold(face):
                continue
            bbox = np.asarray(face.bbox, dtype=np.float32).tolist()
            x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
            detected.append(
                DetectedFace(
                    track_id=index,
                    person_bbox=(x1, y1, x2, y2),
                    face_bbox=(x1, y1, x2, y2),
                    face_score=float(getattr(face, "det_score", 0.0)),
                    embedding=self._model.normalize_embedding(face),
                )
            )
        return detected

    def _crop_person(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[np.ndarray, int, int]:
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        margin_x = int(round(width * self._config.crop_margin_ratio))
        margin_y = int(round(height * self._config.crop_margin_ratio))
        crop_x1 = max(0, x1 - margin_x)
        crop_y1 = max(0, y1 - margin_y)
        crop_x2 = min(frame_width, x2 + margin_x)
        crop_y2 = min(frame_height, y2 + margin_y)
        return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy(), crop_x1, crop_y1

    def _choose_best_face(self, faces: list[Any]) -> Any | None:
        candidates = [face for face in faces if self._passes_threshold(face)]
        if not candidates:
            return None
        return max(candidates, key=self._face_rank)

    def _passes_threshold(self, face: Any) -> bool:
        bbox = np.asarray(face.bbox, dtype=np.float32).tolist()
        x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
        if (x2 - x1) < self._config.min_face_box_size or (y2 - y1) < self._config.min_face_box_size:
            return False
        return float(getattr(face, "det_score", 0.0)) >= self._config.min_face_score

    def _face_rank(self, face: Any) -> tuple[float, float]:
        bbox = np.asarray(face.bbox, dtype=np.float32).tolist()
        x1, y1, x2, y2 = [int(round(value)) for value in bbox[:4]]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        score = float(getattr(face, "det_score", 0.0))
        return score, float(area)

