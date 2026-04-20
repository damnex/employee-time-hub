from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None

try:
    from ultralytics import YOLO  # type: ignore
    from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore
except ImportError:  # pragma: no cover
    YOLO = None
    BYTETracker = None


LOGGER = logging.getLogger("vision_service.detector")
PERSON_CLASS_ID = 0


@dataclass(slots=True)
class DetectorConfig:
    model_path: str = "yolov8n.pt"
    tracker_config: str = str(Path(__file__).resolve().with_name("bytetrack.yaml"))
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    max_det: int = 100
    device: str | int | None = None
    half: bool | None = None
    frame_rate: int = 30


@dataclass(slots=True)
class PersonDetection:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(slots=True)
class TrackedPerson:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float


def _normalize_device(device: str | int | None) -> str | int:
    if isinstance(device, str):
        candidate = device.strip()
        if not candidate:
            device = None
        elif candidate.isdigit():
            return int(candidate)
        else:
            return candidate

    if device is not None:
        return device

    if torch is not None and torch.cuda.is_available():
        return 0
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _should_use_half(device: str | int, requested: bool | None) -> bool:
    if requested is not None:
        return requested
    if isinstance(device, int):
        return True
    return str(device).startswith("cuda")


class VisionDetector:
    def __init__(self, config: DetectorConfig) -> None:
        if YOLO is None or BYTETracker is None:  # pragma: no cover
            raise RuntimeError(
                "Ultralytics is required for the vision service. Install vision_service/requirements.txt first."
            )
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required for the vision service tracker configuration.")

        self._config = config
        self._device = _normalize_device(config.device)
        self._use_half = _should_use_half(self._device, config.half)
        self._model = YOLO(config.model_path)
        self._tracker = BYTETracker(self._load_tracker_args(config.tracker_config), frame_rate=max(config.frame_rate, 1))
        LOGGER.info(
            "Loaded YOLO model %s on device %s with ByteTrack config %s.",
            config.model_path,
            self._device,
            config.tracker_config,
        )

    @property
    def device(self) -> str | int:
        return self._device

    def warmup(self) -> None:
        dummy = np.zeros((self._config.imgsz, self._config.imgsz, 3), dtype=np.uint8)
        self.detect_and_track(dummy)

    def reset(self) -> None:
        self._tracker.reset()

    def detect_and_track(self, frame: np.ndarray) -> tuple[list[PersonDetection], list[TrackedPerson]]:
        results = self._model.predict(
            source=[frame],
            classes=[PERSON_CLASS_ID],
            conf=self._config.conf,
            iou=self._config.iou,
            imgsz=self._config.imgsz,
            max_det=self._config.max_det,
            agnostic_nms=True,
            device=self._device,
            half=self._use_half,
            verbose=False,
        )
        if not results:
            return [], []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            tracked = self._tracker.update(boxes, result.orig_img, None) if boxes is not None else np.empty((0, 8))
            return [], self._parse_tracks(tracked)

        detections = self._parse_detections(boxes)
        tracked = self._tracker.update(boxes, result.orig_img, None)
        return detections, self._parse_tracks(tracked)

    def _load_tracker_args(self, tracker_config_path: str) -> SimpleNamespace:
        config_path = Path(tracker_config_path)
        if not config_path.exists():
            raise RuntimeError(f"ByteTrack config file not found: {tracker_config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        if payload.get("tracker_type") != "bytetrack":
            raise RuntimeError("vision_service only supports ByteTrack tracker_type=bytetrack.")

        return SimpleNamespace(**payload)

    def _parse_detections(self, boxes: Any) -> list[PersonDetection]:
        xyxy = boxes.xyxy.int().cpu().tolist()
        confidences = boxes.conf.float().cpu().tolist() if boxes.conf is not None else [0.0] * len(xyxy)
        classes = boxes.cls.int().cpu().tolist() if boxes.cls is not None else [PERSON_CLASS_ID] * len(xyxy)

        detections: list[PersonDetection] = []
        for bbox, confidence, class_id in zip(xyxy, confidences, classes):
            if int(class_id) != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = [int(value) for value in bbox]
            detections.append(
                PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=round(float(confidence), 4),
                )
            )
        return detections

    def _parse_tracks(self, tracked: np.ndarray) -> list[TrackedPerson]:
        if tracked.size == 0:
            return []

        parsed_tracks: list[TrackedPerson] = []
        for row in tracked.tolist():
            if len(row) < 7:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in row[:4]]
            track_id = int(row[4])
            confidence = round(float(row[5]), 4)
            class_id = int(row[6])
            if class_id != PERSON_CLASS_ID:
                continue
            center = (int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2)))
            parsed_tracks.append(
                TrackedPerson(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    confidence=confidence,
                )
            )
        return parsed_tracks
