from __future__ import annotations

import copy
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2  # type: ignore
import numpy as np

from .detector import DetectorConfig, TrackedPerson, VisionDetector
from .direction import DirectionConfig, DirectionEngine


LOGGER = logging.getLogger("vision_service.tracker")
DEFAULT_TRACKER_CONFIG = str(Path(__file__).resolve().with_name("bytetrack.yaml"))


@dataclass(slots=True)
class VisionServiceConfig:
    source: str = "0"
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: float = 30.0
    ready_timeout_ms: int = 5000
    reconnect_delay_ms: int = 500
    model_path: str = "yolov8n.pt"
    tracker_config: str = DEFAULT_TRACKER_CONFIG
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    max_det: int = 100
    device: str | int | None = None
    half: bool | None = None
    max_processing_fps: float = 25.0
    line_position_fraction: float = 0.5
    line_deadband_px: int = 32
    direction_cooldown_ms: int = 1500
    state_ttl_ms: int = 5000
    stable_track_min_frames: int = 10
    show_debug: bool = False
    debug_window_name: str = "vision-service"


@dataclass(slots=True)
class VisionStatus:
    running: bool = False
    ready: bool = False
    source: str = "0"
    model_path: str = "yolov8n.pt"
    tracker_config: str = DEFAULT_TRACKER_CONFIG
    device: str = "cpu"
    frame_width: int = 0
    frame_height: int = 0
    line_x: int = 0
    entry_zone_max_x: int = 0
    exit_zone_min_x: int = 0
    fps: float = 0.0
    inference_ms: float = 0.0
    detected_people: int = 0
    tracked_people: int = 0
    frames_processed: int = 0
    last_timestamp: str | None = None
    last_error: str | None = None
    debug_window: bool = False


def parse_video_source(source: str) -> int | str:
    candidate = source.strip()
    if re.fullmatch(r"-?\d+", candidate):
        return int(candidate)
    return candidate


def open_video_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_config_from_env() -> VisionServiceConfig:
    return VisionServiceConfig(
        source=os.getenv("VISION_SOURCE", "0"),
        camera_width=int(os.getenv("VISION_CAMERA_WIDTH", "1280")),
        camera_height=int(os.getenv("VISION_CAMERA_HEIGHT", "720")),
        camera_fps=float(os.getenv("VISION_CAMERA_FPS", "30")),
        ready_timeout_ms=int(os.getenv("VISION_READY_TIMEOUT_MS", "5000")),
        reconnect_delay_ms=int(os.getenv("VISION_RECONNECT_DELAY_MS", "500")),
        model_path=os.getenv("VISION_MODEL_PATH", "yolov8n.pt"),
        tracker_config=os.getenv("VISION_TRACKER_CONFIG", DEFAULT_TRACKER_CONFIG),
        imgsz=int(os.getenv("VISION_IMGSZ", "640")),
        conf=float(os.getenv("VISION_CONF", "0.25")),
        iou=float(os.getenv("VISION_IOU", "0.5")),
        max_det=int(os.getenv("VISION_MAX_DET", "100")),
        device=os.getenv("VISION_DEVICE"),
        half=_parse_bool(os.getenv("VISION_HALF"), default=False) if os.getenv("VISION_HALF") is not None else None,
        max_processing_fps=float(os.getenv("VISION_MAX_PROCESS_FPS", "25")),
        line_position_fraction=float(os.getenv("VISION_LINE_POSITION_FRACTION", "0.5")),
        line_deadband_px=int(os.getenv("VISION_LINE_DEADBAND_PX", "32")),
        direction_cooldown_ms=int(os.getenv("VISION_DIRECTION_COOLDOWN_MS", "1500")),
        state_ttl_ms=int(os.getenv("VISION_STATE_TTL_MS", "5000")),
        stable_track_min_frames=int(os.getenv("VISION_STABLE_TRACK_MIN_FRAMES", "10")),
        show_debug=_parse_bool(os.getenv("VISION_SHOW_DEBUG"), default=False),
        debug_window_name=os.getenv("VISION_DEBUG_WINDOW_NAME", "vision-service"),
    )


class VisionService:
    def __init__(self, config: VisionServiceConfig) -> None:
        self._config = config
        self._source = parse_video_source(config.source)
        self._detector_config = DetectorConfig(
            model_path=config.model_path,
            tracker_config=config.tracker_config,
            imgsz=config.imgsz,
            conf=config.conf,
            iou=config.iou,
            max_det=config.max_det,
            device=config.device,
            half=config.half,
            frame_rate=max(int(round(config.camera_fps)) or 30, 1),
        )
        self._direction_engine = DirectionEngine(
            DirectionConfig(
                line_position_fraction=config.line_position_fraction,
                deadband_px=config.line_deadband_px,
                event_cooldown_ms=config.direction_cooldown_ms,
                state_ttl_ms=config.state_ttl_ms,
                stable_track_min_frames=config.stable_track_min_frames,
            )
        )
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._detector: VisionDetector | None = None
        self._latest_payload: dict[str, object] = {
            "timestamp": None,
            "tracks": [],
        }
        self._status = VisionStatus(
            source=str(config.source),
            model_path=config.model_path,
            tracker_config=config.tracker_config,
            debug_window=config.show_debug,
        )
        self._last_processed_at = 0.0
        self._last_processing_started_at = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._processing_loop, name="vision-service", daemon=True)
        self._thread.start()

        if not self._ready.wait(self._config.ready_timeout_ms / 1000):
            raise RuntimeError(self.get_status().get("last_error") or "Timed out waiting for the vision service.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._release_capture()
        if self._config.show_debug:
            try:
                cv2.destroyWindow(self._config.debug_window_name)
            except cv2.error:
                pass
        with self._lock:
            self._status.running = False
            self._status.ready = False

    def get_tracks(self) -> dict[str, object]:
        with self._lock:
            return copy.deepcopy(self._latest_payload)

    def get_status(self) -> dict[str, object]:
        with self._lock:
            return asdict(self._status)

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        if self._config.camera_width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.camera_width)
        if self._config.camera_height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.camera_height)
        if self._config.camera_fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self._config.camera_fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _processing_loop(self) -> None:
        try:
            self._detector = VisionDetector(self._detector_config)
            self._detector.warmup()
            with self._lock:
                self._status.device = str(self._detector.device)
                self._status.running = True
                self._status.last_error = None
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed to initialize vision detector: %s", exc)
            with self._lock:
                self._status.last_error = str(exc)
                self._status.running = False
            self._ready.set()
            return

        while not self._stop_event.is_set():
            capture = open_video_capture(self._source)
            self._configure_capture(capture)
            if not capture.isOpened():
                capture.release()
                error = f"Unable to open video source {self._source!r}."
                LOGGER.warning("%s", error)
                with self._lock:
                    self._status.last_error = error
                    self._status.ready = False
                self._ready.clear()
                self._stop_event.wait(self._config.reconnect_delay_ms / 1000)
                continue

            self._capture = capture
            if self._detector is not None:
                self._detector.reset()

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    error = f"Video source {self._source!r} stopped delivering frames."
                    LOGGER.warning("%s", error)
                    with self._lock:
                        self._status.last_error = error
                        self._status.ready = False
                    self._ready.clear()
                    break

                now_utc = datetime.now(UTC)
                timestamp_ms = int(now_utc.timestamp() * 1000)
                timestamp_iso = now_utc.isoformat()
                self._limit_processing_rate()
                inference_started = time.perf_counter()

                try:
                    assert self._detector is not None
                    detections, tracks = self._detector.detect_and_track(frame)
                    inference_ms = round((time.perf_counter() - inference_started) * 1000, 2)
                    direction_update = self._direction_engine.update(tracks, frame.shape[1], timestamp_ms)
                    payload = self._build_payload(timestamp_iso, timestamp_ms, tracks, direction_update)

                    if self._config.show_debug:
                        self._show_debug_frame(frame, tracks, direction_update)

                    self._update_metrics(
                        timestamp_iso=timestamp_iso,
                        frame_width=int(frame.shape[1]),
                        frame_height=int(frame.shape[0]),
                        line_x=direction_update.line_x,
                        entry_zone_max_x=direction_update.entry_zone_max_x,
                        exit_zone_min_x=direction_update.exit_zone_min_x,
                        detected_people=len(detections),
                        tracked_people=len(tracks),
                        inference_ms=inference_ms,
                        payload=payload,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Vision processing failed: %s", exc)
                    with self._lock:
                        self._status.last_error = str(exc)
                    self._stop_event.wait(0.05)

            capture.release()
            self._capture = None
            if not self._stop_event.is_set():
                self._stop_event.wait(self._config.reconnect_delay_ms / 1000)

        with self._lock:
            self._status.running = False
            self._status.ready = False

    def _update_metrics(
        self,
        *,
        timestamp_iso: str,
        frame_width: int,
        frame_height: int,
        line_x: int,
        entry_zone_max_x: int,
        exit_zone_min_x: int,
        detected_people: int,
        tracked_people: int,
        inference_ms: float,
        payload: dict[str, object],
    ) -> None:
        now = time.perf_counter()
        fps = self._status.fps
        if self._last_processed_at > 0:
            frame_interval = now - self._last_processed_at
            if frame_interval > 0:
                instant_fps = 1.0 / frame_interval
                fps = instant_fps if fps <= 0 else (fps * 0.8) + (instant_fps * 0.2)
        self._last_processed_at = now

        with self._lock:
            self._latest_payload = payload
            self._status.ready = True
            self._status.frame_width = frame_width
            self._status.frame_height = frame_height
            self._status.line_x = line_x
            self._status.entry_zone_max_x = entry_zone_max_x
            self._status.exit_zone_min_x = exit_zone_min_x
            self._status.detected_people = detected_people
            self._status.tracked_people = tracked_people
            self._status.inference_ms = inference_ms
            self._status.last_timestamp = timestamp_iso
            self._status.frames_processed += 1
            self._status.fps = round(fps, 2)
            self._status.last_error = None
        self._ready.set()

    def _build_payload(
        self,
        timestamp_iso: str,
        timestamp_ms: int,
        tracks: list[TrackedPerson],
        direction_update,
    ) -> dict[str, object]:
        return {
            "timestamp": timestamp_iso,
            "timestamp_ms": timestamp_ms,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "bbox": list(track.bbox),
                    "center": list(track.center),
                    "confidence": track.confidence,
                    "direction": direction_update.decisions.get(track.track_id).direction,
                    "zone": direction_update.decisions.get(track.track_id).zone,
                    "age_frames": direction_update.decisions.get(track.track_id).age_frames,
                    "stable": direction_update.decisions.get(track.track_id).stable,
                }
                for track in tracks
            ],
        }

    def _show_debug_frame(
        self,
        frame: np.ndarray,
        tracks: list[TrackedPerson],
        direction_update,
    ) -> None:
        annotated = frame.copy()
        cv2.line(
            annotated,
            (direction_update.entry_zone_max_x, 0),
            (direction_update.entry_zone_max_x, annotated.shape[0]),
            (0, 180, 255),
            2,
        )
        cv2.line(
            annotated,
            (direction_update.exit_zone_min_x, 0),
            (direction_update.exit_zone_min_x, annotated.shape[0]),
            (0, 180, 255),
            2,
        )
        cv2.rectangle(
            annotated,
            (direction_update.entry_zone_max_x, 0),
            (direction_update.exit_zone_min_x, annotated.shape[0]),
            (255, 220, 120),
            1,
        )
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            decision = direction_update.decisions.get(track.track_id)
            direction = decision.direction if decision else None
            zone = decision.zone if decision else "buffer"
            stable = decision.stable if decision else False
            age_frames = decision.age_frames if decision else 0
            color = (0, 200, 0) if direction == "ENTRY" else (0, 0, 220) if direction == "EXIT" else (255, 170, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{track.track_id} {track.confidence:.2f} {zone} f={age_frames}"
            if direction:
                label = f"{label} {direction}"
            if not stable:
                label = f"{label} warmup"
            cv2.putText(
                annotated,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(annotated, track.center, 4, color, -1)

        try:
            cv2.imshow(self._config.debug_window_name, annotated)
            cv2.waitKey(1)
        except cv2.error as exc:
            LOGGER.warning("Disabling debug window after OpenCV GUI error: %s", exc)
            self._config.show_debug = False
            with self._lock:
                self._status.debug_window = False

    def _limit_processing_rate(self) -> None:
        if self._config.max_processing_fps <= 0:
            return

        min_interval = 1.0 / self._config.max_processing_fps
        now = time.perf_counter()
        if self._last_processing_started_at > 0:
            remaining = min_interval - (now - self._last_processing_started_at)
            if remaining > 0:
                time.sleep(remaining)
                now = time.perf_counter()
        self._last_processing_started_at = now

    def _release_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
