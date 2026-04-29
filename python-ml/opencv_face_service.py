#!/usr/bin/env python3
"""Persistent OpenCV face worker for low-latency gate verification."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore

DATA_URL_PATTERN = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.+)$")


@dataclass
class LabelRecord:
    id: int
    folder_name: str
    display_name: str
    employee_code: str | None
    department: str | None
    rfid_uid: str | None
    sample_count: int
    included_in_training: bool


@dataclass
class FramePrediction:
    label: LabelRecord | None
    distance: float | None
    box: tuple[int, int, int, int] | None
    center_x: float | None
    area_ratio: float | None


@dataclass
class LiveRecognitionFace:
    label: LabelRecord | None
    distance: float | None
    box: tuple[int, int, int, int]
    confidence: float
    verified: bool
    area_ratio: float
    center_offset: float


@dataclass
class CapturedCameraFrame:
    frame: np.ndarray
    timestamp_ms: int
    sequence: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent OpenCV face verification worker.")
    parser.add_argument("--model", required=True, type=Path, help="Path to lbph-model.yml")
    parser.add_argument("--labels", required=True, type=Path, help="Path to lbph-labels.json")
    parser.add_argument("--distance-threshold", type=float, help="Override the threshold from the label map.")
    parser.add_argument("--resize-width", type=int, default=800, help="Resize frames before detection. Use 0 to disable.")
    parser.add_argument("--scale-factor", type=float, default=1.08, help="Haar cascade scale factor.")
    parser.add_argument("--min-neighbors", type=int, default=6, help="Haar cascade minNeighbors.")
    parser.add_argument("--min-face-size", type=int, default=52, help="Minimum face size in pixels.")
    parser.add_argument("--camera-source", default="0", help="OpenCV camera source. Use 0 for default webcam or RTSP URL.")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested camera width for live capture.")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested camera height for live capture.")
    parser.add_argument("--camera-fps", type=float, default=20.0, help="Requested camera FPS for live capture.")
    parser.add_argument("--camera-ready-timeout-ms", type=int, default=2500, help="How long to wait for the first live frame.")
    parser.add_argument("--camera-reconnect-delay-ms", type=int, default=500, help="Delay before reconnecting a dropped capture.")
    parser.add_argument("--frame-freshness-ms", type=int, default=1500, help="Maximum acceptable age for a live frame.")
    return parser.parse_args()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def round_float(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None

    rounded = float(np.round(value, digits))
    return rounded if math.isfinite(rounded) else None


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]

    return value


def create_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")
    return detector



def create_eye_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye_tree_eyeglasses.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load eye cascade: {cascade_path}")
    return detector


def parse_camera_source(source: str) -> int | str:
    candidate = source.strip()
    if re.fullmatch(r"-?\d+", candidate):
        return int(candidate)
    return candidate


def open_camera_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture
        capture.release()
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)


class LatestFrameCamera:
    def __init__(self, args: argparse.Namespace) -> None:
        self._source = parse_camera_source(str(args.camera_source))
        self._camera_width = int(args.camera_width)
        self._camera_height = int(args.camera_height)
        self._camera_fps = float(args.camera_fps)
        self._ready_timeout_ms = int(args.camera_ready_timeout_ms)
        self._reconnect_delay_ms = int(args.camera_reconnect_delay_ms)
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp_ms = 0
        self._latest_sequence = 0
        self._last_error: str | None = None

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        if self._camera_width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._camera_width)
        if self._camera_height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._camera_height)
        if self._camera_fps > 0:
            capture.set(cv2.CAP_PROP_FPS, self._camera_fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._reader_loop, name="opencv-face-camera", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._ready_timeout_ms / 1000):
            raise RuntimeError(self._last_error or "Timed out waiting for the camera to provide a live frame.")

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._release_capture()

    def _release_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            capture = open_camera_capture(self._source)
            self._configure_capture(capture)
            if not capture.isOpened():
                self._last_error = f"Unable to open OpenCV camera source {self._source!r}."
                capture.release()
                self._ready.clear()
                self._stop_event.wait(self._reconnect_delay_ms / 1000)
                continue

            self._capture = capture
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    self._last_error = f"Camera source {self._source!r} stopped delivering frames."
                    self._ready.clear()
                    break

                timestamp_ms = int(time.time() * 1000)
                with self._condition:
                    self._latest_frame = frame.copy()
                    self._latest_timestamp_ms = timestamp_ms
                    self._latest_sequence += 1
                    self._last_error = None
                    self._ready.set()
                    self._condition.notify_all()

            capture.release()
            self._capture = None
            if not self._stop_event.is_set():
                self._stop_event.wait(self._reconnect_delay_ms / 1000)

    def _snapshot_latest_locked(self) -> CapturedCameraFrame | None:
        if self._latest_frame is None or self._latest_sequence <= 0:
            return None
        return CapturedCameraFrame(
            frame=self._latest_frame.copy(),
            timestamp_ms=self._latest_timestamp_ms,
            sequence=self._latest_sequence,
        )

    def snapshot_latest(self) -> CapturedCameraFrame | None:
        with self._condition:
            return self._snapshot_latest_locked()

    def wait_for_next_frame(self, after_sequence: int, timeout_ms: int) -> CapturedCameraFrame | None:
        timeout_seconds = max(timeout_ms, 1) / 1000
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while not self._stop_event.is_set():
                snapshot = self._snapshot_latest_locked()
                if snapshot is not None and snapshot.sequence > after_sequence:
                    return snapshot
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
        return None

    def _is_fresh_snapshot(
        self,
        snapshot: CapturedCameraFrame | None,
        trigger_timestamp_ms: int,
        freshness_ms: int,
    ) -> bool:
        if snapshot is None:
            return False

        now_ms = int(time.time() * 1000)
        return (
            snapshot.timestamp_ms >= trigger_timestamp_ms - freshness_ms
            and now_ms - snapshot.timestamp_ms <= freshness_ms
        )

    def capture_latest(
        self,
        trigger_timestamp_ms: int,
        freshness_ms: int,
        timeout_ms: int,
    ) -> CapturedCameraFrame | None:
        self.start()

        latest = self.snapshot_latest()
        if self._is_fresh_snapshot(latest, trigger_timestamp_ms, freshness_ms):
            return latest

        last_sequence = latest.sequence if latest is not None else 0

        deadline = time.monotonic() + max(timeout_ms, 1) / 1000
        while True:
            remaining_ms = int(max(0.0, (deadline - time.monotonic()) * 1000))
            if remaining_ms <= 0:
                return None

            next_frame = self.wait_for_next_frame(last_sequence, remaining_ms)
            if next_frame is None:
                return None

            last_sequence = next_frame.sequence
            if self._is_fresh_snapshot(next_frame, trigger_timestamp_ms, freshness_ms):
                return next_frame


def is_probable_face(
    gray_frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    eye_detector: cv2.CascadeClassifier,
) -> bool:
    x, y, width, height = face_box
    if width <= 0 or height <= 0:
        return False

    frame_area = float(gray_frame.shape[0] * gray_frame.shape[1])
    area_ratio = (width * height) / max(frame_area, 1.0)
    if area_ratio < 0.01 or area_ratio > 0.45:
        return False

    aspect_ratio = width / float(height)
    if aspect_ratio < 0.6 or aspect_ratio > 1.6:
        return False

    roi = gray_frame[y : y + height, x : x + width]
    min_eye_size = max(12, int(min(width, height) * 0.18))
    eyes = (
        eye_detector.detectMultiScale(
            roi,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(min_eye_size, min_eye_size),
        )
        if not eye_detector.empty()
        else []
    )
    if len(eyes) >= 1:
        return True

    laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
    return 0.82 <= aspect_ratio <= 1.32 and 0.02 <= area_ratio <= 0.32 and laplacian_var >= 42.0


def load_labels(path: Path) -> tuple[dict[int, LabelRecord], dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    labels = {
        int(item["id"]): LabelRecord(
            id=int(item["id"]),
            folder_name=item.get("folderName") or f"label-{item['id']}",
            display_name=item.get("displayName") or item.get("folderName") or f"label-{item['id']}",
            employee_code=item.get("employeeCode"),
            department=item.get("department"),
            rfid_uid=item.get("rfidUid"),
            sample_count=int(item.get("sampleCount") or 0),
            included_in_training=bool(item.get("includedInTraining", True)),
        )
        for item in payload.get("labels", [])
    }
    return labels, payload


def decode_frame(frame_payload: str) -> np.ndarray:
    match = DATA_URL_PATTERN.match(frame_payload)
    encoded = match.group(1) if match else frame_payload
    image_bytes = base64.b64decode(encoded)
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode gate frame.")
    return image


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    mean_intensity = float(np.mean(v_channel)) / 255.0
    if mean_intensity > 0:
        gamma = math.log(0.55) / math.log(max(mean_intensity, 1e-3))
        gamma = float(np.clip(gamma, 0.6, 1.6))
        v_gamma = np.power(v_channel / 255.0, gamma)
        v_channel = np.clip(v_gamma * 255.0, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v_channel)
    hsv[:, :, 2] = v_eq
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def resize_frame(frame: np.ndarray, resize_width: int) -> tuple[np.ndarray, float]:
    if resize_width <= 0:
        return frame, 1.0

    height, width = frame.shape[:2]
    if width <= resize_width:
        return frame, 1.0

    scale_factor = resize_width / float(width)
    resized_height = max(1, int(round(height * scale_factor)))
    resized = cv2.resize(frame, (resize_width, resized_height))
    return resized, scale_factor


def detect_faces(
    gray_frame: np.ndarray,
    detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
    args: argparse.Namespace,
) -> list[tuple[int, int, int, int]]:
    def run(scale_factor: float, min_face: int) -> list[tuple[int, int, int, int]]:
        faces = detector.detectMultiScale(
            gray_frame,
            scaleFactor=scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(min_face, min_face),
        )
        sorted_faces = sorted(
            [
                (int(x), int(y), int(width), int(height))
                for x, y, width, height in faces
            ],
            key=lambda item: item[2] * item[3],
            reverse=True,
        )
        if sorted_faces and not eye_detector.empty():
            plausible = [
                face_box
                for face_box in sorted_faces
                if is_probable_face(gray_frame, face_box, eye_detector)
            ]
            if plausible:
                return plausible
        return sorted_faces

    primary = run(args.scale_factor, args.min_face_size)
    if primary:
        return primary

    fallback_size = max(24, int(args.min_face_size * 0.7))
    fallback_scale = max(1.02, min(args.scale_factor - 0.02, 1.12))
    return run(fallback_scale, fallback_size)



def prepare_face_crop(
    gray_frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    image_size: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], float, float]:
    x, y, width, height = face_box
    padding_x = int(width * 0.15)
    padding_y = int(height * 0.15)
    start_x = max(0, x - padding_x)
    start_y = max(0, y - padding_y)
    end_x = min(gray_frame.shape[1], x + width + padding_x)
    end_y = min(gray_frame.shape[0], y + height + padding_y)
    crop = gray_frame[start_y:end_y, start_x:end_x]
    resized = cv2.resize(crop, (image_size, image_size))
    equalized = cv2.equalizeHist(resized)
    top = start_y
    right = end_x
    bottom = end_y
    left = start_x
    center_x = (left + right) / 2 / max(1, gray_frame.shape[1])
    area_ratio = ((right - left) * (bottom - top)) / max(1, gray_frame.shape[0] * gray_frame.shape[1])
    return equalized, (top, right, bottom, left), center_x, area_ratio


def scale_box(box: tuple[int, int, int, int], scale_factor: float) -> tuple[int, int, int, int]:
    if scale_factor == 1.0:
        return box

    x, y, width, height = box
    return (
        int(round(x / scale_factor)),
        int(round(y / scale_factor)),
        int(round(width / scale_factor)),
        int(round(height / scale_factor)),
    )


def predict_face_box(
    gray_frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    recognizer: Any,
    labels: dict[int, LabelRecord],
    image_size: int,
    distance_threshold: float,
) -> FramePrediction:
    prepared_face, output_box, center_x, area_ratio = prepare_face_crop(gray_frame, face_box, image_size)
    predicted_label_id, distance = recognizer.predict(prepared_face)
    label = labels.get(int(predicted_label_id))
    if label is None or not label.included_in_training or float(distance) > distance_threshold:
        return FramePrediction(
            label=None,
            distance=float(distance),
            box=output_box,
            center_x=center_x,
            area_ratio=area_ratio,
        )

    return FramePrediction(
        label=label,
        distance=float(distance),
        box=output_box,
        center_x=center_x,
        area_ratio=area_ratio,
    )


def predict_frame(
    frame_payload: str,
    recognizer: Any,
    detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
    labels: dict[int, LabelRecord],
    image_size: int,
    distance_threshold: float,
    args: argparse.Namespace,
) -> FramePrediction:
    image = normalize_lighting(decode_frame(frame_payload))
    working_frame, scale_factor = resize_frame(image, args.resize_width)
    gray_working = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
    detected_faces = detect_faces(gray_working, detector, eye_detector, args)
    if not detected_faces:
        return FramePrediction(label=None, distance=None, box=None, center_x=None, area_ratio=None)

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    best_prediction: FramePrediction | None = None
    for face_box in detected_faces[:5]:
        scaled_box = scale_box(face_box, scale_factor)
        prediction = predict_face_box(
            gray_frame,
            scaled_box,
            recognizer,
            labels,
            image_size,
            distance_threshold,
        )
        if prediction.box is None:
            continue
        if best_prediction is None:
            best_prediction = prediction
            continue
        if prediction.label is not None and best_prediction.label is None:
            best_prediction = prediction
            continue
        prediction_distance = (
            prediction.distance if prediction.distance is not None else float("inf")
        )
        best_distance = (
            best_prediction.distance if best_prediction.distance is not None else float("inf")
        )
        if prediction_distance < best_distance:
            best_prediction = prediction

    return best_prediction or FramePrediction(label=None, distance=None, box=None, center_x=None, area_ratio=None)


def infer_direction(
    predictions: list[FramePrediction],
    entry_horizontal_direction: str,
    entry_depth_direction: str,
) -> tuple[str, str, float]:
    movement_points = [
        prediction
        for prediction in predictions
        if prediction.box is not None and prediction.center_x is not None and prediction.area_ratio is not None
    ]
    if len(movement_points) < 3:
        return "UNKNOWN", "none", 0.0

    window_size = 1 if len(movement_points) < 4 else min(2, len(movement_points) // 2)
    start_points = movement_points[:window_size]  # type: ignore
    end_points = movement_points[-window_size:]  # type: ignore
    start_center_x = float(np.mean([point.center_x for point in start_points]))
    end_center_x = float(np.mean([point.center_x for point in end_points]))
    start_area = float(np.mean([point.area_ratio for point in start_points]))
    end_area = float(np.mean([point.area_ratio for point in end_points]))
    horizontal_delta = end_center_x - start_center_x
    depth_delta = end_area - start_area

    horizontal_steps = [
        (movement_points[index + 1].center_x or 0.0) - (movement_points[index].center_x or 0.0)
        for index in range(len(movement_points) - 1)
    ]
    depth_steps = [
        (movement_points[index + 1].area_ratio or 0.0) - (movement_points[index].area_ratio or 0.0)
        for index in range(len(movement_points) - 1)
    ]
    horizontal_consistency = (
        sum(
            1
            for step in horizontal_steps
            if step != 0 and np.sign(step) == np.sign(horizontal_delta or 1)
        )
        / max(1, len(horizontal_steps))
    )
    depth_consistency = (
        sum(
            1
            for step in depth_steps
            if step != 0 and np.sign(step) == np.sign(depth_delta or 1)
        )
        / max(1, len(depth_steps))
    )

    horizontal_travel = abs(horizontal_delta)
    depth_travel = abs(depth_delta)
    horizontal_confidence = clamp(((horizontal_travel - 0.04) / 0.14) * 0.7 + horizontal_consistency * 0.3, 0.0, 1.0)
    depth_confidence = clamp(((depth_travel - 0.006) / 0.035) * 0.7 + depth_consistency * 0.3, 0.0, 1.0)

    if horizontal_travel >= 0.04 and horizontal_confidence >= depth_confidence and horizontal_consistency >= 0.55:
        moving_entry = horizontal_delta > 0 if entry_horizontal_direction == "left-to-right" else horizontal_delta < 0
        return ("ENTRY" if moving_entry else "EXIT", "horizontal", round_float(horizontal_confidence))

    if depth_travel >= 0.006 and depth_consistency >= 0.55:
        moving_entry = depth_delta > 0 if entry_depth_direction == "approaching" else depth_delta < 0
        return ("ENTRY" if moving_entry else "EXIT", "depth", round_float(depth_confidence))

    return "UNKNOWN", "none", round_float(max(horizontal_confidence, depth_confidence))


def build_response(
    predictions: list[FramePrediction],
    distance_threshold: float,
    entry_horizontal_direction: str,
    entry_depth_direction: str,
) -> dict[str, Any]:
    verified_predictions = [prediction for prediction in predictions if prediction.label is not None]
    direction, axis, direction_confidence = infer_direction(
        predictions,
        entry_horizontal_direction,
        entry_depth_direction,
    )

    if not verified_predictions:
        best_distance = min(
            (prediction.distance for prediction in predictions if prediction.distance is not None),
            default=None,
        )
        return {
            "verified": False,
            "employee": None,
            "matchConfidence": 0.0,
            "bestDistance": round_float(best_distance, 3) if best_distance is not None else None,
            "distanceThreshold": distance_threshold,
            "movementDirection": direction,
            "movementAxis": axis,
            "movementConfidence": direction_confidence,
            "framesProcessed": len(predictions),
            "framesWithFace": len([prediction for prediction in predictions if prediction.box is not None]),
            "bestBox": None,
        }

    vote_counts = Counter(prediction.label.folder_name for prediction in verified_predictions if prediction.label is not None)  # type: ignore
    best_folder_name = vote_counts.most_common(1)[0][0]
    chosen_predictions = [
        prediction for prediction in verified_predictions if prediction.label and prediction.label.folder_name == best_folder_name  # type: ignore
    ]
    best_prediction = min(
        chosen_predictions,
        key=lambda prediction: prediction.distance if prediction.distance is not None else float("inf"),
    )
    average_distance = float(np.mean([prediction.distance for prediction in chosen_predictions if prediction.distance is not None]))
    match_confidence = clamp(1.0 - (average_distance / max(distance_threshold, 1.0)), 0.0, 1.0)

    best_employee = best_prediction.label
    best_box = best_prediction.box

    return {
        "verified": True,
        "employee": {
            "folderName": best_employee.folder_name,  # type: ignore
            "displayName": best_employee.display_name,  # type: ignore
            "employeeCode": best_employee.employee_code,  # type: ignore
            "department": best_employee.department,  # type: ignore
            "rfidUid": best_employee.rfid_uid,  # type: ignore
            "sampleCount": best_employee.sample_count,  # type: ignore
        },
        "matchConfidence": round_float(match_confidence),
        "bestDistance": round_float(
            best_prediction.distance if best_prediction.distance is not None else average_distance,
            3,
        ),
        "distanceThreshold": distance_threshold,
        "movementDirection": direction,
        "movementAxis": axis,
        "movementConfidence": direction_confidence,
        "framesProcessed": len(predictions),
        "framesWithFace": len([prediction for prediction in predictions if prediction.box is not None]),
        "bestBox": {
            "top": best_box[0],
            "right": best_box[1],
            "bottom": best_box[2],
            "left": best_box[3],
        } if best_box is not None else None,
    }


def recognize_faces_in_image(
    image: np.ndarray,
    recognizer: Any,
    detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
    labels: dict[int, LabelRecord],
    distance_threshold: float,
    image_size: int,
    args: argparse.Namespace,
    max_faces: int,
) -> tuple[list[LiveRecognitionFace], int, int]:
    normalized = normalize_lighting(image)
    frame_height, frame_width = normalized.shape[:2]
    working_frame, scale_factor = resize_frame(normalized, args.resize_width)
    gray_working = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    detected_faces = detect_faces(gray_working, detector, eye_detector, args)

    recognized_faces: list[LiveRecognitionFace] = []
    for face_box in detected_faces[: max(1, min(max_faces, 50))]:
        scaled_box = scale_box(face_box, scale_factor)
        prediction = predict_face_box(
            gray_frame,
            scaled_box,
            recognizer,
            labels,
            image_size,
            distance_threshold,
        )
        if prediction.box is None:
            continue

        raw_distance: float | None = (
            float(prediction.distance) if prediction.distance is not None else None
        )
        distance: float | None = raw_distance if raw_distance is not None and math.isfinite(raw_distance) else None
        confidence = (
            clamp(1.0 - (distance / max(distance_threshold, 1.0)), 0.0, 1.0)
            if distance is not None
            else 0.0
        )
        center_x = float(prediction.center_x) if prediction.center_x is not None else 0.5
        area_ratio = float(prediction.area_ratio) if prediction.area_ratio is not None else 0.0
        recognized_faces.append(
            LiveRecognitionFace(
                label=prediction.label,
                distance=distance,
                box=prediction.box,
                confidence=confidence,
                verified=prediction.label is not None,
                area_ratio=area_ratio,
                center_offset=abs(center_x - 0.5),
            )
        )

    return recognized_faces, int(frame_width), int(frame_height)


def choose_best_live_face(faces: list[LiveRecognitionFace]) -> LiveRecognitionFace | None:
    if not faces:
        return None

    return max(
        faces,
        key=lambda face: (
            1 if face.verified else 0,
            face.confidence,
            face.area_ratio,
            -face.center_offset,
        ),
    )


def recognize_face(
    frame: np.ndarray,
    recognizer: Any,
    detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
    labels: dict[int, LabelRecord],
    distance_threshold: float,
    image_size: int,
    args: argparse.Namespace,
    max_faces: int,
) -> dict[str, Any]:
    faces, frame_width, frame_height = recognize_faces_in_image(
        frame,
        recognizer,
        detector,
        eye_detector,
        labels,
        distance_threshold,
        image_size,
        args,
        max_faces,
    )
    best_face = choose_best_live_face(faces)

    if best_face is None:
        return {
            "name": None,
            "confidence": 0.0,
            "status": "NO_FACE",
            "employeeCode": None,
            "department": None,
            "rfidUid": None,
            "facesDetected": 0,
            "multipleFaces": False,
            "bestBox": None,
            "frameWidth": frame_width,
            "frameHeight": frame_height,
        }

    recognized_name = best_face.label.display_name if best_face.label is not None else None  # type: ignore
    recognized_employee_code = best_face.label.employee_code if best_face.label is not None else None  # type: ignore
    recognized_department = best_face.label.department if best_face.label is not None else None  # type: ignore
    recognized_rfid_uid = best_face.label.rfid_uid if best_face.label is not None else None  # type: ignore
    is_unknown = not best_face.verified or recognized_name is None or best_face.confidence < 0.45

    return {
        "name": "UNKNOWN" if is_unknown else recognized_name,
        "confidence": round_float(best_face.confidence) or 0.0,
        "status": "UNKNOWN" if is_unknown else "MATCH",
        "employeeCode": None if is_unknown else recognized_employee_code,
        "department": None if is_unknown else recognized_department,
        "rfidUid": None if is_unknown else recognized_rfid_uid,
        "facesDetected": len(faces),
        "multipleFaces": len(faces) > 1,
        "bestBox": {
            "top": best_face.box[0],
            "right": best_face.box[1],
            "bottom": best_face.box[2],
            "left": best_face.box[3],
        },
        "frameWidth": frame_width,
        "frameHeight": frame_height,
    }


def select_best_live_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None

    status_priority = {
        "MATCH": 2,
        "UNKNOWN": 1,
        "NO_FACE": 0,
    }
    return max(
        results,
        key=lambda result: (
            status_priority.get(str(result.get("status")), -1),
            float(result.get("confidence") or 0.0),
            int(result.get("facesDetected") or 0),
        ),
    )


def handle_verify_burst(
    request: dict[str, Any],
    recognizer: Any,
    detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
    labels: dict[int, LabelRecord],
    distance_threshold: float,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frames = request.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("No frames were provided for verification.")

    predictions = [
        predict_frame(str(frame_payload), recognizer, detector, eye_detector, labels, image_size, distance_threshold, args)
        for frame_payload in frames
    ]

    return build_response(
        predictions,
        distance_threshold,
        str(request.get("entryHorizontalDirection", "left-to-right")),
        str(request.get("entryDepthDirection", "approaching")),
    )


def handle_recognize_frame(
    request: dict[str, Any],
    recognizer: Any,
    detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
    labels: dict[int, LabelRecord],
    distance_threshold: float,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_payload = request.get("frame")
    if not isinstance(frame_payload, str) or not frame_payload.strip():
        raise ValueError("No frame was provided for live recognition.")

    max_faces = int(request.get("maxFaces", 50) or 50)
    recognized_faces, frame_width, frame_height = recognize_faces_in_image(
        decode_frame(str(frame_payload)),
        recognizer,
        detector,
        eye_detector,
        labels,
        distance_threshold,
        image_size,
        args,
        max_faces,
    )

    return {
        "faces": [
            {
                "label": face.label.display_name if face.label is not None else "Unknown Face",  # type: ignore
                "employeeCode": face.label.employee_code if face.label is not None else None,  # type: ignore
                "department": face.label.department if face.label is not None else None,  # type: ignore
                "rfidUid": face.label.rfid_uid if face.label is not None else None,  # type: ignore
                "verified": face.verified,
                "confidence": round_float(face.confidence),
                "distance": round_float(face.distance, 3) if face.distance is not None else None,  # type: ignore
                "box": {
                    "top": face.box[0],
                    "right": face.box[1],
                    "bottom": face.box[2],
                    "left": face.box[3],
                },
            }
            for face in recognized_faces
        ],
        "frameWidth": int(frame_width),
        "frameHeight": int(frame_height),
    }


def handle_recognize_live_camera(
    request: dict[str, Any],
    camera: LatestFrameCamera,
    recognizer: Any,
    detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
    labels: dict[int, LabelRecord],
    distance_threshold: float,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trigger_timestamp_ms = int(request.get("timestamp") or 0)
    if trigger_timestamp_ms <= 0:
        raise ValueError("RFID trigger timestamp is required for live camera recognition.")

    frame_count = max(1, min(int(request.get("frameCount") or 3), 3))
    max_faces = max(1, min(int(request.get("maxFaces") or 5), 10))
    freshness_ms = max(200, int(request.get("freshnessMs") or args.frame_freshness_ms))
    captured_frame = camera.capture_latest(
        trigger_timestamp_ms=trigger_timestamp_ms,
        freshness_ms=freshness_ms,
        timeout_ms=max(args.camera_ready_timeout_ms, freshness_ms),
    )
    if captured_frame is None:
        raise RuntimeError("No fresh live camera frame was available after the RFID trigger.")

    # Keep RFID recognition event-driven: one snapped frame, repeated recognition passes.
    results = [
        recognize_face(
            captured_frame.frame,
            recognizer,
            detector,
            eye_detector,
            labels,
            distance_threshold,
            image_size,
            args,
            max_faces,
        )
        for _ in range(frame_count)
    ]
    best_result = select_best_live_result(results)
    if best_result is None:
        raise RuntimeError("Live face recognition returned no result.")

    timestamp_delta_ms = abs(captured_frame.timestamp_ms - trigger_timestamp_ms)
    return {
        **best_result,
        "timestamp": captured_frame.timestamp_ms,
        "rfidTimestamp": trigger_timestamp_ms,
        "timestampDeltaMs": timestamp_delta_ms,
        "frameCount": frame_count,
        "frameLatencyMs": timestamp_delta_ms,
    }


def emit_response(payload: dict[str, Any]) -> None:
    safe_payload = sanitize_json_value(payload)
    sys.stdout.write(json.dumps(safe_payload, allow_nan=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    args = parse_args()
    if not hasattr(cv2, "face"):
        print("OpenCV face module is unavailable.", file=sys.stderr)
        return 1

    labels, labels_payload = load_labels(args.labels)
    distance_threshold = (
        float(args.distance_threshold)
        if args.distance_threshold is not None
        else float(labels_payload.get("threshold", 65.0))
    )
    image_size = int(labels_payload.get("imageSize", 200))
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(args.model.resolve()))
    detector = create_detector()
    eye_detector = create_eye_detector()
    camera = LatestFrameCamera(args)

    try:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            request_id: str | None = None
            try:
                request = json.loads(line)
                request_id = request.get("requestId")
                action = request.get("action", "verify_burst")
                if action == "verify_burst":
                    result = handle_verify_burst(
                        request,
                        recognizer,
                        detector,
                        eye_detector,
                        labels,
                        distance_threshold,
                        image_size,
                        args,
                    )
                elif action == "recognize_frame":
                    result = handle_recognize_frame(
                        request,
                        recognizer,
                        detector,
                        eye_detector,
                        labels,
                        distance_threshold,
                        image_size,
                        args,
                    )
                elif action == "recognize_live_camera":
                    result = handle_recognize_live_camera(
                        request,
                        camera,
                        recognizer,
                        detector,
                        eye_detector,
                        labels,
                        distance_threshold,
                        image_size,
                        args,
                    )
                else:
                    raise ValueError(f"Unsupported worker action: {action}")
                emit_response({
                    "requestId": request_id,
                    "ok": True,
                    "result": result,
                })
            except Exception as error:  # noqa: BLE001
                emit_response({
                    "requestId": request_id,
                    "ok": False,
                    "error": str(error),
                })
    finally:
        camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())







