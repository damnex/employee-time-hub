#!/usr/bin/env python3
"""Persistent OpenCV face worker for low-latency gate verification."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent OpenCV face verification worker.")
    parser.add_argument("--model", required=True, type=Path, help="Path to lbph-model.yml")
    parser.add_argument("--labels", required=True, type=Path, help="Path to lbph-labels.json")
    parser.add_argument("--distance-threshold", type=float, help="Override the threshold from the label map.")
    parser.add_argument("--resize-width", type=int, default=640, help="Resize frames before detection. Use 0 to disable.")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Haar cascade scale factor.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Haar cascade minNeighbors.")
    parser.add_argument("--min-face-size", type=int, default=72, help="Minimum face size in pixels.")
    return parser.parse_args()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def create_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")
    return detector


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
    args: argparse.Namespace,
) -> list[tuple[int, int, int, int]]:
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=args.scale_factor,
        minNeighbors=args.min_neighbors,
        minSize=(args.min_face_size, args.min_face_size),
    )
    if len(faces) == 0:
        return []

    return sorted(
        [
            (int(x), int(y), int(width), int(height))
            for x, y, width, height in faces
        ],
        key=lambda item: item[2] * item[3],
        reverse=True,
    )


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
    labels: dict[int, LabelRecord],
    image_size: int,
    distance_threshold: float,
    args: argparse.Namespace,
) -> FramePrediction:
    image = decode_frame(frame_payload)
    working_frame, scale_factor = resize_frame(image, args.resize_width)
    gray_working = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
    detected_faces = detect_faces(gray_working, detector, args)
    if not detected_faces:
        return FramePrediction(label=None, distance=None, box=None, center_x=None, area_ratio=None)

    scaled_box = scale_box(detected_faces[0], scale_factor)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return predict_face_box(
        gray_frame,
        scaled_box,
        recognizer,
        labels,
        image_size,
        distance_threshold,
    )


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
    if len(movement_points) < 4:
        return "UNKNOWN", "none", 0.0

    window_size = min(2, len(movement_points) // 2)
    start_points = movement_points[:window_size]
    end_points = movement_points[-window_size:]
    start_center_x = float(np.mean([point.center_x for point in start_points]))
    end_center_x = float(np.mean([point.center_x for point in end_points]))
    start_area = float(np.mean([point.area_ratio for point in start_points]))
    end_area = float(np.mean([point.area_ratio for point in end_points]))
    horizontal_delta = end_center_x - start_center_x
    depth_delta = end_area - start_area

    horizontal_steps = [
        movement_points[index + 1].center_x - movement_points[index].center_x
        for index in range(len(movement_points) - 1)
    ]
    depth_steps = [
        movement_points[index + 1].area_ratio - movement_points[index].area_ratio
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

    vote_counts = Counter(prediction.label.folder_name for prediction in verified_predictions if prediction.label is not None)
    best_folder_name = vote_counts.most_common(1)[0][0]
    chosen_predictions = [
        prediction for prediction in verified_predictions if prediction.label and prediction.label.folder_name == best_folder_name
    ]
    best_prediction = min(chosen_predictions, key=lambda prediction: prediction.distance or float("inf"))
    average_distance = float(np.mean([prediction.distance for prediction in chosen_predictions if prediction.distance is not None]))
    match_confidence = clamp(1.0 - (average_distance / max(distance_threshold, 1.0)), 0.0, 1.0)

    return {
        "verified": True,
        "employee": {
            "folderName": best_prediction.label.folder_name,
            "displayName": best_prediction.label.display_name,
            "employeeCode": best_prediction.label.employee_code,
            "department": best_prediction.label.department,
            "rfidUid": best_prediction.label.rfid_uid,
            "sampleCount": best_prediction.label.sample_count,
        },
        "matchConfidence": round_float(match_confidence),
        "bestDistance": round_float(best_prediction.distance or average_distance, 3),
        "distanceThreshold": distance_threshold,
        "movementDirection": direction,
        "movementAxis": axis,
        "movementConfidence": direction_confidence,
        "framesProcessed": len(predictions),
        "framesWithFace": len([prediction for prediction in predictions if prediction.box is not None]),
        "bestBox": {
            "top": best_prediction.box[0],
            "right": best_prediction.box[1],
            "bottom": best_prediction.box[2],
            "left": best_prediction.box[3],
        } if best_prediction.box is not None else None,
    }


def handle_verify_burst(
    request: dict[str, Any],
    recognizer: Any,
    detector: cv2.CascadeClassifier,
    labels: dict[int, LabelRecord],
    distance_threshold: float,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frames = request.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("No frames were provided for verification.")

    predictions = [
        predict_frame(str(frame_payload), recognizer, detector, labels, image_size, distance_threshold, args)
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
    labels: dict[int, LabelRecord],
    distance_threshold: float,
    image_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame_payload = request.get("frame")
    if not isinstance(frame_payload, str) or not frame_payload.strip():
        raise ValueError("No frame was provided for live recognition.")

    image = decode_frame(frame_payload)
    frame_height, frame_width = image.shape[:2]
    working_frame, scale_factor = resize_frame(image, args.resize_width)
    gray_working = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = detect_faces(gray_working, detector, args)
    max_faces = int(request.get("maxFaces", 50) or 50)

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

        distance = float(prediction.distance) if prediction.distance is not None else None
        confidence = (
            clamp(1.0 - (distance / max(distance_threshold, 1.0)), 0.0, 1.0)
            if distance is not None
            else 0.0
        )
        recognized_faces.append(
            LiveRecognitionFace(
                label=prediction.label,
                distance=distance,
                box=prediction.box,
                confidence=confidence,
                verified=prediction.label is not None,
            )
        )

    return {
        "faces": [
            {
                "label": face.label.display_name if face.label is not None else "Unknown Face",
                "employeeCode": face.label.employee_code if face.label is not None else None,
                "department": face.label.department if face.label is not None else None,
                "rfidUid": face.label.rfid_uid if face.label is not None else None,
                "verified": face.verified,
                "confidence": round_float(face.confidence),
                "distance": round_float(face.distance, 3) if face.distance is not None else None,
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


def emit_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
