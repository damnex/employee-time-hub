#!/usr/bin/env python3
"""Verify a burst of gate frames with OpenCV face recognition and infer movement direction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from opencv_face_backend import (
    GRAY_NN_MODEL_TYPE,
    LBPH_MODEL_TYPE,
    create_lbph_recognizer,
    load_gray_nn_model,
    resolve_prediction,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a burst of gate frames with the trained OpenCV face model and infer direction."
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to lbph-model.yml")
    parser.add_argument("--labels", required=True, type=Path, help="Path to lbph-labels.json")
    parser.add_argument("--input-json", required=True, type=Path, help="JSON file that lists frame paths.")
    parser.add_argument("--distance-threshold", type=float, help="Override the threshold from the label map.")
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


def detect_largest_face(
    gray_frame: np.ndarray,
    detector: cv2.CascadeClassifier,
) -> tuple[int, int, int, int] | None:
    faces = detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
    return int(x), int(y), int(width), int(height)


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


def predict_frame(
    frame_path: Path,
    recognizer: Any,
    detector: cv2.CascadeClassifier,
    labels: dict[int, LabelRecord],
    image_size: int,
    distance_threshold: float,
    score_margin_threshold: float | None,
    centroid_margin_threshold: float | None,
) -> FramePrediction:
    image = cv2.imread(str(frame_path.resolve()))
    if image is None:
        return FramePrediction(label=None, distance=None, box=None, center_x=None, area_ratio=None)

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_box = detect_largest_face(gray_frame, detector)
    if face_box is None:
        return FramePrediction(label=None, distance=None, box=None, center_x=None, area_ratio=None)

    prepared_face, output_box, center_x, area_ratio = prepare_face_crop(gray_frame, face_box, image_size)
    predicted_label_id, distance, accepted = resolve_prediction(
        recognizer,
        prepared_face,
        distance_threshold,
        score_margin_threshold,
        centroid_margin_threshold,
    )
    label = labels.get(int(predicted_label_id))
    if label is None or not label.included_in_training or not accepted:
        return FramePrediction(label=None, distance=float(distance), box=output_box, center_x=center_x, area_ratio=area_ratio)

    return FramePrediction(
        label=label,
        distance=float(distance),
        box=output_box,
        center_x=center_x,
        area_ratio=area_ratio,
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

    window_size = min(3, len(movement_points) // 2)
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
    horizontal_confidence = clamp(((horizontal_travel - 0.06) / 0.18) * 0.7 + horizontal_consistency * 0.3, 0.0, 1.0)
    depth_confidence = clamp(((depth_travel - 0.01) / 0.05) * 0.7 + depth_consistency * 0.3, 0.0, 1.0)

    if horizontal_travel >= 0.06 and horizontal_confidence >= depth_confidence and horizontal_consistency >= 0.55:
        moving_entry = horizontal_delta > 0 if entry_horizontal_direction == "left-to-right" else horizontal_delta < 0
        return ("ENTRY" if moving_entry else "EXIT", "horizontal", round_float(horizontal_confidence))

    if depth_travel >= 0.01 and depth_consistency >= 0.55:
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
    min_verified_votes = max(2, min(3, max(1, len(predictions) // 2)))
    if len(chosen_predictions) < min_verified_votes:
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


def main() -> int:
    args = parse_args()

    labels, labels_payload = load_labels(args.labels)
    model_type = str(labels_payload.get("modelType") or LBPH_MODEL_TYPE)
    image_size = int(labels_payload.get("imageSize", 200))
    labels_threshold = float(labels_payload.get("threshold", 65.0 if model_type == LBPH_MODEL_TYPE else 0.28))
    score_margin_threshold = (
        float(labels_payload["scoreMarginThreshold"])
        if labels_payload.get("scoreMarginThreshold") is not None
        else None
    )
    centroid_margin_threshold = (
        float(labels_payload["centroidMarginThreshold"])
        if labels_payload.get("centroidMarginThreshold") is not None
        else None
    )
    request_payload = json.loads(args.input_json.resolve().read_text(encoding="utf-8"))
    frame_paths = [Path(frame_path) for frame_path in request_payload.get("frames", [])]
    recognizer: Any
    if model_type == LBPH_MODEL_TYPE:
        recognizer = create_lbph_recognizer()
        recognizer.read(str(args.model.resolve()))
    elif model_type == GRAY_NN_MODEL_TYPE:
        recognizer, stored_threshold, stored_image_size = load_gray_nn_model(args.model.resolve())
        labels_threshold = stored_threshold
        image_size = stored_image_size
    else:
        raise RuntimeError(f"Unsupported face model type: {model_type}")

    distance_threshold = float(args.distance_threshold) if args.distance_threshold is not None else labels_threshold
    detector = create_detector()

    predictions = [
        predict_frame(
            frame_path,
            recognizer,
            detector,
            labels,
            image_size,
            distance_threshold,
            score_margin_threshold,
            centroid_margin_threshold,
        )
        for frame_path in frame_paths
    ]

    response = build_response(
        predictions,
        distance_threshold,
        request_payload.get("entryHorizontalDirection", "left-to-right"),
        request_payload.get("entryDepthDirection", "approaching"),
    )
    print(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
