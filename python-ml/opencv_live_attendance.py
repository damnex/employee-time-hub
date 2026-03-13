#!/usr/bin/env python3
"""Run live attendance using OpenCV LBPH with webcam first and RTSP later."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import cv2
import numpy as np

EVENT_FIELDS = [
    "timestamp",
    "source",
    "employeeLabel",
    "displayName",
    "employeeCode",
    "department",
    "rfidUid",
    "distance",
    "threshold",
    "sampleCount",
    "faceIndex",
    "top",
    "right",
    "bottom",
    "left",
]


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
class DetectionResult:
    face_index: int
    box: tuple[int, int, int, int]
    verified: bool
    label: LabelRecord | None
    distance: float | None


@dataclass
class SessionAttendance:
    first_seen_at: str
    last_seen_at: str
    times_marked: int
    display_name: str
    employee_code: str | None
    department: str | None
    rfid_uid: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run laptop-webcam or RTSP attendance using an OpenCV LBPH model. "
            "Use --source 0 for the laptop webcam now."
        )
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to lbph-model.yml")
    parser.add_argument("--labels", required=True, type=Path, help="Path to lbph-labels.json")
    parser.add_argument("--source", default="0", help="Video source. Use 0 for laptop webcam or an RTSP URL later.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/live-attendance-opencv"),
        help="Where attendance logs, summaries, and optional snapshots will be written.",
    )
    parser.add_argument(
        "--process-every-nth-frame",
        type=int,
        default=2,
        help="Process every Nth frame to keep recognition smooth.",
    )
    parser.add_argument(
        "--min-consecutive-detections",
        type=int,
        default=2,
        help="How many processed frames a person must appear in before attendance is marked.",
    )
    parser.add_argument(
        "--repeat-after-seconds",
        type=float,
        default=0.0,
        help="0 means mark an employee once per session. Set a positive value to allow repeats later.",
    )
    parser.add_argument("--camera-width", type=int, default=1280, help="Preferred webcam width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Preferred webcam height.")
    parser.add_argument("--resize-width", type=int, default=960, help="Resize frames before detection. Use 0 to disable.")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Haar cascade scale factor.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Haar cascade minNeighbors.")
    parser.add_argument("--min-face-size", type=int, default=80, help="Minimum face size in pixels.")
    parser.add_argument(
        "--distance-threshold",
        type=float,
        help="Override the trained LBPH threshold. Lower distances are better.",
    )
    parser.add_argument("--save-snapshots", action="store_true", help="Save a frame image whenever attendance is marked.")
    parser.add_argument("--no-display", action="store_true", help="Run without showing the live OpenCV window.")
    return parser.parse_args()


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int):
        if sys.platform.startswith("win"):
            capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if capture.isOpened():
                return capture
        return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)


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


def scale_boxes(boxes: list[tuple[int, int, int, int]], scale_factor: float) -> list[tuple[int, int, int, int]]:
    if scale_factor == 1.0:
        return boxes
    return [
        (
            int(round(top / scale_factor)),
            int(round(right / scale_factor)),
            int(round(bottom / scale_factor)),
            int(round(left / scale_factor)),
        )
        for top, right, bottom, left in boxes
    ]


def create_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Unable to load Haar cascade: {cascade_path}")
    return detector


def load_labels(path: Path) -> tuple[dict[int, LabelRecord], float, int]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    threshold = float(payload.get("threshold", 65.0))
    image_size = int(payload.get("imageSize", 200))
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
    return labels, threshold, image_size


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
    return [(int(y), int(x + w), int(y + h), int(x)) for x, y, w, h in faces]


def prepare_face_crop(gray_frame: np.ndarray, box: tuple[int, int, int, int], image_size: int) -> np.ndarray:
    top, right, bottom, left = box
    width = max(1, right - left)
    height = max(1, bottom - top)
    padding_x = int(width * 0.15)
    padding_y = int(height * 0.15)
    start_x = max(0, left - padding_x)
    start_y = max(0, top - padding_y)
    end_x = min(gray_frame.shape[1], right + padding_x)
    end_y = min(gray_frame.shape[0], bottom + padding_y)
    crop = gray_frame[start_y:end_y, start_x:end_x]
    resized = cv2.resize(crop, (image_size, image_size))
    return cv2.equalizeHist(resized)


def recognize_frame(
    frame: np.ndarray,
    recognizer: Any,
    detector: cv2.CascadeClassifier,
    labels: dict[int, LabelRecord],
    image_size: int,
    threshold: float,
    args: argparse.Namespace,
) -> list[DetectionResult]:
    working_frame, scale_factor = resize_frame(frame, args.resize_width)
    gray_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
    face_boxes = detect_faces(gray_frame, detector, args)
    scaled_boxes = scale_boxes(face_boxes, scale_factor)

    detections: list[DetectionResult] = []
    for index, box in enumerate(scaled_boxes):
        prepared_face = prepare_face_crop(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), box, image_size)
        label_id, distance = recognizer.predict(prepared_face)
        label_record = labels.get(int(label_id))
        verified = label_record is not None and distance <= threshold and label_record.included_in_training
        detections.append(
            DetectionResult(
                face_index=index,
                box=box,
                verified=verified,
                label=label_record if verified else None,
                distance=float(distance),
            )
        )

    return detections


def ensure_csv_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()


def append_attendance_event(path: Path, row: dict[str, Any]) -> None:
    ensure_csv_header(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writerow(row)


def should_mark_attendance(
    label: str,
    now_ts: float,
    logged_once: set[str],
    last_logged_at: dict[str, float],
    repeat_after_seconds: float,
) -> bool:
    if repeat_after_seconds <= 0:
        return label not in logged_once
    previous = last_logged_at.get(label)
    return previous is None or (now_ts - previous) >= repeat_after_seconds


def create_event_row(
    source_label: str,
    detection: DetectionResult,
    threshold: float,
    timestamp_iso: str,
) -> dict[str, Any]:
    top, right, bottom, left = detection.box
    label = detection.label
    return {
        "timestamp": timestamp_iso,
        "source": source_label,
        "employeeLabel": label.folder_name if label else "",
        "displayName": label.display_name if label else "",
        "employeeCode": label.employee_code if label and label.employee_code else "",
        "department": label.department if label and label.department else "",
        "rfidUid": label.rfid_uid if label and label.rfid_uid else "",
        "distance": round(float(detection.distance or 0.0), 3),
        "threshold": threshold,
        "sampleCount": label.sample_count if label else "",
        "faceIndex": detection.face_index,
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left,
    }


def update_session_attendance(
    session_attendance: dict[str, SessionAttendance],
    detection: DetectionResult,
    timestamp_iso: str,
) -> None:
    if detection.label is None:
        return
    key = detection.label.folder_name
    existing = session_attendance.get(key)
    if existing is None:
        session_attendance[key] = SessionAttendance(
            first_seen_at=timestamp_iso,
            last_seen_at=timestamp_iso,
            times_marked=1,
            display_name=detection.label.display_name,
            employee_code=detection.label.employee_code,
            department=detection.label.department,
            rfid_uid=detection.label.rfid_uid,
        )
        return
    existing.last_seen_at = timestamp_iso
    existing.times_marked += 1


def save_snapshot(frame: np.ndarray, snapshot_dir: Path, timestamp_token: str, label: str) -> None:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{timestamp_token}_{label}.jpg"
    cv2.imwrite(str(snapshot_path), frame)


def draw_overlay(
    frame: np.ndarray,
    detections: list[DetectionResult],
    roster_size: int,
    attendance_count: int,
    source_label: str,
    threshold: float,
    status_message: str,
) -> np.ndarray:
    canvas = frame.copy()
    for detection in detections:
        top, right, bottom, left = detection.box
        if detection.verified and detection.label is not None:
            color = (60, 200, 80)
            text = f"{detection.label.display_name} d={detection.distance:.1f}"
        else:
            color = (30, 30, 220)
            text = f"Unknown d={detection.distance:.1f}" if detection.distance is not None else "Unknown"
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        cv2.rectangle(canvas, (left, max(0, top - 28)), (right, top), color, -1)
        cv2.putText(
            canvas,
            text[:44],
            (left + 6, max(18, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    header_lines = [
        f"Source: {source_label}",
        f"Roster: {roster_size} employees",
        f"Marked this session: {attendance_count}",
        f"LBPH threshold: {threshold:.1f}",
        status_message,
    ]
    for index, line in enumerate(header_lines):
        cv2.putText(
            canvas,
            line,
            (16, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
    return canvas


def write_session_summary(
    path: Path,
    source_label: str,
    roster_size: int,
    threshold: float,
    session_started_at: str,
    session_attendance: dict[str, SessionAttendance],
) -> None:
    payload = {
        "sessionStartedAt": session_started_at,
        "sessionEndedAt": datetime.now(UTC).isoformat(),
        "source": source_label,
        "modelType": "opencv-lbph",
        "threshold": threshold,
        "rosterSize": roster_size,
        "attendanceMarked": len(session_attendance),
        "employees": [
            {
                "employeeLabel": label,
                "displayName": item.display_name,
                "employeeCode": item.employee_code,
                "department": item.department,
                "rfidUid": item.rfid_uid,
                "firstSeenAt": item.first_seen_at,
                "lastSeenAt": item.last_seen_at,
                "timesMarked": item.times_marked,
            }
            for label, item in sorted(session_attendance.items())
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not hasattr(cv2, "face"):
        print(
            "OpenCV face module is unavailable. Install requirements-webcam.txt with opencv-contrib-python.",
            file=sys.stderr,
        )
        return 1

    labels, trained_threshold, image_size = load_labels(args.labels)
    threshold = float(args.distance_threshold) if args.distance_threshold is not None else trained_threshold
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(args.model.resolve()))
    detector = create_detector()

    source_value = parse_source(args.source)
    source_label = f"webcam:{source_value}" if isinstance(source_value, int) else str(source_value)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = output_dir / "attendance-events.csv"
    summary_path = output_dir / "session-summary.json"
    snapshot_dir = output_dir / "snapshots"

    capture = open_capture(source_value)
    if not capture.isOpened():
        print(f"Unable to open video source: {args.source}", file=sys.stderr)
        return 1
    if isinstance(source_value, int):
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    frame_counter = 0
    consecutive_hits: Counter[str] = Counter()
    logged_once: set[str] = set()
    last_logged_at: dict[str, float] = {}
    session_attendance: dict[str, SessionAttendance] = {}
    latest_detections: list[DetectionResult] = []
    status_message = "Show your face to the laptop webcam. Press Q to quit."
    session_started_at = datetime.now(UTC).isoformat()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                status_message = "Frame read failed. Check the webcam or RTSP source."
                if args.no_display:
                    print(status_message, file=sys.stderr)
                    break
                time.sleep(0.1)
                continue

            frame_counter += 1
            if frame_counter % max(1, args.process_every_nth_frame) == 0:
                latest_detections = recognize_frame(frame, recognizer, detector, labels, image_size, threshold, args)
                current_labels = {
                    detection.label.folder_name
                    for detection in latest_detections
                    if detection.verified and detection.label is not None
                }
                for label in list(consecutive_hits.keys()):
                    if label not in current_labels:
                        consecutive_hits[label] = 0

                now_ts = time.time()
                for detection in latest_detections:
                    if not detection.verified or detection.label is None:
                        continue
                    label_key = detection.label.folder_name
                    consecutive_hits[label_key] += 1
                    if consecutive_hits[label_key] < max(1, args.min_consecutive_detections):
                        continue
                    if not should_mark_attendance(
                        label_key,
                        now_ts,
                        logged_once,
                        last_logged_at,
                        args.repeat_after_seconds,
                    ):
                        continue

                    timestamp_iso = datetime.now(UTC).isoformat()
                    append_attendance_event(event_log_path, create_event_row(source_label, detection, threshold, timestamp_iso))
                    update_session_attendance(session_attendance, detection, timestamp_iso)
                    logged_once.add(label_key)
                    last_logged_at[label_key] = now_ts
                    consecutive_hits[label_key] = 0
                    status_message = f"Attendance marked for {detection.label.display_name} at {timestamp_iso}."
                    if args.save_snapshots:
                        timestamp_token = timestamp_iso.replace(":", "-").replace(".", "-")
                        save_snapshot(frame, snapshot_dir, timestamp_token, label_key)

            if args.no_display:
                continue

            overlay = draw_overlay(
                frame,
                latest_detections,
                len([label for label in labels.values() if label.included_in_training]),
                len(session_attendance),
                source_label,
                threshold,
                status_message,
            )
            cv2.imshow("OpenCV Live Attendance", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    write_session_summary(
        summary_path,
        source_label,
        len([label for label in labels.values() if label.included_in_training]),
        threshold,
        session_started_at,
        session_attendance,
    )
    print(f"Attendance events: {event_log_path}")
    print(f"Session summary: {summary_path}")
    print(f"Employees marked this session: {len(session_attendance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
