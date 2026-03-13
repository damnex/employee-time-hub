#!/usr/bin/env python3
"""Run live webcam or RTSP attendance using the exported face profiles."""

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

import cv2  # type: ignore
import face_recognition  # type: ignore
import numpy as np  # type: ignore

FACE_MATCH_THRESHOLD = 0.87
FACE_PRIMARY_THRESHOLD = 0.88
FACE_ANCHOR_AVG_THRESHOLD = 0.85
FACE_ANCHOR_RATIO_THRESHOLD = 0.8
EVENT_FIELDS = [
    "timestamp",
    "source",
    "employeeLabel",
    "displayName",
    "employeeCode",
    "department",
    "rfidUid",
    "confidence",
    "primaryConfidence",
    "anchorAverage",
    "strongAnchorRatio",
    "sampleCount",
    "faceIndex",
    "top",
    "right",
    "bottom",
    "left",
]


@dataclass
class LoadedProfile:
    folder_name: str
    display_name: str
    sample_count: int | None
    employee_code: str | None
    department: str | None
    rfid_uid: str | None
    primary_descriptor: np.ndarray
    anchor_descriptors: list[np.ndarray]


@dataclass
class MatchResult:
    profile: LoadedProfile
    metrics: dict[str, float]


@dataclass
class DetectionResult:
    face_index: int
    box: tuple[int, int, int, int]
    verified: bool
    best_match: MatchResult | None
    top_matches: list[MatchResult]


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
            "Run live attendance from a laptop webcam or RTSP CCTV stream. "
            "Use --source 0 for the laptop webcam now, then switch to an RTSP URL later."
        )
    )
    parser.add_argument("--profiles", required=True, type=Path, help="Path to output/profiles.json")
    parser.add_argument("--source", default="0", help="Video source. Use 0 for the laptop webcam or an RTSP URL later.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/live-attendance"),
        help="Where attendance logs, summaries, and optional snapshots will be written.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="How many best matches to keep per face.")
    parser.add_argument("--max-faces", type=int, default=12, help="Maximum faces to process per frame.")
    parser.add_argument(
        "--process-every-nth-frame",
        type=int,
        default=3,
        help="Process every Nth frame to keep webcam performance smooth with a 50-person roster.",
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
        help="0 means mark an employee once per session. Set a positive value to allow repeats after a cooldown.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=960,
        help="Resize frames before detection for speed. Use 0 to disable resizing.",
    )
    parser.add_argument("--camera-width", type=int, default=1280, help="Preferred webcam width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Preferred webcam height.")
    parser.add_argument("--upsample", type=int, default=0, help="Face detection upsample count.")
    parser.add_argument("--detection-model", choices=("hog", "cnn"), default="hog", help="Face detector backend.")
    parser.add_argument(
        "--landmark-model",
        choices=("small", "large"),
        default="small",
        help="Landmark model used by face_recognition.face_encodings.",
    )
    parser.add_argument("--match-threshold", type=float, default=FACE_MATCH_THRESHOLD, help="Minimum final confidence.")
    parser.add_argument("--primary-threshold", type=float, default=FACE_PRIMARY_THRESHOLD, help="Minimum primary confidence.")
    parser.add_argument(
        "--anchor-threshold",
        type=float,
        default=FACE_ANCHOR_AVG_THRESHOLD,
        help="Minimum anchor average confidence.",
    )
    parser.add_argument(
        "--anchor-ratio-threshold",
        type=float,
        default=FACE_ANCHOR_RATIO_THRESHOLD,
        help="Minimum strong-anchor ratio.",
    )
    parser.add_argument("--save-snapshots", action="store_true", help="Save a frame image whenever attendance is marked.")
    parser.add_argument("--no-display", action="store_true", help="Run without showing the live OpenCV window.")
    return parser.parse_args()


def round_float(value: float, digits: int = 4) -> float:
    return float(np.round(value, digits))


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def calculate_legacy_match_confidence(v1: np.ndarray, v2: np.ndarray) -> float:
    if v1.size == 0 or v1.shape != v2.shape:
        return 0.0

    centered_v1 = v1 - np.mean(v1)
    centered_v2 = v2 - np.mean(v2)
    magnitude_v1 = float(np.linalg.norm(centered_v1))
    magnitude_v2 = float(np.linalg.norm(centered_v2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0

    score = float(np.dot(centered_v1 / magnitude_v1, centered_v2 / magnitude_v2))
    return round_float((score + 1.0) / 2.0)


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


def load_profiles(path: Path) -> list[LoadedProfile]:
    raw_profiles = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(raw_profiles, list):
        raise ValueError("profiles.json must contain a list")

    profiles: list[LoadedProfile] = []
    for raw in raw_profiles:
        metadata = raw.get("metadata") or {}
        face_descriptor = raw.get("faceDescriptor") or {}
        profiles.append(
            LoadedProfile(
                folder_name=raw.get("folderName") or raw.get("label") or "unknown",
                display_name=raw.get("displayName") or raw.get("folderName") or "Unknown",
                sample_count=raw.get("sampleCount"),
                employee_code=raw.get("employeeCode") or metadata.get("employeeCode"),
                department=raw.get("department") or metadata.get("department"),
                rfid_uid=raw.get("rfidUid") or metadata.get("rfidUid"),
                primary_descriptor=np.asarray(face_descriptor.get("primaryDescriptor", []), dtype=np.float32),
                anchor_descriptors=[
                    np.asarray(descriptor, dtype=np.float32)
                    for descriptor in face_descriptor.get("anchorDescriptors", [])
                ],
            )
        )

    return [profile for profile in profiles if profile.primary_descriptor.size > 0]


def build_match_metrics(live_descriptor: np.ndarray, profile: LoadedProfile) -> dict[str, float]:
    base_confidence = calculate_legacy_match_confidence(live_descriptor, profile.primary_descriptor)
    anchor_scores = [
        max(base_confidence, calculate_legacy_match_confidence(live_descriptor, anchor))
        for anchor in profile.anchor_descriptors
    ]
    if not anchor_scores:
        anchor_scores = [base_confidence]

    primary_confidence = max([base_confidence, *anchor_scores])
    anchor_average = average(anchor_scores)
    peak_anchor_confidence = max(anchor_scores, default=0.0)
    strong_anchor_ratio = len(
        [score for score in anchor_scores if score >= FACE_ANCHOR_AVG_THRESHOLD]
    ) / max(1, len(anchor_scores))
    final_confidence = round_float(
        primary_confidence * 0.45
        + anchor_average * 0.45
        + strong_anchor_ratio * 0.1
    )

    return {
        "primaryConfidence": round_float(primary_confidence),
        "anchorAverage": round_float(anchor_average),
        "peakAnchorConfidence": round_float(peak_anchor_confidence),
        "strongAnchorRatio": round_float(strong_anchor_ratio),
        "finalConfidence": final_confidence,
    }


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


def scale_locations(
    locations: list[tuple[int, int, int, int]],
    scale_factor: float,
) -> list[tuple[int, int, int, int]]:
    if scale_factor == 1.0:
        return locations

    return [
        (
            int(round(top / scale_factor)),
            int(round(right / scale_factor)),
            int(round(bottom / scale_factor)),
            int(round(left / scale_factor)),
        )
        for top, right, bottom, left in locations
    ]


def recognize_frame(
    frame: np.ndarray,
    profiles: list[LoadedProfile],
    args: Any,
) -> list[DetectionResult]:
    working_frame, scale_factor = resize_frame(frame, args.resize_width)
    rgb_frame = cv2.cvtColor(working_frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(
        rgb_frame,
        number_of_times_to_upsample=args.upsample,
        model=args.detection_model,
    )
    locations = locations[: args.max_faces]
    encodings = face_recognition.face_encodings(
        rgb_frame,
        known_face_locations=locations,
        model=args.landmark_model,
    )
    full_locations = scale_locations(locations, scale_factor)

    detections: list[DetectionResult] = []
    for index, (location, encoding) in enumerate(zip(full_locations, encodings)):
        live_descriptor = np.asarray(encoding, dtype=np.float32)
        ranked_matches = [
            MatchResult(profile=profile, metrics=build_match_metrics(live_descriptor, profile))
            for profile in profiles
        ]
        ranked_matches.sort(key=lambda item: item.metrics["finalConfidence"], reverse=True)
        k = int(max(1, args.top_k))
        top_matches = ranked_matches[:k]
        best_match = top_matches[0] if top_matches else None
        verified = False
        if best_match is not None:
            verified = (
                best_match.metrics["finalConfidence"] >= args.match_threshold
                and best_match.metrics["primaryConfidence"] >= args.primary_threshold
                and best_match.metrics["anchorAverage"] >= args.anchor_threshold
                and best_match.metrics["strongAnchorRatio"] >= args.anchor_ratio_threshold
            )

        detections.append(
            DetectionResult(
                face_index=index,
                box=location,
                verified=verified,
                best_match=best_match if verified else None,
                top_matches=top_matches,
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

    last_seen = last_logged_at.get(label)
    return last_seen is None or (now_ts - last_seen) >= repeat_after_seconds


def create_event_row(
    source_label: str,
    detection: DetectionResult,
    match: MatchResult,
    timestamp_iso: str,
) -> dict[str, Any]:
    top, right, bottom, left = detection.box
    return {
        "timestamp": timestamp_iso,
        "source": source_label,
        "employeeLabel": match.profile.folder_name,
        "displayName": match.profile.display_name,
        "employeeCode": match.profile.employee_code or "",
        "department": match.profile.department or "",
        "rfidUid": match.profile.rfid_uid or "",
        "confidence": match.metrics["finalConfidence"],
        "primaryConfidence": match.metrics["primaryConfidence"],
        "anchorAverage": match.metrics["anchorAverage"],
        "strongAnchorRatio": match.metrics["strongAnchorRatio"],
        "sampleCount": match.profile.sample_count or "",
        "faceIndex": detection.face_index,
        "top": top,
        "right": right,
        "bottom": bottom,
        "left": left,
    }


def update_session_attendance(
    session_attendance: dict[str, SessionAttendance],
    match: MatchResult,
    timestamp_iso: str,
) -> None:
    existing = session_attendance.get(match.profile.folder_name)
    if existing is None:
        session_attendance[match.profile.folder_name] = SessionAttendance(
            first_seen_at=timestamp_iso,
            last_seen_at=timestamp_iso,
            times_marked=1,
            display_name=match.profile.display_name,
            employee_code=match.profile.employee_code,
            department=match.profile.department,
            rfid_uid=match.profile.rfid_uid,
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
    status_message: str,
) -> np.ndarray:
    canvas = frame.copy()

    for detection in detections:
        top, right, bottom, left = detection.box
        match = detection.best_match
        if detection.verified and match is not None:
            color = (60, 200, 80)
            label = f"{match.profile.display_name} {match.metrics['finalConfidence']:.2f}"
        else:
            color = (30, 30, 220)
            label = "Unknown"

        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        cv2.rectangle(canvas, (left, max(0, top - 28)), (right, top), color, -1)
        cv2.putText(
            canvas,
            label[:42],
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
    session_started_at: str,
    session_attendance: dict[str, SessionAttendance],
) -> None:
    payload = {
        "sessionStartedAt": session_started_at,
        "sessionEndedAt": datetime.now(UTC).isoformat(),
        "source": source_label,
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
    args: Any = parse_args()
    profiles = load_profiles(args.profiles)
    if not profiles:
        print("No valid profiles were loaded. Train first and check output/profiles.json.", file=sys.stderr)
        return 1

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

            frame_counter = frame_counter + 1
            if frame_counter % max(1, args.process_every_nth_frame) == 0:
                latest_detections = recognize_frame(frame, profiles, args)
                current_labels = {
                    detection.best_match.profile.folder_name
                    for detection in latest_detections
                    if detection.verified and detection.best_match is not None
                }
                for label in list(consecutive_hits.keys()):
                    if label not in current_labels:
                        consecutive_hits[label] = 0

                now_ts = time.time()
                for detection in latest_detections:
                    match = detection.best_match
                    if not detection.verified or match is None:
                        continue

                    label = match.profile.folder_name
                    consecutive_hits[label] = consecutive_hits.get(label, 0) + 1
                    if consecutive_hits[label] < max(1, args.min_consecutive_detections):
                        continue
                    if not should_mark_attendance(
                        label,
                        now_ts,
                        logged_once,
                        last_logged_at,
                        args.repeat_after_seconds,
                    ):
                        continue

                    timestamp_iso = datetime.now(UTC).isoformat()
                    event_row = create_event_row(source_label, detection, match, timestamp_iso)
                    append_attendance_event(event_log_path, event_row)
                    update_session_attendance(session_attendance, match, timestamp_iso)
                    logged_once.add(label)
                    last_logged_at[label] = now_ts
                    consecutive_hits[label] = 0
                    status_message = (
                        f"Attendance marked for {match.profile.display_name} "
                        f"at {timestamp_iso}."
                    )
                    if args.save_snapshots:
                        timestamp_token = timestamp_iso.replace(":", "-").replace(".", "-")
                        save_snapshot(frame, snapshot_dir, timestamp_token, label)

            if args.no_display:
                continue

            overlay = draw_overlay(
                frame,
                latest_detections,
                len(profiles),
                len(session_attendance),
                source_label,
                status_message,
            )
            cv2.imshow("Live Attendance", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    write_session_summary(summary_path, source_label, len(profiles), session_started_at, session_attendance)
    print(f"Attendance events: {event_log_path}")
    print(f"Session summary: {summary_path}")
    print(f"Employees marked this session: {len(session_attendance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
