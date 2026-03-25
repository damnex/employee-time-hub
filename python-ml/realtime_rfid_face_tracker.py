#!/usr/bin/env python3
"""High-performance realtime RFID + face tracking pipeline."""

from __future__ import annotations

import argparse
import csv
import pickle
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore

try:
    import face_recognition  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "face_recognition is required for the realtime tracker. Install python-ml/requirements-realtime.txt in Python 3.11."
    ) from exc

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover
    serial = None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True)
class EmployeeMetadata:
    display_name: str
    employee_code: str | None
    department: str | None
    rfid_uid: str | None


@dataclass(slots=True)
class IndexedPerson:
    label: str
    metadata: EmployeeMetadata
    centroid: np.ndarray
    anchors: np.ndarray
    sample_count: int


@dataclass(slots=True)
class MatchResult:
    person: IndexedPerson
    distance: float
    second_best_distance: float | None
    confidence: float


@dataclass(slots=True)
class TrackedFace:
    track_id: int
    box: tuple[int, int, int, int]
    label: str | None
    display_name: str | None
    employee_code: str | None
    rfid_uid: str | None
    confidence: float
    distance: float | None
    authorized: bool
    status_text: str
    last_seen_at: float


@dataclass(slots=True)
class RFIDEvent:
    tag: str
    seen_at: float


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def round_float(value: float, digits: int = 4) -> float:
    return float(np.round(value, digits))


def parse_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def resize_for_runtime(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        return frame
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def box_area(box: tuple[int, int, int, int]) -> int:
    top, right, bottom, left = box
    return max(0, bottom - top) * max(0, right - left)


def box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    top = max(box_a[0], box_b[0])
    right = min(box_a[1], box_b[1])
    bottom = min(box_a[2], box_b[2])
    left = max(box_a[3], box_b[3])
    intersection = box_area((top, right, bottom, left))
    if intersection <= 0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def expand_box(
    box: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int],
    padding_ratio: float = 0.16,
) -> tuple[int, int, int, int]:
    top, right, bottom, left = box
    height = bottom - top
    width = right - left
    pad_y = int(height * padding_ratio)
    pad_x = int(width * padding_ratio)
    frame_height, frame_width = frame_shape[:2]
    return (
        max(0, top - pad_y),
        min(frame_width, right + pad_x),
        min(frame_height, bottom + pad_y),
        max(0, left - pad_x),
    )


def load_metadata_csv(path: Path | None) -> dict[str, EmployeeMetadata]:
    if path is None or not path.exists():
        return {}

    result: dict[str, EmployeeMetadata] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            folder_name = (row.get("folder_name") or "").strip()
            if not folder_name:
                continue
            result[folder_name] = EmployeeMetadata(
                display_name=(row.get("name") or folder_name).strip(),
                employee_code=(row.get("employeeCode") or "").strip() or None,
                department=(row.get("department") or "").strip() or None,
                rfid_uid=((row.get("rfidUid") or "").strip().upper() or None),
            )
    return result


def iter_person_directories(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.iterdir() if path.is_dir() and not path.name.startswith("."))


def iter_image_files(person_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in person_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def prepare_training_image(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_dimension:
        return image
    scale = max_dimension / float(max(height, width))
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def image_quality(image: np.ndarray, face_box: tuple[int, int, int, int]) -> float:
    top, right, bottom, left = face_box
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray) / 255.0)
    brightness_score = 1.0 - clamp(abs(brightness - 0.55) / 0.45, 0.0, 1.0)
    sharpness_score = clamp(float(sharpness) / 1800.0, 0.0, 1.0)
    face_ratio = box_area(face_box) / max(1, image.shape[0] * image.shape[1])
    size_score = clamp(face_ratio / 0.16, 0.0, 1.0)
    return round_float(size_score * 0.45 + sharpness_score * 0.35 + brightness_score * 0.2, 4)


def build_index(args: argparse.Namespace) -> int:
    metadata = load_metadata_csv(args.metadata)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for person_dir in iter_person_directories(args.dataset):
        label = person_dir.name
        embeddings: list[tuple[np.ndarray, float]] = []

        for image_path in iter_image_files(person_dir):
            try:
                rgb_image = face_recognition.load_image_file(str(image_path))
            except Exception as exc:  # noqa: BLE001
                skipped.append({"label": label, "image": str(image_path), "reason": f"load-error: {exc}"})
                continue

            rgb_image = prepare_training_image(rgb_image, args.max_image_size)
            locations = face_recognition.face_locations(
                rgb_image,
                number_of_times_to_upsample=args.upsample,
                model=args.detection_model,
            )
            if len(locations) != 1:
                skipped.append({"label": label, "image": str(image_path), "reason": f"expected-1-face-found-{len(locations)}"})
                continue

            encodings = face_recognition.face_encodings(
                rgb_image,
                known_face_locations=locations,
                num_jitters=args.jitters,
                model=args.landmark_model,
            )
            if not encodings:
                skipped.append({"label": label, "image": str(image_path), "reason": "encoding-failed"})
                continue

            quality = image_quality(rgb_image, locations[0])
            embeddings.append((normalize_embedding(np.asarray(encodings[0], dtype=np.float32)), quality))

        if len(embeddings) < args.min_samples:
            skipped.append({"label": label, "image": "", "reason": f"not-enough-samples-{len(embeddings)}"})
            continue

        embeddings.sort(key=lambda item: item[1], reverse=True)
        selected = embeddings[: args.max_anchors]
        anchor_vectors = np.vstack([item[0] for item in selected]).astype(np.float32)
        centroid = normalize_embedding(np.mean(anchor_vectors, axis=0))
        info = metadata.get(
            label,
            EmployeeMetadata(
                display_name=label,
                employee_code=label,
                department=None,
                rfid_uid=None,
            ),
        )
        records.append(
            {
                "label": label,
                "displayName": info.display_name,
                "employeeCode": info.employee_code,
                "department": info.department,
                "rfidUid": info.rfid_uid,
                "sampleCount": len(embeddings),
                "anchors": anchor_vectors.tolist(),
                "centroid": centroid.tolist(),
            }
        )

    payload = {
        "version": 1,
        "builtAt": datetime.now(UTC).isoformat(),
        "records": records,
        "skipped": skipped,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(payload, handle)

    print(f"Built realtime face index: {args.output}")
    print(f"People indexed: {len(records)}")
    print(f"Skipped entries: {len(skipped)}")
    return 0


class FaceIndex:
    def __init__(self, path: Path) -> None:
        with path.open("rb") as handle:
            payload = pickle.load(handle)

        raw_records = payload.get("records", [])
        self.people: list[IndexedPerson] = []
        anchor_vectors: list[np.ndarray] = []
        owners: list[int] = []

        for index, raw in enumerate(raw_records):
            anchors = np.asarray(raw.get("anchors", []), dtype=np.float32)
            centroid = np.asarray(raw.get("centroid", []), dtype=np.float32)
            if anchors.ndim != 2 or anchors.shape[0] == 0:
                continue
            person = IndexedPerson(
                label=str(raw.get("label") or f"person-{index}"),
                metadata=EmployeeMetadata(
                    display_name=str(raw.get("displayName") or raw.get("label") or f"Person {index + 1}"),
                    employee_code=(str(raw.get("employeeCode")).strip() if raw.get("employeeCode") else None),
                    department=(str(raw.get("department")).strip() if raw.get("department") else None),
                    rfid_uid=(str(raw.get("rfidUid")).strip().upper() if raw.get("rfidUid") else None),
                ),
                centroid=normalize_embedding(centroid),
                anchors=anchors,
                sample_count=int(raw.get("sampleCount") or anchors.shape[0]),
            )
            self.people.append(person)
            anchor_vectors.append(person.anchors)
            owners.extend([len(self.people) - 1] * person.anchors.shape[0])

        if not self.people:
            raise ValueError(f"No usable records found in {path}")

        self.anchor_matrix = np.vstack(anchor_vectors).astype(np.float32)
        self.anchor_owner_indices = np.asarray(owners, dtype=np.int32)

    def match(self, query_embedding: np.ndarray, threshold: float, ambiguity_margin: float) -> MatchResult | None:
        query = normalize_embedding(query_embedding.astype(np.float32))
        distances = np.linalg.norm(self.anchor_matrix - query, axis=1)
        owner_best = np.full(len(self.people), np.inf, dtype=np.float32)
        np.minimum.at(owner_best, self.anchor_owner_indices, distances)

        best_index = int(np.argmin(owner_best))
        best_distance = float(owner_best[best_index])
        finite_distances = owner_best[np.isfinite(owner_best)]
        second_best = None
        if finite_distances.size > 1:
            second_best = float(np.sort(finite_distances)[1])

        if best_distance > threshold:
            return None
        if second_best is not None and (second_best - best_distance) < ambiguity_margin:
            return None

        confidence = clamp(1.0 - (best_distance / max(threshold, 1e-6)), 0.0, 1.0)
        return MatchResult(
            person=self.people[best_index],
            distance=round_float(best_distance, 4),
            second_best_distance=round_float(second_best, 4) if second_best is not None else None,
            confidence=round_float(confidence, 4),
        )


class ThreadedVideoCapture:
    """Keep camera I/O off the recognition loop so processing uses the freshest frame."""

    def __init__(self, source: int | str, width: int, height: int) -> None:
        if isinstance(source, int) and sys.platform.startswith("win"):
            capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture = cv2.VideoCapture(source)
        else:
            capture = cv2.VideoCapture(source)

        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        if width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.capture = capture
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp = 0.0
        self._frame_counter = 0

    def start(self) -> "ThreadedVideoCapture":
        self._thread.start()
        return self

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_timestamp = time.perf_counter()
                self._frame_counter += 1

    def read(self) -> tuple[bool, np.ndarray | None, float, int]:
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0, self._frame_counter
            return True, self._latest_frame.copy(), self._latest_timestamp, self._frame_counter

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.capture.release()


class RFIDReader:
    def __init__(self, serial_port: str | None, baudrate: int, use_stdin: bool) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial_port = serial_port
        self._baudrate = baudrate
        self._use_stdin = use_stdin

    def start(self) -> "RFIDReader":
        if self._serial_port:
            if serial is None:
                raise RuntimeError("pyserial is required when --rfid-serial-port is used.")
            self._thread = threading.Thread(target=self._serial_loop, daemon=True)
            self._thread.start()
        elif self._use_stdin:
            self._thread = threading.Thread(target=self._stdin_loop, daemon=True)
            self._thread.start()
        return self

    def _serial_loop(self) -> None:
        assert serial is not None
        with serial.Serial(self._serial_port, self._baudrate, timeout=0.2) as handle:
            while not self._stop_event.is_set():
                raw = handle.readline().decode("utf-8", errors="ignore").strip()
                if raw:
                    self._queue.put(raw.upper())

    def _stdin_loop(self) -> None:
        while not self._stop_event.is_set():
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.05)
                continue
            tag = line.strip().upper()
            if tag:
                self._queue.put(tag)

    def poll_latest(self) -> str | None:
        latest: str | None = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                return latest

    def stop(self) -> None:
        self._stop_event.set()


class MediaPipeFaceDetector:
    def __init__(self, model_selection: int, min_detection_confidence: float) -> None:
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, rgb_frame: np.ndarray, max_faces: int) -> list[tuple[int, int, int, int]]:
        frame_height, frame_width = rgb_frame.shape[:2]
        result = self._detector.process(rgb_frame)
        if not result.detections:
            return []

        boxes: list[tuple[int, int, int, int]] = []
        for detection in result.detections:
            relative_box = detection.location_data.relative_bounding_box
            left = int(relative_box.xmin * frame_width)
            top = int(relative_box.ymin * frame_height)
            width = int(relative_box.width * frame_width)
            height = int(relative_box.height * frame_height)
            top = max(0, top)
            left = max(0, left)
            bottom = min(frame_height, top + max(1, height))
            right = min(frame_width, left + max(1, width))
            box = expand_box((top, right, bottom, left), rgb_frame.shape)
            if box_area(box) > 0:
                boxes.append(box)

        boxes.sort(key=box_area, reverse=True)
        return boxes[:max_faces]

    def close(self) -> None:
        self._detector.close()


def encode_faces(rgb_frame: np.ndarray, boxes: list[tuple[int, int, int, int]], landmark_model: str) -> list[np.ndarray]:
    if not boxes:
        return []

    encodings = face_recognition.face_encodings(
        rgb_frame,
        known_face_locations=boxes,
        model=landmark_model,
        num_jitters=1,
    )
    if len(encodings) == len(boxes):
        return [np.asarray(encoding, dtype=np.float32) for encoding in encodings]

    recovered: list[np.ndarray] = []
    for box in boxes:
        encoding = face_recognition.face_encodings(
            rgb_frame,
            known_face_locations=[box],
            model=landmark_model,
            num_jitters=1,
        )
        if encoding:
            recovered.append(np.asarray(encoding[0], dtype=np.float32))
    return recovered


def assign_track_ids(
    current_faces: list[TrackedFace],
    previous_faces: list[TrackedFace],
    next_track_id: int,
    now_ts: float,
    ttl_seconds: float,
) -> tuple[list[TrackedFace], int]:
    active_previous = [face for face in previous_faces if (now_ts - face.last_seen_at) <= ttl_seconds]
    assigned_previous: set[int] = set()
    assigned_current: list[TrackedFace] = []

    for face in sorted(current_faces, key=lambda item: box_area(item.box), reverse=True):
        best_index = -1
        best_iou = 0.0
        for index, previous in enumerate(active_previous):
            if previous.track_id in assigned_previous:
                continue
            iou = box_iou(face.box, previous.box)
            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index >= 0 and best_iou >= 0.2:
            previous = active_previous[best_index]
            assigned_previous.add(previous.track_id)
            face.track_id = previous.track_id
        else:
            face.track_id = next_track_id
            next_track_id += 1

        face.last_seen_at = now_ts
        assigned_current.append(face)

    return assigned_current, next_track_id


def process_runtime_frame(
    runtime_frame: np.ndarray,
    detector: MediaPipeFaceDetector,
    face_index: FaceIndex,
    args: argparse.Namespace,
    active_rfid: RFIDEvent | None,
) -> tuple[list[TrackedFace], float]:
    started_at = time.perf_counter()
    rgb_frame = cv2.cvtColor(runtime_frame, cv2.COLOR_BGR2RGB)
    boxes = detector.detect(rgb_frame, args.max_faces)
    encodings = encode_faces(rgb_frame, boxes, args.landmark_model)
    detections: list[TrackedFace] = []

    for box, encoding in zip(boxes, encodings):
        match = face_index.match(encoding, args.match_threshold, args.ambiguity_margin)
        authorized = bool(
            match is not None
            and active_rfid is not None
            and match.person.metadata.rfid_uid is not None
            and match.person.metadata.rfid_uid.upper() == active_rfid.tag
        )
        if match is None:
            detections.append(
                TrackedFace(
                    track_id=0,
                    box=box,
                    label=None,
                    display_name="Unknown",
                    employee_code=None,
                    rfid_uid=None,
                    confidence=0.0,
                    distance=None,
                    authorized=False,
                    status_text="UNKNOWN",
                    last_seen_at=started_at,
                )
            )
            continue

        status_text = "FACE MATCH"
        if active_rfid is not None:
            status_text = "AUTHORIZED" if authorized else "RFID MISMATCH"

        detections.append(
            TrackedFace(
                track_id=0,
                box=box,
                label=match.person.label,
                display_name=match.person.metadata.display_name,
                employee_code=match.person.metadata.employee_code,
                rfid_uid=match.person.metadata.rfid_uid,
                confidence=match.confidence,
                distance=match.distance,
                authorized=authorized,
                status_text=status_text,
                last_seen_at=started_at,
            )
        )

    latency_ms = (time.perf_counter() - started_at) * 1000.0
    return detections, latency_ms


def draw_overlay(
    frame_bgr: np.ndarray,
    faces: list[TrackedFace],
    active_rfid: RFIDEvent | None,
    latency_ms: float,
    processed_frame: bool,
    frame_counter: int,
) -> np.ndarray:
    canvas = frame_bgr.copy()

    for face in faces:
        top, right, bottom, left = face.box
        if face.authorized:
            color = (34, 197, 94)
        elif face.label:
            color = (255, 191, 0)
        else:
            color = (0, 102, 255)

        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        lines = [
            f"#{face.track_id:02d} {face.display_name or 'Unknown'}",
            f"{face.status_text} {face.confidence * 100:.1f}%",
        ]
        if face.employee_code:
            lines.append(face.employee_code)

        text_y = max(22, top - 10)
        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line[:44],
                (left, max(22, text_y - index * 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    status_lines = [
        f"Frame #{frame_counter}",
        f"Mode: {'recognize' if processed_frame else 'reuse tracks'}",
        f"Latency: {latency_ms:.1f} ms",
        f"RFID: {active_rfid.tag if active_rfid else 'waiting'}",
        f"Faces: {len(faces)}",
    ]
    for index, line in enumerate(status_lines):
        cv2.putText(
            canvas,
            line,
            (12, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )

    return canvas


def run_realtime_pipeline(args: argparse.Namespace) -> int:
    source = parse_source(args.source)
    face_index = FaceIndex(args.index)
    capture = ThreadedVideoCapture(source, args.capture_width, args.capture_height).start()
    detector = MediaPipeFaceDetector(args.mediapipe_model, args.min_detection_confidence)
    rfid_reader = RFIDReader(args.rfid_serial_port, args.rfid_baudrate, args.rfid_stdin).start()

    active_rfid: RFIDEvent | None = None
    tracked_faces: list[TrackedFace] = []
    next_track_id = 1
    last_processed_frame = -1
    last_latency_ms = 0.0
    process_stride = max(1, args.process_every_nth_frame)

    try:
        while True:
            latest_tag = rfid_reader.poll_latest()
            if latest_tag:
                active_rfid = RFIDEvent(tag=latest_tag, seen_at=time.perf_counter())
                print(f"[rfid] received {latest_tag}")

            if active_rfid and (time.perf_counter() - active_rfid.seen_at) > args.rfid_ttl_seconds:
                active_rfid = None

            ok, frame, _, frame_counter = capture.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            runtime_frame = resize_for_runtime(frame, args.runtime_width, args.runtime_height)
            should_process = frame_counter != last_processed_frame and (frame_counter % process_stride == 0)
            if should_process:
                processed_faces, last_latency_ms = process_runtime_frame(runtime_frame, detector, face_index, args, active_rfid)
                tracked_faces, next_track_id = assign_track_ids(
                    processed_faces,
                    tracked_faces,
                    next_track_id,
                    time.perf_counter(),
                    args.track_ttl_seconds,
                )
                last_processed_frame = frame_counter
                if active_rfid is not None and any(face.authorized for face in tracked_faces):
                    matched_face = next(face for face in tracked_faces if face.authorized)
                    print(
                        f"[access] AUTHORIZED tag={active_rfid.tag} employee={matched_face.display_name} "
                        f"distance={matched_face.distance}"
                    )
                    active_rfid = None

            display_frame = draw_overlay(runtime_frame, tracked_faces, active_rfid, last_latency_ms, should_process, frame_counter)
            if not args.no_display:
                cv2.imshow("Realtime RFID Face Tracker", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            elif frame_counter % 30 == 0:
                print(
                    f"[status] frame={frame_counter} latency_ms={last_latency_ms:.1f} "
                    f"faces={len(tracked_faces)} rfid={(active_rfid.tag if active_rfid else 'waiting')}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        capture.stop()
        rfid_reader.stop()
        if not args.no_display:
            cv2.destroyAllWindows()

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Realtime employee tracking with threaded OpenCV capture, MediaPipe face detection, "
            "preloaded face encodings, and RFID matching."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index_parser = subparsers.add_parser("build-index", help="Build a pickle face index from dataset folders.")
    build_index_parser.add_argument("--dataset", required=True, type=Path, help="Root folder with one subfolder per employee.")
    build_index_parser.add_argument("--output", required=True, type=Path, help="Output pickle path.")
    build_index_parser.add_argument("--metadata", type=Path, help="Optional CSV keyed by folder_name.")
    build_index_parser.add_argument("--min-samples", type=int, default=5, help="Minimum encodable images required per employee.")
    build_index_parser.add_argument("--max-anchors", type=int, default=8, help="How many best embeddings to keep per employee.")
    build_index_parser.add_argument("--max-image-size", type=int, default=1600, help="Cap training image dimensions for faster indexing.")
    build_index_parser.add_argument("--upsample", type=int, default=1, help="face_recognition upsample count.")
    build_index_parser.add_argument("--jitters", type=int, default=1, help="Encoding jitter count.")
    build_index_parser.add_argument("--detection-model", choices=("hog", "cnn"), default="hog", help="Offline detector backend.")
    build_index_parser.add_argument("--landmark-model", choices=("small", "large"), default="small", help="Landmark model used by dlib.")
    build_index_parser.set_defaults(handler=build_index)

    run_parser = subparsers.add_parser("run", help="Run realtime face detection + RFID matching.")
    run_parser.add_argument("--index", required=True, type=Path, help="Pickle face index generated by build-index.")
    run_parser.add_argument("--source", default="0", help="Video source. Use 0 for webcam or an RTSP/HTTP stream URL.")
    run_parser.add_argument("--capture-width", type=int, default=1280, help="Requested capture width.")
    run_parser.add_argument("--capture-height", type=int, default=720, help="Requested capture height.")
    run_parser.add_argument("--runtime-width", type=int, default=640, help="Runtime frame width used for detection and drawing.")
    run_parser.add_argument("--runtime-height", type=int, default=480, help="Runtime frame height used for detection and drawing.")
    run_parser.add_argument("--process-every-nth-frame", type=int, default=2, help="Process every Nth frame to lower CPU load.")
    run_parser.add_argument("--max-faces", type=int, default=12, help="Maximum faces to encode per processed frame.")
    run_parser.add_argument("--mediapipe-model", choices=(0, 1), type=int, default=1, help="0=short-range, 1=full-range detector.")
    run_parser.add_argument("--min-detection-confidence", type=float, default=0.55, help="Minimum MediaPipe detection confidence.")
    run_parser.add_argument("--match-threshold", type=float, default=0.48, help="Maximum embedding distance for a face match.")
    run_parser.add_argument("--ambiguity-margin", type=float, default=0.05, help="Required gap between best and second-best match.")
    run_parser.add_argument("--landmark-model", choices=("small", "large"), default="small", help="Encoding landmark model.")
    run_parser.add_argument("--track-ttl-seconds", type=float, default=0.6, help="How long unmatched tracks stay visible.")
    run_parser.add_argument("--rfid-ttl-seconds", type=float, default=4.0, help="How long a scanned RFID tag remains active.")
    run_parser.add_argument("--rfid-stdin", action="store_true", help="Read RFID tags from stdin in a background thread.")
    run_parser.add_argument("--rfid-serial-port", help="Serial port for a live RFID reader, for example COM3 or /dev/ttyUSB0.")
    run_parser.add_argument("--rfid-baudrate", type=int, default=115200, help="RFID serial baudrate.")
    run_parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV preview window.")
    run_parser.set_defaults(handler=run_realtime_pipeline)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

