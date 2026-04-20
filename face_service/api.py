from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .database import FaceDatabase, FaceDatabaseConfig
from .detector import FaceCropDetector, FaceDetectorConfig, TrackedPersonInput
from .matcher import FaceMatcher
from .model import FaceModelConfig, InsightFaceModel, decode_image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("face_service.api")


class TrackInput(BaseModel):
    track_id: int
    bbox: list[int] = Field(min_length=4, max_length=4)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[int]) -> list[int]:
        if value[2] <= value[0] or value[3] <= value[1]:
            raise ValueError("Track bbox must be [x1, y1, x2, y2] with positive width and height.")
        return value


class RegisterFaceRequest(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    rfid_tag: str | None = None
    images: list[str] = Field(min_length=5, max_length=10)


class RecognizeRequest(BaseModel):
    frame: str = Field(min_length=1)
    tracks: list[TrackInput] = Field(default_factory=list)
    frame_index: int = 0
    stream_id: str = "default"


@dataclass(slots=True)
class TrackCacheEntry:
    frame_index: int
    result: dict[str, object]


class FaceRecognitionService:
    def __init__(self) -> None:
        model_name = os.getenv("FACE_MODEL_NAME", "buffalo_l")
        model_root = os.getenv("FACE_MODEL_ROOT")
        det_width = int(os.getenv("FACE_DET_WIDTH", "640"))
        det_height = int(os.getenv("FACE_DET_HEIGHT", "640"))
        similarity_threshold = float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.7"))
        crop_margin_ratio = float(os.getenv("FACE_CROP_MARGIN_RATIO", "0.12"))
        min_face_box_size = int(os.getenv("FACE_MIN_BOX_SIZE", "40"))
        min_face_score = float(os.getenv("FACE_MIN_SCORE", "0.45"))
        self._recognize_every_n_frames = max(1, int(os.getenv("FACE_RECOGNIZE_EVERY_N_FRAMES", "5")))
        self._database = FaceDatabase(
            FaceDatabaseConfig(
                path=os.getenv("FACE_DB_PATH", "face_service/data/faces.json"),
            )
        )
        self._model = InsightFaceModel(
            FaceModelConfig(
                model_name=model_name,
                model_root=model_root,
                det_size=(det_width, det_height),
            )
        )
        self._detector = FaceCropDetector(
            self._model,
            FaceDetectorConfig(
                crop_margin_ratio=crop_margin_ratio,
                min_face_box_size=min_face_box_size,
                min_face_score=min_face_score,
            ),
        )
        self._matcher = FaceMatcher(similarity_threshold=similarity_threshold)
        self._cache_lock = threading.Lock()
        self._track_cache: dict[str, TrackCacheEntry] = {}

    def get_status(self) -> dict[str, object]:
        return {
            "ready": True,
            "model_name": os.getenv("FACE_MODEL_NAME", "buffalo_l"),
            "providers": self._model.providers,
            "ctx_id": self._model.ctx_id,
            "recognize_every_n_frames": self._recognize_every_n_frames,
            "employees": len(self._database.list_faces()),
            "threshold": self._matcher.similarity_threshold,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def list_faces(self) -> dict[str, object]:
        employees = self._database.list_faces()
        return {
            "count": len(employees),
            "faces": [
                {
                    "id": employee.get("id"),
                    "name": employee.get("name"),
                    "rfid_tag": employee.get("rfid_tag"),
                    "embedding_count": employee.get("embedding_count", len(employee.get("embeddings", []))),
                    "updated_at": employee.get("updated_at"),
                }
                for employee in employees
            ],
        }

    def register_face(self, request: RegisterFaceRequest) -> dict[str, object]:
        embeddings: list[list[float]] = []
        for image_payload in request.images:
            image = decode_image(image_payload)
            faces = self._detector.detect_faces_in_image(image)
            if len(faces) != 1:
                raise ValueError(
                    f"Each registration image must contain exactly one usable face. Got {len(faces)} face(s)."
                )
            embeddings.append(faces[0].embedding)

        employee = self._database.upsert_employee(
            employee_id=request.id,
            name=request.name,
            rfid_tag=request.rfid_tag,
            embeddings=embeddings,
        )
        LOGGER.info(
            "Registered face embeddings for employee_id=%s name=%s embeddings=%s",
            employee["id"],
            employee["name"],
            employee["embedding_count"],
        )
        return {
            "success": True,
            "employee": {
                "id": employee["id"],
                "name": employee["name"],
                "rfid_tag": employee["rfid_tag"],
                "embedding_count": employee["embedding_count"],
            },
        }

    def recognize(self, request: RecognizeRequest) -> dict[str, object]:
        frame = decode_image(request.frame)
        tracks = [
            TrackedPersonInput(track_id=track.track_id, bbox=tuple(int(value) for value in track.bbox))
            for track in request.tracks
        ]
        employees = self._database.list_faces()
        detections = self._detector.detect_faces_for_tracks(frame, tracks)
        detections_by_track = {det.track_id: det for det in detections}

        results: list[dict[str, object]] = []
        for track in tracks:
            cached = self._get_cached_result(request.stream_id, track.track_id, request.frame_index)
            if cached is not None and track.track_id not in detections_by_track:
                results.append(self._strip_internal_fields(cached))
                continue
            if cached is not None and request.frame_index - cached["last_recognized_frame"] < self._recognize_every_n_frames:
                results.append(self._strip_internal_fields(cached))
                continue

            detection = detections_by_track.get(track.track_id)
            if detection is None:
                result = {
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "face_bbox": None,
                    "name": "unknown",
                    "person_id": None,
                    "rfid_tag": None,
                    "confidence": 0.0,
                    "similarity": 0.0,
                    "matched": False,
                    "last_recognized_frame": request.frame_index,
                }
                self._store_cached_result(request.stream_id, track.track_id, request.frame_index, result)
                results.append(self._strip_internal_fields(result))
                continue

            match = self._matcher.match(detection.embedding, employees)
            result = {
                "track_id": track.track_id,
                "bbox": list(track.bbox),
                "face_bbox": list(detection.face_bbox),
                "name": match.name,
                "person_id": match.person_id,
                "rfid_tag": match.rfid_tag,
                "confidence": match.confidence,
                "similarity": match.similarity,
                "matched": match.matched,
                "last_recognized_frame": request.frame_index,
            }
            self._store_cached_result(request.stream_id, track.track_id, request.frame_index, result)
            results.append(self._strip_internal_fields(result))

        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "stream_id": request.stream_id,
            "tracks": results,
        }

    def _cache_key(self, stream_id: str, track_id: int) -> str:
        return f"{stream_id}:{track_id}"

    def _get_cached_result(self, stream_id: str, track_id: int, frame_index: int) -> dict[str, object] | None:
        with self._cache_lock:
            entry = self._track_cache.get(self._cache_key(stream_id, track_id))
            if entry is None:
                return None
            if frame_index < entry.frame_index:
                return None
            return dict(entry.result)

    def _store_cached_result(
        self,
        stream_id: str,
        track_id: int,
        frame_index: int,
        result: dict[str, object],
    ) -> None:
        with self._cache_lock:
            self._track_cache[self._cache_key(stream_id, track_id)] = TrackCacheEntry(
                frame_index=frame_index,
                result=dict(result),
            )

    def _strip_internal_fields(self, result: dict[str, object]) -> dict[str, object]:
        output = dict(result)
        output.pop("last_recognized_frame", None)
        return output


service = FaceRecognitionService()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield


app = FastAPI(title="Face Recognition Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status")
def get_status() -> dict[str, object]:
    return service.get_status()


@app.get("/faces")
def get_faces() -> dict[str, object]:
    return service.list_faces()


@app.post("/register-face")
def register_face(request: RegisterFaceRequest) -> dict[str, object]:
    try:
        return service.register_face(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/recognize")
def recognize(request: RecognizeRequest) -> dict[str, object]:
    try:
        return service.recognize(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
