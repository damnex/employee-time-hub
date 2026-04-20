from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any

import cv2  # type: ignore
import numpy as np

try:
    import onnxruntime as ort  # type: ignore
except ImportError:  # pragma: no cover
    ort = None

try:
    from insightface.app import FaceAnalysis  # type: ignore
except ImportError:  # pragma: no cover
    FaceAnalysis = None


LOGGER = logging.getLogger("face_service.model")


@dataclass(slots=True)
class FaceModelConfig:
    model_name: str = "buffalo_l"
    model_root: str | None = None
    providers: list[str] | None = None
    ctx_id: int | None = None
    det_size: tuple[int, int] = (640, 640)


def decode_image(image_payload: str) -> np.ndarray:
    payload = image_payload.strip()
    if not payload:
        raise ValueError("Image payload is empty.")

    if payload.startswith("data:"):
        _, encoded = payload.split(",", 1)
    else:
        encoded = payload

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Invalid image payload. Expected a base64 image or data URL.") from exc

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image payload.")
    return image


def encode_image(image: np.ndarray, extension: str = ".jpg", quality: int = 95) -> str:
    params: list[int] = []
    if extension.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise ValueError("Unable to encode image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("Embedding norm is zero.")
    return vector / norm


def choose_providers(explicit_providers: list[str] | None = None) -> list[str]:
    if explicit_providers:
        return explicit_providers

    if ort is None:
        return ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def choose_ctx_id(providers: list[str], explicit_ctx_id: int | None = None) -> int:
    if explicit_ctx_id is not None:
        return explicit_ctx_id
    return 0 if "CUDAExecutionProvider" in providers else -1


class InsightFaceModel:
    def __init__(self, config: FaceModelConfig) -> None:
        if FaceAnalysis is None:  # pragma: no cover
            raise RuntimeError(
                "InsightFace is not installed. Install face_service/requirements.txt before starting the service."
            )

        providers = choose_providers(config.providers)
        ctx_id = choose_ctx_id(providers, config.ctx_id)
        kwargs: dict[str, Any] = {"name": config.model_name, "providers": providers}
        if config.model_root:
            kwargs["root"] = config.model_root

        self._providers = providers
        self._ctx_id = ctx_id
        self._app = FaceAnalysis(**kwargs)
        self._app.prepare(ctx_id=ctx_id, det_size=config.det_size)
        LOGGER.info(
            "Loaded InsightFace model %s with providers=%s ctx_id=%s det_size=%s",
            config.model_name,
            providers,
            ctx_id,
            config.det_size,
        )

    @property
    def providers(self) -> list[str]:
        return list(self._providers)

    @property
    def ctx_id(self) -> int:
        return self._ctx_id

    def detect(self, image: np.ndarray) -> list[Any]:
        return list(self._app.get(image))

    def normalize_embedding(self, face: Any) -> list[float]:
        if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
            vector = np.asarray(face.normed_embedding, dtype=np.float32)
        elif hasattr(face, "embedding") and face.embedding is not None:
            vector = l2_normalize(np.asarray(face.embedding, dtype=np.float32))
        else:
            raise ValueError("InsightFace face object did not include an embedding.")
        return vector.astype(np.float32).tolist()
