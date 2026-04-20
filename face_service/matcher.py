from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MatchResult:
    person_id: str | None
    name: str
    confidence: float
    similarity: float
    rfid_tag: str | None
    matched: bool


class FaceMatcher:
    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self._similarity_threshold = similarity_threshold

    @property
    def similarity_threshold(self) -> float:
        return self._similarity_threshold

    def compare(self, probe: list[float], gallery: list[list[float]]) -> tuple[float, float]:
        if not gallery:
            return 0.0, 0.0

        probe_vec = self._normalize(np.asarray(probe, dtype=np.float32))
        gallery_mat = np.asarray(gallery, dtype=np.float32)
        gallery_mat = self._normalize_rows(gallery_mat)
        similarities = np.matmul(gallery_mat, probe_vec)
        peak = float(np.max(similarities))
        mean_top = float(np.mean(np.sort(similarities)[-min(3, len(similarities)) :]))
        return peak, mean_top

    def match(self, probe: list[float], employees: list[dict[str, object]]) -> MatchResult:
        best: MatchResult | None = None
        for employee in employees:
            embeddings = employee.get("embeddings")
            if not isinstance(embeddings, list) or not embeddings:
                continue

            peak, mean_top = self.compare(probe, embeddings)
            similarity = (peak * 0.7) + (mean_top * 0.3)
            confidence = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            candidate = MatchResult(
                person_id=str(employee.get("id")),
                name=str(employee.get("name") or employee.get("id") or "unknown"),
                confidence=round(confidence, 4),
                similarity=round(similarity, 4),
                rfid_tag=str(employee.get("rfid_tag")) if employee.get("rfid_tag") is not None else None,
                matched=similarity >= self._similarity_threshold,
            )
            if best is None or candidate.similarity > best.similarity:
                best = candidate

        if best is None:
            return MatchResult(None, "unknown", 0.0, 0.0, None, False)
        if not best.matched:
            return MatchResult(None, "unknown", best.confidence, best.similarity, None, False)
        return best

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Embedding norm is zero.")
        return vector / norm

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms <= 0, 1.0, norms)
        return matrix / norms

