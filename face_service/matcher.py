from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class MatchResult:
    person_id: str | None
    name: str
    confidence: float
    similarity: float
    threshold: float
    rfid_tag: str | None
    matched: bool


class FaceMatcher:
    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self._similarity_threshold = max(0.65, min(0.8, similarity_threshold))

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

    def match(self, probe: list[float], employees: list[dict[str, object]], *, face_score: float | None = None) -> MatchResult:
        best: MatchResult | None = None
        second_best_similarity = 0.0
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
                threshold=self._similarity_threshold,
                rfid_tag=str(employee.get("rfid_tag")) if employee.get("rfid_tag") is not None else None,
                matched=False,
            )
            if best is None or candidate.similarity > best.similarity:
                if best is not None:
                    second_best_similarity = best.similarity
                best = candidate
            elif candidate.similarity > second_best_similarity:
                second_best_similarity = candidate.similarity

        if best is None:
            return MatchResult(None, "unknown", 0.0, 0.0, self._similarity_threshold, None, False)

        adaptive_threshold = self._adaptive_threshold(
            best_similarity=best.similarity,
            second_best_similarity=second_best_similarity,
            face_score=face_score,
        )
        if best.similarity < adaptive_threshold:
            return MatchResult(None, "unknown", best.confidence, best.similarity, adaptive_threshold, None, False)
        best.threshold = adaptive_threshold
        best.matched = True
        return best

    def _adaptive_threshold(
        self,
        *,
        best_similarity: float,
        second_best_similarity: float,
        face_score: float | None,
    ) -> float:
        threshold = self._similarity_threshold
        if face_score is not None:
            if face_score < 0.55:
                threshold += 0.08
            elif face_score < 0.7:
                threshold += 0.04

        similarity_gap = best_similarity - second_best_similarity
        if similarity_gap < 0.05:
            threshold += 0.05
        elif similarity_gap < 0.1:
            threshold += 0.03

        if best_similarity < 0.72:
            threshold += 0.03

        return round(max(0.65, min(0.8, threshold)), 4)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Embedding norm is zero.")
        return vector / norm

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms <= 0, 1.0, norms)
        return matrix / norms
