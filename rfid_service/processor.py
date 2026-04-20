from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass


LOGGER = logging.getLogger("rfid_service.processor")


@dataclass(slots=True)
class TagObservation:
    epc: str
    seen_at: float
    raw_hex: str | None = None


@dataclass(slots=True)
class ActiveTagState:
    epc: str
    first_seen_at: float
    last_seen_at: float
    detections: int = 1


@dataclass(slots=True)
class RegistrationState:
    mode: str = "normal"
    selected_tag: str | None = None
    candidate_tag: str | None = None
    candidate_hits: int = 0
    stable_threshold: int = 5
    multiple_tags_detected: bool = False
    message: str = "Registration mode inactive."
    selected_at: float | None = None
    last_seen_at: float | None = None


class TagProcessor:
    def __init__(
        self,
        exit_timeout_seconds: float = 5.0,
        registration_stable_hits: int = 5,
        registration_window_seconds: float = 1.5,
        registration_gap_seconds: float = 1.0,
        recent_history_limit: int = 100,
    ) -> None:
        self.exit_timeout_seconds = exit_timeout_seconds
        self.registration_stable_hits = registration_stable_hits
        self.registration_window_seconds = registration_window_seconds
        self.registration_gap_seconds = registration_gap_seconds
        self._recent_observations: deque[TagObservation] = deque(maxlen=recent_history_limit)
        self._registration_observations: deque[tuple[str, float]] = deque()
        self.last_seen: dict[str, float] = {}
        self.active_tags: dict[str, ActiveTagState] = {}
        self.registration = RegistrationState(stable_threshold=registration_stable_hits)
        self.last_detected_tag: str | None = None
        self.last_detected_at: float | None = None
        self.last_packet_hex: str | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sweeper: threading.Thread | None = None

    def start(self) -> None:
        if self._sweeper and self._sweeper.is_alive():
            return
        self._stop_event.clear()
        self._sweeper = threading.Thread(target=self._sweeper_loop, name="rfid-tag-sweeper", daemon=True)
        self._sweeper.start()

    def stop(self, *, force_exit: bool = True) -> None:
        self._stop_event.set()
        if self._sweeper and self._sweeper.is_alive():
            self._sweeper.join(timeout=2.0)
        self._sweeper = None
        if force_exit:
            self.clear_active_tags("Reader stopped.")

    def set_mode(self, mode: str) -> None:
        with self._lock:
            self.registration.mode = mode
            self._registration_observations.clear()
            self.registration.selected_tag = None
            self.registration.candidate_tag = None
            self.registration.candidate_hits = 0
            self.registration.multiple_tags_detected = False
            self.registration.selected_at = None
            self.registration.last_seen_at = None
            self.registration.message = (
                "Keep only one tag near the reader."
                if mode == "registration"
                else "Registration mode inactive."
            )

    def process_tags(self, tags: list[str], raw_hex: str | None = None) -> None:
        now = time.time()
        unique_tags = list(dict.fromkeys(tags))
        if not unique_tags:
            return

        with self._lock:
            self.last_detected_tag = unique_tags[0]
            self.last_detected_at = now
            self.last_packet_hex = raw_hex
            for tag in unique_tags:
                self.last_seen[tag] = now
                active_state = self.active_tags.get(tag)
                if active_state is None:
                    active_state = ActiveTagState(epc=tag, first_seen_at=now, last_seen_at=now)
                    self.active_tags[tag] = active_state
                    LOGGER.info("ENTRY %s", tag)
                else:
                    active_state.last_seen_at = now
                    active_state.detections += 1

                self._recent_observations.appendleft(TagObservation(epc=tag, seen_at=now, raw_hex=raw_hex))

            self._update_registration_state(unique_tags, now)

    def clear_active_tags(self, reason: str) -> None:
        with self._lock:
            for active_state in list(self.active_tags.values()):
                LOGGER.info("EXIT %s (%s)", active_state.epc, reason)
            self.active_tags.clear()

    def get_recent_tags(self, limit: int = 20) -> list[dict[str, object]]:
        with self._lock:
            return [asdict(observation) for observation in list(self._recent_observations)[:limit]]

    def get_active_tags(self) -> list[dict[str, object]]:
        now = time.time()
        with self._lock:
            self._expire_inactive_tags_locked(now)
            rows: list[dict[str, object]] = []
            for active_state in self.active_tags.values():
                rows.append(
                    {
                        "epc": active_state.epc,
                        "first_seen_at": active_state.first_seen_at,
                        "last_seen_at": active_state.last_seen_at,
                        "detections": active_state.detections,
                        "age_seconds": round(now - active_state.first_seen_at, 3),
                        "idle_seconds": round(now - active_state.last_seen_at, 3),
                    }
                )
            rows.sort(key=lambda row: row["last_seen_at"], reverse=True)
            return rows

    def get_registration_state(self) -> dict[str, object]:
        with self._lock:
            return asdict(self.registration)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._expire_inactive_tags_locked(time.time())
            return {
                "last_detected_tag": self.last_detected_tag,
                "last_detected_at": self.last_detected_at,
                "last_packet_hex": self.last_packet_hex,
                "active_tag_count": len(self.active_tags),
                "recent_tags": [asdict(observation) for observation in list(self._recent_observations)[:20]],
                "registration": asdict(self.registration),
            }

    def _sweeper_loop(self) -> None:
        sweep_interval = min(0.25, max(0.05, self.exit_timeout_seconds / 4))
        while not self._stop_event.is_set():
            self._expire_inactive_tags()
            time.sleep(sweep_interval)

    def _expire_inactive_tags(self) -> None:
        now = time.time()
        with self._lock:
            self._expire_inactive_tags_locked(now)

    def _expire_inactive_tags_locked(self, now: float) -> None:
        expired = [
            tag
            for tag, active_state in self.active_tags.items()
            if (now - active_state.last_seen_at) >= self.exit_timeout_seconds
        ]
        for tag in expired:
            self.active_tags.pop(tag, None)
            self.last_seen.pop(tag, None)
            LOGGER.info("EXIT %s", tag)

    def _trim_registration_observations(self, now: float) -> None:
        while self._registration_observations and (now - self._registration_observations[0][1]) > self.registration_window_seconds:
            self._registration_observations.popleft()

    def _update_registration_state(self, tags: list[str], now: float) -> None:
        if self.registration.mode != "registration":
            return

        for tag in tags:
            self._registration_observations.append((tag, now))
        self._trim_registration_observations(now)

        distinct_recent_tags = {tag for tag, _ in self._registration_observations}
        if len(set(tags)) > 1 or len(distinct_recent_tags) > 1:
            self.registration.selected_tag = None
            self.registration.selected_at = None
            self.registration.candidate_tag = None
            self.registration.candidate_hits = 0
            self.registration.last_seen_at = now
            self.registration.multiple_tags_detected = True
            self.registration.message = "Multiple tags detected. Keep only one tag near the reader."
            return

        tag = tags[0]
        if (
            self.registration.candidate_tag == tag
            and self.registration.last_seen_at is not None
            and (now - self.registration.last_seen_at) <= self.registration_gap_seconds
        ):
            self.registration.candidate_hits += 1
        else:
            self.registration.candidate_tag = tag
            self.registration.candidate_hits = 1

        self.registration.last_seen_at = now
        self.registration.multiple_tags_detected = False

        if self.registration.candidate_hits >= self.registration_stable_hits:
            self.registration.selected_tag = tag
            self.registration.selected_at = now
            self.registration.message = f"Stable registration tag selected: {tag}"
        else:
            self.registration.message = (
                f"Reading a single tag... stability {self.registration.candidate_hits}/{self.registration_stable_hits}"
            )
