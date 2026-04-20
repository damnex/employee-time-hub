from __future__ import annotations

from dataclasses import dataclass

from .detector import TrackedPerson


@dataclass(slots=True)
class DirectionConfig:
    line_position_fraction: float = 0.5
    deadband_px: int = 16
    event_cooldown_ms: int = 1500
    state_ttl_ms: int = 5000


@dataclass(slots=True)
class TrackDirectionState:
    stable_side: str | None = None
    last_seen_ms: int = 0
    last_event_ms: int = 0


class DirectionEngine:
    def __init__(self, config: DirectionConfig) -> None:
        self._config = config
        self._states: dict[int, TrackDirectionState] = {}

    def update(self, tracks: list[TrackedPerson], frame_width: int, timestamp_ms: int) -> tuple[int, dict[int, str | None]]:
        line_x = int(round(frame_width * self._config.line_position_fraction))
        directions: dict[int, str | None] = {track.track_id: None for track in tracks}

        active_ids = {track.track_id for track in tracks}
        for track in tracks:
            state = self._states.get(track.track_id)
            if state is None:
                state = TrackDirectionState()
                self._states[track.track_id] = state

            current_side = self._classify_side(track.center[0], line_x)
            if (
                current_side in {"left", "right"}
                and state.stable_side in {"left", "right"}
                and current_side != state.stable_side
                and (timestamp_ms - state.last_event_ms) >= self._config.event_cooldown_ms
            ):
                directions[track.track_id] = "ENTRY" if state.stable_side == "left" else "EXIT"
                state.last_event_ms = timestamp_ms

            if current_side in {"left", "right"}:
                state.stable_side = current_side
            state.last_seen_ms = timestamp_ms

        self._prune_inactive_states(active_ids, timestamp_ms)
        return line_x, directions

    def _classify_side(self, center_x: int, line_x: int) -> str:
        if center_x <= line_x - self._config.deadband_px:
            return "left"
        if center_x >= line_x + self._config.deadband_px:
            return "right"
        return "center"

    def _prune_inactive_states(self, active_ids: set[int], timestamp_ms: int) -> None:
        expired_ids = [
            track_id
            for track_id, state in self._states.items()
            if track_id not in active_ids and (timestamp_ms - state.last_seen_ms) > self._config.state_ttl_ms
        ]
        for track_id in expired_ids:
            self._states.pop(track_id, None)
