from __future__ import annotations

from dataclasses import dataclass

from .detector import TrackedPerson


@dataclass(slots=True)
class DirectionConfig:
    line_position_fraction: float = 0.5
    deadband_px: int = 32
    event_cooldown_ms: int = 1500
    state_ttl_ms: int = 5000
    zone_hold_frames: int = 2
    stable_track_min_frames: int = 10


@dataclass(slots=True)
class TrackDirectionState:
    committed_zone: str | None = None
    candidate_zone: str | None = None
    candidate_hits: int = 0
    age_frames: int = 0
    last_seen_ms: int = 0
    last_event_ms: int = 0


@dataclass(slots=True)
class TrackDirectionDecision:
    direction: str | None
    zone: str
    age_frames: int
    stable: bool


@dataclass(slots=True)
class DirectionUpdate:
    line_x: int
    entry_zone_max_x: int
    exit_zone_min_x: int
    decisions: dict[int, TrackDirectionDecision]


class DirectionEngine:
    def __init__(self, config: DirectionConfig) -> None:
        self._config = config
        self._states: dict[int, TrackDirectionState] = {}

    def update(self, tracks: list[TrackedPerson], frame_width: int, timestamp_ms: int) -> DirectionUpdate:
        line_x = int(round(frame_width * self._config.line_position_fraction))
        entry_zone_max_x = line_x - self._config.deadband_px
        exit_zone_min_x = line_x + self._config.deadband_px
        decisions: dict[int, TrackDirectionDecision] = {}

        active_ids = {track.track_id for track in tracks}
        for track in tracks:
            state = self._states.get(track.track_id)
            if state is None:
                state = TrackDirectionState()
                self._states[track.track_id] = state

            state.age_frames += 1
            zone = self._classify_zone(track.center[0], entry_zone_max_x, exit_zone_min_x)
            if zone == state.candidate_zone:
                state.candidate_hits += 1
            else:
                state.candidate_zone = zone
                state.candidate_hits = 1

            previous_committed = state.committed_zone
            committed_zone = previous_committed
            if zone in {"entry", "exit"} and state.candidate_hits >= self._config.zone_hold_frames:
                committed_zone = zone

            stable = state.age_frames >= self._config.stable_track_min_frames
            direction: str | None = None
            if (
                stable
                and committed_zone in {"entry", "exit"}
                and previous_committed in {"entry", "exit"}
                and committed_zone != previous_committed
                and (timestamp_ms - state.last_event_ms) >= self._config.event_cooldown_ms
            ):
                direction = "ENTRY" if previous_committed == "entry" else "EXIT"
                state.last_event_ms = timestamp_ms

            state.committed_zone = committed_zone
            state.last_seen_ms = timestamp_ms
            decisions[track.track_id] = TrackDirectionDecision(
                direction=direction,
                zone=zone,
                age_frames=state.age_frames,
                stable=stable,
            )

        self._prune_inactive_states(active_ids, timestamp_ms)
        return DirectionUpdate(
            line_x=line_x,
            entry_zone_max_x=entry_zone_max_x,
            exit_zone_min_x=exit_zone_min_x,
            decisions=decisions,
        )

    def _classify_zone(self, center_x: int, entry_zone_max_x: int, exit_zone_min_x: int) -> str:
        if center_x <= entry_zone_max_x:
            return "entry"
        if center_x >= exit_zone_min_x:
            return "exit"
        return "buffer"

    def _prune_inactive_states(self, active_ids: set[int], timestamp_ms: int) -> None:
        expired_ids = [
            track_id
            for track_id, state in self._states.items()
            if track_id not in active_ids and (timestamp_ms - state.last_seen_ms) > self._config.state_ttl_ms
        ]
        for track_id in expired_ids:
            self._states.pop(track_id, None)
