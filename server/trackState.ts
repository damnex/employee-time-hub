export type TrackDirection = "ENTRY" | "EXIT" | "UNKNOWN";

export interface TrackPositionSample {
  x: number;
  y: number;
  t: number;
}

export interface TrackState {
  track_id: number;
  positions: TrackPositionSample[];
  first_seen: number;
  last_seen: number;
  stable: boolean;
  direction: TrackDirection;
}

export interface TrackStateEvent {
  track_id: number;
  position: {
    x: number;
    y: number;
  };
  timestamp?: number;
}

const MAX_TRACK_POSITIONS = 10;
const STABLE_TRACK_POSITION_COUNT = 5;
const TRACK_VALID_TTL_MS = 1000;
const TRACK_STATE_TTL_MS = 2000;
const TRACK_STATE_CLEANUP_INTERVAL_MS = 500;
const DIRECTION_DELTA_THRESHOLD_PX = Number(process.env.TRACK_DIRECTION_MIN_DELTA_PX ?? "40");

export const trackStateMap = new Map<number, TrackState>();

function normalizeTimestamp(timestamp?: number) {
  if (typeof timestamp === "number" && Number.isFinite(timestamp) && timestamp > 0) {
    return Math.trunc(timestamp);
  }

  return Date.now();
}

function cloneTrackState(state: TrackState): TrackState {
  return {
    ...state,
    positions: state.positions.map((position) => ({ ...position })),
  };
}

function logActiveTrackCount(reason: string) {
  console.debug("[track-state]", reason, {
    activeTracks: trackStateMap.size,
  });
}

export function detectDirection(state: TrackState): TrackDirection {
  const first = state.positions[0];
  const last = state.positions[state.positions.length - 1];

  if (!first || !last) {
    return "UNKNOWN";
  }

  const deltaX = last.x - first.x;

  if (deltaX >= DIRECTION_DELTA_THRESHOLD_PX) {
    return "ENTRY";
  }

  if (deltaX <= -DIRECTION_DELTA_THRESHOLD_PX) {
    return "EXIT";
  }

  return "UNKNOWN";
}

export function updateTrack(trackEvent: TrackStateEvent): TrackState {
  const timestamp = normalizeTimestamp(trackEvent.timestamp);
  const nextPosition: TrackPositionSample = {
    x: trackEvent.position.x,
    y: trackEvent.position.y,
    t: timestamp,
  };

  const existingState = trackStateMap.get(trackEvent.track_id);
  const isNewTrack = !existingState;
  const state: TrackState = existingState
    ? {
      ...existingState,
      positions: [...existingState.positions, nextPosition].slice(-MAX_TRACK_POSITIONS),
      last_seen: timestamp,
    }
    : {
      track_id: trackEvent.track_id,
      positions: [nextPosition],
      first_seen: timestamp,
      last_seen: timestamp,
      stable: false,
      direction: "UNKNOWN",
    };

  state.stable = state.positions.length > STABLE_TRACK_POSITION_COUNT;
  const nextDirection = detectDirection(state);
  if (nextDirection !== state.direction) {
    console.debug("[track-state] direction change", {
      track_id: state.track_id,
      from: state.direction,
      to: nextDirection,
    });
  }
  state.direction = nextDirection;

  trackStateMap.set(trackEvent.track_id, state);
  logActiveTrackCount("update");

  return cloneTrackState(state);
}

export function isValidTrack(state: TrackState, nowMs = Date.now()) {
  return state.stable && (nowMs - state.last_seen) < TRACK_VALID_TTL_MS && state.direction !== "UNKNOWN";
}

export function cleanupTrackStates(nowMs = Date.now()) {
  let removed = 0;

  for (const [trackId, state] of Array.from(trackStateMap.entries())) {
    if ((nowMs - state.last_seen) > TRACK_STATE_TTL_MS) {
      trackStateMap.delete(trackId);
      removed += 1;
    }
  }

  if (removed > 0) {
    logActiveTrackCount("cleanup");
  }
}

export function getTrackState(trackId: number) {
  const state = trackStateMap.get(trackId);
  return state ? cloneTrackState(state) : undefined;
}

export function clearTrackState(trackId: number) {
  trackStateMap.delete(trackId);
}

export function getTrackStateSnapshot() {
  const snapshot = Array.from(trackStateMap.values()).map((state) => cloneTrackState(state));

  return {
    activeTracks: snapshot.length,
    validTracks: snapshot.filter((state) => isValidTrack(state)).length,
    tracks: snapshot,
  };
}

const cleanupTimer = setInterval(() => {
  cleanupTrackStates();
}, TRACK_STATE_CLEANUP_INTERVAL_MS);

cleanupTimer.unref?.();



