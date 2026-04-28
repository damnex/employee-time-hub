export interface RFIDEvent {
  tag: string;
  timestamp: number;
  rssi?: number;
}

export interface TrackPosition {
  x: number;
  y: number;
}

export interface TrackHistoryPosition extends TrackPosition {
  timestamp: number;
}

export interface TrackEvent {
  track_id: number;
  position: TrackPosition;
  timestamp: number;
  history: TrackHistoryPosition[];
}

export interface FaceEvent {
  track_id: number;
  name?: string;
  confidence?: number;
  embedding?: number[];
  timestamp: number;
}

export interface CurrentSessionSnapshot {
  rfid: RFIDEvent[];
  tracks: TrackEvent[];
  faces: FaceEvent[];
}

type RFIDEventInput = Omit<RFIDEvent, "timestamp"> & { timestamp?: number };
type TrackEventInput = Omit<TrackEvent, "timestamp" | "history"> & { timestamp?: number };
type FaceEventInput = Omit<FaceEvent, "timestamp"> & { timestamp?: number };
type TimestampedEvent = { timestamp: number };

export const BUFFER_WINDOW = 1500;

const CLEANUP_INTERVAL_MS = 500;

export const rfidBuffer: RFIDEvent[] = [];
export const trackBuffer: TrackEvent[] = [];
export const faceBuffer: FaceEvent[] = [];

function normalizeTimestamp(timestamp?: number) {
  if (typeof timestamp === "number" && Number.isFinite(timestamp) && timestamp > 0) {
    return Math.trunc(timestamp);
  }

  return Date.now();
}

function removeExpiredEvents<T extends TimestampedEvent>(buffer: T[], cutoffMs: number) {
  for (let index = buffer.length - 1; index >= 0; index -= 1) {
    if (buffer[index]!.timestamp < cutoffMs) {
      buffer.splice(index, 1);
    }
  }
}

function logBufferSizes(reason: string) {
  console.debug("[event-buffer]", reason, {
    rfid: rfidBuffer.length,
    tracks: trackBuffer.length,
    faces: faceBuffer.length,
  });
}

function cloneTrackEvent(event: TrackEvent): TrackEvent {
  return {
    ...event,
    position: { ...event.position },
    history: event.history.map((position) => ({ ...position })),
  };
}

export function cleanupBuffers(nowMs = Date.now()) {
  const cutoffMs = nowMs - BUFFER_WINDOW;
  const previousSizes = {
    rfid: rfidBuffer.length,
    tracks: trackBuffer.length,
    faces: faceBuffer.length,
  };

  removeExpiredEvents(rfidBuffer, cutoffMs);
  removeExpiredEvents(trackBuffer, cutoffMs);
  removeExpiredEvents(faceBuffer, cutoffMs);

  if (
    previousSizes.rfid !== rfidBuffer.length
    || previousSizes.tracks !== trackBuffer.length
    || previousSizes.faces !== faceBuffer.length
  ) {
    logBufferSizes("cleanup");
  }
}

function buildTrackHistory(trackId: number, cutoffMs: number) {
  return trackBuffer
    .filter((event) => event.track_id === trackId && event.timestamp >= cutoffMs)
    .map((event) => ({
      x: event.position.x,
      y: event.position.y,
      timestamp: event.timestamp,
    }));
}

export function addRFID(event: RFIDEventInput): RFIDEvent {
  const bufferedEvent: RFIDEvent = {
    tag: event.tag.trim().toUpperCase(),
    timestamp: normalizeTimestamp(event.timestamp),
    ...(event.rssi === undefined ? {} : { rssi: event.rssi }),
  };

  rfidBuffer.push(bufferedEvent);
  cleanupBuffers();
  logBufferSizes("addRFID");

  return { ...bufferedEvent };
}

export function addTrack(event: TrackEventInput): TrackEvent {
  const timestamp = normalizeTimestamp(event.timestamp);
  const cutoffMs = timestamp - BUFFER_WINDOW;
  const bufferedEvent: TrackEvent = {
    track_id: event.track_id,
    position: { ...event.position },
    timestamp,
    history: buildTrackHistory(event.track_id, cutoffMs),
  };

  trackBuffer.push(bufferedEvent);
  cleanupBuffers();
  logBufferSizes("addTrack");

  return cloneTrackEvent(bufferedEvent);
}

export function addFace(event: FaceEventInput): FaceEvent {
  const bufferedEvent: FaceEvent = {
    track_id: event.track_id,
    timestamp: normalizeTimestamp(event.timestamp),
    ...(event.name === undefined ? {} : { name: event.name }),
    ...(event.confidence === undefined ? {} : { confidence: event.confidence }),
    ...(event.embedding === undefined ? {} : { embedding: [...event.embedding] }),
  };

  faceBuffer.push(bufferedEvent);
  cleanupBuffers();
  logBufferSizes("addFace");

  return {
    ...bufferedEvent,
    ...(bufferedEvent.embedding === undefined ? {} : { embedding: [...bufferedEvent.embedding] }),
  };
}

export function getCurrentSession(nowMs = Date.now()): CurrentSessionSnapshot {
  cleanupBuffers(nowMs);
  const cutoffMs = nowMs - BUFFER_WINDOW;

  return {
    rfid: rfidBuffer
      .filter((event) => event.timestamp >= cutoffMs)
      .map((event) => ({ ...event })),
    tracks: trackBuffer
      .filter((event) => event.timestamp >= cutoffMs)
      .map((event) => cloneTrackEvent(event)),
    faces: faceBuffer
      .filter((event) => event.timestamp >= cutoffMs)
      .map((event) => ({
        ...event,
        ...(event.embedding === undefined ? {} : { embedding: [...event.embedding] }),
      })),
  };
}

const cleanupTimer = setInterval(() => {
  cleanupBuffers();

  if (process.env.EVENT_BUFFER_LOG_SESSION === "1") {
    console.debug("[event-buffer] session", getCurrentSession());
  }
}, CLEANUP_INTERVAL_MS);

cleanupTimer.unref?.();
