import type { ScanTechnology } from "@shared/schema";

export type MovementDirection = "ENTRY" | "EXIT" | "UNKNOWN";
export type TrackZone = "ENTRY_ZONE" | "EXIT_ZONE" | "BUFFER_ZONE";

export interface RfidObservation {
  id: string;
  deviceId: string;
  rfidTag: string;
  timestampMs: number;
  scanTechnology: ScanTechnology;
}

export interface VisionTrackObservation {
  deviceId: string;
  trackId: number;
  timestampMs: number;
  bbox: [number, number, number, number];
  center: [number, number];
  direction: MovementDirection;
  zone?: TrackZone;
  ageFrames?: number;
  stable?: boolean;
  confidence?: number;
  frameWidth?: number;
  frameHeight?: number;
}

export interface FaceObservation {
  deviceId: string;
  trackId: number;
  timestampMs: number;
  name: string;
  confidence: number;
  similarity?: number;
  personId?: string | null;
  rfidTag?: string | null;
  matched: boolean;
  bbox?: [number, number, number, number] | null;
}

export interface MatchingEngineConfig {
  matchWindowMs: number;
  minBufferMs: number;
  pendingTtlMs: number;
  faceTtlMs: number;
  visionTtlMs: number;
  trackCooldownMs: number;
  minStableTrackAgeFrames: number;
  entryZoneMaxFraction: number;
  exitZoneMinFraction: number;
  gateAnchorXFraction: number;
  gateAnchorYFraction: number;
}

export interface MatchIdentity {
  employeeId?: number;
  employeeCode?: string;
  employeeName?: string;
  rfidTag: string;
}

export interface CandidateMatch {
  rfid: RfidObservation;
  track: VisionTrackObservation;
  face: FaceObservation | null;
  timeDeltaMs: number;
  spatialDistance: number;
  faceMatched: boolean;
  directionValid: boolean;
  resolvedDirection: MovementDirection;
  zone: TrackZone;
  stableTrack: boolean;
  score: number;
}

const RFID_SCORE = 50;
const DIRECTION_SCORE = 30;
const FACE_SCORE = 20;

const DEFAULT_CONFIG: MatchingEngineConfig = {
  matchWindowMs: 1500,
  minBufferMs: 250,
  pendingTtlMs: 2000,
  faceTtlMs: 2000,
  visionTtlMs: 2000,
  trackCooldownMs: 500,
  minStableTrackAgeFrames: 10,
  entryZoneMaxFraction: 0.4,
  exitZoneMinFraction: 0.6,
  gateAnchorXFraction: 0.5,
  gateAnchorYFraction: 0.5,
};

function normalizeDirection(direction?: string | null): MovementDirection {
  if (direction === "ENTRY" || direction === "EXIT") {
    return direction;
  }
  return "UNKNOWN";
}

function normalizeZone(zone?: string | null): TrackZone | undefined {
  if (zone === "ENTRY_ZONE" || zone === "EXIT_ZONE" || zone === "BUFFER_ZONE") {
    return zone;
  }
  if (zone === "entry") {
    return "ENTRY_ZONE";
  }
  if (zone === "exit") {
    return "EXIT_ZONE";
  }
  if (zone === "buffer") {
    return "BUFFER_ZONE";
  }
  return undefined;
}

function faceMatchesIdentity(face: FaceObservation | null, identity: MatchIdentity) {
  if (!face || !face.matched) {
    return false;
  }

  const normalizedTag = identity.rfidTag.trim().toUpperCase();
  const normalizedName = identity.employeeName?.trim().toLowerCase();
  const normalizedCode = identity.employeeCode?.trim().toLowerCase();
  const normalizedEmployeeId = identity.employeeId != null ? String(identity.employeeId) : null;
  const faceTag = face.rfidTag?.trim().toUpperCase();
  const facePersonId = face.personId?.trim().toLowerCase();
  const faceName = face.name.trim().toLowerCase();

  return Boolean(
    (faceTag && faceTag === normalizedTag)
    || (normalizedCode && facePersonId && facePersonId === normalizedCode)
    || (normalizedEmployeeId && facePersonId && facePersonId === normalizedEmployeeId)
    || (normalizedName && faceName === normalizedName)
  );
}

export class MatchingEngine {
  private readonly config: MatchingEngineConfig;
  private readonly pendingRfid = new Map<string, RfidObservation>();
  private readonly visionByDevice = new Map<string, Map<number, VisionTrackObservation>>();
  private readonly faceByDeviceTrack = new Map<string, FaceObservation>();
  private readonly reservedTracks = new Map<string, number>();

  constructor(config: Partial<MatchingEngineConfig> = {}) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
    };
  }

  recordRfid(input: Omit<RfidObservation, "id">): RfidObservation {
    const normalizedTag = input.rfidTag.trim().toUpperCase();
    const observation: RfidObservation = {
      ...input,
      rfidTag: normalizedTag,
      id: `${input.deviceId}:${normalizedTag}:${input.timestampMs}`,
    };
    this.pendingRfid.set(observation.id, observation);
    this.prune(observation.timestampMs);
    return observation;
  }

  recordVisionBatch(observations: VisionTrackObservation[]) {
    for (const observation of observations) {
      const byTrack = this.visionByDevice.get(observation.deviceId) ?? new Map<number, VisionTrackObservation>();
      const normalizedTrack = {
        ...observation,
        direction: normalizeDirection(observation.direction),
        zone: observation.zone ?? this.classifyZone(observation),
        stable: observation.stable ?? ((observation.ageFrames ?? 0) >= this.config.minStableTrackAgeFrames),
      } satisfies VisionTrackObservation;
      byTrack.set(observation.trackId, normalizedTrack);
      this.visionByDevice.set(observation.deviceId, byTrack);
    }

    if (observations[0]) {
      this.prune(observations[0].timestampMs);
    }
  }

  recordFaceBatch(observations: FaceObservation[]) {
    for (const observation of observations) {
      this.faceByDeviceTrack.set(this.getFaceKey(observation.deviceId, observation.trackId), {
        ...observation,
        name: observation.name || "unknown",
      });
    }

    if (observations[0]) {
      this.prune(observations[0].timestampMs);
    }
  }

  getPendingRfid(deviceId?: string) {
    const items = Array.from(this.pendingRfid.values()).sort((left, right) => left.timestampMs - right.timestampMs);
    return deviceId ? items.filter((item) => item.deviceId === deviceId) : items;
  }

  isReadyToResolve(rfid: RfidObservation, nowMs = Date.now()) {
    return (nowMs - rfid.timestampMs) >= this.config.minBufferMs;
  }

  hasExpired(rfid: RfidObservation, nowMs = Date.now()) {
    return (nowMs - rfid.timestampMs) > this.config.pendingTtlMs;
  }

  consumeRfid(observationId: string) {
    this.pendingRfid.delete(observationId);
  }

  reserveTrack(deviceId: string, trackId: number, timestampMs: number) {
    this.reservedTracks.set(this.getFaceKey(deviceId, trackId), timestampMs);
  }

  clearTrackContext(deviceId: string, trackId: number) {
    const deviceTracks = this.visionByDevice.get(deviceId);
    deviceTracks?.delete(trackId);
    if (deviceTracks && !deviceTracks.size) {
      this.visionByDevice.delete(deviceId);
    }

    this.faceByDeviceTrack.delete(this.getFaceKey(deviceId, trackId));
  }

  getSnapshot(nowMs = Date.now()) {
    this.prune(nowMs);
    return {
      pendingRfid: this.pendingRfid.size,
      reservedTracks: this.reservedTracks.size,
      devices: Array.from(this.visionByDevice.entries()).map(([deviceId, tracks]) => ({
        deviceId,
        tracks: tracks.size,
      })),
      faceCache: this.faceByDeviceTrack.size,
    };
  }

  findBestCandidate(rfid: RfidObservation, identity: MatchIdentity): CandidateMatch | null {
    const tracks = Array.from(this.visionByDevice.get(rfid.deviceId)?.values() ?? []);
    const eligible = tracks
      .filter((track) => Math.abs(track.timestampMs - rfid.timestampMs) <= this.config.matchWindowMs)
      .filter((track) => !this.isTrackReserved(track.deviceId, track.trackId, rfid.timestampMs))
      .map((track) => this.buildCandidate(rfid, track, identity))
      .filter((candidate): candidate is CandidateMatch => candidate !== null)
      .sort((left, right) => {
        if (right.score !== left.score) {
          return right.score - left.score;
        }
        const leftTrackAge = left.track.ageFrames ?? 0;
        const rightTrackAge = right.track.ageFrames ?? 0;
        if (rightTrackAge !== leftTrackAge) {
          return rightTrackAge - leftTrackAge;
        }
        if (left.spatialDistance !== right.spatialDistance) {
          return left.spatialDistance - right.spatialDistance;
        }
        if (left.timeDeltaMs !== right.timeDeltaMs) {
          return left.timeDeltaMs - right.timeDeltaMs;
        }
        return (right.face?.confidence ?? 0) - (left.face?.confidence ?? 0);
      });

    return eligible[0] ?? null;
  }

  prune(nowMs = Date.now()) {
    for (const [id, observation] of Array.from(this.pendingRfid.entries())) {
      if ((nowMs - observation.timestampMs) > this.config.pendingTtlMs) {
        this.pendingRfid.delete(id);
      }
    }

    for (const [deviceId, tracks] of Array.from(this.visionByDevice.entries())) {
      for (const [trackId, track] of Array.from(tracks.entries())) {
        if ((nowMs - track.timestampMs) > this.config.visionTtlMs) {
          tracks.delete(trackId);
        }
      }
      if (!tracks.size) {
        this.visionByDevice.delete(deviceId);
      }
    }

    for (const [key, face] of Array.from(this.faceByDeviceTrack.entries())) {
      if ((nowMs - face.timestampMs) > this.config.faceTtlMs) {
        this.faceByDeviceTrack.delete(key);
      }
    }

    for (const [key, reservedAt] of Array.from(this.reservedTracks.entries())) {
      if ((nowMs - reservedAt) > this.config.trackCooldownMs) {
        this.reservedTracks.delete(key);
      }
    }
  }

  private buildCandidate(
    rfid: RfidObservation,
    track: VisionTrackObservation,
    identity: MatchIdentity,
  ): CandidateMatch | null {
    const zone = track.zone ?? this.classifyZone(track);
    const stableTrack = track.stable ?? ((track.ageFrames ?? 0) >= this.config.minStableTrackAgeFrames);
    if (!stableTrack || zone === "BUFFER_ZONE") {
      return null;
    }

    const face = this.getFaceObservation(track.deviceId, track.trackId, rfid.timestampMs);
    const spatialDistance = this.calculateSpatialDistance(track);
    const timeDeltaMs = Math.abs(track.timestampMs - rfid.timestampMs);
    const faceMatched = faceMatchesIdentity(face, identity);
    const resolvedDirection = this.resolveDirection(track.direction, zone);
    const directionValid = resolvedDirection === "ENTRY" || resolvedDirection === "EXIT";
    const score = RFID_SCORE + (directionValid ? DIRECTION_SCORE : 0) + (faceMatched ? FACE_SCORE : 0);

    return {
      rfid,
      track,
      face,
      timeDeltaMs,
      spatialDistance,
      faceMatched,
      directionValid,
      resolvedDirection,
      zone,
      stableTrack,
      score,
    };
  }

  private calculateSpatialDistance(track: VisionTrackObservation) {
    const frameWidth = Math.max(track.frameWidth ?? 0, track.center[0] * 2, 1);
    const frameHeight = Math.max(track.frameHeight ?? 0, track.center[1] * 2, 1);
    const anchorX = frameWidth * this.config.gateAnchorXFraction;
    const anchorY = frameHeight * this.config.gateAnchorYFraction;
    const dx = track.center[0] - anchorX;
    const dy = track.center[1] - anchorY;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private getFaceObservation(deviceId: string, trackId: number, timestampMs: number) {
    const face = this.faceByDeviceTrack.get(this.getFaceKey(deviceId, trackId)) ?? null;
    if (!face) {
      return null;
    }
    if (Math.abs(face.timestampMs - timestampMs) > this.config.matchWindowMs) {
      return null;
    }
    return face;
  }

  private resolveDirection(direction: MovementDirection, zone: TrackZone): MovementDirection {
    if (direction === "ENTRY" || direction === "EXIT") {
      return direction;
    }
    if (zone === "ENTRY_ZONE") {
      return "ENTRY";
    }
    if (zone === "EXIT_ZONE") {
      return "EXIT";
    }
    return "UNKNOWN";
  }

  private classifyZone(track: VisionTrackObservation): TrackZone {
    const zone = normalizeZone(track.zone);
    if (zone) {
      return zone;
    }

    const frameWidth = Math.max(track.frameWidth ?? 0, track.center[0] * 2, 1);
    const entryBoundary = frameWidth * this.config.entryZoneMaxFraction;
    const exitBoundary = frameWidth * this.config.exitZoneMinFraction;
    if (track.center[0] <= entryBoundary) {
      return "ENTRY_ZONE";
    }
    if (track.center[0] >= exitBoundary) {
      return "EXIT_ZONE";
    }
    return "BUFFER_ZONE";
  }

  private isTrackReserved(deviceId: string, trackId: number, timestampMs: number) {
    const reservedAt = this.reservedTracks.get(this.getFaceKey(deviceId, trackId));
    if (reservedAt == null) {
      return false;
    }
    return (timestampMs - reservedAt) <= this.config.trackCooldownMs;
  }

  private getFaceKey(deviceId: string, trackId: number) {
    return `${deviceId}:${trackId}`;
  }
}
