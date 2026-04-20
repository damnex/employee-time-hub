import type { ScanTechnology } from "@shared/schema";

export type MovementDirection = "ENTRY" | "EXIT" | "UNKNOWN";

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
  frameWidth?: number;
  frameHeight?: number;
}

export interface FaceObservation {
  deviceId: string;
  trackId: number;
  timestampMs: number;
  name: string;
  confidence: number;
  personId?: string | null;
  rfidTag?: string | null;
  matched: boolean;
  bbox?: [number, number, number, number] | null;
}

export interface MatchingEngineConfig {
  matchWindowMs: number;
  pendingTtlMs: number;
  faceTtlMs: number;
  visionTtlMs: number;
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
  score: number;
}

const DEFAULT_CONFIG: MatchingEngineConfig = {
  matchWindowMs: 1000,
  pendingTtlMs: 2000,
  faceTtlMs: 1500,
  visionTtlMs: 1500,
  gateAnchorXFraction: 0.5,
  gateAnchorYFraction: 0.5,
};

function normalizeDirection(direction?: string | null): MovementDirection {
  if (direction === "ENTRY" || direction === "EXIT") {
    return direction;
  }
  return "UNKNOWN";
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
    || (normalizedName && faceName === normalizedName),
  );
}

export class MatchingEngine {
  private readonly config: MatchingEngineConfig;
  private readonly pendingRfid = new Map<string, RfidObservation>();
  private readonly visionByDevice = new Map<string, Map<number, VisionTrackObservation>>();
  private readonly faceByDeviceTrack = new Map<string, FaceObservation>();

  constructor(config: Partial<MatchingEngineConfig> = {}) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
    };
  }

  recordRfid(input: Omit<RfidObservation, "id">): RfidObservation {
    const observation: RfidObservation = {
      ...input,
      rfidTag: input.rfidTag.trim().toUpperCase(),
      id: `${input.deviceId}:${input.rfidTag.trim().toUpperCase()}:${input.timestampMs}`,
    };
    this.pendingRfid.set(observation.id, observation);
    this.prune(observation.timestampMs);
    return observation;
  }

  recordVisionBatch(observations: VisionTrackObservation[]) {
    for (const observation of observations) {
      const byTrack = this.visionByDevice.get(observation.deviceId) ?? new Map<number, VisionTrackObservation>();
      byTrack.set(observation.trackId, {
        ...observation,
        direction: normalizeDirection(observation.direction),
      });
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
    const items = Array.from(this.pendingRfid.values());
    return deviceId ? items.filter((item) => item.deviceId === deviceId) : items;
  }

  consumeRfid(observationId: string) {
    this.pendingRfid.delete(observationId);
  }

  getSnapshot(nowMs = Date.now()) {
    this.prune(nowMs);
    return {
      pendingRfid: this.pendingRfid.size,
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
      .map((track) => {
        const face = this.getFaceObservation(track.deviceId, track.trackId, rfid.timestampMs);
        const spatialDistance = this.calculateSpatialDistance(track);
        const timeDeltaMs = Math.abs(track.timestampMs - rfid.timestampMs);
        const faceMatched = faceMatchesIdentity(face, identity);
        const directionValid = track.direction === "ENTRY" || track.direction === "EXIT";
        let score = 40;
        if (faceMatched) {
          score += 40;
        }
        if (directionValid) {
          score += 20;
        }

        return {
          rfid,
          track,
          face,
          timeDeltaMs,
          spatialDistance,
          faceMatched,
          directionValid,
          score,
        } satisfies CandidateMatch;
      })
      .sort((left, right) => {
        if (right.score !== left.score) {
          return right.score - left.score;
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

  private getFaceKey(deviceId: string, trackId: number) {
    return `${deviceId}:${trackId}`;
  }
}
