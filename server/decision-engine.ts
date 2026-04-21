import type { Attendance, Employee, GateDecision, ScanTechnology } from "@shared/schema";
import { storage } from "./storage";
import {
  MatchingEngine,
  type CandidateMatch,
  type FaceObservation,
  type MovementDirection,
  type VisionTrackObservation,
} from "./matching-engine";

interface FinalEvent {
  name: string;
  employeeCode: string;
  employeeId: number;
  rfidTag: string;
  direction: "ENTRY" | "EXIT";
  timestamp: string;
  deviceId: string;
  trackId: number;
  score: number;
  confidence: number;
  correlation: {
    timeDeltaMs: number;
    spatialDistance: number;
  };
}

interface RejectedEvent {
  rfidTag: string;
  deviceId: string;
  timestamp: string;
  reason: string;
  score: number;
}

const SCORE_THRESHOLD = 70;
const DUPLICATE_COOLDOWN_MS = 5000;

function toIso(timestampMs: number) {
  return new Date(timestampMs).toISOString();
}

function toDateOnly(timestampMs: number) {
  return toIso(timestampMs).split("T")[0];
}

function normalizeDirection(direction?: string | null): MovementDirection {
  if (direction === "ENTRY" || direction === "EXIT") {
    return direction;
  }
  return "UNKNOWN";
}

export class DecisionEngine {
  private readonly matchingEngine = new MatchingEngine({
    matchWindowMs: Number(process.env.INTEGRATION_MATCH_WINDOW_MS ?? "1500"),
    minBufferMs: Number(process.env.INTEGRATION_MIN_BUFFER_MS ?? "250"),
    pendingTtlMs: Number(process.env.INTEGRATION_PENDING_TTL_MS ?? "2000"),
    faceTtlMs: Number(process.env.INTEGRATION_FACE_TTL_MS ?? "2000"),
    visionTtlMs: Number(process.env.INTEGRATION_VISION_TTL_MS ?? "2000"),
    trackCooldownMs: Number(process.env.INTEGRATION_TRACK_COOLDOWN_MS ?? "1500"),
    minStableTrackAgeFrames: Number(process.env.INTEGRATION_MIN_STABLE_TRACK_FRAMES ?? "10"),
    entryZoneMaxFraction: Number(process.env.INTEGRATION_ENTRY_ZONE_MAX ?? "0.4"),
    exitZoneMinFraction: Number(process.env.INTEGRATION_EXIT_ZONE_MIN ?? "0.6"),
    gateAnchorXFraction: Number(process.env.INTEGRATION_GATE_ANCHOR_X ?? "0.5"),
    gateAnchorYFraction: Number(process.env.INTEGRATION_GATE_ANCHOR_Y ?? "0.5"),
  });
  private readonly lastEventByTag = new Map<string, number>();
  private readonly finalizedEvents: FinalEvent[] = [];
  private readonly rejectedEvents: RejectedEvent[] = [];

  async ingestRfid(input: {
    deviceId: string;
    rfidTag: string;
    timestampMs?: number;
    scanTechnology?: ScanTechnology;
  }) {
    const observation = this.matchingEngine.recordRfid({
      deviceId: input.deviceId.trim(),
      rfidTag: input.rfidTag.trim().toUpperCase(),
      timestampMs: input.timestampMs ?? Date.now(),
      scanTechnology: input.scanTechnology ?? "UHF_RFID",
    });
    const resolved = await this.resolvePending(observation.deviceId);
    return {
      observation,
      resolved,
    };
  }

  async ingestVision(input: { deviceId: string; tracks: VisionTrackObservation[] }) {
    this.matchingEngine.recordVisionBatch(
      input.tracks.map((track) => ({
        ...track,
        deviceId: input.deviceId.trim(),
        direction: normalizeDirection(track.direction),
      })),
    );
    const resolved = await this.resolvePending(input.deviceId.trim());
    return {
      tracks: input.tracks.length,
      resolved,
    };
  }

  async ingestFace(input: { deviceId: string; faces: FaceObservation[] }) {
    this.matchingEngine.recordFaceBatch(
      input.faces.map((face) => ({
        ...face,
        deviceId: input.deviceId.trim(),
        rfidTag: face.rfidTag?.trim().toUpperCase() ?? null,
      })),
    );
    const resolved = await this.resolvePending(input.deviceId.trim());
    return {
      faces: input.faces.length,
      resolved,
    };
  }

  getStatus() {
    return {
      engine: this.matchingEngine.getSnapshot(),
      scoreThreshold: SCORE_THRESHOLD,
      duplicateCooldownMs: DUPLICATE_COOLDOWN_MS,
      recentValidEvents: this.finalizedEvents.slice(0, 20),
      recentRejectedEvents: this.rejectedEvents.slice(0, 20),
    };
  }

  private async resolvePending(deviceId: string) {
    const resolved: FinalEvent[] = [];
    const pending = this.matchingEngine.getPendingRfid(deviceId);
    const nowMs = Date.now();

    for (const rfid of pending) {
      const employee = await storage.getEmployeeByRfid(rfid.rfidTag);
      if (!employee) {
        this.rejectedEvents.unshift({
          rfidTag: rfid.rfidTag,
          deviceId: rfid.deviceId,
          timestamp: toIso(rfid.timestampMs),
          reason: "Unknown RFID tag.",
          score: 50,
        });
        this.matchingEngine.consumeRfid(rfid.id);
        continue;
      }

      if (!this.matchingEngine.isReadyToResolve(rfid, nowMs)) {
        continue;
      }

      const candidate = this.matchingEngine.findBestCandidate(rfid, {
        employeeId: employee.id,
        employeeCode: employee.employeeCode,
        employeeName: employee.name,
        rfidTag: employee.rfidUid,
      });

      if (!candidate) {
        if (this.matchingEngine.hasExpired(rfid, nowMs)) {
          this.rejectedEvents.unshift({
            rfidTag: rfid.rfidTag,
            deviceId: rfid.deviceId,
            timestamp: toIso(rfid.timestampMs),
            reason: "No stable vision candidate reached the matching window in time.",
            score: 50,
          });
          this.matchingEngine.consumeRfid(rfid.id);
        }
        continue;
      }

      if (candidate.score < SCORE_THRESHOLD) {
        if (this.matchingEngine.hasExpired(rfid, nowMs)) {
          this.pushRejected(candidate, "Candidate evidence stayed below the required score before the match window expired.");
          this.matchingEngine.consumeRfid(rfid.id);
        }
        continue;
      }

      const finalized = await this.finalizeCandidate(employee, candidate);
      this.matchingEngine.consumeRfid(rfid.id);
      if (finalized) {
        resolved.push(finalized);
      }
    }

    return resolved;
  }

  private async finalizeCandidate(employee: Employee, candidate: CandidateMatch) {
    const direction = normalizeDirection(candidate.resolvedDirection);
    if (direction !== "ENTRY" && direction !== "EXIT") {
      this.pushRejected(candidate, "Direction was not valid.");
      return null;
    }

    const duplicateKey = candidate.rfid.rfidTag;
    const duplicateAt = this.lastEventByTag.get(duplicateKey);
    if (duplicateAt && (candidate.rfid.timestampMs - duplicateAt) < DUPLICATE_COOLDOWN_MS) {
      this.pushRejected(candidate, `Duplicate tag ${candidate.rfid.rfidTag} blocked by cooldown.`);
      return null;
    }

    const attendance = await this.persistAttendance(employee, candidate, direction);
    if (!attendance) {
      this.pushRejected(candidate, direction === "ENTRY"
        ? "Open attendance already exists for this employee."
        : "No open attendance was available to close."
      );
      return null;
    }

    await storage.createGateEvent({
      employeeId: employee.id,
      date: toDateOnly(candidate.rfid.timestampMs),
      rfidUid: candidate.rfid.rfidTag,
      deviceId: candidate.rfid.deviceId,
      scanTechnology: candidate.rfid.scanTechnology,
      decision: direction as GateDecision,
      verificationStatus: direction,
      eventMessage: candidate.faceMatched
        ? `${direction} validated by RFID + direction with face confirmation.`
        : `${direction} validated by RFID + direction with face optional.`,
      movementDirection: direction,
      movementAxis: "horizontal",
      movementConfidence: 1,
      matchConfidence: Number((candidate.face?.confidence ?? 0).toFixed(4)),
      faceQuality: null,
      faceConsistency: null,
      faceCaptureMode: null,
    });

    this.lastEventByTag.set(duplicateKey, candidate.rfid.timestampMs);
    this.matchingEngine.reserveTrack(candidate.track.deviceId, candidate.track.trackId, candidate.rfid.timestampMs);

    const event: FinalEvent = {
      name: employee.name,
      employeeCode: employee.employeeCode,
      employeeId: employee.id,
      rfidTag: candidate.rfid.rfidTag,
      direction,
      timestamp: toIso(candidate.rfid.timestampMs),
      deviceId: candidate.rfid.deviceId,
      trackId: candidate.track.trackId,
      score: candidate.score,
      confidence: Number((candidate.face?.confidence ?? 0).toFixed(4)),
      correlation: {
        timeDeltaMs: candidate.timeDeltaMs,
        spatialDistance: Number(candidate.spatialDistance.toFixed(2)),
      },
    };
    this.finalizedEvents.unshift(event);
    this.finalizedEvents.splice(20);
    return event;
  }

  private async persistAttendance(
    employee: Employee,
    candidate: CandidateMatch,
    direction: "ENTRY" | "EXIT",
  ): Promise<Attendance | null> {
    const eventDate = toDateOnly(candidate.rfid.timestampMs);
    const eventTime = new Date(candidate.rfid.timestampMs);
    const openAttendance = await storage.getOpenAttendance(employee.id, eventDate);

    if (direction === "ENTRY") {
      if (openAttendance) {
        return null;
      }

      return storage.createAttendance({
        employeeId: employee.id,
        date: eventDate,
        entryTime: eventTime,
        exitTime: null,
        workingHours: null,
        verificationStatus: "ENTRY",
        deviceId: candidate.rfid.deviceId,
      });
    }

    if (!openAttendance) {
      return null;
    }

    const workingHoursMs = eventTime.getTime() - (openAttendance.entryTime?.getTime() ?? eventTime.getTime());
    const workingHours = Number((workingHoursMs / (1000 * 60 * 60)).toFixed(2));
    const updated = await storage.updateAttendance(openAttendance.id, {
      exitTime: eventTime,
      workingHours,
      verificationStatus: "EXIT",
      deviceId: candidate.rfid.deviceId,
    });
    return updated ?? null;
  }

  private pushRejected(candidate: CandidateMatch, reason: string) {
    this.rejectedEvents.unshift({
      rfidTag: candidate.rfid.rfidTag,
      deviceId: candidate.rfid.deviceId,
      timestamp: toIso(candidate.rfid.timestampMs),
      reason,
      score: candidate.score,
    });
    this.rejectedEvents.splice(20);
  }
}

export const decisionEngine = new DecisionEngine();
