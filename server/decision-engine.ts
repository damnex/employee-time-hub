import type { Server } from "http";
import type { Attendance, Employee, GateDecision, ScanTechnology } from "@shared/schema";
import { WebSocket, WebSocketServer } from "ws";
import { clearFaceState } from "./faceState";
import { storage } from "./storage";
import { clearTrackState } from "./trackState";
import {
  MatchingEngine,
  type FaceObservation,
  type MovementDirection,
  type VisionTrackObservation,
} from "./matching-engine";

export interface DecisionMatch {
  tag: string;
  track_id: number;
  name?: string;
  direction: "ENTRY" | "EXIT" | null;
  confidence: number;
  timestamp: number;
  deviceId: string;
  employeeId?: number | null;
  employeeCode?: string | null;
  scanTechnology?: ScanTechnology;
  score?: number;
  correlation?: {
    timeDeltaMs: number;
    spatialDistance: number;
  };
}

export interface ValidatedDecision {
  tag: string;
  track_id: number;
  name: string;
  direction: "ENTRY" | "EXIT";
  confidence: number;
  timestamp: number;
  deviceId: string;
  employeeId: number | null;
  employeeCode: string | null;
  scanTechnology: ScanTechnology;
  score: number;
  correlation?: {
    timeDeltaMs: number;
    spatialDistance: number;
  };
}

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

interface RejectedMatch {
  rfidTag: string;
  deviceId: string;
  trackId: number | null;
  timestamp: string;
  reason: string;
  confidence: number;
}

const MATCH_SCORE_THRESHOLD = 70;
const VALIDATION_CONFIDENCE_THRESHOLD = 0.6;
const DECISION_COOLDOWN = 3000;
const RECENT_DECISION_RETENTION_MS = 5000;
const RECENT_DECISION_CLEANUP_INTERVAL_MS = 1000;
const RECENT_LOG_LIMIT = 20;
const DECISION_WS_PATH = "/ws/gate-events";

let decisionSocketServer: WebSocketServer | null = null;

function toIso(timestampMs: number) {
  return new Date(timestampMs).toISOString();
}

function toDateOnly(timestampMs: number) {
  return toIso(timestampMs).split("T")[0];
}

function normalizeTag(tag: string) {
  return tag.trim().toUpperCase();
}

function normalizeDirection(direction?: string | null): MovementDirection {
  if (direction === "ENTRY" || direction === "EXIT") {
    return direction;
  }
  return "UNKNOWN";
}

function clampConfidence(confidence: number) {
  if (!Number.isFinite(confidence)) {
    return 0;
  }

  return Math.max(0, Math.min(1, confidence));
}

function safeTrackId(trackId?: number | null) {
  return typeof trackId === "number" && Number.isFinite(trackId) ? trackId : 0;
}

function getDecisionSocketClientCount() {
  return decisionSocketServer?.clients.size ?? 0;
}

function broadcastGateEvent(decision: ValidatedDecision) {
  if (!decisionSocketServer) {
    return;
  }

  const payload = JSON.stringify({
    event: "gate-event",
    data: decision,
  });

  for (const client of Array.from(decisionSocketServer.clients)) {
    if (client.readyState !== WebSocket.OPEN) {
      continue;
    }

    try {
      client.send(payload);
    } catch (error) {
      console.warn("[decision-engine] Failed to emit gate-event:", error);
    }
  }
}

export function attachDecisionUiEmitter(server: Server) {
  if (decisionSocketServer) {
    return decisionSocketServer;
  }

  decisionSocketServer = new WebSocketServer({
    server,
    path: DECISION_WS_PATH,
  });

  decisionSocketServer.on("connection", (socket) => {
    socket.send(JSON.stringify({
      event: "ready",
      data: {
        path: DECISION_WS_PATH,
      },
    }));
  });

  return decisionSocketServer;
}

export class DecisionEngine {
  private readonly matchingEngine = new MatchingEngine({
    matchWindowMs: Number(process.env.INTEGRATION_MATCH_WINDOW_MS ?? "1500"),
    minBufferMs: Number(process.env.INTEGRATION_MIN_BUFFER_MS ?? "250"),
    pendingTtlMs: Number(process.env.INTEGRATION_PENDING_TTL_MS ?? "2000"),
    faceTtlMs: Number(process.env.INTEGRATION_FACE_TTL_MS ?? "2000"),
    visionTtlMs: Number(process.env.INTEGRATION_VISION_TTL_MS ?? "2000"),
    trackCooldownMs: Number(process.env.INTEGRATION_TRACK_COOLDOWN_MS ?? "500"),
    minStableTrackAgeFrames: Number(process.env.INTEGRATION_MIN_STABLE_TRACK_FRAMES ?? "10"),
    entryZoneMaxFraction: Number(process.env.INTEGRATION_ENTRY_ZONE_MAX ?? "0.4"),
    exitZoneMinFraction: Number(process.env.INTEGRATION_EXIT_ZONE_MIN ?? "0.6"),
    gateAnchorXFraction: Number(process.env.INTEGRATION_GATE_ANCHOR_X ?? "0.5"),
    gateAnchorYFraction: Number(process.env.INTEGRATION_GATE_ANCHOR_Y ?? "0.5"),
  });
  private readonly recentDecisions = new Map<string, number>();
  private readonly finalizedEvents: FinalEvent[] = [];
  private readonly rejectedMatches: RejectedMatch[] = [];

  constructor() {
    const cleanupTimer = setInterval(() => {
      this.cleanupRecentDecisions();
    }, RECENT_DECISION_CLEANUP_INTERVAL_MS);

    cleanupTimer.unref?.();
  }

  async ingestRfid(input: {
    deviceId: string;
    rfidTag: string;
    timestampMs?: number;
    scanTechnology?: ScanTechnology;
  }) {
    const observation = this.matchingEngine.recordRfid({
      deviceId: input.deviceId.trim(),
      rfidTag: normalizeTag(input.rfidTag),
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

  isValidMatch(match: DecisionMatch | null | undefined): match is DecisionMatch & { direction: "ENTRY" | "EXIT" } {
    return Boolean(
      match
      && match.direction
      && (match.direction === "ENTRY" || match.direction === "EXIT")
      && clampConfidence(match.confidence) >= VALIDATION_CONFIDENCE_THRESHOLD,
    );
  }

  isDuplicate(tag: string, timestampMs = Date.now()) {
    this.cleanupRecentDecisions(timestampMs);
    const normalizedTag = normalizeTag(tag);
    const lastProcessedAt = this.recentDecisions.get(normalizedTag);

    return typeof lastProcessedAt === "number" && (timestampMs - lastProcessedAt) < DECISION_COOLDOWN;
  }

  processMatches(matches: DecisionMatch[]) {
    const decisions: ValidatedDecision[] = [];

    for (const match of matches) {
      const processedAt = match.timestamp || Date.now();
      if (!this.isValidMatch(match)) {
        this.pushRejected({
          rfidTag: normalizeTag(match.tag),
          deviceId: match.deviceId.trim(),
          trackId: safeTrackId(match.track_id),
          timestamp: toIso(processedAt),
          reason: "Match was rejected because confidence stayed below 0.6 or direction was missing.",
          confidence: clampConfidence(match.confidence),
        });
        console.debug("[decision-engine] Rejected match", {
          tag: normalizeTag(match.tag),
          track_id: safeTrackId(match.track_id),
          confidence: Number(clampConfidence(match.confidence).toFixed(4)),
          reason: "invalid",
        });
        continue;
      }

      const normalizedTag = normalizeTag(match.tag);
      if (this.isDuplicate(normalizedTag, processedAt)) {
        this.pushRejected({
          rfidTag: normalizedTag,
          deviceId: match.deviceId.trim(),
          trackId: safeTrackId(match.track_id),
          timestamp: toIso(processedAt),
          reason: `Duplicate tag ${normalizedTag} skipped inside decision cooldown.`,
          confidence: clampConfidence(match.confidence),
        });
        console.debug("[decision-engine] Duplicate skip", {
          tag: normalizedTag,
          track_id: safeTrackId(match.track_id),
          timestamp: toIso(processedAt),
        });
        continue;
      }

      const decision: ValidatedDecision = {
        tag: normalizedTag,
        track_id: safeTrackId(match.track_id),
        name: match.name?.trim() || "Unknown",
        direction: match.direction,
        confidence: clampConfidence(match.confidence),
        timestamp: processedAt,
        deviceId: match.deviceId.trim(),
        employeeId: match.employeeId ?? null,
        employeeCode: match.employeeCode ?? null,
        scanTechnology: match.scanTechnology ?? "UHF_RFID",
        score: Number((match.score ?? match.confidence).toFixed(4)),
        correlation: match.correlation,
      };

      this.recentDecisions.set(decision.tag, decision.timestamp);
      decisions.push(decision);
    }

    return decisions;
  }

  async saveToDB(decision: ValidatedDecision) {
    try {
      const employee =
        decision.employeeId != null
          ? await storage.getEmployee(decision.employeeId) ?? await storage.getEmployeeByRfid(decision.tag)
          : await storage.getEmployeeByRfid(decision.tag);

      if (employee) {
        const attendance = await this.persistAttendance(employee, decision);
        if (!attendance) {
          this.pushRejected({
            rfidTag: decision.tag,
            deviceId: decision.deviceId,
            trackId: decision.track_id,
            timestamp: toIso(decision.timestamp),
            reason: decision.direction === "ENTRY"
              ? "Open attendance already exists for this employee."
              : "No open attendance was available to close.",
            confidence: decision.confidence,
          });
          return false;
        }
      }

      try {
        await storage.createGateEvent({
          employeeId: employee?.id ?? null,
          date: toDateOnly(decision.timestamp),
          rfidUid: decision.tag,
          deviceId: decision.deviceId,
          scanTechnology: decision.scanTechnology,
          decision: decision.direction as GateDecision,
          verificationStatus: decision.direction,
          eventMessage: `${decision.direction} validated by final decision engine for ${decision.name}.`,
          movementDirection: decision.direction,
          movementAxis: "horizontal",
          movementConfidence: Number(decision.confidence.toFixed(4)),
          matchConfidence: Number(decision.confidence.toFixed(4)),
          faceQuality: null,
          faceConsistency: null,
          faceCaptureMode: null,
        });
      } catch (error) {
        console.warn("[decision-engine] Gate event persistence failed, but attendance was kept:", error);
      }

      return true;
    } catch (error) {
      this.pushRejected({
        rfidTag: decision.tag,
        deviceId: decision.deviceId,
        trackId: decision.track_id,
        timestamp: toIso(decision.timestamp),
        reason: error instanceof Error ? error.message : "Database write failed unexpectedly.",
        confidence: decision.confidence,
      });
      return false;
    }
  }

  emitToUI(decision: ValidatedDecision) {
    broadcastGateEvent(decision);
  }

  async handleMatchedPairs(matches: DecisionMatch[]) {
    const decisions = this.processMatches(matches);
    const resolved: FinalEvent[] = [];

    for (const decision of decisions) {
      const saved = await this.saveToDB(decision);
      if (!saved) {
        continue;
      }

      this.matchingEngine.reserveTrack(decision.deviceId, decision.track_id, decision.timestamp);
      this.matchingEngine.clearTrackContext(decision.deviceId, decision.track_id);
      clearTrackState(decision.track_id);
      clearFaceState(decision.track_id);
      this.emitToUI(decision);

      const event = this.toFinalEvent(decision);
      this.finalizedEvents.unshift(event);
      this.finalizedEvents.splice(RECENT_LOG_LIMIT);
      resolved.push(event);

      console.debug("[decision-engine] Accepted decision", {
        tag: decision.tag,
        track_id: decision.track_id,
        direction: decision.direction,
        confidence: Number(decision.confidence.toFixed(4)),
      });
    }

    return resolved;
  }

  getStatus() {
    return {
      engine: this.matchingEngine.getSnapshot(),
      matchScoreThreshold: MATCH_SCORE_THRESHOLD,
      validationConfidenceThreshold: VALIDATION_CONFIDENCE_THRESHOLD,
      decisionCooldownMs: DECISION_COOLDOWN,
      decisionRetentionMs: RECENT_DECISION_RETENTION_MS,
      recentDecisionCount: this.recentDecisions.size,
      ui: {
        websocketPath: DECISION_WS_PATH,
        clients: getDecisionSocketClientCount(),
      },
      recentValidEvents: this.finalizedEvents.slice(0, RECENT_LOG_LIMIT),
      recentRejectedEvents: this.rejectedMatches.slice(0, RECENT_LOG_LIMIT),
    };
  }

  private async resolvePending(deviceId: string) {
    const matches: DecisionMatch[] = [];
    const pending = this.matchingEngine.getPendingRfid(deviceId);
    const nowMs = Date.now();

    for (const rfid of pending) {
      const employee = await storage.getEmployeeByRfid(rfid.rfidTag);
      if (!employee) {
        this.pushRejected({
          rfidTag: rfid.rfidTag,
          deviceId: rfid.deviceId,
          trackId: null,
          timestamp: toIso(rfid.timestampMs),
          reason: "Unknown RFID tag.",
          confidence: 0,
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
          this.pushRejected({
            rfidTag: rfid.rfidTag,
            deviceId: rfid.deviceId,
            trackId: null,
            timestamp: toIso(rfid.timestampMs),
            reason: "No stable vision candidate reached the matching window in time.",
            confidence: 0,
          });
          this.matchingEngine.consumeRfid(rfid.id);
        }
        continue;
      }

      if (candidate.score < MATCH_SCORE_THRESHOLD) {
        if (this.matchingEngine.hasExpired(rfid, nowMs)) {
          this.pushRejected({
            rfidTag: candidate.rfid.rfidTag,
            deviceId: candidate.rfid.deviceId,
            trackId: candidate.track.trackId,
            timestamp: toIso(candidate.rfid.timestampMs),
            reason: "Candidate evidence stayed below the required score before the match window expired.",
            confidence: Number((candidate.score / 100).toFixed(4)),
          });
          this.matchingEngine.consumeRfid(rfid.id);
        }
        continue;
      }

      this.matchingEngine.consumeRfid(rfid.id);
      matches.push(this.toDecisionMatch(employee, candidate));
    }

    return this.handleMatchedPairs(matches);
  }

  private async persistAttendance(
    employee: Employee,
    decision: ValidatedDecision,
  ): Promise<Attendance | null> {
    const eventDate = toDateOnly(decision.timestamp);
    const eventTime = new Date(decision.timestamp);
    const openAttendance = await storage.getOpenAttendance(employee.id, eventDate);

    if (decision.direction === "ENTRY") {
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
        deviceId: decision.deviceId,
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
      deviceId: decision.deviceId,
    });

    return updated ?? null;
  }

  private cleanupRecentDecisions(nowMs = Date.now()) {
    for (const [tag, processedAt] of Array.from(this.recentDecisions.entries())) {
      if ((nowMs - processedAt) > RECENT_DECISION_RETENTION_MS) {
        this.recentDecisions.delete(tag);
      }
    }
  }

  private pushRejected(input: RejectedMatch) {
    this.rejectedMatches.unshift(input);
    this.rejectedMatches.splice(RECENT_LOG_LIMIT);
  }

  private toDecisionMatch(employee: Employee, candidate: {
    rfid: {
      rfidTag: string;
      timestampMs: number;
      deviceId: string;
      scanTechnology: ScanTechnology;
    };
    track: {
      trackId: number;
    };
    resolvedDirection: MovementDirection;
    score: number;
    correlation?: {
      timeDeltaMs: number;
      spatialDistance: number;
    };
    timeDeltaMs: number;
    spatialDistance: number;
  }): DecisionMatch {
    const direction = normalizeDirection(candidate.resolvedDirection);
    return {
      tag: candidate.rfid.rfidTag,
      track_id: candidate.track.trackId,
      name: employee.name,
      direction: direction === "ENTRY" || direction === "EXIT" ? direction : null,
      confidence: Number((candidate.score / 100).toFixed(4)),
      timestamp: candidate.rfid.timestampMs,
      deviceId: candidate.rfid.deviceId,
      employeeId: employee.id,
      employeeCode: employee.employeeCode,
      scanTechnology: candidate.rfid.scanTechnology,
      score: candidate.score,
      correlation: {
        timeDeltaMs: candidate.timeDeltaMs,
        spatialDistance: Number(candidate.spatialDistance.toFixed(2)),
      },
    };
  }

  private toFinalEvent(decision: ValidatedDecision): FinalEvent {
    return {
      name: decision.name,
      employeeCode: decision.employeeCode ?? "",
      employeeId: decision.employeeId ?? 0,
      rfidTag: decision.tag,
      direction: decision.direction,
      timestamp: toIso(decision.timestamp),
      deviceId: decision.deviceId,
      trackId: decision.track_id,
      score: decision.score,
      confidence: Number(decision.confidence.toFixed(4)),
      correlation: decision.correlation ?? {
        timeDeltaMs: 0,
        spatialDistance: 0,
      },
    };
  }
}

export const decisionEngine = new DecisionEngine();
