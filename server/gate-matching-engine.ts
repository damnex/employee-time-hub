import { randomUUID } from "crypto";
import type { MovementAxis, ScanTechnology } from "@shared/schema";

export type AttendanceAction = "ENTRY" | "EXIT";
export type MovementDirection = AttendanceAction | "UNKNOWN";
export type SessionOutcome = AttendanceAction | "REJECTED" | "IGNORED" | "LOW_CONFIDENCE";
export type SessionState = "IDLE" | "PRESENT" | "DISAPPEARED";
export type GateConfidenceTier = "VALID" | "LOW_CONFIDENCE" | "REJECT";
export type GateDecisionAction = AttendanceAction | "IGNORE" | "REJECT";

export const MATCH_WINDOW_MS = 2_000;
export const FRAGMENT_TTL_MS = 12_000;
export const SESSION_TTL_MS = 12 * 60 * 60 * 1_000;
export const SESSION_EXIT_TIMEOUT_MS = 5_000;
export const HF_DUPLICATE_WINDOW_MS = 3_500;
export const UHF_DUPLICATE_WINDOW_MS = 1_200;

const RFID_SCORE = 40;
const FACE_SCORE = 40;
const DIRECTION_SCORE = 20;

type RfidSource =
  | "reader_detected"
  | "api_scan"
  | "presence_start"
  | "presence_ping"
  | "presence_end";

interface GateFragmentBase {
  id: string;
  deviceId: string;
  occurredAtMs: number;
  consumedAtMs: number | null;
}

export interface GateRfidEvent extends GateFragmentBase {
  kind: "rfid";
  rfidUid: string;
  scanTechnology: ScanTechnology;
  source: RfidSource;
}

export interface GateFaceEvent extends GateFragmentBase {
  kind: "face";
  rfidUidHint?: string;
  hasFrames: boolean;
  hasDescriptor: boolean;
}

export interface GateDirectionEvent extends GateFragmentBase {
  kind: "direction";
  movementDirection: MovementDirection;
  movementAxis: MovementAxis;
  movementConfidence?: number;
}

interface GateSessionState {
  sessionId: string;
  deviceId: string;
  rfidUid: string;
  scanTechnology: ScanTechnology;
  state: SessionState;
  lastSeenAtMs: number;
  lastRfidSeenAtMs: number | null;
  lastFaceSeenAtMs: number | null;
  lastDirectionSeenAtMs: number | null;
  lastDecisionAtMs: number | null;
  lastOutcome: SessionOutcome | null;
  lastAction: AttendanceAction | null;
  duplicateWindowMs: number;
  timeoutExitArmedAtMs: number | null;
  timeoutExitReportedAtMs: number | null;
}

export interface GateSessionSnapshot {
  sessionId: string;
  deviceId: string;
  rfidUid: string;
  scanTechnology: ScanTechnology;
  state: SessionState;
  lastSeenAtMs: number;
  lastRfidSeenAtMs: number | null;
  lastFaceSeenAtMs: number | null;
  lastDirectionSeenAtMs: number | null;
  lastDecisionAtMs: number | null;
  lastOutcome: SessionOutcome | null;
  lastAction: AttendanceAction | null;
  duplicateWindowMs: number;
  timeoutExitArmedAtMs: number | null;
  timeoutExitReportedAtMs: number | null;
}

export interface GateCorrelationMatch {
  correlationId: string;
  occurredAt: Date;
  rfid: GateRfidEvent;
  face?: GateFaceEvent;
  direction?: GateDirectionEvent;
  session: GateSessionSnapshot;
}

export interface GateSignalInput {
  deviceId: string;
  occurredAt?: Date;
  rfidUid?: string;
  scanTechnology?: ScanTechnology;
  faceFrames?: string[];
  faceDescriptor?: number[];
  faceAnchorDescriptors?: number[][];
  faceRfidUidHint?: string;
  movementDirection?: MovementDirection;
  movementAxis?: MovementAxis;
  movementConfidence?: number;
}

export interface GateValidationInput {
  correlation: GateCorrelationMatch | null;
  facePresent?: boolean;
  faceMatched?: boolean;
  faceRfidUidHint?: string | null;
  directionDetected?: boolean;
  strictFaceRequired?: boolean;
}

export interface GateValidationResult {
  tier: GateConfidenceTier;
  hardRejected: boolean;
  rfidPresent: boolean;
  facePresent: boolean;
  faceMatched: boolean;
  directionPresent: boolean;
  rfidUid?: string;
  faceRfidUidHint?: string;
  reasons: string[];
}

export interface GateConfidenceScore {
  tier: GateConfidenceTier;
  total: number;
  breakdown: {
    rfid: number;
    face: number;
    direction: number;
  };
}

export interface GateDecisionInput {
  correlation: GateCorrelationMatch | null;
  validation: GateValidationResult;
  confidence: GateConfidenceScore;
  hasOpenAttendance: boolean;
  movementDirection?: MovementDirection;
  timedOut?: boolean;
}

export interface GateDecisionResult {
  action: GateDecisionAction;
  tier: GateConfidenceTier | "IGNORE";
  reason: string;
  timeoutFallback: boolean;
}

export interface GateTimeoutExitCandidate {
  occurredAt: Date;
  session: GateSessionSnapshot;
}

export interface CollectedGateEvents {
  occurredAtMs: number;
  face?: GateFaceEvent | null;
  direction?: GateDirectionEvent | null;
}

function toOccurredAtMs(occurredAt?: Date) {
  return occurredAt?.getTime() ?? Date.now();
}

function normalizeRfidUid(rfidUid: string) {
  return rfidUid.trim().toUpperCase();
}

function normalizeOptionalRfidUid(rfidUid?: string | null) {
  return rfidUid?.trim() ? normalizeRfidUid(rfidUid) : undefined;
}

function duplicateWindowForTechnology(scanTechnology: ScanTechnology) {
  return scanTechnology === "UHF_RFID" ? UHF_DUPLICATE_WINDOW_MS : HF_DUPLICATE_WINDOW_MS;
}

function hasFacePayload(signal: GateSignalInput) {
  return Boolean(
    signal.faceFrames?.length
    || signal.faceDescriptor?.length
    || signal.faceAnchorDescriptors?.length,
  );
}

function hasDirectionPayload(signal: GateSignalInput) {
  return Boolean(
    signal.movementDirection
    || signal.movementAxis
    || typeof signal.movementConfidence === "number",
  );
}

function hasUsableDirection(direction?: GateDirectionEvent | null) {
  return direction?.movementDirection === "ENTRY" || direction?.movementDirection === "EXIT";
}

export class GateMatchingEngine {
  private readonly rfidEvents: GateRfidEvent[] = [];
  private readonly faceEvents: GateFaceEvent[] = [];
  private readonly directionEvents: GateDirectionEvent[] = [];
  private readonly sessions = new Map<string, GateSessionState>();

  private buildSessionKey(deviceId: string, rfidUid: string) {
    return `${deviceId}::${normalizeRfidUid(rfidUid)}`;
  }

  private snapshotSession(session: GateSessionState): GateSessionSnapshot {
    return {
      sessionId: session.sessionId,
      deviceId: session.deviceId,
      rfidUid: session.rfidUid,
      scanTechnology: session.scanTechnology,
      state: session.state,
      lastSeenAtMs: session.lastSeenAtMs,
      lastRfidSeenAtMs: session.lastRfidSeenAtMs,
      lastFaceSeenAtMs: session.lastFaceSeenAtMs,
      lastDirectionSeenAtMs: session.lastDirectionSeenAtMs,
      lastDecisionAtMs: session.lastDecisionAtMs,
      lastOutcome: session.lastOutcome,
      lastAction: session.lastAction,
      duplicateWindowMs: session.duplicateWindowMs,
      timeoutExitArmedAtMs: session.timeoutExitArmedAtMs,
      timeoutExitReportedAtMs: session.timeoutExitReportedAtMs,
    };
  }

  private getOrCreateSession(args: {
    deviceId: string;
    rfidUid: string;
    scanTechnology: ScanTechnology;
    occurredAtMs: number;
  }) {
    const key = this.buildSessionKey(args.deviceId, args.rfidUid);
    let session = this.sessions.get(key);

    if (!session) {
      session = {
        sessionId: randomUUID(),
        deviceId: args.deviceId,
        rfidUid: args.rfidUid,
        scanTechnology: args.scanTechnology,
        state: "IDLE",
        lastSeenAtMs: args.occurredAtMs,
        lastRfidSeenAtMs: null,
        lastFaceSeenAtMs: null,
        lastDirectionSeenAtMs: null,
        lastDecisionAtMs: null,
        lastOutcome: null,
        lastAction: null,
        duplicateWindowMs: duplicateWindowForTechnology(args.scanTechnology),
        timeoutExitArmedAtMs: null,
        timeoutExitReportedAtMs: null,
      };
      this.sessions.set(key, session);
    }

    session.scanTechnology = args.scanTechnology;
    session.duplicateWindowMs = duplicateWindowForTechnology(args.scanTechnology);
    session.lastSeenAtMs = Math.max(session.lastSeenAtMs, args.occurredAtMs);
    session.lastRfidSeenAtMs = Math.max(session.lastRfidSeenAtMs ?? 0, args.occurredAtMs);
    session.state = "PRESENT";
    session.timeoutExitArmedAtMs = null;
    session.timeoutExitReportedAtMs = null;

    return session;
  }

  private touchSessionFace(session: GateSessionState, occurredAtMs: number) {
    session.lastSeenAtMs = Math.max(session.lastSeenAtMs, occurredAtMs);
    session.lastFaceSeenAtMs = Math.max(session.lastFaceSeenAtMs ?? 0, occurredAtMs);
    session.timeoutExitArmedAtMs = null;
    session.timeoutExitReportedAtMs = null;
  }

  private touchSessionDirection(session: GateSessionState, occurredAtMs: number) {
    session.lastSeenAtMs = Math.max(session.lastSeenAtMs, occurredAtMs);
    session.lastDirectionSeenAtMs = Math.max(session.lastDirectionSeenAtMs ?? 0, occurredAtMs);
  }

  private cleanup(nowMs: number) {
    const keepUnexpired = <T extends GateFragmentBase>(event: T) => {
      return nowMs - event.occurredAtMs <= FRAGMENT_TTL_MS;
    };

    const keepAvailable = <T extends GateFragmentBase>(event: T) => {
      return event.consumedAtMs === null || nowMs - event.consumedAtMs <= MATCH_WINDOW_MS;
    };

    const rfidEvents = this.rfidEvents.filter((event) => {
      return keepUnexpired(event) && keepAvailable(event);
    });
    const faceEvents = this.faceEvents.filter((event) => {
      return keepUnexpired(event) && keepAvailable(event);
    });
    const directionEvents = this.directionEvents.filter((event) => {
      return keepUnexpired(event) && keepAvailable(event);
    });

    this.rfidEvents.length = 0;
    this.rfidEvents.push(...rfidEvents);
    this.faceEvents.length = 0;
    this.faceEvents.push(...faceEvents);
    this.directionEvents.length = 0;
    this.directionEvents.push(...directionEvents);

    Array.from(this.sessions.entries()).forEach(([key, session]) => {
      const lastTouchedAt = Math.max(
        session.lastSeenAtMs,
        session.lastDecisionAtMs ?? 0,
        session.timeoutExitArmedAtMs ?? 0,
        session.timeoutExitReportedAtMs ?? 0,
      );

      if (nowMs - lastTouchedAt > SESSION_TTL_MS) {
        this.sessions.delete(key);
      }
    });
  }

  private findBestRfidEvent(args: {
    deviceId: string;
    occurredAtMs: number;
    rfidUidHint?: string;
  }) {
    const normalizedHint = normalizeOptionalRfidUid(args.rfidUidHint);
    const candidates = this.rfidEvents
      .filter((event) => event.deviceId === args.deviceId)
      .filter((event) => event.consumedAtMs === null)
      .filter((event) => Math.abs(event.occurredAtMs - args.occurredAtMs) <= MATCH_WINDOW_MS)
      .filter((event) => !normalizedHint || event.rfidUid === normalizedHint)
      .sort((left, right) => {
        const leftDelta = Math.abs(left.occurredAtMs - args.occurredAtMs);
        const rightDelta = Math.abs(right.occurredAtMs - args.occurredAtMs);
        return leftDelta - rightDelta || right.occurredAtMs - left.occurredAtMs;
      });

    return candidates[0];
  }

  private findBestFaceEvent(args: {
    deviceId: string;
    occurredAtMs: number;
    rfidUidHint?: string;
  }) {
    const normalizedHint = normalizeOptionalRfidUid(args.rfidUidHint);
    const candidates = this.faceEvents
      .filter((event) => event.deviceId === args.deviceId)
      .filter((event) => event.consumedAtMs === null)
      .filter((event) => Math.abs(event.occurredAtMs - args.occurredAtMs) <= MATCH_WINDOW_MS)
      .filter((event) => {
        return !normalizedHint || !event.rfidUidHint || event.rfidUidHint === normalizedHint;
      })
      .sort((left, right) => {
        const leftDelta = Math.abs(left.occurredAtMs - args.occurredAtMs);
        const rightDelta = Math.abs(right.occurredAtMs - args.occurredAtMs);
        return leftDelta - rightDelta || right.occurredAtMs - left.occurredAtMs;
      });

    return candidates[0];
  }

  private findBestDirectionEvent(args: {
    deviceId: string;
    occurredAtMs: number;
  }) {
    const candidates = this.directionEvents
      .filter((event) => event.deviceId === args.deviceId)
      .filter((event) => event.consumedAtMs === null)
      .filter((event) => Math.abs(event.occurredAtMs - args.occurredAtMs) <= MATCH_WINDOW_MS)
      .sort((left, right) => {
        const leftDelta = Math.abs(left.occurredAtMs - args.occurredAtMs);
        const rightDelta = Math.abs(right.occurredAtMs - args.occurredAtMs);
        return leftDelta - rightDelta || right.occurredAtMs - left.occurredAtMs;
      });

    return candidates[0];
  }

  recordRfidDetection(args: {
    deviceId: string;
    rfidUid: string;
    scanTechnology?: ScanTechnology;
    occurredAt?: Date;
    source?: RfidSource;
  }) {
    const occurredAtMs = toOccurredAtMs(args.occurredAt);
    const scanTechnology = args.scanTechnology ?? "HF_RFID";
    const rfidUid = normalizeRfidUid(args.rfidUid);

    this.cleanup(occurredAtMs);

    const session = this.getOrCreateSession({
      deviceId: args.deviceId,
      rfidUid,
      scanTechnology,
      occurredAtMs,
    });

    const event: GateRfidEvent = {
      id: randomUUID(),
      kind: "rfid",
      deviceId: args.deviceId,
      rfidUid,
      scanTechnology,
      source: args.source ?? "reader_detected",
      occurredAtMs,
      consumedAtMs: null,
    };

    this.rfidEvents.push(event);

    return {
      event,
      session: this.snapshotSession(session),
    };
  }

  recordFaceSignal(signal: GateSignalInput) {
    if (!hasFacePayload(signal)) {
      return null;
    }

    const occurredAtMs = toOccurredAtMs(signal.occurredAt);
    this.cleanup(occurredAtMs);

    const event: GateFaceEvent = {
      id: randomUUID(),
      kind: "face",
      deviceId: signal.deviceId,
      rfidUidHint: normalizeOptionalRfidUid(signal.faceRfidUidHint),
      hasFrames: Boolean(signal.faceFrames?.length),
      hasDescriptor: Boolean(
        signal.faceDescriptor?.length || signal.faceAnchorDescriptors?.length,
      ),
      occurredAtMs,
      consumedAtMs: null,
    };

    this.faceEvents.push(event);
    return event;
  }

  recordDirectionSignal(signal: GateSignalInput) {
    if (!hasDirectionPayload(signal)) {
      return null;
    }

    const occurredAtMs = toOccurredAtMs(signal.occurredAt);
    this.cleanup(occurredAtMs);

    const event: GateDirectionEvent = {
      id: randomUUID(),
      kind: "direction",
      deviceId: signal.deviceId,
      movementDirection: signal.movementDirection ?? "UNKNOWN",
      movementAxis: signal.movementAxis ?? "none",
      movementConfidence: signal.movementConfidence,
      occurredAtMs,
      consumedAtMs: null,
    };

    this.directionEvents.push(event);
    return event;
  }

  collectEvents(signal: GateSignalInput): CollectedGateEvents {
    const occurredAtMs = toOccurredAtMs(signal.occurredAt);
    this.cleanup(occurredAtMs);

    return {
      occurredAtMs,
      face: this.recordFaceSignal(signal),
      direction: this.recordDirectionSignal(signal),
    };
  }

  matchEvents(signal: GateSignalInput): GateCorrelationMatch | null {
    const collected = this.collectEvents(signal);
    const occurredAtMs = collected.occurredAtMs;
    let face = collected.face ?? undefined;
    let direction = collected.direction ?? undefined;
    let rfid = signal.rfidUid
      ? this.findBestRfidEvent({
          deviceId: signal.deviceId,
          occurredAtMs,
          rfidUidHint: signal.rfidUid,
        })
      : this.findBestRfidEvent({
          deviceId: signal.deviceId,
          occurredAtMs,
        });

    if (!face) {
      face = this.findBestFaceEvent({
        deviceId: signal.deviceId,
        occurredAtMs,
      });
    }

    if (!direction) {
      direction = this.findBestDirectionEvent({
        deviceId: signal.deviceId,
        occurredAtMs,
      });
    }

    if (!rfid && signal.rfidUid) {
      rfid = this.recordRfidDetection({
        deviceId: signal.deviceId,
        rfidUid: signal.rfidUid,
        scanTechnology: signal.scanTechnology,
        occurredAt: new Date(occurredAtMs),
        source: "api_scan",
      }).event;
    }

    if (!rfid) {
      return null;
    }

    const session = this.getOrCreateSession({
      deviceId: rfid.deviceId,
      rfidUid: rfid.rfidUid,
      scanTechnology: rfid.scanTechnology,
      occurredAtMs,
    });

    rfid.consumedAtMs = occurredAtMs;
    if (face) {
      face.consumedAtMs = occurredAtMs;
      this.touchSessionFace(session, face.occurredAtMs);
    }
    if (direction) {
      direction.consumedAtMs = occurredAtMs;
      this.touchSessionDirection(session, direction.occurredAtMs);
    }

    return {
      correlationId: randomUUID(),
      occurredAt: new Date(occurredAtMs),
      rfid: { ...rfid },
      face: face ? { ...face } : undefined,
      direction: direction ? { ...direction } : undefined,
      session: this.snapshotSession(session),
    };
  }

  ingestCompositeSignal(signal: GateSignalInput) {
    return this.matchEvents(signal);
  }

  validateEvent(input: GateValidationInput): GateValidationResult {
    const correlation = input.correlation;
    const rfidPresent = Boolean(correlation?.rfid);
    const facePresent = input.facePresent ?? Boolean(correlation?.face);
    const faceMatched = input.faceMatched ?? false;
    const faceRfidUidHint = normalizeOptionalRfidUid(
      input.faceRfidUidHint ?? correlation?.face?.rfidUidHint,
    );
    const rfidUid = correlation?.rfid.rfidUid;
    const directionPresent = input.directionDetected ?? hasUsableDirection(correlation?.direction);
    const reasons: string[] = [];
    let hardRejected = false;
    let tier: GateConfidenceTier = "VALID";

    if (!rfidPresent || !rfidUid) {
      reasons.push("RFID event is required before a gate decision can be made.");
    }

    if (rfidUid && faceRfidUidHint && faceRfidUidHint !== rfidUid) {
      reasons.push(`RFID-face mismatch detected. Face hint ${faceRfidUidHint} does not match badge ${rfidUid}.`);
      hardRejected = true;
    }

    if (facePresent && !faceMatched) {
      reasons.push("Face evidence was captured but did not validate against the RFID owner.");
      hardRejected = true;
    }

    if (!facePresent) {
      reasons.push("Face evidence is missing for this gate window.");
      if (input.strictFaceRequired ?? true) {
        tier = "LOW_CONFIDENCE";
      }
    }

    if (!directionPresent) {
      reasons.push("Direction evidence is missing for this gate window.");
    }

    if (hardRejected) {
      tier = "REJECT";
    }

    return {
      tier,
      hardRejected,
      rfidPresent,
      facePresent,
      faceMatched,
      directionPresent,
      rfidUid,
      faceRfidUidHint,
      reasons,
    };
  }

  calculateConfidence(validation: GateValidationResult): GateConfidenceScore {
    const breakdown = {
      rfid: validation.rfidPresent ? RFID_SCORE : 0,
      face: validation.faceMatched ? FACE_SCORE : 0,
      direction: validation.directionPresent ? DIRECTION_SCORE : 0,
    };
    const total = breakdown.rfid + breakdown.face + breakdown.direction;
    let tier: GateConfidenceTier;

    if (validation.hardRejected || total < 50) {
      tier = "REJECT";
    } else if (total >= 80) {
      tier = "VALID";
    } else {
      tier = "LOW_CONFIDENCE";
    }

    return {
      tier,
      total,
      breakdown,
    };
  }

  decideAction(input: GateDecisionInput): GateDecisionResult {
    const movementDirection = input.movementDirection;
    const hasExitDirection = movementDirection === "EXIT";
    const hasEntryDirection = movementDirection === "ENTRY";

    if (input.timedOut && input.hasOpenAttendance) {
      return {
        action: "EXIT",
        tier: "LOW_CONFIDENCE",
        reason: `Automatic timeout EXIT after ${SESSION_EXIT_TIMEOUT_MS / 1000} seconds without RFID or face activity.`,
        timeoutFallback: true,
      };
    }

    if (!input.validation.rfidPresent) {
      return {
        action: "IGNORE",
        tier: "IGNORE",
        reason: "Ignoring event because RFID is missing from the matched window.",
        timeoutFallback: false,
      };
    }

    if (input.validation.hardRejected) {
      return {
        action: "REJECT",
        tier: "REJECT",
        reason: input.validation.reasons[0] ?? "Validation failed.",
        timeoutFallback: false,
      };
    }

    if (!input.validation.directionPresent) {
      return {
        action: "IGNORE",
        tier: "IGNORE",
        reason: "Ignoring event because direction is missing from the matched window.",
        timeoutFallback: false,
      };
    }

    if (input.confidence.tier === "LOW_CONFIDENCE") {
      return {
        action: "IGNORE",
        tier: "LOW_CONFIDENCE",
        reason: input.validation.reasons[0] ?? "Confidence is below the required threshold for a strict gate decision.",
        timeoutFallback: false,
      };
    }

    if (input.confidence.tier === "REJECT") {
      return {
        action: "REJECT",
        tier: "REJECT",
        reason: input.validation.reasons[0] ?? "Confidence is too low to approve this gate event.",
        timeoutFallback: false,
      };
    }

    if (input.hasOpenAttendance) {
      if (hasExitDirection) {
        return {
          action: "EXIT",
          tier: input.confidence.tier,
          reason: "Exit approved from a valid RFID-face pair with exit direction.",
          timeoutFallback: false,
        };
      }

      if (hasEntryDirection) {
        return {
          action: "IGNORE",
          tier: "IGNORE",
          reason: "Duplicate ENTRY ignored because the employee is already present.",
          timeoutFallback: false,
        };
      }

      return {
        action: "IGNORE",
        tier: "IGNORE",
        reason: "Open session kept in PRESENT state until EXIT direction or timeout is observed.",
        timeoutFallback: false,
      };
    }

    if (hasExitDirection) {
      return {
        action: "REJECT",
        tier: "REJECT",
        reason: "Exit direction was detected, but no active ENTRY session exists for this RFID tag.",
        timeoutFallback: false,
      };
    }

    return {
      action: "ENTRY",
      tier: input.confidence.tier,
      reason: hasEntryDirection
        ? "Entry approved from a valid RFID-face pair with entry direction."
        : "Entry approved from a valid RFID-face pair within the matching window.",
      timeoutFallback: false,
    };
  }

  buildTimeoutExitDecision(session: GateSessionSnapshot): GateDecisionResult {
    return {
      action: "EXIT",
      tier: "LOW_CONFIDENCE",
      reason: `Automatic timeout EXIT for ${session.rfidUid} after ${SESSION_EXIT_TIMEOUT_MS / 1000} seconds without RFID or face activity.`,
      timeoutFallback: true,
    };
  }

  getDuplicateReason(args: {
    session: GateSessionSnapshot;
    occurredAt: Date;
    movementDirection?: MovementDirection;
  }) {
    const decisionAtMs = args.session.lastDecisionAtMs;
    if (!decisionAtMs) {
      return null;
    }

    if (
      args.session.lastOutcome !== "ENTRY"
      && args.session.lastOutcome !== "EXIT"
      && args.session.lastOutcome !== "IGNORED"
    ) {
      return null;
    }

    if (args.occurredAt.getTime() - decisionAtMs > args.session.duplicateWindowMs) {
      return null;
    }

    if (
      args.movementDirection
      && args.movementDirection !== "UNKNOWN"
      && args.session.lastAction
      && args.movementDirection !== args.session.lastAction
    ) {
      return null;
    }

    if (args.session.lastAction === "ENTRY" || args.session.state === "PRESENT") {
      return "Tag is still active in the gate session. Duplicate scan ignored.";
    }

    if (args.session.lastAction === "EXIT") {
      return "Exit was already recorded for this tag. Duplicate scan ignored.";
    }

    return "Duplicate scan ignored.";
  }

  recordSessionOutcome(args: {
    session: GateSessionSnapshot;
    occurredAt: Date;
    outcome: SessionOutcome;
    action?: AttendanceAction | null;
  }) {
    const key = this.buildSessionKey(args.session.deviceId, args.session.rfidUid);
    const session = this.sessions.get(key);
    if (!session) {
      return null;
    }

    session.lastDecisionAtMs = args.occurredAt.getTime();
    session.lastOutcome = args.outcome;

    if (args.action) {
      session.lastAction = args.action;
      session.state = args.action === "ENTRY" ? "PRESENT" : "DISAPPEARED";
      session.timeoutExitArmedAtMs = null;
      session.timeoutExitReportedAtMs = args.action === "EXIT"
        ? args.occurredAt.getTime()
        : null;
    } else if (args.outcome === "IGNORED" && session.lastAction === "ENTRY") {
      session.state = "PRESENT";
    }

    return this.snapshotSession(session);
  }

  armTimeoutExit(args: {
    session: GateSessionSnapshot;
    occurredAt: Date;
  }) {
    const key = this.buildSessionKey(args.session.deviceId, args.session.rfidUid);
    const session = this.sessions.get(key);
    if (!session) {
      return null;
    }

    session.state = "PRESENT";
    session.timeoutExitArmedAtMs = args.occurredAt.getTime();
    session.timeoutExitReportedAtMs = null;
    return this.snapshotSession(session);
  }

  collectTimedOutSessions(now = new Date()) {
    const nowMs = now.getTime();
    this.cleanup(nowMs);

    const timedOutSessions: GateTimeoutExitCandidate[] = [];
    this.sessions.forEach((session) => {
      if (
        session.state !== "PRESENT"
        || session.lastAction !== "ENTRY"
        || session.timeoutExitArmedAtMs === null
        || session.timeoutExitReportedAtMs !== null
        || nowMs - session.timeoutExitArmedAtMs < SESSION_EXIT_TIMEOUT_MS
      ) {
        return;
      }

      session.timeoutExitReportedAtMs = nowMs;
      timedOutSessions.push({
        occurredAt: new Date(nowMs),
        session: this.snapshotSession(session),
      });
    });

    return timedOutSessions;
  }

  markTagDisappeared(args: {
    deviceId: string;
    rfidUid: string;
    occurredAt?: Date;
  }) {
    const key = this.buildSessionKey(args.deviceId, args.rfidUid);
    const session = this.sessions.get(key);
    if (!session) {
      return null;
    }

    const occurredAtMs = toOccurredAtMs(args.occurredAt);
    session.lastSeenAtMs = occurredAtMs;
    session.state = "DISAPPEARED";
    return this.snapshotSession(session);
  }
}

export const gateMatchingEngine = new GateMatchingEngine();
