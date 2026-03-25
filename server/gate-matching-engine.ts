import { randomUUID } from "crypto";
import type { MovementAxis, ScanTechnology } from "@shared/schema";

export type AttendanceAction = "ENTRY" | "EXIT";
export type MovementDirection = AttendanceAction | "UNKNOWN";
export type SessionOutcome = AttendanceAction | "REJECTED" | "IGNORED";

const MATCH_WINDOW_MS = 2_000;
const FRAGMENT_TTL_MS = 12_000;
const SESSION_TTL_MS = 12 * 60 * 60 * 1_000;
const HF_DUPLICATE_WINDOW_MS = 3_500;
const UHF_DUPLICATE_WINDOW_MS = 1_200;

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

type SessionState = "IDLE" | "PRESENT" | "DISAPPEARED";

interface GateSessionState {
  sessionId: string;
  deviceId: string;
  rfidUid: string;
  scanTechnology: ScanTechnology;
  state: SessionState;
  lastSeenAtMs: number;
  lastDecisionAtMs: number | null;
  lastOutcome: SessionOutcome | null;
  lastAction: AttendanceAction | null;
  duplicateWindowMs: number;
}

export interface GateSessionSnapshot {
  sessionId: string;
  deviceId: string;
  rfidUid: string;
  scanTechnology: ScanTechnology;
  state: SessionState;
  lastSeenAtMs: number;
  lastDecisionAtMs: number | null;
  lastOutcome: SessionOutcome | null;
  lastAction: AttendanceAction | null;
  duplicateWindowMs: number;
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
  movementDirection?: MovementDirection;
  movementAxis?: MovementAxis;
  movementConfidence?: number;
}

function toOccurredAtMs(occurredAt?: Date) {
  return occurredAt?.getTime() ?? Date.now();
}

function normalizeRfidUid(rfidUid: string) {
  return rfidUid.trim().toUpperCase();
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
      lastDecisionAtMs: session.lastDecisionAtMs,
      lastOutcome: session.lastOutcome,
      lastAction: session.lastAction,
      duplicateWindowMs: session.duplicateWindowMs,
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
        lastDecisionAtMs: null,
        lastOutcome: null,
        lastAction: null,
        duplicateWindowMs: duplicateWindowForTechnology(args.scanTechnology),
      };
      this.sessions.set(key, session);
    }

    session.scanTechnology = args.scanTechnology;
    session.duplicateWindowMs = duplicateWindowForTechnology(args.scanTechnology);
    session.lastSeenAtMs = Math.max(session.lastSeenAtMs, args.occurredAtMs);
    session.state = "PRESENT";

    return session;
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
    const normalizedHint = args.rfidUidHint ? normalizeRfidUid(args.rfidUidHint) : undefined;
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
    const normalizedHint = args.rfidUidHint ? normalizeRfidUid(args.rfidUidHint) : undefined;
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
      rfidUidHint: signal.rfidUid ? normalizeRfidUid(signal.rfidUid) : undefined,
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

  ingestCompositeSignal(signal: GateSignalInput): GateCorrelationMatch | null {
    const occurredAtMs = toOccurredAtMs(signal.occurredAt);
    this.cleanup(occurredAtMs);

    let face = this.recordFaceSignal(signal);
    let direction = this.recordDirectionSignal(signal);
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
        rfidUidHint: signal.rfidUid,
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
    }
    if (direction) {
      direction.consumedAtMs = occurredAtMs;
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
    } else if (args.outcome === "IGNORED" && session.lastAction === "ENTRY") {
      session.state = "PRESENT";
    }

    return this.snapshotSession(session);
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
