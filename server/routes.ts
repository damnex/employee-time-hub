import type { Express } from "express";
import type { Server } from "http";
import { promises as fs } from "fs";
import { storage } from "./storage";
import { allowInsecureFaceFallback, useTriggeredCameraFaceRecognition } from "./env";
import { api } from "@shared/routes";
import {
  faceProfileSchema,
  type Attendance,
  type Employee,
  type FaceCaptureMode,
  type FacePose,
  type MovementAxis,
  type GateDecision,
  type ScanTechnology,
  scanTechnologySchema,
} from "@shared/schema";
import { z } from "zod";
import {
  buildPythonFaceDescriptorMeta,
  PYTHON_DATASET_ROOT,
  type PythonLiveRecognitionFace,
  PYTHON_LBPH_LABELS_PATH,
  PYTHON_LBPH_MODEL_PATH,
  readPythonFaceDescriptorMeta,
  recognizeRfidTriggeredFaceWithPython,
  recognizeLiveFrameWithPython,
  removeEmployeeDataset,
  retrainPythonFaceModel,
  appendEmployeeDatasetFrames,
  parseDataUrl,
  saveEmployeeDatasetPhotos,
  verifyGateFramesWithPython,
  warmPythonFaceWorker,
} from "./python-face";
import path from "path";
import {
  gateMatchingEngine,
  type GateCorrelationMatch,
} from "./gate-matching-engine";
import { registerRfidProxyRoutes } from "./rfid-proxy";
import { stopManagedRfidService, warmRfidService } from "./rfid-service";
import { handleRfidIntegration } from "./rfid-handler";
import { handleVisionIntegration } from "./vision-handler";
import { handleFaceIntegration } from "./face-handler";
import { decisionEngine } from "./decision-engine";
const SESSION_TIMEOUT_SWEEP_MS = 1000;
type AttendanceAction = "ENTRY" | "EXIT";
type MovementDirection = AttendanceAction | "UNKNOWN";

interface ProcessScanInput {
  rfidUid: string;
  deviceId: string;
  faceFrames?: string[];
  faceDescriptor?: number[];
  faceAnchorDescriptors?: number[][];
  faceConsistency?: number;
  faceQuality?: number;
  faceCaptureMode?: FaceCaptureMode;
  facePose?: FacePose;
  faceYaw?: number;
  facePitch?: number;
  faceRoll?: number;
  faceLiveConfidence?: number;
  faceRealConfidence?: number;
  scanTechnology?: ScanTechnology;
  movementDirection?: MovementDirection;
  movementAxis?: MovementAxis;
  movementConfidence?: number;
}

interface ProcessedScanResult {
  success: boolean;
  ignored?: boolean;
  message: string;
  employee?: Employee;
  attendance?: Attendance;
  matchConfidence?: number;
  matchDetails?: FaceMatchDetails;
  action?: AttendanceAction;
  movementDirection?: MovementDirection;
  movementConfidence?: number;
  detectedFaceLabel?: string;
  detectedFaceBox?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  } | null;
}

interface NormalizedFaceProfile {
  primaryDescriptor: number[];
  anchorDescriptors: number[][];
  consistency: number;
  averageQuality: number;
  captureMode: FaceCaptureMode | "legacy" | "unknown";
  secureCapture: boolean;
  legacy: boolean;
  engine: "legacy" | "heuristic" | "human";
  poseEmbeddings?: Array<{
    pose: FacePose;
    descriptor: number[];
    averageQuality: number;
    sampleCount: number;
  }>;
  pose?: FacePose;
  yaw: number;
  pitch: number;
  roll: number;
  liveConfidence: number | null;
  realConfidence: number | null;
}

interface FaceMatchDetails {
  primaryConfidence: number;
  anchorAverage: number;
  peakAnchorConfidence: number;
  strongAnchorRatio: number;
  liveConsistency: number;
  poseConfidence?: number;
  liveLiveness?: number;
  liveRealness?: number;
}

function roundMetric(value: number, digits = 4) {
  return Number(value.toFixed(digits));
}

function calculateLegacyMatchConfidence(v1: number[], v2: number[]): number {
  if (!v1 || !v2 || v1.length !== v2.length || !v1.length) {
    return 0;
  }

  const meanV1 = v1.reduce((sum, value) => sum + value, 0) / v1.length;
  const meanV2 = v2.reduce((sum, value) => sum + value, 0) / v2.length;
  const centeredV1 = v1.map((value) => value - meanV1);
  const centeredV2 = v2.map((value) => value - meanV2);
  const magnitudeV1 = Math.sqrt(centeredV1.reduce((sum, value) => sum + value * value, 0));
  const magnitudeV2 = Math.sqrt(centeredV2.reduce((sum, value) => sum + value * value, 0));
  if (!magnitudeV1 || !magnitudeV2) {
    return 0;
  }

  let dotProduct = 0;
  for (let i = 0; i < centeredV1.length; i++) {
    dotProduct += (centeredV1[i] / magnitudeV1) * (centeredV2[i] / magnitudeV2);
  }

  return roundMetric((dotProduct + 1) / 2);
}

function calculateHumanMatchConfidence(v1: number[], v2: number[]) {
  if (!v1 || !v2 || v1.length !== v2.length || !v1.length) {
    return 0;
  }

  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    const diff = v1[i] - v2[i];
    sum += diff * diff;
  }

  const distance = 25 * sum;
  if (distance === 0) {
    return 1;
  }

  const root = Math.sqrt(distance);
  const normalized = (1 - root / 100 - 0.2) / 0.6;
  return roundMetric(Math.max(Math.min(normalized, 1), 0));
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function normalizeStoredFaceProfile(faceDescriptor: unknown): NormalizedFaceProfile | null {
  if (!faceDescriptor) {
    return null;
  }

  if (Array.isArray(faceDescriptor) && faceDescriptor.every((value) => typeof value === "number")) {
    return {
      primaryDescriptor: faceDescriptor,
      anchorDescriptors: [faceDescriptor],
      consistency: 1,
      averageQuality: 0.55,
      captureMode: "legacy",
      secureCapture: false,
      legacy: true,
      engine: "legacy",
      yaw: 0,
      pitch: 0,
      roll: 0,
      liveConfidence: null,
      realConfidence: null,
    };
  }

  const parsedProfile = faceProfileSchema.safeParse(faceDescriptor);
  if (!parsedProfile.success) {
    return null;
  }

  return {
    primaryDescriptor: parsedProfile.data.primaryDescriptor,
    anchorDescriptors: parsedProfile.data.anchorDescriptors,
    consistency: parsedProfile.data.consistency,
    averageQuality: parsedProfile.data.averageQuality,
    captureMode: parsedProfile.data.captureMode ?? "unknown",
    secureCapture: parsedProfile.data.captureMode === "detected",
    legacy: parsedProfile.data.version !== 3,
    engine: parsedProfile.data.version === 3 ? "human" : "heuristic",
    poseEmbeddings: parsedProfile.data.version === 3
      ? parsedProfile.data.poseEmbeddings.map((embedding) => ({
          pose: embedding.pose,
          descriptor: embedding.descriptor,
          averageQuality: embedding.averageQuality,
          sampleCount: embedding.sampleCount,
        }))
      : undefined,
    pose: "unknown",
    yaw: 0,
    pitch: 0,
    roll: 0,
    liveConfidence: parsedProfile.data.version === 3 ? (parsedProfile.data.averageLive ?? null) : null,
    realConfidence: parsedProfile.data.version === 3 ? (parsedProfile.data.averageReal ?? null) : null,
  };
}

function normalizeLiveFaceProfile(input: ProcessScanInput): NormalizedFaceProfile | null {
  if (!input.faceDescriptor?.length) {
    return null;
  }

  const anchorDescriptors = input.faceAnchorDescriptors?.filter((descriptor) => {
    return descriptor.length === input.faceDescriptor?.length;
  });

  return {
    primaryDescriptor: input.faceDescriptor,
    anchorDescriptors: anchorDescriptors?.length ? anchorDescriptors : [input.faceDescriptor],
    consistency: input.faceConsistency ?? 1,
    averageQuality: input.faceQuality ?? input.faceConsistency ?? 0.35,
    captureMode: input.faceCaptureMode ?? "fallback",
    secureCapture: input.faceCaptureMode === "detected",
    legacy: false,
    engine:
      input.facePose
      || input.faceLiveConfidence !== undefined
      || input.faceRealConfidence !== undefined
        ? "human"
        : "heuristic",
    poseEmbeddings: input.facePose
      ? [{
          pose: input.facePose,
          descriptor: input.faceDescriptor,
          averageQuality: input.faceQuality ?? input.faceConsistency ?? 0.35,
          sampleCount: 1,
        }]
      : undefined,
    pose: input.facePose ?? "unknown",
    yaw: input.faceYaw ?? 0,
    pitch: input.facePitch ?? 0,
    roll: input.faceRoll ?? 0,
    liveConfidence: input.faceLiveConfidence ?? null,
    realConfidence: input.faceRealConfidence ?? null,
  };
}

function calculateProfileMatchMetrics(
  liveProfile: NormalizedFaceProfile,
  storedProfile: NormalizedFaceProfile,
) {
  const useHumanEmbeddingMatch =
    liveProfile.engine === "human" && storedProfile.engine === "human";
  const similarity = useHumanEmbeddingMatch
    ? calculateHumanMatchConfidence
    : calculateLegacyMatchConfidence;
  const strongAnchorThreshold = useHumanEmbeddingMatch
    ? HUMAN_FACE_ANCHOR_AVG_THRESHOLD
    : FACE_ANCHOR_AVG_THRESHOLD;
  const primaryConfidence = Math.max(
    similarity(liveProfile.primaryDescriptor, storedProfile.primaryDescriptor),
    ...storedProfile.anchorDescriptors.map((anchor) => {
      return similarity(liveProfile.primaryDescriptor, anchor);
    }),
  );
  const anchorScores = liveProfile.anchorDescriptors.map((liveAnchor) => {
    return Math.max(
      similarity(liveAnchor, storedProfile.primaryDescriptor),
      ...storedProfile.anchorDescriptors.map((storedAnchor) => {
        return similarity(liveAnchor, storedAnchor);
      }),
    );
  });
  const anchorAverage = average(anchorScores);
  const peakAnchorConfidence = Math.max(...anchorScores, 0);
  const strongAnchorRatio =
    anchorScores.filter((score) => score >= strongAnchorThreshold).length
    / Math.max(1, anchorScores.length);
  const poseScores = storedProfile.poseEmbeddings?.map((poseEmbedding) => {
    return {
      pose: poseEmbedding.pose,
      similarity: similarity(liveProfile.primaryDescriptor, poseEmbedding.descriptor),
    };
  }) ?? [];
  const poseConfidence = poseScores.length
    ? Math.max(
        ...poseScores.map((poseScore) => poseScore.similarity),
        liveProfile.pose && liveProfile.pose !== "unknown"
          ? poseScores.find((poseScore) => poseScore.pose === liveProfile.pose)?.similarity ?? 0
          : 0,
      )
    : primaryConfidence;
  const finalConfidence = roundMetric(
    useHumanEmbeddingMatch
      ? (
          primaryConfidence * 0.38
          + anchorAverage * 0.27
          + poseConfidence * 0.25
          + strongAnchorRatio * 0.1
        )
      : (
          primaryConfidence * 0.45
          + anchorAverage * 0.45
          + strongAnchorRatio * 0.1
        ),
  );

  return {
    primaryConfidence: roundMetric(primaryConfidence),
    anchorAverage: roundMetric(anchorAverage),
    peakAnchorConfidence: roundMetric(peakAnchorConfidence),
    strongAnchorRatio: roundMetric(strongAnchorRatio),
    poseConfidence: roundMetric(poseConfidence),
    finalConfidence,
  };
}

const FACE_MATCH_THRESHOLD = 0.87;
const FACE_PRIMARY_THRESHOLD = 0.88;
const FACE_ANCHOR_AVG_THRESHOLD = 0.85;
const FACE_ANCHOR_RATIO_THRESHOLD = 0.8;
const FACE_SCAN_CONSISTENCY_THRESHOLD = 0.88;
const INSECURE_FACE_MATCH_THRESHOLD = 0.94;
const INSECURE_FACE_PRIMARY_THRESHOLD = 0.95;
const INSECURE_FACE_ANCHOR_AVG_THRESHOLD = 0.93;
const INSECURE_FACE_ANCHOR_RATIO_THRESHOLD = 0.9;
const INSECURE_FACE_SCAN_CONSISTENCY_THRESHOLD = 0.94;
const LEGACY_FACE_MATCH_THRESHOLD = 0.9;
const MIN_STORED_FACE_QUALITY = 0.18;
const MIN_LIVE_FACE_QUALITY = 0.16;
const MIN_INSECURE_LIVE_FACE_QUALITY = 0.2;
const HUMAN_FACE_MATCH_THRESHOLD = 0.57;
const HUMAN_FACE_PRIMARY_THRESHOLD = 0.58;
const HUMAN_FACE_ANCHOR_AVG_THRESHOLD = 0.54;
const HUMAN_FACE_ANCHOR_RATIO_THRESHOLD = 0.55;
const HUMAN_FACE_POSE_THRESHOLD = 0.55;
const HUMAN_FACE_SCAN_CONSISTENCY_THRESHOLD = 0.6;
const MIN_HUMAN_STORED_FACE_QUALITY = 0.48;
const MIN_HUMAN_LIVE_FACE_QUALITY = 0.42;
const MIN_HUMAN_LIVE_LIVENESS = 0.45;
const MIN_HUMAN_LIVE_REALNESS = 0.35;
const DIRECTION_CONFIDENCE_THRESHOLD = 0.58;

function normalizeScanTechnology(scanTechnology?: ScanTechnology): ScanTechnology {
  return scanTechnology ?? "UHF_RFID";
}

async function logGateEvent(args: {
  date: string;
  input: ProcessScanInput;
  employee?: Employee;
  verificationStatus: Attendance["verificationStatus"];
  decision: GateDecision;
  message: string;
  matchConfidence?: number;
  liveFaceProfile?: NormalizedFaceProfile | null;
}) {
  try {
    await storage.createGateEvent({
      employeeId: args.employee?.id ?? null,
      date: args.date,
      rfidUid: args.input.rfidUid.trim().toUpperCase(),
      deviceId: args.input.deviceId,
      scanTechnology: normalizeScanTechnology(args.input.scanTechnology),
      decision: args.decision,
      verificationStatus: args.verificationStatus,
      eventMessage: args.message,
      movementDirection: args.input.movementDirection ?? "UNKNOWN",
      movementAxis: args.input.movementAxis ?? "none",
      movementConfidence: args.input.movementConfidence,
      matchConfidence: args.matchConfidence,
      faceQuality: args.liveFaceProfile?.averageQuality,
      faceConsistency: args.liveFaceProfile?.consistency,
      faceCaptureMode: args.liveFaceProfile?.captureMode,
    });
  } catch (error) {
    console.warn("[gate-events] Skipping raw gate event persistence:", error);
  }
}

let timeoutSweepInFlight = false;

async function processTimedOutGateSessions() {
  if (timeoutSweepInFlight) {
    return;
  }

  timeoutSweepInFlight = true;
  try {
    const timeoutCandidates = gateMatchingEngine.collectTimedOutSessions(new Date());
    for (const candidate of timeoutCandidates) {
      const employee = await storage.getEmployeeByRfid(candidate.session.rfidUid);
      if (!employee) {
        continue;
      }

      const timeoutDecision = gateMatchingEngine.buildTimeoutExitDecision(candidate.session);
      const timeoutDate = candidate.occurredAt.toISOString().split("T")[0];
      const openEntry = await storage.getOpenAttendance(employee.id, timeoutDate);

      if (!openEntry) {
        gateMatchingEngine.recordSessionOutcome({
          session: candidate.session,
          occurredAt: candidate.occurredAt,
          outcome: "EXIT",
          action: "EXIT",
        });
        continue;
      }

      const attendance = await closeOpenAttendance(openEntry, candidate.occurredAt);
      if (!attendance) {
        console.warn("[gate-timeout] Timed-out exit could not be recorded:", candidate.session.rfidUid);
        continue;
      }

      gateMatchingEngine.recordSessionOutcome({
        session: candidate.session,
        occurredAt: candidate.occurredAt,
        outcome: "EXIT",
        action: "EXIT",
      });

      await logGateEvent({
        date: timeoutDate,
        input: {
          rfidUid: candidate.session.rfidUid,
          deviceId: candidate.session.deviceId,
          scanTechnology: candidate.session.scanTechnology,
          movementDirection: "UNKNOWN",
          movementAxis: "none",
        },
        employee,
        verificationStatus: "EXIT",
        decision: "EXIT",
        message: timeoutDecision.reason,
      });
    }
  } finally {
    timeoutSweepInFlight = false;
  }
}

async function createFailureAttendance(
  employeeId: number,
  date: string,
  now: Date,
  deviceId: string,
  verificationStatus: "FAILED_FACE" | "FAILED_DIRECTION",
) {
  return storage.createAttendance({
    employeeId,
    date,
    entryTime: now,
    verificationStatus,
    deviceId,
  });
}

async function createEntryAttendance(
  employeeId: number,
  date: string,
  now: Date,
  deviceId: string,
) {
  return storage.createAttendance({
    employeeId,
    date,
    entryTime: now,
    verificationStatus: "ENTRY",
    deviceId,
  });
}

async function closeOpenAttendance(openEntry: Attendance, now: Date) {
  const workingHoursMs = now.getTime() - (openEntry.entryTime?.getTime() || now.getTime());
  const workingHours = workingHoursMs / (1000 * 60 * 60);

  return storage.updateAttendance(openEntry.id, {
    exitTime: now,
    workingHours: Number(workingHours.toFixed(2)),
    verificationStatus: "EXIT",
  });
}

function rememberSessionOutcome(
  correlation: GateCorrelationMatch | null | undefined,
  now: Date,
  outcome: "ENTRY" | "EXIT" | "REJECTED" | "IGNORED" | "LOW_CONFIDENCE",
  action?: AttendanceAction,
) {
  if (!correlation) {
    return;
  }

  gateMatchingEngine.recordSessionOutcome({
    session: correlation.session,
    occurredAt: now,
    outcome,
    action,
  });
}

function logGateDecisionPipeline(args: {
  correlation: GateCorrelationMatch | null;
  employee: Employee;
  validation: ReturnType<typeof gateMatchingEngine.validateEvent>;
  confidence: ReturnType<typeof gateMatchingEngine.calculateConfidence>;
  decision: ReturnType<typeof gateMatchingEngine.decideAction>;
}) {
  const { correlation, employee, validation, confidence, decision } = args;
  const matchedSummary = {
    correlationId: correlation?.correlationId ?? null,
    deviceId: correlation?.rfid.deviceId ?? null,
    rfidUid: correlation?.rfid.rfidUid ?? null,
    faceRfidUidHint: correlation?.face?.rfidUidHint ?? null,
    movementDirection: correlation?.direction?.movementDirection ?? "UNKNOWN",
    movementConfidence: correlation?.direction?.movementConfidence ?? null,
    sessionState: correlation?.session.state ?? null,
  };

  console.info(
    `[gate-engine] ${employee.employeeCode} ${employee.name} `
    + `matched=${JSON.stringify(matchedSummary)} `
    + `validation=${JSON.stringify(validation)} `
    + `confidence=${JSON.stringify(confidence)} `
    + `decision=${JSON.stringify(decision)}`,
  );
}

async function resolveAttendanceDecision(
  employee: Employee,
  now: Date,
  todayDate: string,
  deviceId: string,
  matchConfidence: number,
  correlation: GateCorrelationMatch | null,
  facePresent: boolean,
  faceMatched: boolean,
  faceRfidUidHint?: string | null,
  movementDirection?: MovementDirection,
  movementConfidence?: number,
): Promise<ProcessedScanResult> {
  const duplicateReason = correlation
    ? gateMatchingEngine.getDuplicateReason({
        session: correlation.session,
        occurredAt: now,
        movementDirection,
      })
    : null;

  if (duplicateReason) {
    console.info(
      `[gate-engine] ${employee.employeeCode} ${employee.name} `
      + `duplicate=${JSON.stringify({
        correlationId: correlation?.correlationId ?? null,
        deviceId: correlation?.rfid.deviceId ?? null,
        rfidUid: correlation?.rfid.rfidUid ?? null,
        movementDirection: movementDirection ?? "UNKNOWN",
        sessionState: correlation?.session.state ?? null,
      })} decision=${JSON.stringify({
        action: "IGNORE",
        tier: "IGNORE",
        reason: duplicateReason,
      })}`,
    );
    rememberSessionOutcome(correlation, now, "IGNORED");
    return {
      success: true,
      ignored: true,
      message: duplicateReason,
      employee,
      matchConfidence,
      movementDirection,
      movementConfidence,
    };
  }

  const openEntry = await storage.getOpenAttendance(employee.id, todayDate);
  const directionIsConfident =
    (movementDirection === "ENTRY" || movementDirection === "EXIT")
    && (movementConfidence ?? 0) >= DIRECTION_CONFIDENCE_THRESHOLD;
  const validation = gateMatchingEngine.validateEvent({
    correlation,
    facePresent,
    faceMatched,
    faceRfidUidHint,
    directionDetected: directionIsConfident,
  });
  const confidence = gateMatchingEngine.calculateConfidence(validation);
  const decision = gateMatchingEngine.decideAction({
    correlation,
    validation,
    confidence,
    hasOpenAttendance: Boolean(openEntry),
    movementDirection: directionIsConfident ? movementDirection : undefined,
  });
  logGateDecisionPipeline({
    correlation,
    employee,
    validation,
    confidence,
    decision,
  });

  if (decision.action === "IGNORE") {
    if (correlation && openEntry && !directionIsConfident) {
      gateMatchingEngine.armTimeoutExit({
        session: correlation.session,
        occurredAt: now,
      });
    }
    rememberSessionOutcome(
      correlation,
      now,
      decision.tier === "LOW_CONFIDENCE" ? "LOW_CONFIDENCE" : "IGNORED",
    );
    return {
      success: true,
      ignored: true,
      message: decision.reason,
      employee,
      matchConfidence,
      movementDirection,
      movementConfidence,
    };
  }

  if (decision.action === "REJECT") {
    const verificationStatus: "FAILED_FACE" | "FAILED_DIRECTION" =
      validation.hardRejected ? "FAILED_FACE" : "FAILED_DIRECTION";
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      deviceId,
      verificationStatus,
    );
    rememberSessionOutcome(
      correlation,
      now,
      decision.tier === "LOW_CONFIDENCE" ? "LOW_CONFIDENCE" : "REJECTED",
    );

    return {
      success: false,
      message: decision.reason,
      employee,
      attendance,
      matchConfidence,
      movementDirection,
      movementConfidence,
    };
  }

  if (decision.action === "EXIT") {
    if (!openEntry) {
      rememberSessionOutcome(correlation, now, "REJECTED");
      return {
        success: false,
        message: "Exit could not be recorded because no active entry exists.",
        employee,
        matchConfidence,
        movementDirection,
        movementConfidence,
      };
    }

    const attendance = await closeOpenAttendance(openEntry, now);
    if (!attendance) {
      rememberSessionOutcome(correlation, now, "REJECTED");
      return {
        success: false,
        message: "Exit could not be recorded. Please retry the scan.",
        employee,
        matchConfidence,
        movementDirection,
        movementConfidence,
      };
    }

    rememberSessionOutcome(correlation, now, "EXIT", "EXIT");
    return {
      success: true,
      message: decision.reason,
      employee,
      attendance,
      matchConfidence,
      action: "EXIT",
      movementDirection,
      movementConfidence,
    };
  }

  const attendance = await createEntryAttendance(employee.id, todayDate, now, deviceId);
  rememberSessionOutcome(correlation, now, "ENTRY", "ENTRY");

  return {
    success: true,
    message: decision.reason,
    employee,
    attendance,
    matchConfidence,
    action: "ENTRY",
    movementDirection,
    movementConfidence,
  };
}

type LogAndReturnFn = (
  result: ProcessedScanResult,
  options: {
    verificationStatus: Attendance["verificationStatus"];
    decision: GateDecision;
    liveFaceProfile?: NormalizedFaceProfile | null;
  },
) => Promise<ProcessedScanResult>;

async function ensurePythonFaceModelExists() {
  await Promise.all([
    fs.access(PYTHON_LBPH_MODEL_PATH),
    fs.access(PYTHON_LBPH_LABELS_PATH),
  ]);
}

function buildPythonMatchDetails(matchConfidence: number): FaceMatchDetails {
  return {
    primaryConfidence: matchConfidence,
    anchorAverage: matchConfidence,
    peakAnchorConfidence: matchConfidence,
    strongAnchorRatio: matchConfidence > 0 ? 1 : 0,
    liveConsistency: matchConfidence,
  };
}

function buildTriggeredCameraMatchDetails(matchConfidence: number): FaceMatchDetails {
  return {
    primaryConfidence: matchConfidence,
    anchorAverage: matchConfidence,
    peakAnchorConfidence: matchConfidence,
    strongAnchorRatio: matchConfidence > 0 ? 1 : 0,
    liveConsistency: matchConfidence,
  };
}

async function syncPythonFaceMetadataForEmployees(
  employees: Employee[],
  options: {
    labels?: Array<{
      folderName: string;
      sampleCount: number;
      includedInTraining: boolean;
    }>;
    failureMessage?: string;
  } = {},
) {
  const labelsByFolder = new Map(
    (options.labels ?? []).map((label) => [label.folderName, label]),
  );

  await Promise.all(
    employees.map(async (employee) => {
      const pythonMeta = readPythonFaceDescriptorMeta(employee.faceDescriptor);
      if (!pythonMeta) {
        return;
      }

      const label = labelsByFolder.get(pythonMeta.folderName);
      const status = options.failureMessage
        ? "failed"
        : label?.includedInTraining
          ? "trained"
          : "failed";
      const message = options.failureMessage
        ?? (label?.includedInTraining
          ? "Python model trained successfully."
          : "Python training skipped this employee. Capture clearer dataset images and retry.");

      await storage.updateEmployee(employee.id, {
        faceDescriptor: buildPythonFaceDescriptorMeta({
          folderName: pythonMeta.folderName,
          datasetSampleCount: label?.sampleCount ?? pythonMeta.datasetSampleCount,
          status,
          trainedAt: status === "trained" ? new Date().toISOString() : null,
          lastTrainingMessage: message,
        }),
      });
    }),
  );
}

async function retrainAndSyncPythonFaceModel() {
  const employees = await storage.getEmployees();
  const trainingSummary = await retrainPythonFaceModel(employees);
  await syncPythonFaceMetadataForEmployees(employees, {
    labels: trainingSummary.labels,
  });
}

async function processPythonRfidScan(args: {
  input: ProcessScanInput;
  employee: Employee;
  now: Date;
  todayDate: string;
  correlation: GateCorrelationMatch | null;
  logAndReturn: LogAndReturnFn;
}) {
  const { input, employee, now, todayDate, correlation, logAndReturn } = args;

  async function salvageBadgeOwnerFromFrames(
    faceFrames: string[],
    targetEmployee: Employee,
  ): Promise<{
    face: PythonLiveRecognitionFace | null;
    frameSize: { width: number; height: number } | null;
  }> {
    let bestFace: PythonLiveRecognitionFace | null = null;
    let bestFrameSize: { width: number; height: number } | null = null;
    for (const frame of faceFrames) {
      try {
        const recognition = await recognizeLiveFrameWithPython(frame, 12);
        for (const face of recognition.faces) {
          const code = face.employeeCode?.trim();
          if (!face.verified || !code || code !== targetEmployee.employeeCode) {
            continue;
          }

          if (!bestFace || face.confidence > bestFace.confidence) {
            bestFace = face;
            bestFrameSize = { width: recognition.frameWidth, height: recognition.frameHeight };
          }
        }
        if (bestFace && (bestFace.confidence ?? 0) >= 0.65) {
          break;
        }
      } catch (err) {
        console.warn("Salvage frame recognition failed:", err);
      }
    }
    return { face: bestFace, frameSize: bestFrameSize };
  }

  if (!input.faceFrames?.length) {
    const result = await resolveAttendanceDecision(
      employee,
      now,
      todayDate,
      input.deviceId,
      0,
      correlation,
      false,
      false,
      null,
      input.movementDirection,
      input.movementConfidence,
    );

    return logAndReturn({
      ...result,
      message: "Live gate frames were not captured, so face confidence stayed low.",
    }, {
      verificationStatus: "FAILED_FACE",
      decision: result.action ?? (result.ignored ? "UNKNOWN" : "REJECTED"),
    });
  }

  try {
    await ensurePythonFaceModelExists();
  } catch {
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: "Python face model is not trained yet. Enroll employees and refresh the Python training first.",
      employee,
      attendance,
      matchConfidence: 0,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
    });
  }

  try {
    const verification = await verifyGateFramesWithPython(input.faceFrames);
    const matchConfidence = verification.matchConfidence;
    const matchDetails = buildPythonMatchDetails(matchConfidence);
    const detectedFaceLabel = verification.employee?.displayName
      ?? (verification.framesWithFace ? "Unknown Face" : undefined);
    const detectedFaceBox = verification.bestBox ?? null;

    let matchedEmployeeCode = verification.employee?.employeeCode?.trim();
    let matchedRfidUid = verification.employee?.rfidUid?.trim().toUpperCase();
    let matchesBadgeOwner = Boolean(
      (matchedEmployeeCode && matchedEmployeeCode === employee.employeeCode)
      || (matchedRfidUid && matchedRfidUid === employee.rfidUid.toUpperCase()),
    );

    // Salvage multi-face scenes: re-run recognition across all frames and prefer the badge owner.
    if (!matchesBadgeOwner && input.faceFrames.length) {
      const { face: salvageFace, frameSize } = await salvageBadgeOwnerFromFrames(input.faceFrames, employee);
      if (salvageFace) {
        const goodConfidence = (salvageFace.confidence ?? 0) >= 0.42;
        const goodDistance = salvageFace.distance !== null && verification.distanceThreshold
          ? salvageFace.distance <= verification.distanceThreshold * 1.15
          : false;
        if (goodConfidence || goodDistance) {
        matchesBadgeOwner = true;
        matchedEmployeeCode = employee.employeeCode;
        matchedRfidUid = employee.rfidUid.toUpperCase();
        verification.verified = true;
        verification.employee = {
          folderName: salvageFace.label,
          displayName: employee.name,
          employeeCode: employee.employeeCode,
          department: employee.department,
          rfidUid: employee.rfidUid,
          sampleCount: verification.employee?.sampleCount ?? undefined,
        };
        verification.matchConfidence = salvageFace.confidence;
        verification.bestDistance = salvageFace.distance ?? verification.bestDistance;
        verification.bestBox = {
          top: salvageFace.box.top,
          right: salvageFace.box.right,
          bottom: salvageFace.box.bottom,
          left: salvageFace.box.left,
        };
          verification.framesWithFace = Math.max(verification.framesWithFace, 1);
          verification.framesProcessed = Math.max(verification.framesProcessed, input.faceFrames.length);
          if (frameSize) {
            verification.previewFrameSize = { width: frameSize.width, height: frameSize.height };
          }
        }
      } else {
        // prevent mis-assignment to another person
        verification.verified = false;
      }
    }

    // enforce confidence floor to avoid weak false matches
    if (verification.verified && verification.matchConfidence < 0.45) {
      verification.verified = false;
    }

    if (!verification.verified || !verification.employee) {
      const result = await resolveAttendanceDecision(
        employee,
        now,
        todayDate,
        input.deviceId,
        matchConfidence,
        correlation,
        verification.framesWithFace > 0,
        false,
        verification.employee?.rfidUid ?? null,
        verification.movementDirection,
        verification.movementConfidence,
      );

      return logAndReturn({
        ...result,
        message: verification.framesWithFace
          ? result.message
          : "No face was detected clearly in the gate camera frames, so the event stayed low confidence.",
        matchDetails,
        detectedFaceLabel,
        detectedFaceBox,
      }, {
        verificationStatus: "FAILED_FACE",
        decision: result.action ?? (result.ignored ? "UNKNOWN" : "REJECTED"),
      });
      }

      if (!matchesBadgeOwner) {
        const result = await resolveAttendanceDecision(
          employee,
          now,
          todayDate,
          input.deviceId,
          matchConfidence,
          correlation,
          true,
          false,
          matchedRfidUid ?? verification.employee.rfidUid ?? null,
          verification.movementDirection,
          verification.movementConfidence,
        );

      return logAndReturn({
        ...result,
        matchDetails,
        detectedFaceLabel,
        detectedFaceBox,
      }, {
        verificationStatus: "FAILED_FACE",
        decision: result.action ?? (result.ignored ? "UNKNOWN" : "REJECTED"),
      });
    }

    // Enrich training data with the latest gate frames for this employee (non-blocking to avoid latency).
    if (verification.employee.folderName) {
      void appendEmployeeDatasetFrames({
        folderName: verification.employee.folderName,
        frames: input.faceFrames,
      }).catch((error) => {
        console.warn("[python-face] Failed to save live gate frames:", error);
      });
    }

    const result = await resolveAttendanceDecision(
      employee,
      now,
      todayDate,
      input.deviceId,
      matchConfidence,
      correlation,
      true,
      true,
      matchedRfidUid,
      verification.movementDirection,
      verification.movementConfidence,
    );

    return logAndReturn({
      ...result,
      matchDetails,
      movementDirection: verification.movementDirection,
      movementConfidence: verification.movementConfidence,
      detectedFaceLabel,
      detectedFaceBox,
    }, {
      verificationStatus: result.attendance?.verificationStatus ?? (result.success ? "ENTRY" : "FAILED_DIRECTION"),
      decision: result.action ?? (result.success ? "UNKNOWN" : "REJECTED"),
    });
  } catch (error) {
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message:
        error instanceof Error
          ? `Python verification failed: ${error.message}`
          : "Python verification failed unexpectedly.",
      employee,
      attendance,
      matchConfidence: 0,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
    });
  }
}

async function processTriggeredCameraFaceScan(args: {
  input: ProcessScanInput;
  employee: Employee;
  normalizedRfidUid: string;
  now: Date;
  todayDate: string;
  correlation: GateCorrelationMatch | null;
  logAndReturn: LogAndReturnFn;
}) {
  const { input, employee, normalizedRfidUid, now, todayDate, correlation, logAndReturn } = args;

  try {
    const recognition = await recognizeRfidTriggeredFaceWithPython({
      rfidTag: normalizedRfidUid,
      timestamp: now.getTime(),
      frameCount: 1,
      maxFaces: 2,
      freshnessMs: 900,
      captureSpacingMs: 80,
    });
    const recognitionTimestampDeltaMs = recognition.timestampDeltaMs;
    const recognitionIsFresh = recognitionTimestampDeltaMs <= 2_000;
    const detectedFaceLabel = recognition.name ?? undefined;
    const detectedFaceBox = recognition.bestBox ?? null;
    const matchConfidence = recognition.confidence;
    const matchDetails = buildTriggeredCameraMatchDetails(matchConfidence);
    const recognizedRfidUid = recognition.rfidUid?.trim().toUpperCase() ?? null;
    const facePresent = recognition.status !== "NO_FACE" && recognition.facesDetected > 0 && recognitionIsFresh;
    const faceMatched = Boolean(
      recognitionIsFresh
      && recognition.status === "MATCH"
      && recognizedRfidUid
      && recognizedRfidUid === normalizedRfidUid
    );

    const result = await resolveAttendanceDecision(
      employee,
      now,
      todayDate,
      input.deviceId,
      matchConfidence,
      correlation,
      facePresent,
      faceMatched,
      recognizedRfidUid,
      input.movementDirection,
      input.movementConfidence,
    );

    const message = !recognitionIsFresh
      ? "Triggered live face result arrived outside the allowed 2 second window, so the scan stayed low confidence."
      : recognition.status === "NO_FACE"
        ? "RFID was detected, but no live face was visible in the camera window."
        : recognition.status === "UNKNOWN"
          ? recognition.multipleFaces
            ? result.message
            : "A live face was captured after the RFID trigger, but the person could not be verified confidently."
          : result.message;

    return await logAndReturn({
      ...result,
      message,
      matchConfidence,
      matchDetails,
      detectedFaceLabel,
      detectedFaceBox,
      movementDirection: input.movementDirection,
      movementConfidence: input.movementConfidence,
    }, {
      verificationStatus: result.attendance?.verificationStatus ?? (result.success ? "ENTRY" : "FAILED_FACE"),
      decision: result.action ?? (result.ignored ? "UNKNOWN" : "REJECTED"),
      liveFaceProfile: null,
    });
  } catch (error) {
    console.warn("[gate-camera] Triggered face recognition failed. Falling back to alternate face flow:", error);
    return null;
  }
}

async function processRfidScan(input: ProcessScanInput): Promise<ProcessedScanResult> {
  const normalizedRfidUid = input.rfidUid.trim().toUpperCase();
  input.rfidUid = normalizedRfidUid;
  const now = new Date();
  const todayDate = now.toISOString().split("T")[0];
  const correlation = gateMatchingEngine.ingestCompositeSignal({
    ...input,
    occurredAt: now,
  });
  const employee = await storage.getEmployeeByRfid(normalizedRfidUid);
  const logAndReturn = async (
    result: ProcessedScanResult,
    options: {
      verificationStatus: Attendance["verificationStatus"];
      decision: GateDecision;
      liveFaceProfile?: NormalizedFaceProfile | null;
    },
  ) => {
    await logGateEvent({
      date: todayDate,
      input,
      employee: result.employee ?? employee,
      verificationStatus: options.verificationStatus,
      decision: options.decision,
      message: result.message,
      matchConfidence: result.matchConfidence,
      liveFaceProfile: options.liveFaceProfile,
    });

    return result;
  };

  if (!employee) {
    rememberSessionOutcome(correlation, now, "REJECTED");
    return logAndReturn({
      success: false,
      message: "Unknown RFID card rejected.",
    }, {
      verificationStatus: "UNKNOWN_RFID",
      decision: "REJECTED",
    });
  }

  const triggeredCameraFaceEnabled = useTriggeredCameraFaceRecognition();
  // The web gate UI already captures frames from the same browser camera the
  // operator is looking at. Prefer those frames when present so we do not
  // silently switch to a different OpenCV device (camera source 0 / RTSP)
  // mid-scan and create "camera mismatch" failures.
  if (triggeredCameraFaceEnabled && !input.faceFrames?.length) {
    const triggeredResult = await processTriggeredCameraFaceScan({
      input,
      employee,
      normalizedRfidUid,
      now,
      todayDate,
      correlation,
      logAndReturn,
    });
    if (triggeredResult) {
      return triggeredResult;
    }
  }

  if (input.faceFrames?.length) {
    return processPythonRfidScan({
      input,
      employee,
      now,
      todayDate,
      correlation,
      logAndReturn,
    });
  }

  let matchConfidence = 0;
  let faceMatched = false;
  let matchDetails: FaceMatchDetails | undefined;
  const insecureFallbackAllowed = allowInsecureFaceFallback();
  const liveFaceProfile = normalizeLiveFaceProfile(input);
  const storedFaceProfile = normalizeStoredFaceProfile(employee.faceDescriptor);

  if (!storedFaceProfile) {
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: "No secure face profile is stored for this employee. Re-enroll the employee in Chrome or Edge.",
      employee,
      attendance,
      matchConfidence,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
      liveFaceProfile,
    });
  }

  if (!liveFaceProfile) {
    const result = await resolveAttendanceDecision(
      employee,
      now,
      todayDate,
      input.deviceId,
      matchConfidence,
      correlation,
      false,
      false,
      null,
      input.movementDirection,
      input.movementConfidence,
    );

    return logAndReturn({
      ...result,
      message: "Live face verification data was not captured, so the scan stayed low confidence.",
      movementDirection: input.movementDirection,
      movementConfidence: input.movementConfidence,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: result.action ?? (result.ignored ? "UNKNOWN" : "REJECTED"),
      liveFaceProfile,
    });
  }

  if (!storedFaceProfile.secureCapture && !insecureFallbackAllowed) {
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: "This employee face profile was enrolled with an insecure fallback flow. Re-enroll in Chrome or Edge.",
      employee,
      attendance,
      matchConfidence,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
      liveFaceProfile,
    });
  }

  if (!liveFaceProfile?.secureCapture && !insecureFallbackAllowed) {
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: "Secure live face detection is unavailable. Use Chrome or Edge on the gate terminal.",
      employee,
      attendance,
      matchConfidence,
      movementDirection: input.movementDirection,
      movementConfidence: input.movementConfidence,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
      liveFaceProfile,
    });
  }

  if (storedFaceProfile.engine !== liveFaceProfile.engine) {
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: storedFaceProfile.engine === "human"
        ? "The gate is still using the older biometric engine. Refresh the terminal and retry the scan."
        : "This employee is enrolled with the previous biometric engine. Re-enroll the employee with the new ML face capture.",
      employee,
      attendance,
      matchConfidence,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
      liveFaceProfile,
    });
  }

  if (storedFaceProfile.engine === "human") {
    if (storedFaceProfile.averageQuality < MIN_HUMAN_STORED_FACE_QUALITY) {
      const attendance = await createFailureAttendance(
        employee.id,
        todayDate,
        now,
        input.deviceId,
        "FAILED_FACE",
      );

      return logAndReturn({
        success: false,
        message: "Stored ML face profile quality is too low. Re-enroll this employee with front, left, and right capture.",
        employee,
        attendance,
        matchConfidence,
      }, {
        verificationStatus: "FAILED_FACE",
        decision: "REJECTED",
        liveFaceProfile,
      });
    }

    if (liveFaceProfile.averageQuality < MIN_HUMAN_LIVE_FACE_QUALITY) {
      const attendance = await createFailureAttendance(
        employee.id,
        todayDate,
        now,
        input.deviceId,
        "FAILED_FACE",
      );

      return logAndReturn({
        success: false,
        message: "Live ML face quality is too low. Move closer, reduce backlight, and keep the tracked face box stable before retrying.",
        employee,
        attendance,
        matchConfidence,
        movementDirection: input.movementDirection,
        movementConfidence: input.movementConfidence,
      }, {
        verificationStatus: "FAILED_FACE",
        decision: "REJECTED",
        liveFaceProfile,
      });
    }

    if ((liveFaceProfile.liveConfidence ?? 0) < MIN_HUMAN_LIVE_LIVENESS) {
      const attendance = await createFailureAttendance(
        employee.id,
        todayDate,
        now,
        input.deviceId,
        "FAILED_FACE",
      );

      return logAndReturn({
        success: false,
        message: "Liveness verification failed. Ask the person to blink or move naturally and retry in better lighting.",
        employee,
        attendance,
        matchConfidence,
      }, {
        verificationStatus: "FAILED_FACE",
        decision: "REJECTED",
        liveFaceProfile,
      });
    }

    if ((liveFaceProfile.realConfidence ?? 0) < MIN_HUMAN_LIVE_REALNESS) {
      const attendance = await createFailureAttendance(
        employee.id,
        todayDate,
        now,
        input.deviceId,
        "FAILED_FACE",
      );

      return logAndReturn({
        success: false,
        message: "Anti-spoof verification failed. Remove glare, masks, or screen reflections and retry with a real face in view.",
        employee,
        attendance,
        matchConfidence,
      }, {
        verificationStatus: "FAILED_FACE",
        decision: "REJECTED",
        liveFaceProfile,
      });
    }
  }

  if (
    storedFaceProfile.engine !== "human"
    && !storedFaceProfile.legacy
    && storedFaceProfile.averageQuality < MIN_STORED_FACE_QUALITY
  ) {
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: "Stored face profile quality is too low for reliable verification. Re-enroll this employee in brighter, front-facing lighting.",
      employee,
      attendance,
      matchConfidence,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
      liveFaceProfile,
    });
  }

  const requiredLiveQuality = liveFaceProfile.secureCapture
    ? MIN_LIVE_FACE_QUALITY
    : MIN_INSECURE_LIVE_FACE_QUALITY;

  if (liveFaceProfile.engine !== "human" && liveFaceProfile.averageQuality < requiredLiveQuality) {
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: "Live face quality is too low. Move closer, improve lighting, and keep the full face inside the frame before retrying.",
      employee,
      attendance,
      matchConfidence,
      movementDirection: input.movementDirection,
      movementConfidence: input.movementConfidence,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
      liveFaceProfile,
    });
  }

  if (storedFaceProfile.legacy) {
    matchConfidence = calculateLegacyMatchConfidence(
      liveFaceProfile.primaryDescriptor,
      storedFaceProfile.primaryDescriptor,
    );
    faceMatched = matchConfidence >= LEGACY_FACE_MATCH_THRESHOLD;
    matchDetails = {
      primaryConfidence: matchConfidence,
      anchorAverage: matchConfidence,
      peakAnchorConfidence: matchConfidence,
      strongAnchorRatio: matchConfidence >= LEGACY_FACE_MATCH_THRESHOLD ? 1 : 0,
      liveConsistency: roundMetric(liveFaceProfile.consistency),
    };
  } else {
    const metrics = calculateProfileMatchMetrics(liveFaceProfile, storedFaceProfile);
    matchConfidence = metrics.finalConfidence;
    matchDetails = {
      primaryConfidence: metrics.primaryConfidence,
      anchorAverage: metrics.anchorAverage,
      peakAnchorConfidence: metrics.peakAnchorConfidence,
      strongAnchorRatio: metrics.strongAnchorRatio,
      liveConsistency: roundMetric(liveFaceProfile.consistency),
      poseConfidence: metrics.poseConfidence,
      liveLiveness: liveFaceProfile.liveConfidence ?? undefined,
      liveRealness: liveFaceProfile.realConfidence ?? undefined,
    };

    if (storedFaceProfile.engine === "human" && liveFaceProfile.engine === "human") {
      faceMatched =
        metrics.finalConfidence >= HUMAN_FACE_MATCH_THRESHOLD
        && metrics.primaryConfidence >= HUMAN_FACE_PRIMARY_THRESHOLD
        && metrics.anchorAverage >= HUMAN_FACE_ANCHOR_AVG_THRESHOLD
        && metrics.strongAnchorRatio >= HUMAN_FACE_ANCHOR_RATIO_THRESHOLD
        && metrics.poseConfidence >= HUMAN_FACE_POSE_THRESHOLD
        && liveFaceProfile.consistency >= HUMAN_FACE_SCAN_CONSISTENCY_THRESHOLD;
    } else {
      const securePair = storedFaceProfile.secureCapture && liveFaceProfile.secureCapture;
      faceMatched = securePair
        ? (
            metrics.finalConfidence >= FACE_MATCH_THRESHOLD
            && metrics.primaryConfidence >= FACE_PRIMARY_THRESHOLD
            && metrics.anchorAverage >= FACE_ANCHOR_AVG_THRESHOLD
            && metrics.strongAnchorRatio >= FACE_ANCHOR_RATIO_THRESHOLD
            && liveFaceProfile.consistency >= FACE_SCAN_CONSISTENCY_THRESHOLD
          )
        : (
            metrics.finalConfidence >= INSECURE_FACE_MATCH_THRESHOLD
            && metrics.primaryConfidence >= INSECURE_FACE_PRIMARY_THRESHOLD
            && metrics.anchorAverage >= INSECURE_FACE_ANCHOR_AVG_THRESHOLD
            && metrics.strongAnchorRatio >= INSECURE_FACE_ANCHOR_RATIO_THRESHOLD
            && liveFaceProfile.consistency >= INSECURE_FACE_SCAN_CONSISTENCY_THRESHOLD
          );
    }
  }

  if (!faceMatched) {
    if (matchDetails) {
      console.warn(
        `[Biometrics] Face rejected for ${employee.employeeCode} (${employee.name}) on ${input.deviceId}: `
        + `final=${matchConfidence.toFixed(4)} `
        + `primary=${matchDetails.primaryConfidence.toFixed(4)} `
        + `anchors=${matchDetails.anchorAverage.toFixed(4)} `
        + `peak=${matchDetails.peakAnchorConfidence.toFixed(4)} `
        + `ratio=${matchDetails.strongAnchorRatio.toFixed(4)} `
        + `consistency=${matchDetails.liveConsistency.toFixed(4)} `
        + `pose=${matchDetails.poseConfidence?.toFixed(4) ?? "--"} `
        + `liveQuality=${liveFaceProfile.averageQuality.toFixed(4)} `
        + `storedQuality=${storedFaceProfile.averageQuality.toFixed(4)} `
        + `liveness=${liveFaceProfile.liveConfidence?.toFixed(4) ?? "--"} `
        + `real=${liveFaceProfile.realConfidence?.toFixed(4) ?? "--"}`,
      );
    }

    const result = await resolveAttendanceDecision(
      employee,
      now,
      todayDate,
      input.deviceId,
      matchConfidence,
      correlation,
      true,
      false,
      null,
      input.movementDirection,
      input.movementConfidence,
    );

    return logAndReturn({
      ...result,
      message: storedFaceProfile?.legacy
        ? "Face verification failed. Access denied. Re-enroll this employee for stricter biometric matching."
        : result.message,
      matchDetails,
      movementDirection: input.movementDirection,
      movementConfidence: input.movementConfidence,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: result.action ?? (result.ignored ? "UNKNOWN" : "REJECTED"),
      liveFaceProfile,
    });
  }

  const result = await resolveAttendanceDecision(
    employee,
    now,
    todayDate,
    input.deviceId,
    matchConfidence,
    correlation,
    true,
    true,
    employee.rfidUid,
    input.movementDirection,
    input.movementConfidence,
  );

  return logAndReturn({
    ...result,
    matchDetails,
  }, {
    verificationStatus: result.attendance?.verificationStatus ?? (result.success ? "ENTRY" : "FAILED_DIRECTION"),
    decision: result.action ?? (result.success ? "UNKNOWN" : "REJECTED"),
    liveFaceProfile,
  });
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  void warmPythonFaceWorker().catch((error) => {
    console.warn("[python-face] Warm-up skipped:", error);
  });
  void warmRfidService().catch((error) => {
    console.warn("[rfid-service] Warm-up skipped:", error);
  });
  registerRfidProxyRoutes(app);
  const timeoutExitSweep = setInterval(() => {
    void processTimedOutGateSessions().catch((error) => {
      console.error("[gate-timeout] Sweep failed:", error);
    });
  }, SESSION_TIMEOUT_SWEEP_MS);

  httpServer.once("close", () => {
    clearInterval(timeoutExitSweep);
    void stopManagedRfidService();
  });
  
  async function validateEmployeeIdentityConflicts(
    input: { employeeCode?: string; rfidUid?: string },
    currentEmployeeId?: number,
  ) {
    const employees = await storage.getEmployees();

    if (input.employeeCode) {
      const existingByCode = employees.find((employee) => {
        return employee.employeeCode === input.employeeCode
          && employee.id !== currentEmployeeId;
      });

      if (existingByCode) {
        return {
          field: "employeeCode",
          message: `Employee code already belongs to ${existingByCode.name}.`,
        };
      }
    }

    if (input.rfidUid) {
      const normalizedRfidUid = input.rfidUid.trim().toUpperCase();
      const existingByRfid = employees.find((employee) => {
        return employee.rfidUid.toUpperCase() === normalizedRfidUid
          && employee.id !== currentEmployeeId;
      });

      if (existingByRfid) {
        return {
          field: "rfidUid",
          message: `RFID badge already mapped to ${existingByRfid.name}.`,
        };
      }
    }

    return null;
  }

  // Employees API
  app.get(api.employees.list.path, async (req, res) => {
    const employees = await storage.getEmployees();
    res.json(employees);
  });

  app.get(api.employees.get.path, async (req, res) => {
    const employee = await storage.getEmployee(Number(req.params.id));
    if (!employee) {
      return res.status(404).json({ message: "Employee not found" });
    }
    res.json(employee);
  });

  app.get("/api/employees/:id/photo", async (req, res) => {
    const employeeId = Number(req.params.id);
    if (!Number.isFinite(employeeId)) {
      return res.status(400).json({ message: "Invalid employee id" });
    }

    const employee = await storage.getEmployee(employeeId);
    if (!employee) {
      return res.status(404).json({ message: "Employee not found" });
    }

    const pythonMeta = readPythonFaceDescriptorMeta(employee.faceDescriptor);
    const folderName = pythonMeta?.folderName || employee.employeeCode;
    if (!folderName) {
      return res.status(404).json({ message: "No face dataset available for this employee." });
    }

    const datasetDir = path.join(PYTHON_DATASET_ROOT, folderName);
    try {
      const entries = await fs.readdir(datasetDir, { withFileTypes: true });
      const profile = entries.find((entry) => entry.isFile() && /^profile\.(jpe?g|png)$/i.test(entry.name));
      const jpgs = entries
        .filter((entry) => entry.isFile() && /\.(jpe?g|png)$/i.test(entry.name))
        .map((entry) => entry.name)
        .sort();
      const sampleFile = profile?.name ?? jpgs[0];
      if (!sampleFile) {
        return res.status(404).json({ message: "No dataset images found for this employee." });
      }

      const filePath = path.join(datasetDir, sampleFile);
      res.setHeader("Cache-Control", "public, max-age=3600");
      return res.sendFile(filePath);
    } catch (error) {
      console.warn("[employee-photo] Unable to read dataset folder:", error);
      return res.status(404).json({ message: "Employee photo unavailable." });
    }
  });

  app.get("/api/employees/:id/photo/meta", async (req, res) => {
    const employeeId = Number(req.params.id);
    if (!Number.isFinite(employeeId)) {
      return res.status(400).json({ message: "Invalid employee id" });
    }

    const employee = await storage.getEmployee(employeeId);
    if (!employee) {
      return res.status(404).json({ message: "Employee not found" });
    }

    const pythonMeta = readPythonFaceDescriptorMeta(employee.faceDescriptor);
    const folderName = pythonMeta?.folderName || employee.employeeCode;
    if (!folderName) {
      return res.json({ hasProfilePhoto: false });
    }

    const datasetDir = path.join(PYTHON_DATASET_ROOT, folderName);
    try {
      const entries = await fs.readdir(datasetDir, { withFileTypes: true });
      const profile = entries.find((entry) => entry.isFile() && /^profile\.(jpe?g|png)$/i.test(entry.name));
      return res.json({ hasProfilePhoto: Boolean(profile) });
    } catch (error) {
      console.warn("[employee-photo-meta] Unable to read dataset folder:", error);
      return res.json({ hasProfilePhoto: false });
    }
  });

  app.post(api.employees.enrollPython.path, async (req, res) => {
    try {
      const input = api.employees.enrollPython.input.parse(req.body);
      input.rfidUid = input.rfidUid.trim().toUpperCase();

      const conflict = await validateEmployeeIdentityConflicts(input);
      if (conflict) {
        return res.status(400).json(conflict);
      }

      const dataset = await saveEmployeeDatasetPhotos({
        folderName: input.employeeCode,
        datasetPhotos: input.datasetPhotos,
      });
      if (input.profilePhoto) {
        const profilePath = path.join(dataset.directory, "profile.jpg");
        await fs.writeFile(profilePath, parseDataUrl(input.profilePhoto));
      }
      const employee = await storage.createEmployee({
        employeeCode: input.employeeCode,
        name: input.name,
        department: input.department,
        phone: input.phone,
        email: input.email,
        rfidUid: input.rfidUid,
        isActive: input.isActive ?? true,
        faceDescriptor: buildPythonFaceDescriptorMeta({
          folderName: dataset.folderName,
          datasetSampleCount: dataset.datasetSampleCount,
          status: "training",
          trainedAt: null,
          lastTrainingMessage: "Python training is running.",
        }),
      });

      try {
        await retrainAndSyncPythonFaceModel();
      } catch (error) {
        await storage.updateEmployee(employee.id, {
          faceDescriptor: buildPythonFaceDescriptorMeta({
            folderName: dataset.folderName,
            datasetSampleCount: dataset.datasetSampleCount,
            status: "failed",
            trainedAt: null,
            lastTrainingMessage:
              error instanceof Error
                ? error.message
                : "Python training failed unexpectedly.",
          }),
        });
      }

      const latestEmployee = await storage.getEmployee(employee.id);
      res.status(201).json(latestEmployee ?? employee);
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join(".") });
      }
      res.status(500).json({ message: err instanceof Error ? err.message : "Internal server error" });
    }
  });

  app.post(api.employees.create.path, async (req, res) => {
    try {
      const input = api.employees.create.input.parse(req.body);
      input.rfidUid = input.rfidUid.trim().toUpperCase();

      const conflict = await validateEmployeeIdentityConflicts(input);
      if (conflict) {
        return res.status(400).json(conflict);
      }

      const employee = await storage.createEmployee(input);
      res.status(201).json(employee);
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      }
      res.status(500).json({ message: "Internal server error" });
    }
  });

  app.patch(api.employees.update.path, async (req, res) => {
    try {
      const input = api.employees.update.input.parse(req.body);
      const { profilePhoto, ...updatePayload } = input;
      if (updatePayload.rfidUid) {
        updatePayload.rfidUid = updatePayload.rfidUid.trim().toUpperCase();
      }

      const employeeId = Number(req.params.id);
      const conflict = await validateEmployeeIdentityConflicts(updatePayload, employeeId);
      if (conflict) {
        return res.status(400).json(conflict);
      }

      const employee = await storage.updateEmployee(employeeId, updatePayload);
      if (!employee) {
        return res.status(404).json({ message: "Employee not found" });
      }

      if (profilePhoto) {
        const pythonMeta = readPythonFaceDescriptorMeta(employee.faceDescriptor);
        const folderName = pythonMeta?.folderName || employee.employeeCode;
        if (!folderName) {
          return res.status(400).json({ message: "No Python dataset exists for this employee. Re-enroll before adding a profile photo." });
        }
        const datasetDir = path.join(PYTHON_DATASET_ROOT, folderName);
        await fs.mkdir(datasetDir, { recursive: true });
        const profilePath = path.join(datasetDir, "profile.jpg");
        await fs.writeFile(profilePath, parseDataUrl(profilePhoto));
      }

      res.json(employee);
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      }
      res.status(500).json({ message: "Internal server error" });
    }
  });

  app.delete(api.employees.delete.path, async (req, res) => {
    try {
      const employeeId = Number(req.params.id);
      const deletedEmployee = await storage.deleteEmployee(employeeId);

      if (!deletedEmployee) {
        return res.status(404).json({ message: "Employee not found" });
      }

      const pythonMeta = readPythonFaceDescriptorMeta(deletedEmployee.faceDescriptor);
      if (pythonMeta) {
        await removeEmployeeDataset(pythonMeta.folderName);
        try {
          await retrainAndSyncPythonFaceModel();
        } catch (error) {
          console.error("[python-face] Retraining after delete failed:", error);
        }
      }

      res.json(deletedEmployee);
    } catch (err) {
      res.status(500).json({ message: "Internal server error" });
    }
  });

  // Attendances API
  app.get(api.attendances.list.path, async (req, res) => {
    try {
      const parsedInput = api.attendances.list.input?.safeParse(req.query);
      if (parsedInput && !parsedInput.success) {
        return res.status(400).json({
          message: parsedInput.error.errors[0]?.message ?? "Invalid attendance filter.",
          field: parsedInput.error.errors[0]?.path.join("."),
        });
      }

      const attendances = await storage.getAttendances(parsedInput?.data);
      res.json(attendances);
    } catch (err) {
      res.status(500).json({ message: "Internal server error" });
    }
  });

  app.get(api.gateEvents.list.path, async (req, res) => {
    try {
      const parsedInput = api.gateEvents.list.input?.safeParse(req.query);
      if (parsedInput && !parsedInput.success) {
        return res.status(400).json({
          message: parsedInput.error.errors[0]?.message ?? "Invalid gate event filter.",
          field: parsedInput.error.errors[0]?.path.join("."),
        });
      }

      const gateEventRows = await storage.getGateEvents(parsedInput?.data);
      res.json(gateEventRows);
    } catch (err) {
      res.status(500).json({ message: "Internal server error" });
    }
  });

  // Stats API
  app.get(api.stats.dashboard.path, async (req, res) => {
    const stats = await storage.getDashboardStats();
    res.json(stats);
  });

  app.post(api.scan.liveFaces.path, async (req, res) => {
    try {
      const input = api.scan.liveFaces.input.parse(req.body);
      const processedAt = new Date().toISOString();

      try {
        await ensurePythonFaceModelExists();
      } catch {
        return res.json({
          success: false,
          message: "Python face model is not trained yet. Enroll employees and refresh Python training first.",
          processedAt,
          faces: [],
        });
      }

      const recognition = await recognizeLiveFrameWithPython(input.frame, input.maxFaces ?? 50);
      const faces = recognition.faces.map((face) => ({
        label: face.verified ? face.label : "Unknown Face",
        employeeCode: face.verified ? face.employeeCode ?? undefined : undefined,
        department: face.verified ? face.department ?? undefined : undefined,
        rfidUid: face.verified ? face.rfidUid ?? undefined : undefined,
        confidence: face.confidence,
        distance: face.distance,
        verified: face.verified,
        box: face.box,
      }));
      const matchedCount = faces.filter((face) => face.verified).length;
      const message = !faces.length
        ? "No faces detected in the live camera frame."
        : matchedCount
          ? `Live recognition active for ${matchedCount} visible employee${matchedCount === 1 ? "" : "s"}.`
          : "Faces detected, but none matched the trained employee roster.";

      return res.json({
        success: true,
        message,
        processedAt,
        frameWidth: recognition.frameWidth,
        frameHeight: recognition.frameHeight,
        faces,
      });
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      }

      console.error("Live recognition error:", err);
      return res.json({
        success: false,
        message: err instanceof Error ? err.message : "Live recognition failed unexpectedly.",
        processedAt: new Date().toISOString(),
        faces: [],
      });
    }
  });

  app.post(api.scan.cameraFace.path, async (req, res) => {
    try {
      const input = api.scan.cameraFace.input.parse(req.body);
      const processedAt = new Date().toISOString();

      try {
        await ensurePythonFaceModelExists();
      } catch {
        return res.json({
          success: false,
          message: "Python face model is not trained yet. Enroll employees and refresh Python training first.",
          processedAt,
          rfidTimestamp: input.timestamp,
          timestampDeltaMs: null,
          status: "NO_FACE" as const,
          facesDetected: 0,
          multipleFaces: false,
          frameCount: 0,
          frameLatencyMs: null,
          bestBox: null,
        });
      }

      const recognition = await recognizeRfidTriggeredFaceWithPython({
        rfidTag: input.rfidTag,
        timestamp: input.timestamp,
        frameCount: input.frameCount ?? 1,
        maxFaces: input.maxFaces ?? 2,
        freshnessMs: input.freshnessMs ?? 900,
        captureSpacingMs: input.captureSpacingMs ?? 80,
      });

      const message = recognition.status === "MATCH"
        ? `Live face match captured for ${recognition.name ?? "employee"} after RFID trigger.`
        : recognition.status === "UNKNOWN"
          ? recognition.multipleFaces
            ? "Multiple faces were visible after the RFID trigger, and the best live match stayed unknown."
            : "A live face was captured after the RFID trigger, but confidence stayed below the known-employee threshold."
          : "No face was detected in the live camera window after the RFID trigger.";

      return res.json({
        success: recognition.status === "MATCH",
        message,
        processedAt,
        name: recognition.name ?? null,
        confidence: recognition.confidence,
        timestamp: recognition.timestamp,
        rfidTimestamp: recognition.rfidTimestamp,
        timestampDeltaMs: recognition.timestampDeltaMs,
        status: recognition.status,
        employeeCode: recognition.employeeCode ?? null,
        department: recognition.department ?? null,
        rfidUid: recognition.rfidUid ?? null,
        facesDetected: recognition.facesDetected,
        multipleFaces: recognition.multipleFaces,
        frameCount: recognition.frameCount,
        frameLatencyMs: recognition.frameLatencyMs,
        bestBox: recognition.bestBox ?? null,
        frameWidth: recognition.frameWidth,
        frameHeight: recognition.frameHeight,
      });
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      }

      console.error("Triggered live camera recognition error:", err);
      return res.json({
        success: false,
        message: err instanceof Error ? err.message : "Triggered live camera recognition failed unexpectedly.",
        processedAt: new Date().toISOString(),
        status: "NO_FACE" as const,
        facesDetected: 0,
        multipleFaces: false,
        frameCount: 0,
        frameLatencyMs: null,
        bestBox: null,
      });
    }
  });

  app.post("/api/rfid-event", async (req, res) => {
    try {
      const input = z.object({
        rfidUid: z.string().trim().min(1),
        deviceId: z.string().trim().min(1),
        scanTechnology: scanTechnologySchema.optional(),
      }).parse(req.body);

      const normalizedRfidUid = input.rfidUid.trim().toUpperCase();
      const detection = gateMatchingEngine.recordRfidDetection({
        deviceId: input.deviceId,
        rfidUid: normalizedRfidUid,
        scanTechnology: input.scanTechnology ?? "UHF_RFID",
        source: "reader_detected",
      });
      const fusion = await decisionEngine.ingestRfid({
        deviceId: input.deviceId,
        rfidTag: normalizedRfidUid,
        scanTechnology: input.scanTechnology ?? "UHF_RFID",
      });
      const employee = await storage.getEmployeeByRfid(normalizedRfidUid);

      return res.json({
        success: true,
        rfidUid: normalizedRfidUid,
        employee,
        session: detection.session,
        fusionResolved: fusion.resolved,
      });
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({
          message: err.errors[0]?.message ?? "Invalid RFID event payload.",
          field: err.errors[0]?.path.join("."),
        });
      }

      return res.status(500).json({ message: "Unable to process RFID event." });
    }
  });

  app.post("/api/integration/rfid", async (req, res) => {
    try {
      const result = await handleRfidIntegration(req.body);
      return res.json({ success: true, ...result });
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({
          message: err.errors[0]?.message ?? "Invalid RFID integration payload.",
          field: err.errors[0]?.path.join("."),
        });
      }

      console.error("[integration-rfid] Failed:", err);
      return res.status(500).json({ message: "Unable to process RFID integration event." });
    }
  });

  app.post("/api/integration/vision", async (req, res) => {
    try {
      const result = await handleVisionIntegration(req.body);
      return res.json({ success: true, ...result });
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({
          message: err.errors[0]?.message ?? "Invalid vision integration payload.",
          field: err.errors[0]?.path.join("."),
        });
      }

      console.error("[integration-vision] Failed:", err);
      return res.status(500).json({ message: "Unable to process vision integration event." });
    }
  });

  app.post("/api/integration/face", async (req, res) => {
    try {
      const result = await handleFaceIntegration(req.body);
      return res.json({ success: true, ...result });
    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({
          message: err.errors[0]?.message ?? "Invalid face integration payload.",
          field: err.errors[0]?.path.join("."),
        });
      }

      console.error("[integration-face] Failed:", err);
      return res.status(500).json({ message: "Unable to process face integration event." });
    }
  });

  app.get("/api/integration/status", (_req, res) => {
    return res.json(decisionEngine.getStatus());
  });
  // RFID Scan Endpoint (Core Logic)
  app.post(api.scan.rfid.path, async (req, res) => {
    try {
      const input = api.scan.rfid.input.parse(req.body);
      input.rfidUid = input.rfidUid.trim().toUpperCase();
      const result = await processRfidScan(input);
      return res.json(result);

    } catch (err) {
      if (err instanceof z.ZodError) {
        return res.status(400).json({ message: err.errors[0].message, field: err.errors[0].path.join('.') });
      }
      console.error("Scan error:", err);
      res.status(500).json({ message: "Internal server error" });
    }
  });

  return httpServer;
}


