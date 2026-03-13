import type { Express } from "express";
import type { IncomingMessage, Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { promises as fs } from "fs";
import { storage } from "./storage";
import { allowInsecureFaceFallback } from "./env";
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
} from "@shared/schema";
import { z } from "zod";
import {
  buildPythonFaceDescriptorMeta,
  PYTHON_LBPH_LABELS_PATH,
  PYTHON_LBPH_MODEL_PATH,
  readPythonFaceDescriptorMeta,
  recognizeLiveFrameWithPython,
  removeEmployeeDataset,
  retrainPythonFaceModel,
  appendEmployeeDatasetFrames,
  saveEmployeeDatasetPhotos,
  verifyGateFramesWithPython,
} from "./python-face";

type ClientType = "browser" | "device";

interface ClientConnection {
  ws: WebSocket;
  deviceId: string;
  clientType: ClientType;
}

const activeConnections = new Set<ClientConnection>();
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
  return scanTechnology ?? "HF_RFID";
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

async function resolveAttendanceDecision(
  employee: Employee,
  now: Date,
  todayDate: string,
  deviceId: string,
  matchConfidence: number,
  movementDirection?: MovementDirection,
  movementConfidence?: number,
): Promise<ProcessedScanResult> {
  const openEntry = await storage.getOpenAttendance(employee.id, todayDate);
  const hasDirectionSignal = movementDirection !== undefined;
  const directionIsConfident =
    (movementDirection === "ENTRY" || movementDirection === "EXIT")
    && (movementConfidence ?? 0) >= DIRECTION_CONFIDENCE_THRESHOLD;

  if (hasDirectionSignal && directionIsConfident) {
    if (movementDirection === "ENTRY") {
      if (openEntry) {
        const attendance = await createFailureAttendance(
          employee.id,
          todayDate,
          now,
          deviceId,
          "FAILED_DIRECTION",
        );

        return {
          success: false,
          message: "An entry is already open. Show the exit-side face before marking another scan.",
          employee,
          attendance,
          matchConfidence,
          movementDirection,
          movementConfidence,
        };
      }

      const attendance = await createEntryAttendance(employee.id, todayDate, now, deviceId);

      return {
        success: true,
        message: "Entry marked successfully.",
        employee,
        attendance,
        matchConfidence,
        action: "ENTRY",
        movementDirection,
        movementConfidence,
      };
    }

    if (!openEntry) {
      const attendance = await createFailureAttendance(
        employee.id,
        todayDate,
        now,
        deviceId,
        "FAILED_DIRECTION",
      );

      return {
        success: false,
        message: "No active entry was found. Mark an entry first while the correct entry-side face is visible.",
        employee,
        attendance,
        matchConfidence,
        movementDirection,
        movementConfidence,
      };
    }

    const attendance = await closeOpenAttendance(openEntry, now);
    if (!attendance) {
      return {
        success: false,
        message: "Exit could not be recorded. Please retry the scan.",
        employee,
        matchConfidence,
        movementDirection,
        movementConfidence,
      };
    }

    return {
      success: true,
      message: "Exit marked successfully.",
      employee,
      attendance,
      matchConfidence,
      action: "EXIT",
      movementDirection,
      movementConfidence,
    };
  }

  if (openEntry) {
    const attendance = await closeOpenAttendance(openEntry, now);
    if (!attendance) {
      return {
        success: false,
        message: "Exit could not be recorded. Please retry the scan.",
        employee,
        matchConfidence,
      };
    }

    return {
      success: true,
      message: "Exit marked successfully.",
      employee,
      attendance,
      matchConfidence,
      action: "EXIT",
    };
  }

  const attendance = await createEntryAttendance(employee.id, todayDate, now, deviceId);

  return {
    success: true,
    message: "Entry marked successfully.",
    employee,
    attendance,
    matchConfidence,
    action: "ENTRY",
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
  logAndReturn: LogAndReturnFn;
}) {
  const { input, employee, now, todayDate, logAndReturn } = args;

  if (!input.faceFrames?.length) {
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: "Live gate frames were not captured. Retry the scan.",
      employee,
      attendance,
      matchConfidence: 0,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
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

    if (!verification.verified || !verification.employee) {
      const attendance = await createFailureAttendance(
        employee.id,
        todayDate,
        now,
        input.deviceId,
        "FAILED_FACE",
      );

      return logAndReturn({
        success: false,
        message: verification.framesWithFace
          ? "Python face verification failed. Access denied."
          : "No face was detected clearly in the gate camera frames. Retry the scan.",
        employee,
        attendance,
        matchConfidence,
        matchDetails,
        movementDirection: verification.movementDirection,
        movementConfidence: verification.movementConfidence,
        detectedFaceLabel,
        detectedFaceBox,
      }, {
        verificationStatus: "FAILED_FACE",
        decision: "REJECTED",
      });
    }

    const matchedEmployeeCode = verification.employee.employeeCode?.trim();
    const matchedRfidUid = verification.employee.rfidUid?.trim().toUpperCase();
    const matchesBadgeOwner = Boolean(
      (matchedEmployeeCode && matchedEmployeeCode === employee.employeeCode)
      || (matchedRfidUid && matchedRfidUid === employee.rfidUid.toUpperCase()),
    );

    if (!matchesBadgeOwner) {
      const attendance = await createFailureAttendance(
        employee.id,
        todayDate,
        now,
        input.deviceId,
        "FAILED_FACE",
      );

      return logAndReturn({
        success: false,
        message: verification.employee.displayName
          ? `ID-face mismatch. Badge ${employee.rfidUid} belongs to ${employee.name}, but Python verification matched ${verification.employee.displayName}.`
          : "Python face verification matched an unknown roster profile. Re-enroll the employee dataset and retry.",
        employee,
        attendance,
        matchConfidence,
        matchDetails,
        movementDirection: verification.movementDirection,
        movementConfidence: verification.movementConfidence,
        detectedFaceLabel,
        detectedFaceBox,
      }, {
        verificationStatus: "FAILED_FACE",
        decision: "REJECTED",
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

async function processRfidScan(input: ProcessScanInput): Promise<ProcessedScanResult> {
  const normalizedRfidUid = input.rfidUid.trim().toUpperCase();
  input.rfidUid = normalizedRfidUid;
  const now = new Date();
  const todayDate = now.toISOString().split("T")[0];
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
    return logAndReturn({
      success: false,
      message: "Unknown RFID card rejected.",
    }, {
      verificationStatus: "UNKNOWN_RFID",
      decision: "REJECTED",
    });
  }

  if (input.faceFrames?.length) {
    return processPythonRfidScan({
      input,
      employee,
      now,
      todayDate,
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
    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: "Live face verification data was not captured. Retry the scan.",
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

    const attendance = await createFailureAttendance(
      employee.id,
      todayDate,
      now,
      input.deviceId,
      "FAILED_FACE",
    );

    return logAndReturn({
      success: false,
      message: storedFaceProfile?.legacy
        ? "Face verification failed. Access denied. Re-enroll this employee for stricter biometric matching."
        : "Face verification failed. Access denied.",
      employee,
      attendance,
      matchConfidence,
      matchDetails,
      movementDirection: input.movementDirection,
      movementConfidence: input.movementConfidence,
    }, {
      verificationStatus: "FAILED_FACE",
      decision: "REJECTED",
      liveFaceProfile,
    });
  }

  const result = await resolveAttendanceDecision(
    employee,
    now,
    todayDate,
    input.deviceId,
    matchConfidence,
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

function toSocketScanResult(result: ProcessedScanResult, rfidUid: string) {
  return {
    type: "scan_result",
    success: result.success,
    message: result.message,
    employee: result.employee
      ? { id: result.employee.id, name: result.employee.name }
      : undefined,
    action: result.action,
    matchConfidence: result.matchConfidence,
    matchDetails: result.matchDetails,
    movementDirection: result.movementDirection,
    movementConfidence: result.movementConfidence,
    detectedFaceLabel: result.detectedFaceLabel,
    detectedFaceBox: result.detectedFaceBox,
    rfidUid,
  };
}

function getConnectionIp(req: IncomingMessage) {
  const forwardedFor = req.headers["x-forwarded-for"];
  const forwardedIp = Array.isArray(forwardedFor)
    ? forwardedFor[0]
    : forwardedFor?.split(",")[0];

  return (forwardedIp || req.socket.remoteAddress || "unknown")
    .replace(/^::ffff:/, "");
}

function broadcastMessage(
  payload: Record<string, unknown>,
  predicate?: (connection: ClientConnection) => boolean,
) {
  const message = JSON.stringify(payload);

  activeConnections.forEach((connection) => {
    if (connection.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    if (predicate && !predicate(connection)) {
      return;
    }

    connection.ws.send(message);
  });
}

function replaceExistingConnection(deviceId: string, clientType: ClientType) {
  activeConnections.forEach((connection) => {
    if (connection.deviceId !== deviceId || connection.clientType !== clientType) {
      return;
    }

    activeConnections.delete(connection);

    if (
      connection.ws.readyState === WebSocket.OPEN
      || connection.ws.readyState === WebSocket.CONNECTING
    ) {
      connection.ws.close(4002, "Replaced by newer connection");
    }
  });
}

function sendDevicePresence(ws: WebSocket, deviceId: string, online: boolean) {
  ws.send(JSON.stringify({
    type: "device_presence",
    deviceId,
    online,
  }));
}

function broadcastDevicePresence(deviceId: string, online: boolean) {
  broadcastMessage(
    {
      type: "device_presence",
      deviceId,
      online,
    },
    (client) => client.clientType === "browser",
  );
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  // Setup WebSocket server for real ESP8266 device communication
  const wss = new WebSocketServer({
    server: httpServer,
    path: "/ws/device",
    perMessageDeflate: false,
  });

  wss.on("error", (error) => {
    console.error("[WebSocket] Server startup error:", error);
  });
  
  wss.on("connection", (ws: WebSocket, req) => {
    req.socket.setNoDelay(true);
    const searchParams = new URL(
      `http://${req.headers.host}${req.url}`,
    ).searchParams;
    const deviceId =
      searchParams.get("deviceId") || `device-${Date.now()}`;
    const clientType = (searchParams.get("clientType") === "device"
      ? "device"
      : "browser") as ClientType;
    const connectionIp = getConnectionIp(req);
    const connection: ClientConnection = { ws, deviceId, clientType };

    replaceExistingConnection(deviceId, clientType);

    if (clientType === "device") {
      console.log(
        `[WebSocket] Reader connected: ${deviceId} from ${connectionIp}. Reader is online and ready to read badges.`,
      );
      broadcastDevicePresence(deviceId, true);
    } else {
      console.log(`[WebSocket] ${clientType} connected: ${deviceId} from ${connectionIp}`);
    }
    activeConnections.add(connection);
    
    // On browser connect, send snapshot of current device presence
    if (clientType === "browser") {
      activeConnections.forEach((existing) => {
        if (existing.clientType === "device") {
          sendDevicePresence(ws, existing.deviceId, true);
        }
      });
    }

    ws.send(JSON.stringify({
      type: "connected",
      deviceId,
      message: "Connected to attendance system",
    }));
    
    ws.on("message", async (data) => {
      try {
        const message = JSON.parse(data.toString());

        if (message.type === "rfid_detected") {
          const rfidUid = String(message.rfidUid || "").trim().toUpperCase();

          if (!rfidUid) {
            ws.send(JSON.stringify({
              type: "error",
              message: "RFID UID is required.",
            }));
            return;
          }

          const mappedEmployee = await storage.getEmployeeByRfid(rfidUid);
          const payload = {
            type: "rfid_detected",
            message: mappedEmployee
              ? `Badge already mapped to ${mappedEmployee.name}.`
              : "Badge detected and ready to assign.",
            rfidUid,
            available: !mappedEmployee,
            employee: mappedEmployee
              ? { id: mappedEmployee.id, name: mappedEmployee.name }
              : undefined,
            deviceId,
          };

          ws.send(JSON.stringify(payload));
          broadcastMessage(payload, (client) => client.clientType === "browser");
          return;
        }
        
        // Handle RFID scan from ESP8266 device
        if (message.type === "rfid_scan") {
          const rfidUid = String(message.rfidUid || "").trim().toUpperCase();
          const result = await processRfidScan({
            rfidUid,
            deviceId,
            faceDescriptor: Array.isArray(message.faceDescriptor)
              ? message.faceDescriptor
              : undefined,
            faceAnchorDescriptors: Array.isArray(message.faceAnchorDescriptors)
              ? message.faceAnchorDescriptors.filter((value: unknown) => {
                  return Array.isArray(value) && value.every((item) => typeof item === "number");
                }) as number[][]
              : undefined,
            faceConsistency:
              typeof message.faceConsistency === "number"
                ? message.faceConsistency
                : undefined,
            faceQuality:
              typeof message.faceQuality === "number"
                ? message.faceQuality
                : undefined,
            facePose:
              message.facePose === "front"
              || message.facePose === "left"
              || message.facePose === "right"
              || message.facePose === "up"
              || message.facePose === "down"
              || message.facePose === "unknown"
                ? message.facePose
                : undefined,
            faceYaw:
              typeof message.faceYaw === "number"
                ? message.faceYaw
                : undefined,
            facePitch:
              typeof message.facePitch === "number"
                ? message.facePitch
                : undefined,
            faceRoll:
              typeof message.faceRoll === "number"
                ? message.faceRoll
                : undefined,
            faceLiveConfidence:
              typeof message.faceLiveConfidence === "number"
                ? message.faceLiveConfidence
                : undefined,
            faceRealConfidence:
              typeof message.faceRealConfidence === "number"
                ? message.faceRealConfidence
                : undefined,
            scanTechnology:
              message.scanTechnology === "HF_RFID" || message.scanTechnology === "UHF_RFID"
                ? message.scanTechnology
                : undefined,
            movementDirection:
              message.movementDirection === "ENTRY"
              || message.movementDirection === "EXIT"
              || message.movementDirection === "UNKNOWN"
                ? message.movementDirection
                : undefined,
            movementAxis:
              message.movementAxis === "horizontal"
              || message.movementAxis === "depth"
              || message.movementAxis === "none"
                ? message.movementAxis
                : undefined,
            movementConfidence:
              typeof message.movementConfidence === "number"
                ? message.movementConfidence
                : undefined,
          });

          ws.send(JSON.stringify(toSocketScanResult(result, rfidUid)));
          return;
        }
      } catch (error) {
        console.error("[WebSocket] Error processing message:", error);
        ws.send(JSON.stringify({ type: "error", message: "Error processing request" }));
      }
    });

    ws.on("error", (error) => {
      console.error(`[WebSocket] ${clientType} socket error for ${deviceId}:`, error);
      activeConnections.delete(connection);
    });
    
    ws.on("close", () => {
      if (clientType === "device") {
        console.log(
          `[WebSocket] Reader disconnected: ${deviceId} from ${connectionIp}. Reader is no longer ready.`,
        );
        broadcastDevicePresence(deviceId, false);
      } else {
        console.log(`[WebSocket] ${clientType} disconnected: ${deviceId} from ${connectionIp}`);
      }
      activeConnections.delete(connection);
    });
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
      if (input.rfidUid) {
        input.rfidUid = input.rfidUid.trim().toUpperCase();
      }

      const employeeId = Number(req.params.id);
      const conflict = await validateEmployeeIdentityConflicts(input, employeeId);
      if (conflict) {
        return res.status(400).json(conflict);
      }

      const employee = await storage.updateEmployee(employeeId, input);
      if (!employee) {
        return res.status(404).json({ message: "Employee not found" });
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


