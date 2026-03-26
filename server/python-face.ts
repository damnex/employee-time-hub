import { randomUUID } from "crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "child_process";
import { promises as fs } from "fs";
import path from "path";
import type { Employee } from "@shared/schema";
import { loadEnvironment } from "./env";

const PYTHON_ML_ROOT = path.resolve(process.cwd(), "python-ml");
const DATASET_ROOT = path.join(PYTHON_ML_ROOT, "dataset");
const OUTPUT_ROOT = path.join(PYTHON_ML_ROOT, "output", "opencv");
const TMP_ROOT = path.join(PYTHON_ML_ROOT, "output", "tmp");
const GENERATED_METADATA_FILE = path.join(PYTHON_ML_ROOT, "metadata.generated.csv");
const TRAIN_SCRIPT = path.join(PYTHON_ML_ROOT, "opencv_lbph_train.py");
const FACE_SERVICE_SCRIPT = path.join(PYTHON_ML_ROOT, "opencv_face_service.py");
export const PYTHON_LBPH_MODEL_PATH = path.join(OUTPUT_ROOT, "lbph-model.yml");
export const PYTHON_LBPH_LABELS_PATH = path.join(OUTPUT_ROOT, "lbph-labels.json");
export const PYTHON_DATASET_ROOT = DATASET_ROOT;

export type PythonFaceStatus = "training" | "trained" | "failed";

export interface PythonFaceDescriptorMeta {
  provider: "python-opencv-lbph";
  status: PythonFaceStatus;
  folderName: string;
  datasetSampleCount: number;
  trainedAt: string | null;
  lastTrainingMessage: string | null;
}

interface PythonTrainingLabel {
  id: number;
  folderName: string;
  displayName: string;
  employeeCode?: string | null;
  department?: string | null;
  rfidUid?: string | null;
  sampleCount: number;
  includedInTraining: boolean;
}

export interface PythonTrainingSummary {
  labels: PythonTrainingLabel[];
}

interface PythonVerifyEmployee {
  folderName: string;
  displayName: string;
  employeeCode?: string | null;
  department?: string | null;
  rfidUid?: string | null;
  sampleCount?: number | null;
}

export interface PythonGateVerificationResult {
  verified: boolean;
  employee?: PythonVerifyEmployee;
  matchConfidence: number;
  bestDistance: number | null;
  distanceThreshold: number;
  movementDirection: "ENTRY" | "EXIT" | "UNKNOWN";
  movementAxis: "horizontal" | "depth" | "none";
  movementConfidence: number;
  framesProcessed: number;
  framesWithFace: number;
  bestBox?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  } | null;
  previewFrameSize?: {
    width: number;
    height: number;
  } | null;
}

export interface PythonLiveRecognitionFace {
  label: string;
  employeeCode?: string | null;
  department?: string | null;
  rfidUid?: string | null;
  confidence: number;
  distance: number | null;
  verified: boolean;
  box: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
}

export interface PythonLiveRecognitionResult {
  faces: PythonLiveRecognitionFace[];
  frameWidth: number;
  frameHeight: number;
}

export interface PythonTriggeredFaceRecognitionRequest {
  rfidTag: string;
  timestamp: number;
  frameCount?: number;
  maxFaces?: number;
  freshnessMs?: number;
  captureSpacingMs?: number;
}

export interface PythonTriggeredFaceRecognitionResult {
  name?: string | null;
  confidence: number;
  timestamp: number;
  rfidTimestamp: number;
  timestampDeltaMs: number;
  status: "MATCH" | "UNKNOWN" | "NO_FACE";
  employeeCode?: string | null;
  department?: string | null;
  rfidUid?: string | null;
  facesDetected: number;
  multipleFaces: boolean;
  frameCount: number;
  frameLatencyMs: number | null;
  bestBox?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  } | null;
  frameWidth?: number;
  frameHeight?: number;
}

interface PythonVerifyBurstRequest {
  requestId: string;
  action: "verify_burst";
  frames: string[];
  entryHorizontalDirection: "left-to-right" | "right-to-left";
  entryDepthDirection: "approaching" | "receding";
}

interface PythonRecognizeFrameRequest {
  requestId: string;
  action: "recognize_frame";
  frame: string;
  maxFaces?: number;
}

interface PythonRecognizeLiveCameraRequest {
  requestId: string;
  action: "recognize_live_camera";
  rfidTag: string;
  timestamp: number;
  frameCount?: number;
  maxFaces?: number;
  freshnessMs?: number;
  captureSpacingMs?: number;
}

type PythonFaceWorkerRequest =
  | PythonVerifyBurstRequest
  | PythonRecognizeFrameRequest
  | PythonRecognizeLiveCameraRequest;

interface PythonFaceWorkerResponse<T> {
  requestId?: string;
  ok: boolean;
  result?: T;
  error?: string;
}

function slugifyDatasetLabel(value: string) {
  return value
    .trim()
    .replace(/[^a-zA-Z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64) || `employee_${Date.now()}`;
}

export function parseDataUrl(dataUrl: string) {
  const match = /^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$/.exec(dataUrl);
  if (!match) {
    throw new Error("Invalid image payload. Expected a base64 data URL.");
  }

  return Buffer.from(match[1], "base64");
}

async function ensureDirectory(targetPath: string) {
  await fs.mkdir(targetPath, { recursive: true });
}

function getPythonFaceWorkerCameraArgs() {
  loadEnvironment();

  const cameraSource = process.env.PYTHON_FACE_CAMERA_SOURCE?.trim() || "0";
  const cameraWidth = process.env.PYTHON_FACE_CAMERA_WIDTH?.trim() || "640";
  const cameraHeight = process.env.PYTHON_FACE_CAMERA_HEIGHT?.trim() || "480";
  const cameraFps = process.env.PYTHON_FACE_CAMERA_FPS?.trim() || "20";
  const cameraReadyTimeoutMs = process.env.PYTHON_FACE_CAMERA_READY_TIMEOUT_MS?.trim() || "2500";
  const cameraReconnectDelayMs = process.env.PYTHON_FACE_CAMERA_RECONNECT_DELAY_MS?.trim() || "500";
  const frameFreshnessMs = process.env.PYTHON_FACE_FRAME_FRESHNESS_MS?.trim() || "1500";

  return [
    "--camera-source",
    cameraSource,
    "--camera-width",
    cameraWidth,
    "--camera-height",
    cameraHeight,
    "--camera-fps",
    cameraFps,
    "--camera-ready-timeout-ms",
    cameraReadyTimeoutMs,
    "--camera-reconnect-delay-ms",
    cameraReconnectDelayMs,
    "--frame-freshness-ms",
    frameFreshnessMs,
  ];
}

export async function appendEmployeeDatasetFrames(args: { folderName: string; frames: string[] }) {
  const { folderName, frames } = args;
  if (!folderName || !frames.length) {
    return;
  }

  const targetDir = path.join(DATASET_ROOT, folderName);
  await ensureDirectory(targetDir);

  const entries = await fs.readdir(targetDir, { withFileTypes: true });
  const sampleRegex = /^sample-(\d+)\.jpg$/i;
  const nextIndex =
    entries.reduce((max, entry) => {
      const match = sampleRegex.exec(entry.name);
      return match ? Math.max(max, Number(match[1])) : max;
    }, 0) + 1;

  await Promise.all(
    frames.map(async (frame, idx) => {
      const targetFile = path.join(
        targetDir,
        `sample-${String(nextIndex + idx).padStart(3, "0")}.jpg`,
      );
      await fs.writeFile(targetFile, parseDataUrl(frame));
    }),
  );
}

async function runPythonScript(scriptPath: string, args: string[], timeoutMs = 180000) {
  await ensureDirectory(OUTPUT_ROOT);
  await ensureDirectory(TMP_ROOT);

  return await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
    const child = spawn("py", ["-3", scriptPath, ...args], {
      cwd: process.cwd(),
      windowsHide: true,
      env: process.env,
    });

    let stdout = "";
    let stderr = "";
    const timeoutId = setTimeout(() => {
      child.kill();
      reject(new Error(`Python script timed out after ${timeoutMs}ms.`));
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });

    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });

    child.on("error", (error) => {
      clearTimeout(timeoutId);
      reject(error);
    });

    child.on("close", (code) => {
      clearTimeout(timeoutId);
      if (code !== 0) {
        reject(new Error(stderr.trim() || stdout.trim() || `Python exited with code ${code}.`));
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

class PythonFaceWorker {
  private child: ChildProcessWithoutNullStreams | null = null;
  private startPromise: Promise<void> | null = null;
  private stdoutBuffer = "";
  private pending = new Map<string, {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
    timeoutId: ReturnType<typeof setTimeout>;
  }>();

  private handleStdout(chunk: string) {
    this.stdoutBuffer += chunk;

    while (true) {
      const newlineIndex = this.stdoutBuffer.indexOf("\n");
      if (newlineIndex === -1) {
        return;
      }

      const line = this.stdoutBuffer.slice(0, newlineIndex).trim();
      this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1);
      if (!line) {
        continue;
      }

      let payload: PythonFaceWorkerResponse<unknown>;
      try {
        payload = JSON.parse(line) as PythonFaceWorkerResponse<unknown>;
      } catch (error) {
        console.warn("[python-face] Unable to parse worker response:", error, line);
        continue;
      }

      const requestId = payload.requestId;
      if (!requestId) {
        continue;
      }

      const pendingRequest = this.pending.get(requestId);
      if (!pendingRequest) {
        continue;
      }

      clearTimeout(pendingRequest.timeoutId);
      this.pending.delete(requestId);

      if (!payload.ok || typeof payload.result === "undefined") {
        pendingRequest.reject(new Error(payload.error || "Python worker returned an empty result."));
        continue;
      }

      pendingRequest.resolve(payload.result);
    }
  }

  private rejectAllPending(message: string) {
    this.pending.forEach((pendingRequest) => {
      clearTimeout(pendingRequest.timeoutId);
      pendingRequest.reject(new Error(message));
    });
    this.pending.clear();
  }

  private async sendRequest<T>(request: PythonFaceWorkerRequest, timeoutMs = 5000) {
    await this.start();

    if (!this.child) {
      throw new Error("Python face worker is unavailable.");
    }

    return await new Promise<T>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pending.delete(request.requestId);
        // Kill the worker to avoid stuck state and force restart on next call.
        if (this.child) {
          this.child.kill();
          this.child = null;
          this.startPromise = null;
        }
        reject(new Error("Python face worker timed out. Restarting worker."));
      }, timeoutMs);

      this.pending.set(request.requestId, {
        resolve: (value) => resolve(value as T),
        reject,
        timeoutId,
      });

      try {
        this.child?.stdin.write(`${JSON.stringify(request)}\n`);
      } catch (error) {
        clearTimeout(timeoutId);
        this.pending.delete(request.requestId);
        reject(error instanceof Error ? error : new Error("Unable to send request to Python worker."));
      }
    });
  }

  async start() {
    if (this.child && !this.child.killed) {
      return;
    }

    if (this.startPromise) {
      await this.startPromise;
      return;
    }

    this.startPromise = (async () => {
      await ensureDirectory(OUTPUT_ROOT);

      const child = spawn("py", [
        "-3",
        "-u",
        FACE_SERVICE_SCRIPT,
        "--model",
        PYTHON_LBPH_MODEL_PATH,
        "--labels",
        PYTHON_LBPH_LABELS_PATH,
        "--resize-width",
        "0",
        "--scale-factor",
        "1.04",
        "--min-face-size",
        "24",
        "--distance-threshold",
        "120",
        ...getPythonFaceWorkerCameraArgs(),
      ], {
        cwd: process.cwd(),
        windowsHide: true,
        env: process.env,
      });

      this.child = child;
      this.stdoutBuffer = "";

      child.stdout.on("data", (chunk) => {
        this.handleStdout(String(chunk));
      });

      child.stderr.on("data", (chunk) => {
        const message = String(chunk).trim();
        if (message) {
          console.warn("[python-face]", message);
        }
      });

      child.on("error", (error) => {
        this.rejectAllPending(`Python face worker failed: ${error.message}`);
        this.child = null;
        this.startPromise = null;
      });

      child.on("close", (code) => {
        this.rejectAllPending(`Python face worker stopped with code ${code ?? "unknown"}.`);
        this.child = null;
        this.startPromise = null;
      });
    })();

    await this.startPromise;
  }

  async stop() {
    if (!this.child) {
      this.startPromise = null;
      return;
    }

    const child = this.child;
    this.child = null;
    this.startPromise = null;
    this.rejectAllPending("Python face worker restarted.");
    child.kill();
  }

  async restart() {
    await this.stop();
    try {
      await Promise.all([
        fs.access(PYTHON_LBPH_MODEL_PATH),
        fs.access(PYTHON_LBPH_LABELS_PATH),
      ]);
    } catch {
      return;
    }

    await this.start();
  }

  async verifyBurst(faceFrames: string[]) {
    const requestId = randomUUID();
    const request: PythonVerifyBurstRequest = {
      requestId,
      action: "verify_burst",
      frames: faceFrames,
      entryHorizontalDirection: "left-to-right",
      entryDepthDirection: "approaching",
    };

    return await this.sendRequest<PythonGateVerificationResult>(request, 30000);
  }

  async recognizeFrame(frame: string, maxFaces = 50) {
    const requestId = randomUUID();
    const request: PythonRecognizeFrameRequest = {
      requestId,
      action: "recognize_frame",
      frame,
      maxFaces,
    };

    return await this.sendRequest<PythonLiveRecognitionResult>(request, 15000);
  }

  async recognizeLiveCamera(input: PythonTriggeredFaceRecognitionRequest) {
    const requestId = randomUUID();
    const request: PythonRecognizeLiveCameraRequest = {
      requestId,
      action: "recognize_live_camera",
      rfidTag: input.rfidTag.trim().toUpperCase(),
      timestamp: input.timestamp,
      frameCount: input.frameCount,
      maxFaces: input.maxFaces,
      freshnessMs: input.freshnessMs,
      captureSpacingMs: input.captureSpacingMs,
    };

    return await this.sendRequest<PythonTriggeredFaceRecognitionResult>(request, 15000);
  }
}

const pythonFaceWorker = new PythonFaceWorker();

export function readPythonFaceDescriptorMeta(faceDescriptor: unknown): PythonFaceDescriptorMeta | null {
  if (!faceDescriptor || typeof faceDescriptor !== "object") {
    return null;
  }

  const candidate = faceDescriptor as Record<string, unknown>;
  if (candidate.provider !== "python-opencv-lbph") {
    return null;
  }

  const status = candidate.status;
  if (status !== "training" && status !== "trained" && status !== "failed") {
    return null;
  }

  return {
    provider: "python-opencv-lbph",
    status,
    folderName: typeof candidate.folderName === "string" ? candidate.folderName : "",
    datasetSampleCount:
      typeof candidate.datasetSampleCount === "number" && Number.isFinite(candidate.datasetSampleCount)
        ? candidate.datasetSampleCount
        : 0,
    trainedAt: typeof candidate.trainedAt === "string" ? candidate.trainedAt : null,
    lastTrainingMessage: typeof candidate.lastTrainingMessage === "string" ? candidate.lastTrainingMessage : null,
  };
}

export function buildPythonFaceDescriptorMeta(args: {
  folderName: string;
  datasetSampleCount: number;
  status: PythonFaceStatus;
  trainedAt?: string | null;
  lastTrainingMessage?: string | null;
}): PythonFaceDescriptorMeta {
  return {
    provider: "python-opencv-lbph",
    status: args.status,
    folderName: args.folderName,
    datasetSampleCount: args.datasetSampleCount,
    trainedAt: args.trainedAt ?? null,
    lastTrainingMessage: args.lastTrainingMessage ?? null,
  };
}

export async function saveEmployeeDatasetPhotos(args: {
  folderName: string;
  datasetPhotos: string[];
}) {
  const normalizedFolderName = slugifyDatasetLabel(args.folderName);
  const targetDir = path.join(DATASET_ROOT, normalizedFolderName);
  await fs.rm(targetDir, { recursive: true, force: true });
  await ensureDirectory(targetDir);

  for (let index = 0; index < args.datasetPhotos.length; index += 1) {
    const targetFile = path.join(
      targetDir,
      `sample-${String(index + 1).padStart(3, "0")}.jpg`,
    );
    await fs.writeFile(targetFile, parseDataUrl(args.datasetPhotos[index]));
  }

  return {
    folderName: normalizedFolderName,
    datasetSampleCount: args.datasetPhotos.length,
    directory: targetDir,
  };
}

export async function removeEmployeeDataset(folderName: string) {
  if (!folderName) {
    return;
  }

  await fs.rm(path.join(DATASET_ROOT, slugifyDatasetLabel(folderName)), {
    recursive: true,
    force: true,
  });
}

async function writeGeneratedMetadataCsv(employees: Employee[]) {
  await ensureDirectory(PYTHON_ML_ROOT);

  const header = "folder_name,employeeCode,name,department,rfidUid,email,phone,isActive";
  const rows = employees.map((employee) => {
    const existingMeta = readPythonFaceDescriptorMeta(employee.faceDescriptor);
    const folderName = existingMeta?.folderName || slugifyDatasetLabel(employee.employeeCode);
    const values = [
      folderName,
      employee.employeeCode,
      employee.name,
      employee.department,
      employee.rfidUid,
      employee.email ?? "",
      employee.phone ?? "",
      employee.isActive ? "true" : "false",
    ];

    return values
      .map((value) => `"${String(value).replace(/"/g, "\"\"")}"`)
      .join(",");
  });

  await fs.writeFile(
    GENERATED_METADATA_FILE,
    [header, ...rows].join("\n"),
    "utf-8",
  );
}

async function listDatasetFolders() {
  await ensureDirectory(DATASET_ROOT);
  const entries = await fs.readdir(DATASET_ROOT, { withFileTypes: true });
  return entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .sort();
}

async function clearModelArtifacts() {
  await pythonFaceWorker.stop();
  await Promise.all([
    fs.rm(PYTHON_LBPH_MODEL_PATH, { force: true }),
    fs.rm(PYTHON_LBPH_LABELS_PATH, { force: true }),
    fs.rm(path.join(OUTPUT_ROOT, "lbph-training-report.json"), { force: true }),
  ]);
}

export async function retrainPythonFaceModel(employees: Employee[]): Promise<PythonTrainingSummary> {
  const datasetFolders = await listDatasetFolders();
  if (!datasetFolders.length) {
    await clearModelArtifacts();
    return { labels: [] };
  }

  await writeGeneratedMetadataCsv(employees);
  await runPythonScript(TRAIN_SCRIPT, [
    "--dataset",
    DATASET_ROOT,
    "--output",
    OUTPUT_ROOT,
    "--metadata",
    GENERATED_METADATA_FILE,
  ]);
  await pythonFaceWorker.restart();

  const labelsPayload = JSON.parse(await fs.readFile(PYTHON_LBPH_LABELS_PATH, "utf-8")) as {
    labels?: PythonTrainingLabel[];
  };

  return {
    labels: Array.isArray(labelsPayload.labels) ? labelsPayload.labels : [],
  };
}

export async function verifyGateFramesWithPython(faceFrames: string[]) {
  if (!faceFrames.length) {
    throw new Error("Live gate frames were not captured.");
  }
  let attempt = 0;
  let lastError: unknown;
  while (attempt < 2) {
    try {
      return await pythonFaceWorker.verifyBurst(faceFrames);
    } catch (err) {
      lastError = err;
      await pythonFaceWorker.restart();
      attempt += 1;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Python verification failed after retry.");
}

export async function recognizeLiveFrameWithPython(frame: string, maxFaces = 50) {
  if (!frame.trim()) {
    throw new Error("Live recognition frame was not captured.");
  }

  let attempt = 0;
  let lastError: unknown;
  while (attempt < 2) {
    try {
      return await pythonFaceWorker.recognizeFrame(frame, maxFaces);
    } catch (err) {
      lastError = err;
      await pythonFaceWorker.restart();
      attempt += 1;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Python live recognition failed after retry.");
}

export async function recognizeRfidTriggeredFaceWithPython(input: PythonTriggeredFaceRecognitionRequest) {
  const normalizedTag = input.rfidTag.trim().toUpperCase();
  if (!normalizedTag) {
    throw new Error("RFID tag is required for triggered face recognition.");
  }

  if (!Number.isFinite(input.timestamp) || input.timestamp <= 0) {
    throw new Error("RFID trigger timestamp must be a valid Unix millisecond value.");
  }

  let attempt = 0;
  let lastError: unknown;
  while (attempt < 2) {
    try {
      return await pythonFaceWorker.recognizeLiveCamera({
        ...input,
        rfidTag: normalizedTag,
      });
    } catch (err) {
      lastError = err;
      await pythonFaceWorker.restart();
      attempt += 1;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Python camera-triggered recognition failed after retry.");
}

export async function warmPythonFaceWorker() {
  try {
    await Promise.all([
      fs.access(PYTHON_LBPH_MODEL_PATH),
      fs.access(PYTHON_LBPH_LABELS_PATH),
    ]);
  } catch {
    return;
  }

  await pythonFaceWorker.start();
}



