import { randomUUID } from "crypto";
import { spawn, type ChildProcessWithoutNullStreams } from "child_process";
import { promises as fs } from "fs";
import path from "path";
import type { Employee } from "@shared/schema";

const PYTHON_ML_ROOT = path.resolve(process.cwd(), "python-ml");
const DATASET_ROOT = path.join(PYTHON_ML_ROOT, "dataset");
const OUTPUT_ROOT = path.join(PYTHON_ML_ROOT, "output", "opencv");
const TMP_ROOT = path.join(PYTHON_ML_ROOT, "output", "tmp");
const GENERATED_METADATA_FILE = path.join(PYTHON_ML_ROOT, "metadata.generated.csv");
const TRAIN_SCRIPT = path.join(PYTHON_ML_ROOT, "opencv_lbph_train.py");
const FACE_SERVICE_SCRIPT = path.join(PYTHON_ML_ROOT, "opencv_face_service.py");
export const PYTHON_LBPH_MODEL_PATH = path.join(OUTPUT_ROOT, "lbph-model.yml");
export const PYTHON_LBPH_LABELS_PATH = path.join(OUTPUT_ROOT, "lbph-labels.json");

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

type PythonFaceWorkerRequest = PythonVerifyBurstRequest | PythonRecognizeFrameRequest;

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

function parseDataUrl(dataUrl: string) {
  const match = /^data:image\/[a-zA-Z0-9.+-]+;base64,(.+)$/.exec(dataUrl);
  if (!match) {
    throw new Error("Invalid image payload. Expected a base64 data URL.");
  }

  return Buffer.from(match[1], "base64");
}

async function ensureDirectory(targetPath: string) {
  await fs.mkdir(targetPath, { recursive: true });
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
        reject(new Error("Python face worker timed out."));
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
        "640",
        "--min-face-size",
        "72",
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

    return await this.sendRequest<PythonGateVerificationResult>(request, 5000);
  }

  async recognizeFrame(frame: string, maxFaces = 50) {
    const requestId = randomUUID();
    const request: PythonRecognizeFrameRequest = {
      requestId,
      action: "recognize_frame",
      frame,
      maxFaces,
    };

    return await this.sendRequest<PythonLiveRecognitionResult>(request, 4000);
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

  return await pythonFaceWorker.verifyBurst(faceFrames);
}

export async function recognizeLiveFrameWithPython(frame: string) {
  if (!frame.trim()) {
    throw new Error("Live recognition frame was not captured.");
  }

  return await pythonFaceWorker.recognizeFrame(frame, 50);
}


