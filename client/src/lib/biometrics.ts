import type { Config, FaceResult } from "@vladmandic/human";
import type {
  FaceCaptureMode,
  FacePose,
  FacePoseCoverage,
  FaceProfile,
  FaceProfileV3,
} from "@shared/schema";

type HumanModule = typeof import("@vladmandic/human");
type HumanInstance = InstanceType<HumanModule["default"]>;

const HUMAN_MODEL_BASE_PATH = "/ml-models/";
const TRACKING_INTERVAL_MS = 220;
const DEFAULT_SAMPLE_DELAY_MS = 110;
const DEFAULT_MIN_SAMPLE_QUALITY = 0.5;
const DEFAULT_MIN_LIVE_CONFIDENCE = 0.45;
const DEFAULT_MIN_REAL_CONFIDENCE = 0.35;
const DEFAULT_MAX_ATTEMPTS_MULTIPLIER = 5;

const RAD_TO_DEG = 180 / Math.PI;

const DEFAULT_POSE_TARGETS: FacePoseCoverage = {
  front: 14,
  left: 11,
  right: 11,
  up: 0,
  down: 0,
  unknown: 0,
};

export interface FaceCropBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface FaceCaptureSample {
  descriptor: number[];
  quality: number;
  pose: FacePose;
  yaw: number;
  pitch: number;
  roll: number;
  liveConfidence: number;
  realConfidence: number;
  distance: number | null;
}

export interface FaceTrackingSnapshot {
  status: "loading" | "ready" | "no-face" | "multiple" | "off-center" | "low-quality" | "unsupported";
  faceCount: number;
  bounds: FaceCropBounds | null;
  quality: number;
  pose: FacePose;
  yaw: number;
  pitch: number;
  roll: number;
  liveConfidence: number;
  realConfidence: number;
  distance: number | null;
  boxScore: number;
  faceScore: number;
  descriptor: number[] | null;
  guidance: string;
}

export interface LiveTrackedFaceDetection {
  bounds: FaceCropBounds;
  quality: number;
  pose: FacePose;
  yaw: number;
  pitch: number;
  roll: number;
  liveConfidence: number;
  realConfidence: number;
  distance: number | null;
  boxScore: number;
  faceScore: number;
}

export interface FaceTemplateCaptureOptions {
  sampleCount: number;
  sampleDelayMs?: number;
  minQuality?: number;
  maxAttempts?: number;
  requireDetector?: boolean;
  poseTargets?: Partial<FacePoseCoverage>;
  minLiveConfidence?: number;
  minRealConfidence?: number;
  onProgress?: (
    acceptedSamples: number,
    attemptCount: number,
    poseCounts: FacePoseCoverage,
    latestPose: FacePose,
  ) => void;
}

export interface FaceTemplateCaptureResult {
  descriptor: number[];
  anchorDescriptors: number[][];
  averageQuality: number;
  acceptedSamples: number;
  attempts: number;
  consistency: number;
  captureMode: FaceCaptureMode;
  profile: FaceProfile;
  poseCounts: FacePoseCoverage;
}

export interface BiometricRuntimeInfo {
  isIOS: boolean;
  isSafari: boolean;
  detectorAvailable: boolean;
  fallbackAllowed: boolean;
  compatibilityMode: boolean;
}

interface AggregatedPoseEmbedding {
  pose: FacePose;
  descriptor: number[];
  sampleCount: number;
  averageQuality: number;
  yaw: number;
  pitch: number;
  roll: number;
  averageLive: number;
  averageReal: number;
}

let humanPromise: Promise<HumanInstance> | null = null;

const HUMAN_CONFIG: Partial<Config> = {
  backend: "webgl",
  modelBasePath: HUMAN_MODEL_BASE_PATH,
  warmup: "face",
  async: true,
  cacheModels: true,
  validateModels: false,
  debug: false,
  skipAllowed: false,
  cacheSensitivity: 0.7,
  filter: {
    enabled: true,
    autoBrightness: true,
    contrast: 0.02,
    sharpness: 0.2,
    return: true,
  },
  gesture: {
    enabled: false,
  },
  face: {
    enabled: true,
    detector: {
      enabled: true,
      modelPath: "blazeface.json",
      rotation: true,
      maxDetected: 1,
      minConfidence: 0.35,
      minSize: 80,
      iouThreshold: 0.2,
      scale: 1.55,
      mask: false,
      return: false,
      skipFrames: 1,
      skipTime: 60,
      square: false,
    },
    mesh: {
      enabled: true,
      modelPath: "facemesh.json",
      skipFrames: 1,
      skipTime: 80,
      keepInvalid: false,
    },
    iris: {
      enabled: true,
      modelPath: "iris.json",
      skipFrames: 1,
      skipTime: 120,
      scale: 2.2,
    },
    description: {
      enabled: true,
      modelPath: "faceres.json",
      skipFrames: 1,
      skipTime: 60,
      minConfidence: 0.2,
    },
    antispoof: {
      enabled: true,
      modelPath: "antispoof.json",
      skipFrames: 1,
      skipTime: 80,
    },
    liveness: {
      enabled: true,
      modelPath: "liveness.json",
      skipFrames: 1,
      skipTime: 80,
    },
    emotion: {
      enabled: false,
      modelPath: "emotion.json",
      minConfidence: 0.1,
      skipFrames: 99,
      skipTime: 1500,
    },
    attention: {
      enabled: false,
      modelPath: "facemesh-attention.json",
      skipFrames: 99,
      skipTime: 1500,
    },
    gear: {
      enabled: false,
      modelPath: "",
      minConfidence: 0.1,
      skipFrames: 99,
      skipTime: 1500,
    },
  },
  body: {
    enabled: false,
    maxDetected: 1,
    minConfidence: 0.2,
    skipFrames: 99,
    skipTime: 1500,
    modelPath: "movenet-lightning.json",
  },
  hand: {
    enabled: false,
    maxDetected: 2,
    minConfidence: 0.2,
    skipFrames: 99,
    skipTime: 1500,
    detector: {
      modelPath: "handtrack.json",
    },
    skeleton: {
      modelPath: "handlandmark-lite.json",
    },
  },
  object: {
    enabled: false,
    maxDetected: 1,
    minConfidence: 0.2,
    skipFrames: 99,
    skipTime: 1500,
    modelPath: "centernet.json",
  },
  segmentation: {
    enabled: false,
    modelPath: "",
  },
};

const TRACKING_CONFIG: Partial<Config> = {
  face: {
    enabled: true,
    detector: {
      enabled: true,
      rotation: true,
      maxDetected: 1,
      minConfidence: 0.42,
      skipFrames: 0,
      skipTime: 0,
    },
    mesh: {
      enabled: true,
      skipFrames: 0,
      skipTime: 0,
      keepInvalid: false,
    },
    iris: {
      enabled: true,
      skipFrames: 0,
      skipTime: 0,
      scale: 2.2,
    },
    description: {
      enabled: false,
      minConfidence: 0.2,
      skipFrames: 0,
      skipTime: 0,
    },
    antispoof: {
      enabled: false,
      skipFrames: 99,
      skipTime: 1500,
    },
    liveness: {
      enabled: false,
      skipFrames: 99,
      skipTime: 1500,
    },
    emotion: {
      enabled: false,
      minConfidence: 0.1,
      skipFrames: 99,
      skipTime: 1500,
    },
    attention: {
      enabled: false,
      skipFrames: 99,
      skipTime: 1500,
    },
    gear: {
      enabled: false,
      minConfidence: 0.1,
      skipFrames: 99,
      skipTime: 1500,
    },
  },
  gesture: {
    enabled: false,
  },
};

const MULTI_FACE_TRACKING_CONFIG: Partial<Config> = {
  face: {
    enabled: true,
    detector: {
      enabled: true,
      rotation: true,
      maxDetected: 50,
      minConfidence: 0.32,
      skipFrames: 1,
      skipTime: 40,
    },
    mesh: {
      enabled: false,
      skipFrames: 0,
      skipTime: 0,
      keepInvalid: false,
    },
    iris: {
      enabled: false,
      skipFrames: 2,
      skipTime: 120,
      scale: 2.2,
    },
    description: {
      enabled: false,
      minConfidence: 0.2,
      skipFrames: 2,
      skipTime: 120,
    },
    antispoof: {
      enabled: false,
      skipFrames: 99,
      skipTime: 1500,
    },
    liveness: {
      enabled: false,
      skipFrames: 99,
      skipTime: 1500,
    },
    emotion: {
      enabled: false,
      minConfidence: 0.1,
      skipFrames: 99,
      skipTime: 1500,
    },
    attention: {
      enabled: false,
      skipFrames: 99,
      skipTime: 1500,
    },
    gear: {
      enabled: false,
      minConfidence: 0.1,
      skipFrames: 99,
      skipTime: 1500,
    },
  },
  gesture: {
    enabled: false,
  },
};

const CAPTURE_CONFIG: Partial<Config> = {
  face: {
    enabled: true,
    detector: {
      enabled: true,
      rotation: true,
      maxDetected: 1,
      minConfidence: 0.35,
      skipFrames: 0,
      skipTime: 0,
    },
    mesh: {
      enabled: true,
      skipFrames: 0,
      skipTime: 0,
      keepInvalid: false,
    },
    iris: {
      enabled: true,
      skipFrames: 0,
      skipTime: 0,
      scale: 2.2,
    },
    description: {
      enabled: true,
      minConfidence: 0.2,
      skipFrames: 0,
      skipTime: 0,
    },
    antispoof: {
      enabled: true,
      skipFrames: 0,
      skipTime: 0,
    },
    liveness: {
      enabled: true,
      skipFrames: 0,
      skipTime: 0,
    },
    emotion: {
      enabled: false,
      minConfidence: 0.1,
      skipFrames: 99,
      skipTime: 1500,
    },
    attention: {
      enabled: false,
      skipFrames: 99,
      skipTime: 1500,
    },
    gear: {
      enabled: false,
      minConfidence: 0.1,
      skipFrames: 99,
      skipTime: 1500,
    },
  },
  gesture: {
    enabled: false,
  },
};

function sleep(durationMs: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs);
  });
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function roundMetric(value: number, digits = 3) {
  return Number(value.toFixed(digits));
}

function createEmptyPoseCoverage(): FacePoseCoverage {
  return {
    front: 0,
    left: 0,
    right: 0,
    up: 0,
    down: 0,
    unknown: 0,
  };
}

function mergePoseCoverage(targets?: Partial<FacePoseCoverage>) {
  return {
    ...createEmptyPoseCoverage(),
    ...(targets ?? {}),
  };
}

function resolvePoseTargets(sampleCount: number, poseTargets?: Partial<FacePoseCoverage>) {
  const explicitTargets = mergePoseCoverage(poseTargets);
  const explicitTotal = Object.values(explicitTargets).reduce((sum, value) => sum + value, 0);
  if (explicitTotal > 0) {
    return explicitTargets;
  }

  if (sampleCount === DEFAULT_POSE_TARGETS.front + DEFAULT_POSE_TARGETS.left + DEFAULT_POSE_TARGETS.right) {
    return { ...DEFAULT_POSE_TARGETS };
  }

  const front = Math.max(6, Math.round(sampleCount * 0.4));
  const left = Math.max(4, Math.round(sampleCount * 0.3));
  const right = Math.max(4, sampleCount - front - left);
  return {
    front,
    left,
    right,
    up: 0,
    down: 0,
    unknown: 0,
  };
}

function isIOSDevice() {
  if (typeof navigator === "undefined") {
    return false;
  }

  const userAgent = navigator.userAgent.toLowerCase();
  const touchMac =
    navigator.platform === "MacIntel"
    && typeof navigator.maxTouchPoints === "number"
    && navigator.maxTouchPoints > 1;

  return /iphone|ipad|ipod/.test(userAgent) || touchMac;
}

function isSafariBrowser() {
  if (typeof navigator === "undefined") {
    return false;
  }

  const userAgent = navigator.userAgent.toLowerCase();
  return /safari/.test(userAgent)
    && !/chrome|crios|edg|edgios|android|fxios/.test(userAgent);
}

function supportsModelRuntime() {
  if (typeof window === "undefined" || typeof navigator === "undefined" || typeof document === "undefined") {
    return false;
  }

  return (
    typeof navigator.mediaDevices?.getUserMedia === "function"
    && typeof WebAssembly !== "undefined"
    && typeof document.createElement === "function"
  );
}

function getFaceBounds(face: FaceResult): FaceCropBounds {
  return {
    x: face.box[0],
    y: face.box[1],
    width: face.box[2],
    height: face.box[3],
  };
}

function getPoseFromFace(face: FaceResult): FacePose {
  if (face.mesh && face.mesh.length > 450) {
    const zDiff = (face.mesh[33]?.[2] || 0) - (face.mesh[263]?.[2] || 0);
    const xDiff = face.mesh[33]?.[0] - face.mesh[263]?.[0];
    const facingRatio = xDiff ? Math.abs(zDiff / xDiff) : 0;
    if (facingRatio > 0.18) {
      return zDiff < 0 ? "left" : "right";
    }

    const chinDepth = face.mesh[152]?.[2] || 0;
    if (chinDepth < -10) {
      return "up";
    }
    if (chinDepth > 10) {
      return "down";
    }
  }

  return "front";
}

function getTrackingQuality(face: FaceResult, video: HTMLVideoElement) {
  const bounds = getFaceBounds(face);
  const centerX = (bounds.x + bounds.width / 2) / Math.max(1, video.videoWidth);
  const centerY = (bounds.y + bounds.height / 2) / Math.max(1, video.videoHeight);
  const widthRatio = bounds.width / Math.max(1, video.videoWidth);
  const heightRatio = bounds.height / Math.max(1, video.videoHeight);
  const centerPenalty = Math.abs(centerX - 0.5) + Math.abs(centerY - 0.48);
  const centerScore = clamp(1 - centerPenalty * 1.45, 0, 1);
  const widthScore = clamp(1 - Math.abs(widthRatio - 0.3) / 0.24, 0, 1);
  const heightScore = clamp(1 - Math.abs(heightRatio - 0.42) / 0.28, 0, 1);
  const boxScore = face.boxScore || 0;
  const faceScore = face.faceScore || face.score || 0;
  const liveScore = face.live ?? 0.65;
  const realScore = face.real ?? 0.65;

  return roundMetric(
    centerScore * 0.18
    + widthScore * 0.12
    + heightScore * 0.12
    + boxScore * 0.18
    + faceScore * 0.18
    + liveScore * 0.11
    + realScore * 0.11,
  );
}

function getGuidanceForSnapshot(snapshot: FaceTrackingSnapshot) {
  switch (snapshot.status) {
    case "loading":
      return "Loading the face model and camera frame.";
    case "no-face":
      return "Step into view so the camera can find a face.";
    case "multiple":
      return "Keep only one face in the camera frame.";
    case "off-center":
      return "Move closer and keep the face inside the tracked frame.";
    case "low-quality":
      return "Improve lighting and face angle for a sharper biometric read.";
    case "unsupported":
      return "This browser cannot run the face ML pipeline reliably.";
    case "ready":
    default:
      return "Face tracked. Hold steady or move through the portal.";
  }
}

function buildUnsupportedSnapshot(message: string): FaceTrackingSnapshot {
  const snapshot: FaceTrackingSnapshot = {
    status: "unsupported",
    faceCount: 0,
    bounds: null,
    quality: 0,
    pose: "unknown",
    yaw: 0,
    pitch: 0,
    roll: 0,
    liveConfidence: 0,
    realConfidence: 0,
    distance: null,
    boxScore: 0,
    faceScore: 0,
    descriptor: null,
    guidance: message,
  };

  return snapshot;
}

function buildTrackingSnapshot(video: HTMLVideoElement, faces: FaceResult[]): FaceTrackingSnapshot {
  if (video.readyState < 2) {
    return {
      status: "loading",
      faceCount: 0,
      bounds: null,
      quality: 0,
      pose: "unknown",
      yaw: 0,
      pitch: 0,
      roll: 0,
      liveConfidence: 0,
      realConfidence: 0,
      distance: null,
      boxScore: 0,
      faceScore: 0,
      descriptor: null,
      guidance: "Waiting for a live camera frame.",
    };
  }

  if (faces.length === 0) {
    const snapshot: FaceTrackingSnapshot = {
      status: "no-face",
      faceCount: 0,
      bounds: null,
      quality: 0,
      pose: "unknown",
      yaw: 0,
      pitch: 0,
      roll: 0,
      liveConfidence: 0,
      realConfidence: 0,
      distance: null,
      boxScore: 0,
      faceScore: 0,
      descriptor: null,
      guidance: "",
    };
    snapshot.guidance = getGuidanceForSnapshot(snapshot);
    return snapshot;
  }

  if (faces.length > 1) {
    const snapshot: FaceTrackingSnapshot = {
      status: "multiple",
      faceCount: faces.length,
      bounds: null,
      quality: 0,
      pose: "unknown",
      yaw: 0,
      pitch: 0,
      roll: 0,
      liveConfidence: 0,
      realConfidence: 0,
      distance: null,
      boxScore: 0,
      faceScore: 0,
      descriptor: null,
      guidance: "",
    };
    snapshot.guidance = getGuidanceForSnapshot(snapshot);
    return snapshot;
  }

  const face = faces[0];
  const bounds = getFaceBounds(face);
  const yaw = roundMetric((face.rotation?.angle.yaw ?? 0) * RAD_TO_DEG, 1);
  const pitch = roundMetric((face.rotation?.angle.pitch ?? 0) * RAD_TO_DEG, 1);
  const roll = roundMetric((face.rotation?.angle.roll ?? 0) * RAD_TO_DEG, 1);
  const quality = getTrackingQuality(face, video);
  const pose = getPoseFromFace(face);
  const centerX = (bounds.x + bounds.width / 2) / Math.max(1, video.videoWidth);
  const centerY = (bounds.y + bounds.height / 2) / Math.max(1, video.videoHeight);
  const widthRatio = bounds.width / Math.max(1, video.videoWidth);
  const heightRatio = bounds.height / Math.max(1, video.videoHeight);
  const centeredEnough =
    centerX > 0.22
    && centerX < 0.78
    && centerY > 0.18
    && centerY < 0.82;
  const sizeEnough =
    widthRatio > 0.16
    && widthRatio < 0.72
    && heightRatio > 0.22
    && heightRatio < 0.88;

  const status = !centeredEnough || !sizeEnough
    ? "off-center"
    : quality < DEFAULT_MIN_SAMPLE_QUALITY
      ? "low-quality"
      : "ready";

  const snapshot: FaceTrackingSnapshot = {
    status,
    faceCount: 1,
    bounds,
    quality,
    pose,
    yaw,
    pitch,
    roll,
    liveConfidence: roundMetric(face.live ?? 0, 3),
    realConfidence: roundMetric(face.real ?? 0, 3),
    distance: typeof face.distance === "number" ? roundMetric(face.distance, 3) : null,
    boxScore: roundMetric(face.boxScore || 0, 3),
    faceScore: roundMetric(face.faceScore || face.score || 0, 3),
    descriptor: face.embedding ?? null,
    guidance: "",
  };
  snapshot.guidance = getGuidanceForSnapshot(snapshot);
  return snapshot;
}

async function loadHumanModule() {
  return import("@vladmandic/human");
}

async function getHuman() {
  if (!humanPromise) {
    humanPromise = loadHumanModule().then(async (module) => {
      const Human = module.default;
      const human = new Human(HUMAN_CONFIG);
      human.env.perfadd = false;
      await human.load();
      await human.warmup();
      return human;
    }).catch((error) => {
      humanPromise = null;
      throw error;
    });
  }

  return humanPromise;
}

async function detectFaces(video: HTMLVideoElement, config: Partial<Config>) {
  const human = await getHuman();
  const result = await human.detect(video, config);
  return result.face ?? [];
}

function buildLiveTrackedFace(video: HTMLVideoElement, face: FaceResult): LiveTrackedFaceDetection {
  const bounds = getFaceBounds(face);
  return {
    bounds,
    quality: getTrackingQuality(face, video),
    pose: getPoseFromFace(face),
    yaw: roundMetric((face.rotation?.angle.yaw ?? 0) * RAD_TO_DEG, 1),
    pitch: roundMetric((face.rotation?.angle.pitch ?? 0) * RAD_TO_DEG, 1),
    roll: roundMetric((face.rotation?.angle.roll ?? 0) * RAD_TO_DEG, 1),
    liveConfidence: roundMetric(face.live ?? 0, 3),
    realConfidence: roundMetric(face.real ?? 0, 3),
    distance: typeof face.distance === "number" ? roundMetric(face.distance, 3) : null,
    boxScore: roundMetric(face.boxScore || 0, 3),
    faceScore: roundMetric(face.faceScore || face.score || 0, 3),
  };
}

export async function detectLiveTrackingFaces(
  video: HTMLVideoElement,
  options: {
    maxDetected?: number;
  } = {},
): Promise<LiveTrackedFaceDetection[]> {
  if (!supportsModelRuntime() || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return [];
  }

  const maxDetected = Math.max(1, Math.min(options.maxDetected ?? 20, 50));
  const faces = await detectFaces(video, {
    face: {
      ...MULTI_FACE_TRACKING_CONFIG.face,
      detector: {
        ...MULTI_FACE_TRACKING_CONFIG.face?.detector,
        maxDetected,
      },
    },
  });

  return faces
    .map((face) => buildLiveTrackedFace(video, face))
    .sort((left, right) => left.bounds.x - right.bounds.x);
}

export function isFaceDetectorAvailable() {
  return supportsModelRuntime();
}

export function allowInsecureFaceFallback() {
  return import.meta.env.VITE_ALLOW_INSECURE_FACE_FALLBACK === "true";
}

export function getBiometricRuntimeInfo(): BiometricRuntimeInfo {
  const detectorAvailable = supportsModelRuntime();
  const fallbackAllowed = allowInsecureFaceFallback();

  return {
    isIOS: isIOSDevice(),
    isSafari: isSafariBrowser(),
    detectorAvailable,
    fallbackAllowed,
    compatibilityMode: !detectorAvailable && fallbackAllowed,
  };
}

export function getBiometricCameraConstraints(): MediaTrackConstraints {
  const runtime = getBiometricRuntimeInfo();

  return {
    facingMode: "user",
    width: { ideal: runtime.isIOS ? 1280 : 960, min: 640 },
    height: { ideal: runtime.isIOS ? 960 : 720, min: 480 },
    frameRate: { ideal: 30, max: 30 },
  };
}

export function describeFacePose(pose: FacePose) {
  switch (pose) {
    case "front":
      return "Front";
    case "left":
      return "Left";
    case "right":
      return "Right";
    case "up":
      return "Up";
    case "down":
      return "Down";
    default:
      return "Unknown";
  }
}

export function getDefaultEnrollmentPoseTargets() {
  return { ...DEFAULT_POSE_TARGETS };
}

function getRequiredPoseList(poseTargets: FacePoseCoverage) {
  return (Object.keys(poseTargets) as FacePose[]).filter((pose) => poseTargets[pose] > 0);
}

function hasRequiredPoseCoverage(coverage: FacePoseCoverage, targets: FacePoseCoverage) {
  return (Object.keys(targets) as FacePose[]).every((pose) => coverage[pose] >= targets[pose]);
}

function getMissingPoseCoverage(coverage: FacePoseCoverage, targets: FacePoseCoverage) {
  return (Object.keys(targets) as FacePose[])
    .filter((pose) => coverage[pose] < targets[pose])
    .map((pose) => `${describeFacePose(pose)} (${coverage[pose]}/${targets[pose]})`);
}

function averageFaceSamplesInternal(descriptors: number[][]) {
  if (!descriptors.length) {
    return [];
  }

  const dimension = descriptors[0].length;
  const sums = new Array<number>(dimension).fill(0);

  descriptors.forEach((descriptor) => {
    descriptor.forEach((value, index) => {
      sums[index] += value;
    });
  });

  return sums.map((value) => roundMetric(value / descriptors.length, 6));
}

export function averageFaceSamples(descriptors: number[][]) {
  return averageFaceSamplesInternal(descriptors);
}

function calculateDescriptorDistance(v1: number[], v2: number[]) {
  if (!v1.length || v1.length !== v2.length) {
    return Number.MAX_SAFE_INTEGER;
  }

  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    const diff = v1[i] - v2[i];
    sum += diff * diff;
  }

  return roundMetric(25 * sum, 4);
}

export function calculateDescriptorSimilarity(v1: number[], v2: number[]) {
  const distance = calculateDescriptorDistance(v1, v2);
  if (distance === Number.MAX_SAFE_INTEGER) {
    return 0;
  }
  if (distance === 0) {
    return 1;
  }

  const root = Math.sqrt(distance);
  const normalized = (1 - root / 100 - 0.2) / 0.6;
  return roundMetric(clamp(normalized, 0, 1), 4);
}

function calculateDescriptorSetConsistency(descriptors: number[][]) {
  if (descriptors.length <= 1) {
    return 1;
  }

  const averagedDescriptor = averageFaceSamplesInternal(descriptors);
  const scores = descriptors.map((descriptor) => {
    return calculateDescriptorSimilarity(descriptor, averagedDescriptor);
  });
  return roundMetric(average(scores), 4);
}

function buildProfileFromSamples(
  samples: FaceCaptureSample[],
  human: HumanInstance,
  targets: FacePoseCoverage,
): FaceProfileV3 {
  const grouped = new Map<FacePose, FaceCaptureSample[]>();
  const coverage = createEmptyPoseCoverage();

  samples.forEach((sample) => {
    coverage[sample.pose] += 1;
    grouped.set(sample.pose, [...(grouped.get(sample.pose) ?? []), sample]);
  });

  const poseEmbeddings = Array.from(grouped.entries())
    .map(([pose, poseSamples]) => {
      const descriptors = poseSamples.map((sample) => sample.descriptor);
      const aggregated: AggregatedPoseEmbedding = {
        pose,
        descriptor: averageFaceSamplesInternal(descriptors),
        sampleCount: poseSamples.length,
        averageQuality: roundMetric(average(poseSamples.map((sample) => sample.quality))),
        yaw: roundMetric(average(poseSamples.map((sample) => sample.yaw)), 2),
        pitch: roundMetric(average(poseSamples.map((sample) => sample.pitch)), 2),
        roll: roundMetric(average(poseSamples.map((sample) => sample.roll)), 2),
        averageLive: roundMetric(average(poseSamples.map((sample) => sample.liveConfidence))),
        averageReal: roundMetric(average(poseSamples.map((sample) => sample.realConfidence))),
      };

      return aggregated;
    })
    .sort((a, b) => b.sampleCount - a.sampleCount);

  const allDescriptors = samples.map((sample) => sample.descriptor);
  const primaryDescriptor = averageFaceSamplesInternal(allDescriptors);
  const anchorDescriptors = poseEmbeddings.map((poseEmbedding) => poseEmbedding.descriptor);

  return {
    version: 3,
    captureMode: "detected",
    engine: {
      provider: "human",
      libraryVersion: human.version,
      descriptionModel: "faceres.json",
      detectorModel: "blazeface.json",
      meshModel: "facemesh.json",
      irisModel: "iris.json",
      livenessModel: "liveness.json",
      antispoofModel: "antispoof.json",
    },
    primaryDescriptor,
    anchorDescriptors,
    averageQuality: roundMetric(average(samples.map((sample) => sample.quality))),
    sampleCount: samples.length,
    consistency: calculateDescriptorSetConsistency(allDescriptors),
    poseEmbeddings: poseEmbeddings.map((poseEmbedding) => ({
      pose: poseEmbedding.pose,
      descriptor: poseEmbedding.descriptor,
      sampleCount: poseEmbedding.sampleCount,
      averageQuality: poseEmbedding.averageQuality,
      yaw: poseEmbedding.yaw,
      pitch: poseEmbedding.pitch,
      roll: poseEmbedding.roll,
      averageLive: poseEmbedding.averageLive,
      averageReal: poseEmbedding.averageReal,
    })),
    poseCoverage: coverage,
    requiredPoses: getRequiredPoseList(targets),
    averageLive: roundMetric(average(samples.map((sample) => sample.liveConfidence))),
    averageReal: roundMetric(average(samples.map((sample) => sample.realConfidence))),
    orientationSpread: {
      minYaw: roundMetric(Math.min(...samples.map((sample) => sample.yaw)), 2),
      maxYaw: roundMetric(Math.max(...samples.map((sample) => sample.yaw)), 2),
      minPitch: roundMetric(Math.min(...samples.map((sample) => sample.pitch)), 2),
      maxPitch: roundMetric(Math.max(...samples.map((sample) => sample.pitch)), 2),
      minRoll: roundMetric(Math.min(...samples.map((sample) => sample.roll)), 2),
      maxRoll: roundMetric(Math.max(...samples.map((sample) => sample.roll)), 2),
    },
  };
}

export async function warmupBiometricModels() {
  if (!supportsModelRuntime()) {
    throw new Error("This browser cannot run the face ML models.");
  }

  await getHuman();
}

export function startFaceTracking(
  video: HTMLVideoElement,
  onUpdate: (snapshot: FaceTrackingSnapshot) => void,
  options: {
    intervalMs?: number;
    mode?: "tracking" | "capture";
  } = {},
) {
  let cancelled = false;
  let inFlight = false;
  let timeoutId = 0;

  const intervalMs = options.intervalMs ?? TRACKING_INTERVAL_MS;
  const detectConfig = options.mode === "capture" ? CAPTURE_CONFIG : TRACKING_CONFIG;

  const loop = async () => {
    if (cancelled) {
      return;
    }

    if (inFlight) {
      timeoutId = window.setTimeout(() => {
        void loop();
      }, intervalMs);
      return;
    }

    if (!supportsModelRuntime()) {
      onUpdate(buildUnsupportedSnapshot("This browser cannot run the face ML pipeline."));
      return;
    }

    inFlight = true;
    try {
      const faces = await detectFaces(video, detectConfig);
      if (!cancelled) {
        onUpdate(buildTrackingSnapshot(video, faces));
      }
    } catch (error) {
      if (!cancelled) {
        onUpdate(
          buildUnsupportedSnapshot(
            error instanceof Error
              ? error.message
              : "The face ML pipeline could not start.",
          ),
        );
      }
    } finally {
      inFlight = false;
      if (!cancelled) {
        timeoutId = window.setTimeout(() => {
          void loop();
        }, intervalMs);
      }
    }
  };

  void loop();

  return () => {
    cancelled = true;
    if (timeoutId) {
      window.clearTimeout(timeoutId);
    }
  };
}

export async function captureFaceTemplate(
  video: HTMLVideoElement,
  _canvas: HTMLCanvasElement,
  options: FaceTemplateCaptureOptions,
): Promise<FaceTemplateCaptureResult> {
  if (!supportsModelRuntime()) {
    throw new Error("This browser cannot run the face ML pipeline. Use a modern Chrome, Edge, Safari, or iPhone browser with camera support.");
  }

  if (video.readyState < 2) {
    throw new Error("Camera preview is not ready. Wait for the live video feed, then retry.");
  }

  const human = await getHuman();
  const targetCoverage = resolvePoseTargets(options.sampleCount, options.poseTargets);
  const requiredCoverage = getRequiredPoseList(targetCoverage);
  const coverage = createEmptyPoseCoverage();
  const acceptedSamples: FaceCaptureSample[] = [];
  const sampleDelayMs = options.sampleDelayMs ?? DEFAULT_SAMPLE_DELAY_MS;
  const minQuality = options.minQuality ?? DEFAULT_MIN_SAMPLE_QUALITY;
  const minLiveConfidence = options.minLiveConfidence ?? DEFAULT_MIN_LIVE_CONFIDENCE;
  const minRealConfidence = options.minRealConfidence ?? DEFAULT_MIN_REAL_CONFIDENCE;
  const maxAttempts = options.maxAttempts ?? (options.sampleCount * DEFAULT_MAX_ATTEMPTS_MULTIPLIER);

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const faces = await detectFaces(video, CAPTURE_CONFIG);
    const snapshot = buildTrackingSnapshot(video, faces);
    const face = faces[0];

    if (
      snapshot.status !== "ready"
      || !face
      || !snapshot.descriptor?.length
      || snapshot.quality < minQuality
      || snapshot.liveConfidence < minLiveConfidence
      || snapshot.realConfidence < minRealConfidence
    ) {
      options.onProgress?.(acceptedSamples.length, attempt, { ...coverage }, snapshot.pose);
      await sleep(sampleDelayMs);
      continue;
    }

    const pose = snapshot.pose;
    const poseTarget = targetCoverage[pose] ?? 0;
    const requiredStillMissing = !hasRequiredPoseCoverage(coverage, targetCoverage);

    if (requiredStillMissing && poseTarget === 0) {
      options.onProgress?.(acceptedSamples.length, attempt, { ...coverage }, pose);
      await sleep(sampleDelayMs);
      continue;
    }

    if (requiredStillMissing && poseTarget > 0 && coverage[pose] >= poseTarget) {
      options.onProgress?.(acceptedSamples.length, attempt, { ...coverage }, pose);
      await sleep(sampleDelayMs);
      continue;
    }

    acceptedSamples.push({
      descriptor: snapshot.descriptor,
      quality: snapshot.quality,
      pose,
      yaw: snapshot.yaw,
      pitch: snapshot.pitch,
      roll: snapshot.roll,
      liveConfidence: snapshot.liveConfidence,
      realConfidence: snapshot.realConfidence,
      distance: snapshot.distance,
    });
    coverage[pose] += 1;
    options.onProgress?.(acceptedSamples.length, attempt, { ...coverage }, pose);

    const enoughSamples = acceptedSamples.length >= options.sampleCount;
    if (enoughSamples && hasRequiredPoseCoverage(coverage, targetCoverage)) {
      const profile = buildProfileFromSamples(acceptedSamples, human, targetCoverage);
      return {
        descriptor: profile.primaryDescriptor,
        anchorDescriptors: profile.anchorDescriptors,
        averageQuality: profile.averageQuality,
        acceptedSamples: acceptedSamples.length,
        attempts: attempt,
        consistency: profile.consistency,
        captureMode: "detected",
        profile,
        poseCounts: profile.poseCoverage,
      };
    }

    await sleep(sampleDelayMs);
  }

  const missingCoverage = getMissingPoseCoverage(coverage, targetCoverage);
  const missingMessage = missingCoverage.length
    ? `Missing pose coverage: ${missingCoverage.join(", ")}.`
    : "The capture did not reach the required ML quality target.";

  throw new Error(
    `${missingMessage} Look front, then left, then right while staying inside the tracked frame with stable lighting.`,
  );
}

