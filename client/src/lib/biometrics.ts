import type { FaceCaptureMode, FaceProfile } from "@shared/schema";

const FACE_GRID_SIZE = 12;
const FACE_CAPTURE_SIZE = 128;
const DEFAULT_SAMPLE_DELAY_MS = 80;
const DEFAULT_MIN_SAMPLE_QUALITY = 0.2;
const DEFAULT_ANCHOR_COUNT = 5;

interface FaceDetectorLike {
  detect: (source: CanvasImageSource) => Promise<Array<{
    boundingBox: FaceCropBounds;
  }>>;
}

interface FaceCropBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

declare global {
  interface Window {
    FaceDetector?: new (options?: { fastMode?: boolean; maxDetectedFaces?: number }) => FaceDetectorLike;
  }
}

let cachedFaceDetector: FaceDetectorLike | false | null = null;

export interface FaceCaptureSample {
  descriptor: number[];
  quality: number;
}

export interface FaceTemplateCaptureOptions {
  sampleCount: number;
  sampleDelayMs?: number;
  minQuality?: number;
  maxAttempts?: number;
  requireDetector?: boolean;
  onProgress?: (acceptedSamples: number, attemptCount: number) => void;
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
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function normalizeDescriptor(values: number[]) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const centered = values.map((value) => value - mean);
  const magnitude = Math.sqrt(
    centered.reduce((sum, value) => sum + value * value, 0),
  );

  if (!magnitude) {
    return centered.map(() => 0);
  }

  return centered.map((value) => Number((value / magnitude).toFixed(6)));
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function isFaceDetectorAvailable() {
  return (
    typeof window !== "undefined"
    && "FaceDetector" in window
    && typeof window.FaceDetector === "function"
  );
}

export function allowInsecureFaceFallback() {
  return import.meta.env.VITE_ALLOW_INSECURE_FACE_FALLBACK === "true";
}

function getFaceCropRegion(
  sourceWidth: number,
  sourceHeight: number,
  faceBounds?: FaceCropBounds | null,
) {
  if (!faceBounds) {
    const cropSize = Math.min(sourceWidth, sourceHeight);
    return {
      offsetX: (sourceWidth - cropSize) / 2,
      offsetY: (sourceHeight - cropSize) / 2,
      cropSize,
    };
  }

  const faceCenterX = faceBounds.x + faceBounds.width / 2;
  const faceCenterY = faceBounds.y + faceBounds.height / 2;
  const cropSize = Math.max(faceBounds.width, faceBounds.height) * 1.8;
  const clampedCropSize = Math.min(Math.max(cropSize, 96), Math.min(sourceWidth, sourceHeight));
  const offsetX = clamp(
    faceCenterX - clampedCropSize / 2,
    0,
    Math.max(0, sourceWidth - clampedCropSize),
  );
  const offsetY = clamp(
    faceCenterY - clampedCropSize / 2,
    0,
    Math.max(0, sourceHeight - clampedCropSize),
  );

  return {
    offsetX,
    offsetY,
    cropSize: clampedCropSize,
  };
}

function drawVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  faceBounds?: FaceCropBounds | null,
) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    return null;
  }

  canvas.width = FACE_CAPTURE_SIZE;
  canvas.height = FACE_CAPTURE_SIZE;

  const sourceWidth = video.videoWidth || FACE_CAPTURE_SIZE;
  const sourceHeight = video.videoHeight || FACE_CAPTURE_SIZE;
  const { cropSize, offsetX, offsetY } = getFaceCropRegion(
    sourceWidth,
    sourceHeight,
    faceBounds,
  );

  ctx.drawImage(
    video,
    offsetX,
    offsetY,
    cropSize,
    cropSize,
    0,
    0,
    FACE_CAPTURE_SIZE,
    FACE_CAPTURE_SIZE,
  );

  return ctx.getImageData(0, 0, FACE_CAPTURE_SIZE, FACE_CAPTURE_SIZE);
}

export function captureFaceSample(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  faceBounds?: FaceCropBounds | null,
): FaceCaptureSample | null {
  if (!video.videoWidth || !video.videoHeight) {
    return null;
  }

  const imageData = drawVideoFrame(video, canvas, faceBounds);
  if (!imageData) {
    return null;
  }

  const { data, width, height } = imageData;
  const grayscale = new Float32Array(width * height);

  for (let i = 0; i < data.length; i += 4) {
    const pixelIndex = i / 4;
    grayscale[pixelIndex] =
      0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
  }

  const blockSize = width / FACE_GRID_SIZE;
  const brightnessFeatures: number[] = [];
  const textureFeatures: number[] = [];

  for (let gridY = 0; gridY < FACE_GRID_SIZE; gridY++) {
    for (let gridX = 0; gridX < FACE_GRID_SIZE; gridX++) {
      let brightnessSum = 0;
      let textureSum = 0;
      let pixelCount = 0;

      const startX = Math.floor(gridX * blockSize);
      const endX = Math.floor((gridX + 1) * blockSize);
      const startY = Math.floor(gridY * blockSize);
      const endY = Math.floor((gridY + 1) * blockSize);

      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const index = y * width + x;
          const value = grayscale[index];
          brightnessSum += value;

          const right = x + 1 < width ? grayscale[index + 1] : value;
          const down = y + 1 < height ? grayscale[index + width] : value;
          const horizontalGradient = value - right;
          const verticalGradient = value - down;
          textureSum += Math.sqrt(
            horizontalGradient * horizontalGradient
              + verticalGradient * verticalGradient,
          );

          pixelCount += 1;
        }
      }

      brightnessFeatures.push(brightnessSum / (pixelCount * 255));
      textureFeatures.push(textureSum / (pixelCount * 255));
    }
  }

  const descriptor = normalizeDescriptor([
    ...brightnessFeatures,
    ...textureFeatures,
  ]);

  const averageBrightness =
    brightnessFeatures.reduce((sum, value) => sum + value, 0)
    / brightnessFeatures.length;
  const contrast = Math.sqrt(
    brightnessFeatures.reduce((sum, value) => {
      const delta = value - averageBrightness;
      return sum + delta * delta;
    }, 0) / brightnessFeatures.length,
  );
  const sharpness =
    textureFeatures.reduce((sum, value) => sum + value, 0)
    / textureFeatures.length;
  const exposure = 1 - Math.min(1, Math.abs(averageBrightness - 0.5) * 2);
  const quality = clamp(
    contrast * 2.4 + sharpness * 3.2 + exposure * 0.45,
    0,
    1,
  );

  return {
    descriptor,
    quality: Number(quality.toFixed(3)),
  };
}

export function averageFaceSamples(descriptors: number[][]) {
  if (!descriptors.length) {
    return [];
  }

  const merged = descriptors[0].map((_, index) => {
    const sum = descriptors.reduce(
      (runningTotal, descriptor) => runningTotal + descriptor[index],
      0,
    );

    return sum / descriptors.length;
  });

  return normalizeDescriptor(merged);
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function calculateDescriptorSimilarity(v1: number[], v2: number[]) {
  if (!v1 || !v2 || v1.length !== v2.length || !v1.length) {
    return 0;
  }

  const normalizedV1 = normalizeDescriptor(v1);
  const normalizedV2 = normalizeDescriptor(v2);
  let dotProduct = 0;

  for (let i = 0; i < normalizedV1.length; i++) {
    dotProduct += normalizedV1[i] * normalizedV2[i];
  }

  return Number((((dotProduct + 1) / 2)).toFixed(4));
}

function calculateDescriptorSetConsistency(descriptors: number[][]) {
  if (descriptors.length < 2) {
    return 1;
  }

  const pairScores: number[] = [];

  for (let i = 0; i < descriptors.length; i++) {
    for (let j = i + 1; j < descriptors.length; j++) {
      pairScores.push(calculateDescriptorSimilarity(descriptors[i], descriptors[j]));
    }
  }

  return Number(average(pairScores).toFixed(4));
}

function selectAnchorDescriptors(samples: FaceCaptureSample[], anchorCount = DEFAULT_ANCHOR_COUNT) {
  if (!samples.length) {
    return [];
  }

  if (samples.length <= anchorCount) {
    return samples.map((sample) => sample.descriptor);
  }

  const anchors: number[][] = [];
  for (let anchorIndex = 0; anchorIndex < anchorCount; anchorIndex++) {
    const sampleIndex = Math.round(
      (anchorIndex * (samples.length - 1)) / Math.max(1, anchorCount - 1),
    );
    anchors.push(samples[sampleIndex].descriptor);
  }

  return anchors;
}

function buildFaceProfile(samples: FaceCaptureSample[], captureMode: FaceCaptureMode): FaceProfile {
  const merged = mergeFaceCaptureSamples(samples);
  const anchorDescriptors = selectAnchorDescriptors(samples);
  const consistency = calculateDescriptorSetConsistency(
    samples.map((sample) => sample.descriptor),
  );

  return {
    version: 2,
    captureMode,
    primaryDescriptor: merged.descriptor,
    anchorDescriptors,
    averageQuality: merged.averageQuality,
    sampleCount: samples.length,
    consistency: Number(consistency.toFixed(3)),
  };
}

async function getFaceDetector(): Promise<FaceDetectorLike | null> {
  if (cachedFaceDetector === false) {
    return null;
  }

  if (cachedFaceDetector) {
    return cachedFaceDetector;
  }

  if (
    typeof window === "undefined"
    || !("FaceDetector" in window)
    || typeof window.FaceDetector !== "function"
  ) {
    cachedFaceDetector = false;
    return null;
  }

  cachedFaceDetector = new window.FaceDetector({
    fastMode: true,
    maxDetectedFaces: 1,
  }) as FaceDetectorLike;

  return cachedFaceDetector;
}

async function detectSingleFaceBounds(video: HTMLVideoElement) {
  const detector = await getFaceDetector();
  if (!detector) {
    return undefined;
  }

  const faces = await detector.detect(video);
  if (faces.length !== 1) {
    return null;
  }

  return faces[0].boundingBox;
}

export function mergeFaceCaptureSamples(samples: FaceCaptureSample[]) {
  if (!samples.length) {
    return {
      descriptor: [],
      averageQuality: 0,
    };
  }

  const descriptorLength = samples[0]?.descriptor.length ?? 0;
  if (!descriptorLength) {
    return {
      descriptor: [],
      averageQuality: 0,
    };
  }

  const merged = Array.from({ length: descriptorLength }, (_, index) => {
    let weightedSum = 0;
    let totalWeight = 0;

    samples.forEach((sample) => {
      const weight = Math.max(0.12, sample.quality);
      weightedSum += sample.descriptor[index] * weight;
      totalWeight += weight;
    });

    return totalWeight ? weightedSum / totalWeight : 0;
  });

  const averageQuality =
    samples.reduce((sum, sample) => sum + sample.quality, 0) / samples.length;

  return {
    descriptor: normalizeDescriptor(merged),
    averageQuality: Number(averageQuality.toFixed(3)),
  };
}

export async function captureFaceTemplate(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  options: FaceTemplateCaptureOptions,
): Promise<FaceTemplateCaptureResult> {
  const {
    sampleCount,
    sampleDelayMs = DEFAULT_SAMPLE_DELAY_MS,
    minQuality = DEFAULT_MIN_SAMPLE_QUALITY,
    maxAttempts = sampleCount * 3,
    requireDetector = false,
    onProgress,
  } = options;
  const captureMode: FaceCaptureMode = isFaceDetectorAvailable() ? "detected" : "fallback";

  if (requireDetector && captureMode !== "detected") {
    throw new Error("Secure face detection is unavailable in this browser. Use Chrome or Edge.");
  }

  const acceptedSamples: FaceCaptureSample[] = [];
  let attempts = 0;

  while (acceptedSamples.length < sampleCount && attempts < maxAttempts) {
    const detectedFaceBounds = await detectSingleFaceBounds(video);
    const sample = detectedFaceBounds === null
      ? null
      : captureFaceSample(video, canvas, detectedFaceBounds);
    attempts += 1;

    if (sample && sample.quality >= minQuality) {
      acceptedSamples.push(sample);
      onProgress?.(acceptedSamples.length, attempts);
    }

    if (acceptedSamples.length < sampleCount) {
      await delay(sampleDelayMs);
    }
  }

  if (acceptedSamples.length < sampleCount) {
    throw new Error(
      `Only ${acceptedSamples.length} of ${sampleCount} clear face samples were captured. Keep one face centered, improve lighting, and retry.`,
    );
  }

  const profile = buildFaceProfile(acceptedSamples, captureMode);

  return {
    descriptor: profile.primaryDescriptor,
    anchorDescriptors: profile.anchorDescriptors,
    averageQuality: profile.averageQuality,
    acceptedSamples: acceptedSamples.length,
    attempts,
    consistency: profile.consistency,
    captureMode,
    profile,
  };
}
