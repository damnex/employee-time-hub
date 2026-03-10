const FACE_GRID_SIZE = 8;
const FACE_CAPTURE_SIZE = 128;
const DEFAULT_SAMPLE_DELAY_MS = 80;
const DEFAULT_MIN_SAMPLE_QUALITY = 0.2;

export interface FaceCaptureSample {
  descriptor: number[];
  quality: number;
}

export interface FaceTemplateCaptureOptions {
  sampleCount: number;
  sampleDelayMs?: number;
  minQuality?: number;
  maxAttempts?: number;
  onProgress?: (acceptedSamples: number, attemptCount: number) => void;
}

export interface FaceTemplateCaptureResult {
  descriptor: number[];
  averageQuality: number;
  acceptedSamples: number;
  attempts: number;
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

function drawVideoFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    return null;
  }

  canvas.width = FACE_CAPTURE_SIZE;
  canvas.height = FACE_CAPTURE_SIZE;

  const sourceWidth = video.videoWidth || FACE_CAPTURE_SIZE;
  const sourceHeight = video.videoHeight || FACE_CAPTURE_SIZE;
  const cropSize = Math.min(sourceWidth, sourceHeight);
  const offsetX = (sourceWidth - cropSize) / 2;
  const offsetY = (sourceHeight - cropSize) / 2;

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
): FaceCaptureSample | null {
  if (!video.videoWidth || !video.videoHeight) {
    return null;
  }

  const imageData = drawVideoFrame(video, canvas);
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
    onProgress,
  } = options;

  const acceptedSamples: FaceCaptureSample[] = [];
  let attempts = 0;

  while (acceptedSamples.length < sampleCount && attempts < maxAttempts) {
    const sample = captureFaceSample(video, canvas);
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
      `Only ${acceptedSamples.length} of ${sampleCount} clear samples were captured. Improve lighting, keep still, and retry.`,
    );
  }

  const { descriptor, averageQuality } = mergeFaceCaptureSamples(acceptedSamples);

  return {
    descriptor,
    averageQuality,
    acceptedSamples: acceptedSamples.length,
    attempts,
  };
}
