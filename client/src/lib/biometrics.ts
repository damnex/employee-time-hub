const FACE_GRID_SIZE = 8;
const FACE_CAPTURE_SIZE = 128;

export interface FaceCaptureSample {
  descriptor: number[];
  quality: number;
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
