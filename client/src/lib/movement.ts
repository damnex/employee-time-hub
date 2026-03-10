export type AttendanceDirection = "ENTRY" | "EXIT" | "UNKNOWN";
export type MovementAxis = "horizontal" | "depth" | "none";

export interface MovementSample {
  timestamp: number;
  centerX: number;
  centerY: number;
  area: number;
}

export interface DirectionInferenceOptions {
  entryHorizontalDirection?: "left-to-right" | "right-to-left";
  entryDepthDirection?: "approaching" | "receding";
  maxSamples?: number;
  maxSampleAgeMs?: number;
}

export interface DirectionInferenceResult {
  direction: AttendanceDirection;
  confidence: number;
  axis: MovementAxis;
  sampleCount: number;
}

const DEFAULT_MAX_SAMPLES = 25;
const DEFAULT_MAX_SAMPLE_AGE_MS = 1800;
const MIN_AVERAGE_BUCKET_SIZE = 4;
const MIN_HORIZONTAL_TRAVEL = 0.11;
const MIN_DEPTH_TRAVEL = 0.02;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function averageSamplePoints(samples: MovementSample[]) {
  return {
    centerX: average(samples.map((sample) => sample.centerX)),
    centerY: average(samples.map((sample) => sample.centerY)),
    area: average(samples.map((sample) => sample.area)),
  };
}

function calculateConsistency(values: number[], direction: number) {
  const relevantSteps = values.filter((value) => Math.abs(value) >= 0.0025);
  if (!relevantSteps.length) {
    return 0;
  }

  const alignedSteps = relevantSteps.filter((value) => Math.sign(value) === direction);
  return alignedSteps.length / relevantSteps.length;
}

export function appendMovementSample(
  samples: MovementSample[],
  sample: MovementSample,
  maxSamples = DEFAULT_MAX_SAMPLES,
) {
  const recentSamples = samples.slice(-(maxSamples - 1));
  return [...recentSamples, sample];
}

export function inferMovementDirection(
  samples: MovementSample[],
  options: DirectionInferenceOptions = {},
): DirectionInferenceResult {
  const {
    entryHorizontalDirection = "left-to-right",
    entryDepthDirection = "approaching",
    maxSamples = DEFAULT_MAX_SAMPLES,
    maxSampleAgeMs = DEFAULT_MAX_SAMPLE_AGE_MS,
  } = options;
  const now = Date.now();
  const recentSamples = samples
    .filter((sample) => now - sample.timestamp <= maxSampleAgeMs)
    .slice(-maxSamples);

  if (recentSamples.length < 8) {
    return {
      direction: "UNKNOWN",
      confidence: 0,
      axis: "none",
      sampleCount: recentSamples.length,
    };
  }

  const bucketSize = Math.max(
    MIN_AVERAGE_BUCKET_SIZE,
    Math.min(6, Math.floor(recentSamples.length / 4)),
  );
  const startWindow = recentSamples.slice(0, bucketSize);
  const endWindow = recentSamples.slice(-bucketSize);
  const start = averageSamplePoints(startWindow);
  const end = averageSamplePoints(endWindow);
  const horizontalDelta = end.centerX - start.centerX;
  const depthDelta = end.area - start.area;
  const horizontalSteps = recentSamples.slice(1).map((sample, index) => {
    return sample.centerX - recentSamples[index].centerX;
  });
  const depthSteps = recentSamples.slice(1).map((sample, index) => {
    return sample.area - recentSamples[index].area;
  });
  const horizontalConsistency = calculateConsistency(
    horizontalSteps,
    Math.sign(horizontalDelta) || 1,
  );
  const depthConsistency = calculateConsistency(
    depthSteps,
    Math.sign(depthDelta) || 1,
  );
  const horizontalTravel = Math.abs(horizontalDelta);
  const depthTravel = Math.abs(depthDelta);
  const horizontalConfidence = clamp(
    ((horizontalTravel - MIN_HORIZONTAL_TRAVEL) / 0.18) * 0.65
      + horizontalConsistency * 0.35,
    0,
    1,
  );
  const depthConfidence = clamp(
    ((depthTravel - MIN_DEPTH_TRAVEL) / 0.06) * 0.65
      + depthConsistency * 0.35,
    0,
    1,
  );

  if (
    horizontalTravel >= MIN_HORIZONTAL_TRAVEL
    && horizontalConsistency >= 0.58
    && horizontalConfidence >= depthConfidence
  ) {
    const movingEntry =
      entryHorizontalDirection === "left-to-right"
        ? horizontalDelta > 0
        : horizontalDelta < 0;

    return {
      direction: movingEntry ? "ENTRY" : "EXIT",
      confidence: Number(horizontalConfidence.toFixed(3)),
      axis: "horizontal",
      sampleCount: recentSamples.length,
    };
  }

  if (depthTravel >= MIN_DEPTH_TRAVEL && depthConsistency >= 0.58) {
    const movingEntry =
      entryDepthDirection === "approaching"
        ? depthDelta > 0
        : depthDelta < 0;

    return {
      direction: movingEntry ? "ENTRY" : "EXIT",
      confidence: Number(depthConfidence.toFixed(3)),
      axis: "depth",
      sampleCount: recentSamples.length,
    };
  }

  return {
    direction: "UNKNOWN",
    confidence: Number(Math.max(horizontalConfidence, depthConfidence).toFixed(3)),
    axis: "none",
    sampleCount: recentSamples.length,
  };
}
