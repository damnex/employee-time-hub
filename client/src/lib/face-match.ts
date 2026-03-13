import { faceProfileSchema, type Employee, type FaceCaptureMode, type FacePose } from "@shared/schema";
import { averageFaceSamples, calculateDescriptorSimilarity } from "@/lib/biometrics";

const HUMAN_FACE_ANCHOR_AVG_THRESHOLD = 0.54;
export const LOCAL_HUMAN_MATCH_THRESHOLD = 0.57;

function roundMetric(value: number, digits = 4) {
  return Number(value.toFixed(digits));
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function calculateLegacyMatchConfidence(v1: number[], v2: number[]) {
  if (!v1.length || v1.length !== v2.length) {
    return 0;
  }

  const meanV1 = average(v1);
  const meanV2 = average(v2);
  const centeredV1 = v1.map((value) => value - meanV1);
  const centeredV2 = v2.map((value) => value - meanV2);
  const magnitudeV1 = Math.sqrt(centeredV1.reduce((sum, value) => sum + value * value, 0));
  const magnitudeV2 = Math.sqrt(centeredV2.reduce((sum, value) => sum + value * value, 0));

  if (!magnitudeV1 || !magnitudeV2) {
    return 0;
  }

  let dotProduct = 0;
  for (let index = 0; index < centeredV1.length; index += 1) {
    dotProduct += (centeredV1[index] / magnitudeV1) * (centeredV2[index] / magnitudeV2);
  }

  return roundMetric((dotProduct + 1) / 2);
}

function calculateDescriptorSetConsistency(descriptors: number[][]) {
  if (descriptors.length <= 1) {
    return 1;
  }

  const averagedDescriptor = averageFaceSamples(descriptors);
  const scores = descriptors.map((descriptor) => calculateDescriptorSimilarity(descriptor, averagedDescriptor));
  return roundMetric(average(scores));
}

function calculateSimilarity(v1: number[], v2: number[], engine: "human" | "legacy" | "heuristic") {
  return engine === "human"
    ? calculateDescriptorSimilarity(v1, v2)
    : calculateLegacyMatchConfidence(v1, v2);
}

export interface TrackingDescriptorSample {
  timestamp: number;
  descriptor: number[];
  quality: number;
  pose: FacePose;
  yaw: number;
  pitch: number;
  roll: number;
  liveConfidence: number;
  realConfidence: number;
}

export interface GateLiveFaceProfile {
  primaryDescriptor: number[];
  anchorDescriptors: number[][];
  consistency: number;
  averageQuality: number;
  captureMode: FaceCaptureMode | "legacy" | "unknown";
  poseEmbeddings: Array<{
    pose: FacePose;
    descriptor: number[];
    averageQuality: number;
    sampleCount: number;
    yaw: number;
    pitch: number;
    roll: number;
    averageLive: number;
    averageReal: number;
  }>;
  pose: FacePose;
  yaw: number;
  pitch: number;
  roll: number;
  averageLive: number;
  averageReal: number;
  sampleCount: number;
  engine: "human" | "legacy" | "heuristic";
}

interface NormalizedEmployeeFaceProfile extends GateLiveFaceProfile {
  employee: Employee;
}

export interface EmployeeFaceMatch {
  employee: Employee;
  confidence: number;
  metrics: {
    primaryConfidence: number;
    anchorAverage: number;
    peakAnchorConfidence: number;
    strongAnchorRatio: number;
    poseConfidence: number;
  };
}

export function normalizeFaceProfileData(faceProfile: unknown): GateLiveFaceProfile | null {
  if (!faceProfile) {
    return null;
  }

  if (Array.isArray(faceProfile) && faceProfile.every((value) => typeof value === "number")) {
    return {
      primaryDescriptor: faceProfile,
      anchorDescriptors: [faceProfile],
      consistency: 1,
      averageQuality: 0.55,
      captureMode: "legacy",
      poseEmbeddings: [],
      pose: "unknown",
      yaw: 0,
      pitch: 0,
      roll: 0,
      averageLive: 0,
      averageReal: 0,
      sampleCount: 1,
      engine: "legacy",
    };
  }

  const parsedProfile = faceProfileSchema.safeParse(faceProfile);
  if (!parsedProfile.success) {
    return null;
  }

  const normalizedPoseEmbeddings = parsedProfile.data.version === 3
    ? parsedProfile.data.poseEmbeddings.map((embedding) => ({
        pose: embedding.pose,
        descriptor: embedding.descriptor,
        averageQuality: embedding.averageQuality,
        sampleCount: embedding.sampleCount,
        yaw: embedding.yaw,
        pitch: embedding.pitch,
        roll: embedding.roll,
        averageLive: embedding.averageLive ?? 0,
        averageReal: embedding.averageReal ?? 0,
      }))
    : [];
  const dominantPoseEmbedding = normalizedPoseEmbeddings[0];

  return {
    primaryDescriptor: parsedProfile.data.primaryDescriptor,
    anchorDescriptors: parsedProfile.data.anchorDescriptors,
    consistency: parsedProfile.data.consistency,
    averageQuality: parsedProfile.data.averageQuality,
    captureMode: parsedProfile.data.captureMode ?? "unknown",
    poseEmbeddings: normalizedPoseEmbeddings,
    pose: dominantPoseEmbedding?.pose ?? "unknown",
    yaw: dominantPoseEmbedding?.yaw ?? 0,
    pitch: dominantPoseEmbedding?.pitch ?? 0,
    roll: dominantPoseEmbedding?.roll ?? 0,
    averageLive: parsedProfile.data.version === 3 ? (parsedProfile.data.averageLive ?? 0) : 0,
    averageReal: parsedProfile.data.version === 3 ? (parsedProfile.data.averageReal ?? 0) : 0,
    sampleCount: parsedProfile.data.sampleCount,
    engine: parsedProfile.data.version === 3 ? "human" : "heuristic",
  };
}

export function buildLiveFaceProfileFromSamples(samples: TrackingDescriptorSample[]): GateLiveFaceProfile | null {
  if (!samples.length) {
    return null;
  }

  const groupedSamples = new Map<FacePose, TrackingDescriptorSample[]>();
  samples.forEach((sample) => {
    groupedSamples.set(sample.pose, [...(groupedSamples.get(sample.pose) ?? []), sample]);
  });

  const poseEmbeddings = Array.from(groupedSamples.entries())
    .map(([pose, poseSamples]) => {
      const descriptors = poseSamples.map((sample) => sample.descriptor);
      return {
        pose,
        descriptor: averageFaceSamples(descriptors),
        averageQuality: roundMetric(average(poseSamples.map((sample) => sample.quality))),
        sampleCount: poseSamples.length,
        yaw: roundMetric(average(poseSamples.map((sample) => sample.yaw)), 2),
        pitch: roundMetric(average(poseSamples.map((sample) => sample.pitch)), 2),
        roll: roundMetric(average(poseSamples.map((sample) => sample.roll)), 2),
        averageLive: roundMetric(average(poseSamples.map((sample) => sample.liveConfidence))),
        averageReal: roundMetric(average(poseSamples.map((sample) => sample.realConfidence))),
      };
    })
    .sort((first, second) => second.sampleCount - first.sampleCount);

  const allDescriptors = samples.map((sample) => sample.descriptor);
  const dominantPoseEmbedding = poseEmbeddings[0];

  return {
    primaryDescriptor: averageFaceSamples(allDescriptors),
    anchorDescriptors: poseEmbeddings.map((embedding) => embedding.descriptor),
    consistency: calculateDescriptorSetConsistency(allDescriptors),
    averageQuality: roundMetric(average(samples.map((sample) => sample.quality))),
    captureMode: "detected",
    poseEmbeddings,
    pose: dominantPoseEmbedding?.pose ?? "unknown",
    yaw: dominantPoseEmbedding?.yaw ?? 0,
    pitch: dominantPoseEmbedding?.pitch ?? 0,
    roll: dominantPoseEmbedding?.roll ?? 0,
    averageLive: roundMetric(average(samples.map((sample) => sample.liveConfidence))),
    averageReal: roundMetric(average(samples.map((sample) => sample.realConfidence))),
    sampleCount: samples.length,
    engine: "human",
  };
}

function calculateMatchMetrics(
  liveProfile: GateLiveFaceProfile,
  storedProfile: GateLiveFaceProfile,
) {
  const similarityEngine =
    liveProfile.engine === "human" && storedProfile.engine === "human"
      ? "human"
      : storedProfile.engine;
  const primaryConfidence = Math.max(
    calculateSimilarity(liveProfile.primaryDescriptor, storedProfile.primaryDescriptor, similarityEngine),
    ...storedProfile.anchorDescriptors.map((anchorDescriptor) => {
      return calculateSimilarity(liveProfile.primaryDescriptor, anchorDescriptor, similarityEngine);
    }),
  );
  const anchorScores = liveProfile.anchorDescriptors.map((liveAnchor) => {
    return Math.max(
      calculateSimilarity(liveAnchor, storedProfile.primaryDescriptor, similarityEngine),
      ...storedProfile.anchorDescriptors.map((storedAnchor) => {
        return calculateSimilarity(liveAnchor, storedAnchor, similarityEngine);
      }),
    );
  });
  const anchorAverage = average(anchorScores);
  const peakAnchorConfidence = Math.max(...anchorScores, 0);
  const strongAnchorRatio =
    anchorScores.filter((score) => score >= HUMAN_FACE_ANCHOR_AVG_THRESHOLD).length
    / Math.max(1, anchorScores.length);
  const poseScores = storedProfile.poseEmbeddings.map((embedding) => ({
    pose: embedding.pose,
    similarity: calculateSimilarity(liveProfile.primaryDescriptor, embedding.descriptor, similarityEngine),
  }));
  const poseConfidence = poseScores.length
    ? Math.max(
        ...poseScores.map((poseScore) => poseScore.similarity),
        liveProfile.pose !== "unknown"
          ? poseScores.find((poseScore) => poseScore.pose === liveProfile.pose)?.similarity ?? 0
          : 0,
      )
    : primaryConfidence;
  const confidence = roundMetric(
    primaryConfidence * 0.38
      + anchorAverage * 0.27
      + poseConfidence * 0.25
      + strongAnchorRatio * 0.1,
  );

  return {
    primaryConfidence: roundMetric(primaryConfidence),
    anchorAverage: roundMetric(anchorAverage),
    peakAnchorConfidence: roundMetric(peakAnchorConfidence),
    strongAnchorRatio: roundMetric(strongAnchorRatio),
    poseConfidence: roundMetric(poseConfidence),
    confidence,
  };
}

function normalizeEmployeeFaceProfile(employee: Employee): NormalizedEmployeeFaceProfile | null {
  const normalizedFaceProfile = normalizeFaceProfileData(employee.faceDescriptor);
  if (!normalizedFaceProfile) {
    return null;
  }

  return {
    ...normalizedFaceProfile,
    employee,
  };
}

export function findEmployeeFaceMatches(
  liveProfile: GateLiveFaceProfile,
  employees: Employee[],
): EmployeeFaceMatch[] {
  return employees
    .filter((employee) => employee.isActive)
    .map(normalizeEmployeeFaceProfile)
    .filter((employeeProfile): employeeProfile is NormalizedEmployeeFaceProfile => employeeProfile !== null)
    .map((employeeProfile) => {
      const metrics = calculateMatchMetrics(liveProfile, employeeProfile);
      return {
        employee: employeeProfile.employee,
        confidence: metrics.confidence,
        metrics: {
          primaryConfidence: metrics.primaryConfidence,
          anchorAverage: metrics.anchorAverage,
          peakAnchorConfidence: metrics.peakAnchorConfidence,
          strongAnchorRatio: metrics.strongAnchorRatio,
          poseConfidence: metrics.poseConfidence,
        },
      };
    })
    .sort((first, second) => second.confidence - first.confidence);
}
