import { faceProfileSchema } from "@shared/schema";
import type { FaceObservation } from "./matching-engine";
import { storage } from "./storage";
import { getCurrentSession } from "./buffer";
import { getTrackState, isValidTrack } from "./trackState";

export interface FaceState {
  track_id: number;
  embedding?: number[];
  name?: string;
  confidence?: number;
  last_updated: number;
  stable: boolean;
}

export interface FaceRecognitionInput {
  deviceId: string;
  track_id: number;
  name?: string;
  confidence?: number;
  similarity?: number;
  embedding?: number[];
  personId?: string | null;
  rfidTag?: string | null;
  matched?: boolean;
  bbox?: [number, number, number, number] | null;
  timestamp?: number;
}

export interface ResolvedFaceTrack {
  faceState: FaceState;
  observation: FaceObservation;
  source: "cache" | "refresh" | "raw";
}

interface EmployeeEmbeddingEntry {
  name: string;
  employeeCode?: string | null;
  rfidUid?: string | null;
  embeddings: number[][];
}

const FACE_MATCH_THRESHOLD = Number(process.env.FACE_STATE_MATCH_THRESHOLD ?? "0.7");
const FACE_REFRESH_MS = Number(process.env.FACE_STATE_REFRESH_MS ?? "2500");
const FACE_STATE_TTL_MS = Number(process.env.FACE_STATE_TTL_MS ?? "3000");
const FACE_STATE_CLEANUP_INTERVAL_MS = 500;
const FACE_EMBEDDING_CACHE_TTL_MS = Number(process.env.FACE_EMBEDDING_CACHE_TTL_MS ?? "5000");
const FACE_STATE_REQUIRE_RFID_SESSION = process.env.FACE_STATE_REQUIRE_RFID_SESSION === "1";

export const faceStateMap = new Map<number, FaceState>();

let cachedEmbeddingIndex: EmployeeEmbeddingEntry[] = [];
let cachedEmbeddingIndexLoadedAt = 0;

function normalizeTimestamp(timestamp?: number) {
  if (typeof timestamp === "number" && Number.isFinite(timestamp) && timestamp > 0) {
    return Math.trunc(timestamp);
  }

  return Date.now();
}

function normalizeName(name?: string | null) {
  const normalized = name?.trim();
  if (!normalized || normalized.toLowerCase() === "unknown") {
    return undefined;
  }

  return normalized;
}

function normalizeRfidTag(rfidTag?: string | null) {
  const normalized = rfidTag?.trim();
  return normalized ? normalized.toUpperCase() : null;
}

function cloneFaceState(state: FaceState): FaceState {
  return {
    ...state,
    ...(state.embedding ? { embedding: [...state.embedding] } : {}),
  };
}

function copyEmbedding(embedding?: number[]) {
  return embedding?.length ? [...embedding] : undefined;
}

function hasActiveRfidSession(nowMs = Date.now()) {
  return getCurrentSession(nowMs).rfid.length > 0;
}

function isFaceStateOutdated(state: FaceState, nowMs = Date.now()) {
  return (nowMs - state.last_updated) > FACE_REFRESH_MS;
}

function isFaceStateExpired(state: FaceState, nowMs = Date.now()) {
  return (nowMs - state.last_updated) > FACE_STATE_TTL_MS;
}

function normalizeReportedScore(similarity?: number, confidence?: number) {
  if (typeof similarity === "number" && Number.isFinite(similarity)) {
    return Math.max(-1, Math.min(1, similarity));
  }

  if (typeof confidence === "number" && Number.isFinite(confidence)) {
    return Math.max(0, Math.min(1, confidence));
  }

  return 0;
}

function cosineSimilarity(left: number[], right: number[]) {
  if (!left.length || left.length !== right.length) {
    return -1;
  }

  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;

  for (let index = 0; index < left.length; index += 1) {
    const leftValue = left[index] ?? 0;
    const rightValue = right[index] ?? 0;
    dot += leftValue * rightValue;
    leftNorm += leftValue * leftValue;
    rightNorm += rightValue * rightValue;
  }

  if (!leftNorm || !rightNorm) {
    return -1;
  }

  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

function extractFaceEmbeddings(faceDescriptor: unknown) {
  if (Array.isArray(faceDescriptor) && faceDescriptor.every((value) => typeof value === "number")) {
    return [faceDescriptor];
  }

  const parsedProfile = faceProfileSchema.safeParse(faceDescriptor);
  if (!parsedProfile.success) {
    return [];
  }

  const embeddings: number[][] = [
    parsedProfile.data.primaryDescriptor,
    ...parsedProfile.data.anchorDescriptors,
  ];

  if (parsedProfile.data.version === 3) {
    embeddings.push(...parsedProfile.data.poseEmbeddings.map((poseEmbedding) => poseEmbedding.descriptor));
  }

  const seen = new Set<string>();
  return embeddings.filter((embedding) => {
    const signature = embedding.join(",");
    if (seen.has(signature)) {
      return false;
    }

    seen.add(signature);
    return true;
  });
}

async function getEmployeeEmbeddingIndex(nowMs = Date.now()) {
  if (cachedEmbeddingIndex.length && (nowMs - cachedEmbeddingIndexLoadedAt) <= FACE_EMBEDDING_CACHE_TTL_MS) {
    return cachedEmbeddingIndex;
  }

  const employees = await storage.getEmployees();
  cachedEmbeddingIndex = employees
    .map((employee) => ({
      name: employee.name,
      employeeCode: employee.employeeCode,
      rfidUid: employee.rfidUid,
      embeddings: extractFaceEmbeddings(employee.faceDescriptor),
    }))
    .filter((employee) => employee.embeddings.length > 0);
  cachedEmbeddingIndexLoadedAt = nowMs;

  return cachedEmbeddingIndex;
}

async function findBestEmbeddingMatch(embedding: number[], nowMs: number) {
  const employees = await getEmployeeEmbeddingIndex(nowMs);
  let bestMatch:
    | {
        name: string;
        employeeCode?: string | null;
        rfidUid?: string | null;
        similarity: number;
      }
    | null = null;

  for (const employee of employees) {
    for (const candidateEmbedding of employee.embeddings) {
      const similarity = cosineSimilarity(embedding, candidateEmbedding);
      if (!bestMatch || similarity > bestMatch.similarity) {
        bestMatch = {
          name: employee.name,
          employeeCode: employee.employeeCode,
          rfidUid: employee.rfidUid,
          similarity,
        };
      }
    }
  }

  return bestMatch && bestMatch.similarity >= FACE_MATCH_THRESHOLD ? bestMatch : null;
}

function writeFaceState(state: FaceState) {
  faceStateMap.set(state.track_id, state);
  return cloneFaceState(state);
}

function buildObservationFromState(args: FaceRecognitionInput, state: FaceState, timestampMs: number): FaceObservation {
  return {
    deviceId: args.deviceId.trim(),
    trackId: args.track_id,
    timestampMs,
    name: state.name ?? args.name ?? "unknown",
    confidence: state.confidence ?? args.confidence ?? 0,
    similarity: args.similarity,
    personId: args.personId ?? null,
    rfidTag: normalizeRfidTag(args.rfidTag),
    matched: state.stable && Boolean(state.name),
    bbox: args.bbox ?? null,
  };
}

function buildRawObservation(args: FaceRecognitionInput, timestampMs: number): FaceObservation {
  return {
    deviceId: args.deviceId.trim(),
    trackId: args.track_id,
    timestampMs,
    name: args.name?.trim() || "unknown",
    confidence: args.confidence ?? 0,
    similarity: args.similarity,
    personId: args.personId ?? null,
    rfidTag: normalizeRfidTag(args.rfidTag),
    matched: Boolean(args.matched && normalizeName(args.name)),
    bbox: args.bbox ?? null,
  };
}

export function shouldRunFaceRecognition(args: {
  track_id: number;
  timestamp?: number;
  requireRfidSession?: boolean;
}) {
  const timestampMs = normalizeTimestamp(args.timestamp);
  const trackState = getTrackState(args.track_id);
  if (!trackState || !isValidTrack(trackState, timestampMs)) {
    return false;
  }

  if ((args.requireRfidSession ?? FACE_STATE_REQUIRE_RFID_SESSION) && !hasActiveRfidSession(timestampMs)) {
    return false;
  }

  const existingState = faceStateMap.get(args.track_id);
  return !existingState || !existingState.stable || isFaceStateOutdated(existingState, timestampMs) || (existingState.confidence ?? 0) < FACE_MATCH_THRESHOLD;
}

export async function resolveFaceTrack(args: FaceRecognitionInput): Promise<ResolvedFaceTrack | null> {
  const timestampMs = normalizeTimestamp(args.timestamp);
  const trackState = getTrackState(args.track_id);
  if (!trackState || !isValidTrack(trackState, timestampMs)) {
    return null;
  }

  const existingState = faceStateMap.get(args.track_id);
  if (existingState?.stable && !isFaceStateOutdated(existingState, timestampMs) && (existingState.confidence ?? 0) >= FACE_MATCH_THRESHOLD) {
    return {
      faceState: cloneFaceState(existingState),
      observation: buildObservationFromState(args, existingState, timestampMs),
      source: "cache",
    };
  }

  if ((FACE_STATE_REQUIRE_RFID_SESSION && !hasActiveRfidSession(timestampMs)) && !existingState?.stable) {
    return null;
  }

  const normalizedName = normalizeName(args.name);
  const reportedScore = normalizeReportedScore(args.similarity, args.confidence);
  const normalizedEmbedding = copyEmbedding(args.embedding);

  let resolvedState: FaceState | null = null;

  if (normalizedEmbedding?.length) {
    try {
      const embeddingMatch = await findBestEmbeddingMatch(normalizedEmbedding, timestampMs);
      if (embeddingMatch) {
        resolvedState = writeFaceState({
          track_id: args.track_id,
          embedding: normalizedEmbedding,
          name: embeddingMatch.name,
          confidence: embeddingMatch.similarity,
          last_updated: timestampMs,
          stable: true,
        });

        console.debug("[face-state] match", {
          track_id: args.track_id,
          name: embeddingMatch.name,
          confidence: Number(embeddingMatch.similarity.toFixed(4)),
          source: "embedding",
        });
      }
    } catch (error) {
      console.warn("[face-state] Embedding match lookup failed:", error);
    }
  }

  if (!resolvedState && normalizedName && Boolean(args.matched ?? true) && reportedScore >= FACE_MATCH_THRESHOLD) {
    resolvedState = writeFaceState({
      track_id: args.track_id,
      embedding: normalizedEmbedding,
      name: normalizedName,
      confidence: reportedScore,
      last_updated: timestampMs,
      stable: true,
    });

    console.debug("[face-state] match", {
      track_id: args.track_id,
      name: normalizedName,
      confidence: Number(reportedScore.toFixed(4)),
      source: "service",
    });
  }

  if (resolvedState) {
    return {
      faceState: resolvedState,
      observation: buildObservationFromState(args, resolvedState, timestampMs),
      source: "refresh",
    };
  }

  if (existingState?.stable && !isFaceStateExpired(existingState, timestampMs) && (existingState.confidence ?? 0) >= FACE_MATCH_THRESHOLD) {
    return {
      faceState: cloneFaceState(existingState),
      observation: buildObservationFromState(args, existingState, timestampMs),
      source: "cache",
    };
  }

  const pendingState = writeFaceState({
    track_id: args.track_id,
    embedding: normalizedEmbedding,
    name: normalizedName,
    confidence: reportedScore || undefined,
    last_updated: timestampMs,
    stable: false,
  });

  return {
    faceState: pendingState,
    observation: buildRawObservation(args, timestampMs),
    source: "raw",
  };
}

export function cleanupFaceStates(nowMs = Date.now()) {
  let removed = 0;

  for (const [trackId, state] of Array.from(faceStateMap.entries())) {
    if ((nowMs - state.last_updated) > FACE_STATE_TTL_MS) {
      faceStateMap.delete(trackId);
      removed += 1;
    }
  }

  if (removed > 0) {
    console.debug("[face-state] cleanup", { activeFaces: faceStateMap.size });
  }
}

export function getFaceState(trackId: number) {
  const state = faceStateMap.get(trackId);
  return state ? cloneFaceState(state) : undefined;
}

export function clearFaceState(trackId: number) {
  faceStateMap.delete(trackId);
}

export function getFaceStateSnapshot() {
  const states = Array.from(faceStateMap.values()).map((state) => cloneFaceState(state));

  return {
    activeFaces: states.length,
    stableFaces: states.filter((state) => state.stable).length,
    faces: states,
  };
}

const cleanupTimer = setInterval(() => {
  cleanupFaceStates();
}, FACE_STATE_CLEANUP_INTERVAL_MS);

cleanupTimer.unref?.();

