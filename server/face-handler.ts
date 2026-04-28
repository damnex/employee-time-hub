import { z } from "zod";
import { decisionEngine } from "./decision-engine";
import { addFace } from "./buffer";
import { resolveFaceTrack } from "./faceState";
import { refreshScoreMatrix } from "./score-matrix";

const bboxSchema = z.tuple([
  z.number(),
  z.number(),
  z.number(),
  z.number(),
]);

const embeddingSchema = z.array(z.number()).min(1);

export const faceIntegrationSchema = z.object({
  deviceId: z.string().trim().min(1),
  timestamp: z.number().int().positive().optional(),
  timestamp_ms: z.number().int().positive().optional(),
  tracks: z.array(z.object({
    trackId: z.number().int().nonnegative().optional(),
    track_id: z.number().int().nonnegative().optional(),
    name: z.string().trim().default("unknown"),
    confidence: z.number().min(0).max(1),
    similarity: z.number().min(-1).max(1).optional(),
    personId: z.string().trim().optional(),
    person_id: z.string().trim().optional(),
    rfidTag: z.string().trim().optional(),
    rfid_tag: z.string().trim().optional(),
    matched: z.boolean().optional(),
    embedding: embeddingSchema.optional(),
    faceEmbedding: embeddingSchema.optional(),
    face_embedding: embeddingSchema.optional(),
    faceBbox: bboxSchema.optional(),
    face_bbox: bboxSchema.optional(),
  }).superRefine((value, ctx) => {
    if (value.trackId == null && value.track_id == null) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: "trackId is required.",
        path: ["trackId"],
      });
    }
  })),
});

export async function handleFaceIntegration(body: unknown) {
  const input = faceIntegrationSchema.parse(body);
  const timestampMs = input.timestamp ?? input.timestamp_ms ?? Date.now();
  const resolvedFaces = await Promise.all(
    input.tracks.map(async (track) => {
      const trackId = track.trackId ?? track.track_id ?? 0;
      const embedding = track.embedding ?? track.faceEmbedding ?? track.face_embedding;

      addFace({
        track_id: trackId,
        name: track.name,
        confidence: track.confidence,
        embedding,
        timestamp: timestampMs,
      });

      return await resolveFaceTrack({
        deviceId: input.deviceId,
        track_id: trackId,
        name: track.name,
        confidence: track.confidence,
        similarity: track.similarity,
        embedding,
        personId: track.personId ?? track.person_id ?? null,
        rfidTag: track.rfidTag ?? track.rfid_tag ?? null,
        matched: track.matched ?? track.name.trim().toLowerCase() !== "unknown",
        bbox: track.faceBbox ?? track.face_bbox ?? null,
        timestamp: timestampMs,
      });
    }),
  );

  const faces = resolvedFaces
    .filter((resolvedFace): resolvedFace is NonNullable<(typeof resolvedFaces)[number]> => Boolean(resolvedFace))
    .map((resolvedFace) => resolvedFace.observation);

  refreshScoreMatrix(timestampMs);

  return decisionEngine.ingestFace({
    deviceId: input.deviceId,
    faces,
  });
}
