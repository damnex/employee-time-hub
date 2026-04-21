import { z } from "zod";
import { decisionEngine } from "./decision-engine";

const bboxSchema = z.tuple([
  z.number(),
  z.number(),
  z.number(),
  z.number(),
]);

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
  return decisionEngine.ingestFace({
    deviceId: input.deviceId,
    faces: input.tracks.map((track) => ({
      deviceId: input.deviceId,
      trackId: track.trackId ?? track.track_id ?? 0,
      timestampMs,
      name: track.name,
      confidence: track.confidence,
      similarity: track.similarity,
      personId: track.personId ?? track.person_id ?? null,
      rfidTag: track.rfidTag ?? track.rfid_tag ?? null,
      matched: track.matched ?? track.name.trim().toLowerCase() !== "unknown",
      bbox: track.faceBbox ?? track.face_bbox ?? null,
    })),
  });
}
