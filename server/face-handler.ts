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
  tracks: z.array(z.object({
    trackId: z.number().int().nonnegative(),
    name: z.string().trim().default("unknown"),
    confidence: z.number().min(0).max(1),
    personId: z.string().trim().optional(),
    rfidTag: z.string().trim().optional(),
    matched: z.boolean().optional(),
    faceBbox: bboxSchema.optional(),
  })),
});

export async function handleFaceIntegration(body: unknown) {
  const input = faceIntegrationSchema.parse(body);
  const timestampMs = input.timestamp ?? Date.now();
  return decisionEngine.ingestFace({
    deviceId: input.deviceId,
    faces: input.tracks.map((track) => ({
      deviceId: input.deviceId,
      trackId: track.trackId,
      timestampMs,
      name: track.name,
      confidence: track.confidence,
      personId: track.personId ?? null,
      rfidTag: track.rfidTag ?? null,
      matched: track.matched ?? track.name.trim().toLowerCase() !== "unknown",
      bbox: track.faceBbox ?? null,
    })),
  });
}
