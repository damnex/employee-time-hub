import { z } from "zod";
import { decisionEngine } from "./decision-engine";

const bboxSchema = z.tuple([
  z.number(),
  z.number(),
  z.number(),
  z.number(),
]);

const centerSchema = z.tuple([z.number(), z.number()]);

const trackIntegrationSchema = z.object({
  trackId: z.number().int().nonnegative().optional(),
  track_id: z.number().int().nonnegative().optional(),
  bbox: bboxSchema,
  center: centerSchema,
  confidence: z.number().min(0).max(1).optional(),
  direction: z.enum(["ENTRY", "EXIT", "UNKNOWN"]).optional(),
  zone: z.enum(["ENTRY_ZONE", "EXIT_ZONE", "BUFFER_ZONE", "entry", "exit", "buffer"]).optional(),
  ageFrames: z.number().int().nonnegative().optional(),
  age_frames: z.number().int().nonnegative().optional(),
  stable: z.boolean().optional(),
  frameWidth: z.number().int().positive().optional(),
  frameHeight: z.number().int().positive().optional(),
}).superRefine((value, ctx) => {
  if (value.trackId == null && value.track_id == null) {
    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      message: "trackId is required.",
      path: ["trackId"],
    });
  }
});

export const visionIntegrationSchema = z.object({
  deviceId: z.string().trim().min(1),
  timestamp: z.number().int().positive().optional(),
  timestamp_ms: z.number().int().positive().optional(),
  tracks: z.array(trackIntegrationSchema),
});

export async function handleVisionIntegration(body: unknown) {
  const input = visionIntegrationSchema.parse(body);
  const timestampMs = input.timestamp ?? input.timestamp_ms ?? Date.now();
  return decisionEngine.ingestVision({
    deviceId: input.deviceId,
    tracks: input.tracks.map((track) => ({
      deviceId: input.deviceId,
      trackId: track.trackId ?? track.track_id ?? 0,
      timestampMs,
      bbox: track.bbox,
      center: track.center,
      confidence: track.confidence,
      direction: track.direction ?? "UNKNOWN",
      zone: track.zone as "ENTRY_ZONE" | "EXIT_ZONE" | "BUFFER_ZONE" | undefined,
      ageFrames: track.ageFrames ?? track.age_frames,
      stable: track.stable,
      frameWidth: track.frameWidth,
      frameHeight: track.frameHeight,
    })),
  });
}
