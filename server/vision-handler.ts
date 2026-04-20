import { z } from "zod";
import { decisionEngine } from "./decision-engine";

const bboxSchema = z.tuple([
  z.number(),
  z.number(),
  z.number(),
  z.number(),
]);

const centerSchema = z.tuple([z.number(), z.number()]);

export const visionIntegrationSchema = z.object({
  deviceId: z.string().trim().min(1),
  timestamp: z.number().int().positive().optional(),
  tracks: z.array(z.object({
    trackId: z.number().int().nonnegative(),
    bbox: bboxSchema,
    center: centerSchema,
    direction: z.enum(["ENTRY", "EXIT", "UNKNOWN"]).optional(),
    frameWidth: z.number().int().positive().optional(),
    frameHeight: z.number().int().positive().optional(),
  })),
});

export async function handleVisionIntegration(body: unknown) {
  const input = visionIntegrationSchema.parse(body);
  const timestampMs = input.timestamp ?? Date.now();
  return decisionEngine.ingestVision({
    deviceId: input.deviceId,
    tracks: input.tracks.map((track) => ({
      deviceId: input.deviceId,
      trackId: track.trackId,
      timestampMs,
      bbox: track.bbox,
      center: track.center,
      direction: track.direction ?? "UNKNOWN",
      frameWidth: track.frameWidth,
      frameHeight: track.frameHeight,
    })),
  });
}
