import { z } from "zod";
import { decisionEngine } from "./decision-engine";
import { addTrack } from "./buffer";
import { updateTrack } from "./trackState";
import { refreshScoreMatrix } from "./score-matrix";

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
  const tracks = input.tracks.map((track) => {
    const trackId = track.trackId ?? track.track_id ?? 0;
    const bufferedTrack = addTrack({
      track_id: trackId,
      position: {
        x: track.center[0],
        y: track.center[1],
      },
      timestamp: timestampMs,
    });
    const trackState = updateTrack({
      track_id: bufferedTrack.track_id,
      position: bufferedTrack.position,
      timestamp: bufferedTrack.timestamp,
    });

    return {
      deviceId: input.deviceId,
      trackId,
      timestampMs,
      bbox: track.bbox,
      center: track.center,
      confidence: track.confidence,
      direction: track.direction ?? trackState.direction,
      zone: track.zone as "ENTRY_ZONE" | "EXIT_ZONE" | "BUFFER_ZONE" | undefined,
      ageFrames: track.ageFrames ?? track.age_frames,
      stable: track.stable ?? trackState.stable,
      frameWidth: track.frameWidth,
      frameHeight: track.frameHeight,
    };
  });

  refreshScoreMatrix(timestampMs);

  return decisionEngine.ingestVision({
    deviceId: input.deviceId,
    tracks,
  });
}
