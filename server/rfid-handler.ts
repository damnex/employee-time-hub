import { z } from "zod";
import { decisionEngine } from "./decision-engine";
import { scanTechnologySchema } from "@shared/schema";
import { addRFID } from "./buffer";
import { refreshScoreMatrix } from "./score-matrix";

export const rfidIntegrationSchema = z.object({
  deviceId: z.string().trim().min(1),
  rfidTag: z.string().trim().min(1),
  timestamp: z.number().int().positive().optional(),
  scanTechnology: scanTechnologySchema.optional(),
});

export async function handleRfidIntegration(body: unknown) {
  const input = rfidIntegrationSchema.parse(body);
  const timestampMs = input.timestamp ?? Date.now();
  const normalizedTag = input.rfidTag.trim().toUpperCase();

  addRFID({
    tag: normalizedTag,
    timestamp: timestampMs,
  });
  refreshScoreMatrix(timestampMs);

  return decisionEngine.ingestRfid({
    deviceId: input.deviceId,
    rfidTag: normalizedTag,
    timestampMs,
    scanTechnology: input.scanTechnology,
  });
}
