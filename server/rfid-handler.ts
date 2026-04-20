import { z } from "zod";
import { decisionEngine } from "./decision-engine";
import { scanTechnologySchema } from "@shared/schema";

export const rfidIntegrationSchema = z.object({
  deviceId: z.string().trim().min(1),
  rfidTag: z.string().trim().min(1),
  timestamp: z.number().int().positive().optional(),
  scanTechnology: scanTechnologySchema.optional(),
});

export async function handleRfidIntegration(body: unknown) {
  const input = rfidIntegrationSchema.parse(body);
  return decisionEngine.ingestRfid({
    deviceId: input.deviceId,
    rfidTag: input.rfidTag,
    timestampMs: input.timestamp,
    scanTechnology: input.scanTechnology,
  });
}
