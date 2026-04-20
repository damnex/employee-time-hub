import { pgTable, text, serial, integer, boolean, timestamp, jsonb, doublePrecision } from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const faceCaptureModeSchema = z.enum(["detected", "fallback"]);
export const scanTechnologySchema = z.enum(["UHF_RFID"]);
export const movementAxisSchema = z.enum(["horizontal", "depth", "pose", "none"]);
export const gateDecisionSchema = z.enum(["ENTRY", "EXIT", "REJECTED", "UNKNOWN"]);
export const facePoseSchema = z.enum(["front", "left", "right", "up", "down", "unknown"]);

const descriptorSchema = z.array(z.number()).min(1);

const faceProfileV2Schema = z.object({
  version: z.literal(2),
  captureMode: faceCaptureModeSchema.optional(),
  primaryDescriptor: descriptorSchema,
  anchorDescriptors: z.array(descriptorSchema).min(1),
  averageQuality: z.number().min(0).max(1),
  sampleCount: z.number().int().min(1),
  consistency: z.number().min(0).max(1),
});

export const facePoseCoverageSchema = z.object({
  front: z.number().int().min(0),
  left: z.number().int().min(0),
  right: z.number().int().min(0),
  up: z.number().int().min(0),
  down: z.number().int().min(0),
  unknown: z.number().int().min(0),
});

export const facePoseEmbeddingSchema = z.object({
  pose: facePoseSchema,
  descriptor: descriptorSchema,
  sampleCount: z.number().int().min(1),
  averageQuality: z.number().min(0).max(1),
  yaw: z.number(),
  pitch: z.number(),
  roll: z.number(),
  averageLive: z.number().min(0).max(1).optional(),
  averageReal: z.number().min(0).max(1).optional(),
});

export const faceProfileV3Schema = z.object({
  version: z.literal(3),
  captureMode: faceCaptureModeSchema.optional(),
  engine: z.object({
    provider: z.literal("human"),
    libraryVersion: z.string(),
    descriptionModel: z.string(),
    detectorModel: z.string(),
    meshModel: z.string(),
    irisModel: z.string().optional(),
    livenessModel: z.string().optional(),
    antispoofModel: z.string().optional(),
  }),
  primaryDescriptor: descriptorSchema,
  anchorDescriptors: z.array(descriptorSchema).min(1),
  averageQuality: z.number().min(0).max(1),
  sampleCount: z.number().int().min(1),
  consistency: z.number().min(0).max(1),
  poseEmbeddings: z.array(facePoseEmbeddingSchema).min(1),
  poseCoverage: facePoseCoverageSchema,
  requiredPoses: z.array(facePoseSchema).min(1),
  averageLive: z.number().min(0).max(1).optional(),
  averageReal: z.number().min(0).max(1).optional(),
  orientationSpread: z.object({
    minYaw: z.number(),
    maxYaw: z.number(),
    minPitch: z.number(),
    maxPitch: z.number(),
    minRoll: z.number(),
    maxRoll: z.number(),
  }),
});

export const faceProfileSchema = z.union([
  faceProfileV2Schema,
  faceProfileV3Schema,
]);

export const employees = pgTable("employees", {
  id: serial("id").primaryKey(),
  employeeCode: text("employee_code").notNull().unique(),
  name: text("name").notNull(),
  department: text("department").notNull(),
  phone: text("phone"),
  email: text("email"),
  rfidUid: text("rfid_uid").notNull().unique(),
  faceDescriptor: jsonb("face_descriptor"), // Legacy descriptor array or structured face profile
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
});

export const devices = pgTable("devices", {
  id: serial("id").primaryKey(),
  deviceId: text("device_id").notNull().unique(),
  location: text("location").notNull(),
  deviceType: text("device_type").notNull(),
});

export const attendances = pgTable("attendances", {
  id: serial("id").primaryKey(),
  employeeId: integer("employee_id").notNull().references(() => employees.id),
  date: text("date").notNull(), // YYYY-MM-DD
  entryTime: timestamp("entry_time"),
  exitTime: timestamp("exit_time"),
  workingHours: doublePrecision("working_hours"),
  verificationStatus: text("verification_status").notNull(), // 'ENTRY', 'EXIT', 'FAILED_FACE', 'FAILED_DIRECTION', 'UNKNOWN_RFID'
  deviceId: text("device_id").notNull(),
});

export const gateEvents = pgTable("gate_events", {
  id: serial("id").primaryKey(),
  employeeId: integer("employee_id").references(() => employees.id),
  date: text("date").notNull(), // YYYY-MM-DD
  occurredAt: timestamp("occurred_at").defaultNow(),
  rfidUid: text("rfid_uid").notNull(),
  deviceId: text("device_id").notNull(),
  scanTechnology: text("scan_technology").notNull().default("UHF_RFID"),
  decision: text("decision").notNull(), // 'ENTRY', 'EXIT', 'REJECTED', 'UNKNOWN'
  verificationStatus: text("verification_status").notNull(), // 'ENTRY', 'EXIT', 'FAILED_FACE', 'FAILED_DIRECTION', 'UNKNOWN_RFID'
  eventMessage: text("event_message").notNull(),
  movementDirection: text("movement_direction"),
  movementAxis: text("movement_axis"),
  movementConfidence: doublePrecision("movement_confidence"),
  matchConfidence: doublePrecision("match_confidence"),
  faceQuality: doublePrecision("face_quality"),
  faceConsistency: doublePrecision("face_consistency"),
  faceCaptureMode: text("face_capture_mode"),
});

export const employeesRelations = relations(employees, ({ many }) => ({
  attendances: many(attendances),
  gateEvents: many(gateEvents),
}));

export const attendancesRelations = relations(attendances, ({ one }) => ({
  employee: one(employees, {
    fields: [attendances.employeeId],
    references: [employees.id],
  }),
}));

export const gateEventsRelations = relations(gateEvents, ({ one }) => ({
  employee: one(employees, {
    fields: [gateEvents.employeeId],
    references: [employees.id],
  }),
}));

export const insertEmployeeSchema = createInsertSchema(employees).omit({ id: true, createdAt: true });
export const insertDeviceSchema = createInsertSchema(devices).omit({ id: true });
export const insertAttendanceSchema = createInsertSchema(attendances).omit({ id: true });
export const insertGateEventSchema = createInsertSchema(gateEvents).omit({ id: true, occurredAt: true });

export type Employee = typeof employees.$inferSelect;
export type InsertEmployee = z.infer<typeof insertEmployeeSchema>;
export type Device = typeof devices.$inferSelect;
export type InsertDevice = z.infer<typeof insertDeviceSchema>;
export type Attendance = typeof attendances.$inferSelect;
export type InsertAttendance = z.infer<typeof insertAttendanceSchema>;
export type GateEvent = typeof gateEvents.$inferSelect;
export type InsertGateEvent = z.infer<typeof insertGateEventSchema>;
export type FaceProfile = z.infer<typeof faceProfileSchema>;
export type FaceProfileV2 = z.infer<typeof faceProfileV2Schema>;
export type FaceProfileV3 = z.infer<typeof faceProfileV3Schema>;
export type FaceCaptureMode = z.infer<typeof faceCaptureModeSchema>;
export type FacePose = z.infer<typeof facePoseSchema>;
export type FacePoseCoverage = z.infer<typeof facePoseCoverageSchema>;
export type ScanTechnology = z.infer<typeof scanTechnologySchema>;
export type MovementAxis = z.infer<typeof movementAxisSchema>;
export type GateDecision = z.infer<typeof gateDecisionSchema>;

export type CreateEmployeeRequest = InsertEmployee;
export type UpdateEmployeeRequest = Partial<InsertEmployee>;

export function isFaceProfile(value: unknown): value is FaceProfile {
  return faceProfileSchema.safeParse(value).success;
}
