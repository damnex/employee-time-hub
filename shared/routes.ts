import { z } from 'zod';
import {
  insertEmployeeSchema,
  employees,
  devices,
  attendances,
  gateEvents,
  faceCaptureModeSchema,
  facePoseSchema,
  movementAxisSchema,
  scanTechnologySchema,
} from './schema';

const movementDirectionSchema = z.enum(["ENTRY", "EXIT", "UNKNOWN"]);
const attendanceStatusSchema = z.enum([
  "ENTRY",
  "EXIT",
  "FAILED_FACE",
  "FAILED_DIRECTION",
  "UNKNOWN_RFID",
]);
const attendanceFiltersSchema = z.object({
  date: z.string().optional(),
  dateFrom: z.string().optional(),
  dateTo: z.string().optional(),
  employeeId: z.coerce.number().optional(),
  status: attendanceStatusSchema.optional(),
  department: z.string().trim().optional(),
  deviceId: z.string().trim().optional(),
  search: z.string().trim().optional(),
}).optional();
const gateEventFiltersSchema = z.object({
  date: z.string().optional(),
  dateFrom: z.string().optional(),
  dateTo: z.string().optional(),
  employeeId: z.coerce.number().optional(),
  status: attendanceStatusSchema.optional(),
  department: z.string().trim().optional(),
  deviceId: z.string().trim().optional(),
  technology: scanTechnologySchema.optional(),
  movementDirection: movementDirectionSchema.optional(),
  search: z.string().trim().optional(),
}).optional();
const matchDetailsSchema = z.object({
  primaryConfidence: z.number(),
  anchorAverage: z.number(),
  peakAnchorConfidence: z.number(),
  strongAnchorRatio: z.number(),
  liveConsistency: z.number(),
  poseConfidence: z.number().optional(),
  liveLiveness: z.number().optional(),
  liveRealness: z.number().optional(),
});
const faceBoxSchema = z.object({
  top: z.number(),
  right: z.number(),
  bottom: z.number(),
  left: z.number(),
});
const liveRecognizedFaceSchema = z.object({
  label: z.string(),
  employeeCode: z.string().nullish(),
  department: z.string().nullish(),
  rfidUid: z.string().nullish(),
  confidence: z.number(),
  distance: z.number().nullish(),
  verified: z.boolean(),
  box: faceBoxSchema,
});

export const errorSchemas = {
  validation: z.object({ message: z.string(), field: z.string().optional() }),
  notFound: z.object({ message: z.string() }),
  internal: z.object({ message: z.string() }),
};

const attendanceWithEmployeeSchema = z.custom<
  typeof attendances.$inferSelect & { employee?: typeof employees.$inferSelect }
>();
const gateEventWithEmployeeSchema = z.custom<
  typeof gateEvents.$inferSelect & { employee?: typeof employees.$inferSelect }
>();

const dashboardStatsResponseSchema = z.object({
  totalEmployees: z.number(),
  presentToday: z.number(),
  absentToday: z.number(),
  recentScans: z.array(attendanceWithEmployeeSchema),
});

export const api = {
  employees: {
    list: {
      method: 'GET' as const,
      path: '/api/employees' as const,
      responses: { 200: z.array(z.custom<typeof employees.$inferSelect>()) },
    },
    get: {
      method: 'GET' as const,
      path: '/api/employees/:id' as const,
      responses: { 200: z.custom<typeof employees.$inferSelect>(), 404: errorSchemas.notFound },
    },
    create: {
      method: 'POST' as const,
      path: '/api/employees' as const,
      input: insertEmployeeSchema,
      responses: { 201: z.custom<typeof employees.$inferSelect>(), 400: errorSchemas.validation },
    },
    enrollPython: {
      method: 'POST' as const,
      path: '/api/employees/enroll-python' as const,
      input: z.object({
        employeeCode: z.string().trim().min(1),
        name: z.string().trim().min(1),
        department: z.string().trim().min(1),
        phone: z.string().trim().optional(),
        email: z.string().trim().optional(),
        rfidUid: z.string().trim().min(1),
        isActive: z.boolean().optional(),
        datasetPhotos: z.array(z.string()).min(12).max(100),
        profilePhoto: z.string().trim().optional(),
      }),
      responses: { 201: z.custom<typeof employees.$inferSelect>(), 400: errorSchemas.validation },
    },
    update: {
      method: 'PATCH' as const,
      path: '/api/employees/:id' as const,
      input: insertEmployeeSchema.partial().extend({
        profilePhoto: z.string().trim().optional(),
      }),
      responses: { 200: z.custom<typeof employees.$inferSelect>(), 400: errorSchemas.validation, 404: errorSchemas.notFound },
    },
    delete: {
      method: 'DELETE' as const,
      path: '/api/employees/:id' as const,
      responses: { 200: z.custom<typeof employees.$inferSelect>(), 404: errorSchemas.notFound },
    }
  },
  attendances: {
    list: {
      method: 'GET' as const,
      path: '/api/attendances' as const,
      input: attendanceFiltersSchema,
      responses: { 200: z.array(attendanceWithEmployeeSchema) },
    }
  },
  gateEvents: {
    list: {
      method: 'GET' as const,
      path: '/api/gate-events' as const,
      input: gateEventFiltersSchema,
      responses: { 200: z.array(gateEventWithEmployeeSchema) },
    }
  },
  scan: {
    rfid: {
      method: 'POST' as const,
      path: '/api/scan-rfid' as const,
      input: z.object({
        rfidUid: z.string(),
        deviceId: z.string(),
        faceFrames: z.array(z.string()).min(3).max(16).optional(),
        faceDescriptor: z.array(z.number()).optional(),
        faceAnchorDescriptors: z.array(z.array(z.number())).optional(),
        faceConsistency: z.number().min(0).max(1).optional(),
        faceQuality: z.number().min(0).max(1).optional(),
        faceCaptureMode: faceCaptureModeSchema.optional(),
        facePose: facePoseSchema.optional(),
        faceYaw: z.number().optional(),
        facePitch: z.number().optional(),
        faceRoll: z.number().optional(),
        faceLiveConfidence: z.number().min(0).max(1).optional(),
        faceRealConfidence: z.number().min(0).max(1).optional(),
        scanTechnology: scanTechnologySchema.optional(),
        movementDirection: movementDirectionSchema.optional(),
        movementAxis: movementAxisSchema.optional(),
        movementConfidence: z.number().min(0).max(1).optional(),
      }),
      responses: {
        200: z.object({
          success: z.boolean(),
          ignored: z.boolean().optional(),
          message: z.string(),
          employee: z.custom<typeof employees.$inferSelect>().optional(),
          attendance: z.custom<typeof attendances.$inferSelect>().optional(),
          matchConfidence: z.number().optional(),
          matchDetails: matchDetailsSchema.optional(),
          action: z.enum(["ENTRY", "EXIT"]).optional(),
          movementDirection: movementDirectionSchema.optional(),
          movementConfidence: z.number().optional(),
          detectedFaceLabel: z.string().optional(),
          detectedFaceBox: faceBoxSchema.nullish(),
        }),
        400: errorSchemas.validation,
        404: errorSchemas.notFound
      }
    },
    liveFaces: {
      method: 'POST' as const,
      path: '/api/scan/live-faces' as const,
      input: z.object({
        deviceId: z.string().trim().min(1),
        frame: z.string().trim().min(1),
        maxFaces: z.number().int().min(1).max(50).optional(),
      }),
      responses: {
        200: z.object({
          success: z.boolean(),
          message: z.string(),
          processedAt: z.string(),
          frameWidth: z.number().optional(),
          frameHeight: z.number().optional(),
          faces: z.array(liveRecognizedFaceSchema),
        }),
        400: errorSchemas.validation,
      }
    }
  },
  stats: {
    dashboard: {
      method: 'GET' as const,
      path: '/api/stats' as const,
      responses: {
        200: dashboardStatsResponseSchema,
      }
    }
  }
};

export function buildUrl(path: string, params?: Record<string, string | number>): string {
  let url = path;
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (url.includes(`:${key}`)) {
        url = url.replace(`:${key}`, String(value));
      }
    });
  }
  return url;
}

