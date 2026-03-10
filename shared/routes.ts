import { z } from 'zod';
import { insertEmployeeSchema, employees, devices, attendances, faceCaptureModeSchema } from './schema';

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
  search: z.string().trim().optional(),
}).optional();
const matchDetailsSchema = z.object({
  primaryConfidence: z.number(),
  anchorAverage: z.number(),
  peakAnchorConfidence: z.number(),
  strongAnchorRatio: z.number(),
  liveConsistency: z.number(),
});

export const errorSchemas = {
  validation: z.object({ message: z.string(), field: z.string().optional() }),
  notFound: z.object({ message: z.string() }),
  internal: z.object({ message: z.string() }),
};

const attendanceWithEmployeeSchema = z.custom<
  typeof attendances.$inferSelect & { employee?: typeof employees.$inferSelect }
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
    update: {
      method: 'PATCH' as const,
      path: '/api/employees/:id' as const,
      input: insertEmployeeSchema.partial(),
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
  scan: {
    rfid: {
      method: 'POST' as const,
      path: '/api/scan-rfid' as const,
      input: z.object({
        rfidUid: z.string(),
        deviceId: z.string(),
        faceDescriptor: z.array(z.number()).optional(),
        faceAnchorDescriptors: z.array(z.array(z.number())).optional(),
        faceConsistency: z.number().min(0).max(1).optional(),
        faceCaptureMode: faceCaptureModeSchema.optional(),
        movementDirection: movementDirectionSchema.optional(),
        movementConfidence: z.number().min(0).max(1).optional(),
      }),
      responses: {
        200: z.object({
          success: z.boolean(),
          message: z.string(),
          employee: z.custom<typeof employees.$inferSelect>().optional(),
          attendance: z.custom<typeof attendances.$inferSelect>().optional(),
          matchConfidence: z.number().optional(),
          matchDetails: matchDetailsSchema.optional(),
          action: z.enum(["ENTRY", "EXIT"]).optional(),
          movementDirection: movementDirectionSchema.optional(),
          movementConfidence: z.number().optional(),
        }),
        400: errorSchemas.validation,
        404: errorSchemas.notFound
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
