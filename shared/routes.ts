import { z } from 'zod';
import { insertEmployeeSchema, employees, devices, attendances } from './schema';

export const errorSchemas = {
  validation: z.object({ message: z.string(), field: z.string().optional() }),
  notFound: z.object({ message: z.string() }),
  internal: z.object({ message: z.string() }),
};

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
      input: z.object({
        date: z.string().optional(),
        employeeId: z.coerce.number().optional()
      }).optional(),
      responses: { 200: z.array(z.custom<typeof attendances.$inferSelect & { employee?: typeof employees.$inferSelect }>()) },
    }
  },
  scan: {
    rfid: {
      method: 'POST' as const,
      path: '/api/scan-rfid' as const,
      input: z.object({
        rfidUid: z.string(),
        deviceId: z.string(),
        faceDescriptor: z.array(z.number()).optional()
      }),
      responses: {
        200: z.object({
          success: z.boolean(),
          message: z.string(),
          employee: z.custom<typeof employees.$inferSelect>().optional(),
          attendance: z.custom<typeof attendances.$inferSelect>().optional(),
          matchConfidence: z.number().optional()
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
        200: z.object({
          totalEmployees: z.number(),
          presentToday: z.number(),
          absentToday: z.number(),
          recentScans: z.array(z.custom<typeof attendances.$inferSelect & { employee?: typeof employees.$inferSelect }>())
        })
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
