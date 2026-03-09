import { pgTable, text, serial, integer, boolean, timestamp, jsonb, doublePrecision } from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const employees = pgTable("employees", {
  id: serial("id").primaryKey(),
  employeeCode: text("employee_code").notNull().unique(),
  name: text("name").notNull(),
  department: text("department").notNull(),
  phone: text("phone"),
  email: text("email"),
  rfidUid: text("rfid_uid").notNull().unique(),
  faceDescriptor: jsonb("face_descriptor"), // Array of 128 numbers representing the face
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
  verificationStatus: text("verification_status").notNull(), // 'ENTRY', 'EXIT', 'FAILED_FACE', 'UNKNOWN_RFID'
  deviceId: text("device_id").notNull(),
});

export const employeesRelations = relations(employees, ({ many }) => ({
  attendances: many(attendances),
}));

export const attendancesRelations = relations(attendances, ({ one }) => ({
  employee: one(employees, {
    fields: [attendances.employeeId],
    references: [employees.id],
  }),
}));

export const insertEmployeeSchema = createInsertSchema(employees).omit({ id: true, createdAt: true });
export const insertDeviceSchema = createInsertSchema(devices).omit({ id: true });
export const insertAttendanceSchema = createInsertSchema(attendances).omit({ id: true });

export type Employee = typeof employees.$inferSelect;
export type InsertEmployee = z.infer<typeof insertEmployeeSchema>;
export type Device = typeof devices.$inferSelect;
export type InsertDevice = z.infer<typeof insertDeviceSchema>;
export type Attendance = typeof attendances.$inferSelect;
export type InsertAttendance = z.infer<typeof insertAttendanceSchema>;

export type CreateEmployeeRequest = InsertEmployee;
export type UpdateEmployeeRequest = Partial<InsertEmployee>;
