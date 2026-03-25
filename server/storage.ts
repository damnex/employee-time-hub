import { db, pool, shouldUseDatabaseStorage, verifyDatabaseConnection } from "./db";
import { eq, and, desc, gte, ilike, lte, or, sql } from "drizzle-orm";
import {
  employees,
  devices,
  attendances,
  gateEvents,
  type Employee,
  type InsertEmployee,
  type Device,
  type InsertDevice,
  type Attendance,
  type InsertAttendance,
  type GateEvent,
  type InsertGateEvent,
} from "@shared/schema";

export interface IStorage {
  // Employees
  getEmployees(): Promise<Employee[]>;
  getEmployee(id: number): Promise<Employee | undefined>;
  getEmployeeByRfid(rfidUid: string): Promise<Employee | undefined>;
  createEmployee(employee: InsertEmployee): Promise<Employee>;
  updateEmployee(id: number, updates: Partial<InsertEmployee>): Promise<Employee | undefined>;
  deleteEmployee(id: number): Promise<Employee | undefined>;
  
  // Devices
  getDevice(deviceId: string): Promise<Device | undefined>;
  createDevice(device: InsertDevice): Promise<Device>;
  
  // Attendances
  getAttendances(filters?: AttendanceFilters): Promise<(Attendance & { employee?: Employee })[]>;
  getOpenAttendance(employeeId: number, date: string): Promise<Attendance | undefined>;
  createAttendance(attendance: InsertAttendance): Promise<Attendance>;
  updateAttendance(id: number, updates: Partial<InsertAttendance>): Promise<Attendance | undefined>;

  // Gate events
  getGateEvents(filters?: GateEventFilters): Promise<(GateEvent & { employee?: Employee })[]>;
  createGateEvent(event: InsertGateEvent): Promise<GateEvent>;
  
  // Stats
  getDashboardStats(): Promise<{ totalEmployees: number; presentToday: number; absentToday: number; recentScans: (Attendance & { employee?: Employee })[] }>;
}

type AttendanceWithEmployee = Attendance & { employee?: Employee };
type GateEventWithEmployee = GateEvent & { employee?: Employee };
export interface AttendanceFilters {
  date?: string;
  dateFrom?: string;
  dateTo?: string;
  employeeId?: number;
  status?: Attendance["verificationStatus"];
  department?: string;
  deviceId?: string;
  search?: string;
}

export interface GateEventFilters {
  date?: string;
  dateFrom?: string;
  dateTo?: string;
  employeeId?: number;
  status?: GateEvent["verificationStatus"];
  department?: string;
  deviceId?: string;
  technology?: GateEvent["scanTechnology"];
  movementDirection?: GateEvent["movementDirection"];
  search?: string;
}

function requireDb() {
  if (!db) {
    throw new Error("DATABASE_URL is not configured.");
  }

  return db;
}

const GATE_EVENTS_SCHEMA_RECHECK_MS = 30_000;
let gateEventsAvailability: "unknown" | "available" | "unavailable" = "unknown";
let gateEventsAvailabilityCheckedAt = 0;

function isGateEventsSchemaError(error: unknown) {
  if (!error || typeof error !== "object") {
    return false;
  }

  const errorWithCode = error as { code?: string; message?: string };
  const message = errorWithCode.message?.toLowerCase() ?? "";
  return (
    errorWithCode.code === "42P01"
    || errorWithCode.code === "42703"
    || errorWithCode.code === "42704"
    || message.includes("gate_events")
    || message.includes("scan_technology")
    || message.includes("movement_axis")
    || message.includes("face_capture_mode")
  );
}

function markGateEventsUnavailable(reason?: unknown) {
  gateEventsAvailabilityCheckedAt = Date.now();

  if (gateEventsAvailability === "unavailable") {
    return;
  }

  gateEventsAvailability = "unavailable";

  const errorMessage =
    reason && typeof reason === "object" && "message" in reason && typeof reason.message === "string"
      ? ` ${reason.message}`
      : "";

  console.warn(
    `[storage] gate_events schema is unavailable. Raw gate event logging is disabled until the database is updated.${errorMessage}`,
  );
}

async function canUseGateEvents() {
  if (!pool) {
    return false;
  }

  const now = Date.now();
  if (gateEventsAvailability === "available") {
    return true;
  }

  if (
    gateEventsAvailability === "unavailable"
    && now - gateEventsAvailabilityCheckedAt < GATE_EVENTS_SCHEMA_RECHECK_MS
  ) {
    return false;
  }

  try {
    const result = await pool.query("select to_regclass('public.gate_events') as gate_events");
    gateEventsAvailabilityCheckedAt = now;
    if (!result.rows[0]?.gate_events) {
      markGateEventsUnavailable();
      return false;
    }

    gateEventsAvailability = "available";
    return true;
  } catch (error) {
    if (isGateEventsSchemaError(error)) {
      markGateEventsUnavailable(error);
      return false;
    }

    throw error;
  }
}

export class DatabaseStorage implements IStorage {
  async getEmployees(): Promise<Employee[]> {
    return await requireDb().select().from(employees).orderBy(employees.id);
  }

  async getEmployee(id: number): Promise<Employee | undefined> {
    const [employee] = await requireDb().select().from(employees).where(eq(employees.id, id));
    return employee;
  }

  async getEmployeeByRfid(rfidUid: string): Promise<Employee | undefined> {
    const [employee] = await requireDb().select().from(employees).where(eq(employees.rfidUid, rfidUid));
    return employee;
  }

  async createEmployee(employee: InsertEmployee): Promise<Employee> {
    const [newEmployee] = await requireDb().insert(employees).values(employee).returning();
    return newEmployee;
  }

  async updateEmployee(id: number, updates: Partial<InsertEmployee>): Promise<Employee | undefined> {
    const [updated] = await requireDb().update(employees).set(updates).where(eq(employees.id, id)).returning();
    return updated;
  }

  async deleteEmployee(id: number): Promise<Employee | undefined> {
    if (await canUseGateEvents()) {
      try {
        await requireDb().delete(gateEvents).where(eq(gateEvents.employeeId, id));
      } catch (error) {
        if (isGateEventsSchemaError(error)) {
          markGateEventsUnavailable(error);
        } else {
          throw error;
        }
      }
    }

    return await requireDb().transaction(async (tx) => {
      await tx.delete(attendances).where(eq(attendances.employeeId, id));
      const [deletedEmployee] = await tx.delete(employees).where(eq(employees.id, id)).returning();
      return deletedEmployee;
    });
  }

  async getDevice(deviceId: string): Promise<Device | undefined> {
    const [device] = await requireDb().select().from(devices).where(eq(devices.deviceId, deviceId));
    return device;
  }

  async createDevice(device: InsertDevice): Promise<Device> {
    const [newDevice] = await requireDb().insert(devices).values(device).returning();
    return newDevice;
  }

  async getAttendances(filters: AttendanceFilters = {}): Promise<(Attendance & { employee?: Employee })[]> {
    let query = requireDb().select({
      attendance: attendances,
      employee: employees
    })
    .from(attendances)
    .leftJoin(employees, eq(attendances.employeeId, employees.id))
    .orderBy(desc(attendances.id));

    const conditions = [];
    if (filters.date) {
      conditions.push(eq(attendances.date, filters.date));
    } else {
      if (filters.dateFrom) conditions.push(gte(attendances.date, filters.dateFrom));
      if (filters.dateTo) conditions.push(lte(attendances.date, filters.dateTo));
    }
    if (filters.employeeId) conditions.push(eq(attendances.employeeId, filters.employeeId));
    if (filters.status) conditions.push(eq(attendances.verificationStatus, filters.status));
    if (filters.department) conditions.push(eq(employees.department, filters.department));
    if (filters.deviceId) conditions.push(eq(attendances.deviceId, filters.deviceId));
    if (filters.search) {
      const searchPattern = `%${filters.search}%`;
      conditions.push(or(
        ilike(employees.name, searchPattern),
        ilike(employees.employeeCode, searchPattern),
        ilike(employees.department, searchPattern),
        ilike(employees.rfidUid, searchPattern),
        ilike(attendances.deviceId, searchPattern),
        ilike(attendances.verificationStatus, searchPattern),
      ));
    }

    if (conditions.length > 0) {
      // @ts-ignore
      query = query.where(and(...conditions));
    }

    const results = await query;
    return results.map(row => ({
      ...row.attendance,
      employee: row.employee || undefined
    }));
  }

  async getGateEvents(filters: GateEventFilters = {}): Promise<GateEventWithEmployee[]> {
    if (!(await canUseGateEvents())) {
      return [];
    }

    try {
      let query = requireDb().select({
        gateEvent: gateEvents,
        employee: employees,
      })
      .from(gateEvents)
      .leftJoin(employees, eq(gateEvents.employeeId, employees.id))
      .orderBy(desc(gateEvents.id));

      const conditions = [];
      if (filters.date) {
        conditions.push(eq(gateEvents.date, filters.date));
      } else {
        if (filters.dateFrom) conditions.push(gte(gateEvents.date, filters.dateFrom));
        if (filters.dateTo) conditions.push(lte(gateEvents.date, filters.dateTo));
      }
      if (filters.employeeId) conditions.push(eq(gateEvents.employeeId, filters.employeeId));
      if (filters.status) conditions.push(eq(gateEvents.verificationStatus, filters.status));
      if (filters.department) conditions.push(eq(employees.department, filters.department));
      if (filters.deviceId) conditions.push(eq(gateEvents.deviceId, filters.deviceId));
      if (filters.technology) conditions.push(eq(gateEvents.scanTechnology, filters.technology));
      if (filters.movementDirection) conditions.push(eq(gateEvents.movementDirection, filters.movementDirection));
      if (filters.search) {
        const searchPattern = `%${filters.search}%`;
        conditions.push(or(
          ilike(employees.name, searchPattern),
          ilike(employees.employeeCode, searchPattern),
          ilike(employees.department, searchPattern),
          ilike(employees.rfidUid, searchPattern),
          ilike(gateEvents.rfidUid, searchPattern),
          ilike(gateEvents.deviceId, searchPattern),
          ilike(gateEvents.eventMessage, searchPattern),
        ));
      }

      if (conditions.length > 0) {
        // @ts-ignore
        query = query.where(and(...conditions));
      }

      const results = await query;
      return results.map((row) => ({
        ...row.gateEvent,
        employee: row.employee || undefined,
      }));
    } catch (error) {
      if (isGateEventsSchemaError(error)) {
        markGateEventsUnavailable(error);
        return [];
      }

      throw error;
    }
  }

  async getOpenAttendance(employeeId: number, date: string): Promise<Attendance | undefined> {
    const [attendance] = await requireDb().select()
      .from(attendances)
      .where(
        and(
          eq(attendances.employeeId, employeeId),
          eq(attendances.date, date),
          eq(attendances.verificationStatus, "ENTRY"),
          sql`${attendances.exitTime} IS NULL`
        )
      )
      .orderBy(desc(attendances.id))
      .limit(1);
    return attendance;
  }

  async createAttendance(attendance: InsertAttendance): Promise<Attendance> {
    const [newAttendance] = await requireDb().insert(attendances).values(attendance).returning();
    return newAttendance;
  }

  async createGateEvent(event: InsertGateEvent): Promise<GateEvent> {
    const [newEvent] = await requireDb().insert(gateEvents).values(event).returning();
    return newEvent;
  }

  async updateAttendance(id: number, updates: Partial<InsertAttendance>): Promise<Attendance | undefined> {
    const [updated] = await requireDb().update(attendances).set(updates).where(eq(attendances.id, id)).returning();
    return updated;
  }

  async getDashboardStats() {
    const today = new Date().toISOString().split('T')[0];
    
    // Total employees
    const [totalEmpResult] = await requireDb().select({ count: sql`count(*)`.mapWith(Number) }).from(employees).where(eq(employees.isActive, true));
    const totalEmployees = totalEmpResult?.count || 0;

    // Present today (has an entry today with verification_status ENTRY or EXIT)
    const presentRows = await requireDb().select({ count: sql`count(distinct ${attendances.employeeId})`.mapWith(Number) })
      .from(attendances)
      .where(and(
        eq(attendances.date, today),
        sql`${attendances.verificationStatus} IN ('ENTRY', 'EXIT')`
      ));
    const presentToday = presentRows[0]?.count || 0;
    
    const absentToday = Math.max(0, totalEmployees - presentToday);

    // Recent scans
    const recentScans = await requireDb().select({
      attendance: attendances,
      employee: employees
    })
    .from(attendances)
    .leftJoin(employees, eq(attendances.employeeId, employees.id))
    .orderBy(desc(attendances.id))
    .limit(10);

    return {
      totalEmployees,
      presentToday,
      absentToday,
      recentScans: recentScans.map(row => ({
        ...row.attendance,
        employee: row.employee || undefined
      }))
    };
  }
}

export class MemoryStorage implements IStorage {
  private employeeIdSequence = 1;
  private deviceIdSequence = 1;
  private attendanceIdSequence = 1;
  private gateEventIdSequence = 1;

  private employeeStore = new Map<number, Employee>();
  private deviceStore = new Map<string, Device>();
  private attendanceStore = new Map<number, Attendance>();
  private gateEventStore = new Map<number, GateEvent>();

  async getEmployees(): Promise<Employee[]> {
    return Array.from(this.employeeStore.values()).sort((a, b) => a.id - b.id);
  }

  async getEmployee(id: number): Promise<Employee | undefined> {
    return this.employeeStore.get(id);
  }

  async getEmployeeByRfid(rfidUid: string): Promise<Employee | undefined> {
    return Array.from(this.employeeStore.values()).find((employee) => employee.rfidUid === rfidUid);
  }

  async createEmployee(employee: InsertEmployee): Promise<Employee> {
    const newEmployee: Employee = {
      id: this.employeeIdSequence++,
      employeeCode: employee.employeeCode,
      name: employee.name,
      department: employee.department,
      phone: employee.phone ?? null,
      email: employee.email ?? null,
      rfidUid: employee.rfidUid,
      faceDescriptor: employee.faceDescriptor ?? null,
      isActive: employee.isActive ?? true,
      createdAt: new Date(),
    };

    this.employeeStore.set(newEmployee.id, newEmployee);
    return newEmployee;
  }

  async updateEmployee(id: number, updates: Partial<InsertEmployee>): Promise<Employee | undefined> {
    const existing = this.employeeStore.get(id);
    if (!existing) {
      return undefined;
    }

    const updated: Employee = {
      ...existing,
      ...updates,
      phone: updates.phone === undefined ? existing.phone : updates.phone ?? null,
      email: updates.email === undefined ? existing.email : updates.email ?? null,
      faceDescriptor:
        updates.faceDescriptor === undefined
          ? existing.faceDescriptor
          : updates.faceDescriptor ?? null,
      isActive: updates.isActive ?? existing.isActive,
    };

    this.employeeStore.set(id, updated);
    return updated;
  }

  async deleteEmployee(id: number): Promise<Employee | undefined> {
    const existing = this.employeeStore.get(id);
    if (!existing) {
      return undefined;
    }

    this.employeeStore.delete(id);

    this.attendanceStore.forEach((attendance, attendanceId) => {
      if (attendance.employeeId === id) {
        this.attendanceStore.delete(attendanceId);
      }
    });

    this.gateEventStore.forEach((gateEvent, gateEventId) => {
      if (gateEvent.employeeId === id) {
        this.gateEventStore.delete(gateEventId);
      }
    });

    return existing;
  }

  async getDevice(deviceId: string): Promise<Device | undefined> {
    return this.deviceStore.get(deviceId);
  }

  async createDevice(device: InsertDevice): Promise<Device> {
    const newDevice: Device = {
      id: this.deviceIdSequence++,
      deviceId: device.deviceId,
      location: device.location,
      deviceType: device.deviceType,
    };

    this.deviceStore.set(newDevice.deviceId, newDevice);
    return newDevice;
  }

  async getAttendances(filters: AttendanceFilters = {}): Promise<AttendanceWithEmployee[]> {
    return Array.from(this.attendanceStore.values())
      .filter((attendance) => {
        if (filters.date) {
          return attendance.date === filters.date;
        }

        if (filters.dateFrom && attendance.date < filters.dateFrom) {
          return false;
        }

        if (filters.dateTo && attendance.date > filters.dateTo) {
          return false;
        }

        return true;
      })
      .filter((attendance) => !filters.employeeId || attendance.employeeId === filters.employeeId)
      .filter((attendance) => !filters.status || attendance.verificationStatus === filters.status)
      .filter((attendance) => !filters.deviceId || attendance.deviceId === filters.deviceId)
      .filter((attendance) => {
        if (!filters.department) {
          return true;
        }

        return this.employeeStore.get(attendance.employeeId)?.department === filters.department;
      })
      .sort((a, b) => b.id - a.id)
      .map((attendance) => ({
        ...attendance,
        employee: this.employeeStore.get(attendance.employeeId),
      }))
      .filter((attendance) => {
        if (!filters.search) {
          return true;
        }

        const searchValue = filters.search.toLowerCase();
        const haystack = [
          attendance.employee?.name,
          attendance.employee?.employeeCode,
          attendance.employee?.department,
          attendance.employee?.rfidUid,
          attendance.deviceId,
          attendance.verificationStatus,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();

        return haystack.includes(searchValue);
      });
  }

  async getGateEvents(filters: GateEventFilters = {}): Promise<GateEventWithEmployee[]> {
    return Array.from(this.gateEventStore.values())
      .filter((gateEvent) => {
        if (filters.date) {
          return gateEvent.date === filters.date;
        }

        if (filters.dateFrom && gateEvent.date < filters.dateFrom) {
          return false;
        }

        if (filters.dateTo && gateEvent.date > filters.dateTo) {
          return false;
        }

        return true;
      })
      .filter((gateEvent) => !filters.employeeId || gateEvent.employeeId === filters.employeeId)
      .filter((gateEvent) => !filters.status || gateEvent.verificationStatus === filters.status)
      .filter((gateEvent) => !filters.deviceId || gateEvent.deviceId === filters.deviceId)
      .filter((gateEvent) => !filters.technology || gateEvent.scanTechnology === filters.technology)
      .filter((gateEvent) => !filters.movementDirection || gateEvent.movementDirection === filters.movementDirection)
      .filter((gateEvent) => {
        if (!filters.department) {
          return true;
        }

        return gateEvent.employeeId != null
          && this.employeeStore.get(gateEvent.employeeId)?.department === filters.department;
      })
      .sort((a, b) => b.id - a.id)
      .map((gateEvent) => ({
        ...gateEvent,
        employee: gateEvent.employeeId != null ? this.employeeStore.get(gateEvent.employeeId) : undefined,
      }))
      .filter((gateEvent) => {
        if (!filters.search) {
          return true;
        }

        const searchValue = filters.search.toLowerCase();
        const haystack = [
          gateEvent.employee?.name,
          gateEvent.employee?.employeeCode,
          gateEvent.employee?.department,
          gateEvent.employee?.rfidUid,
          gateEvent.rfidUid,
          gateEvent.deviceId,
          gateEvent.eventMessage,
          gateEvent.scanTechnology,
          gateEvent.verificationStatus,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();

        return haystack.includes(searchValue);
      });
  }

  async getOpenAttendance(employeeId: number, date: string): Promise<Attendance | undefined> {
    return Array.from(this.attendanceStore.values())
      .filter((attendance) => {
        return attendance.employeeId === employeeId
          && attendance.date === date
          && attendance.verificationStatus === "ENTRY"
          && attendance.exitTime === null;
      })
      .sort((a, b) => b.id - a.id)[0];
  }

  async createAttendance(attendance: InsertAttendance): Promise<Attendance> {
    const newAttendance: Attendance = {
      id: this.attendanceIdSequence++,
      employeeId: attendance.employeeId,
      date: attendance.date,
      entryTime: attendance.entryTime ?? null,
      exitTime: attendance.exitTime ?? null,
      workingHours: attendance.workingHours ?? null,
      verificationStatus: attendance.verificationStatus,
      deviceId: attendance.deviceId,
    };

    this.attendanceStore.set(newAttendance.id, newAttendance);
    return newAttendance;
  }

  async createGateEvent(event: InsertGateEvent): Promise<GateEvent> {
    const newGateEvent: GateEvent = {
      id: this.gateEventIdSequence++,
      employeeId: event.employeeId ?? null,
      date: event.date,
      occurredAt: new Date(),
      rfidUid: event.rfidUid,
      deviceId: event.deviceId,
      scanTechnology: event.scanTechnology ?? "HF_RFID",
      decision: event.decision,
      verificationStatus: event.verificationStatus,
      eventMessage: event.eventMessage,
      movementDirection: event.movementDirection ?? null,
      movementAxis: event.movementAxis ?? null,
      movementConfidence: event.movementConfidence ?? null,
      matchConfidence: event.matchConfidence ?? null,
      faceQuality: event.faceQuality ?? null,
      faceConsistency: event.faceConsistency ?? null,
      faceCaptureMode: event.faceCaptureMode ?? null,
    };

    this.gateEventStore.set(newGateEvent.id, newGateEvent);
    return newGateEvent;
  }

  async updateAttendance(id: number, updates: Partial<InsertAttendance>): Promise<Attendance | undefined> {
    const existing = this.attendanceStore.get(id);
    if (!existing) {
      return undefined;
    }

    const updated: Attendance = {
      ...existing,
      ...updates,
      entryTime: updates.entryTime === undefined ? existing.entryTime : updates.entryTime ?? null,
      exitTime: updates.exitTime === undefined ? existing.exitTime : updates.exitTime ?? null,
      workingHours:
        updates.workingHours === undefined ? existing.workingHours : updates.workingHours ?? null,
    };

    this.attendanceStore.set(id, updated);
    return updated;
  }

  async getDashboardStats(): Promise<{
    totalEmployees: number;
    presentToday: number;
    absentToday: number;
    recentScans: AttendanceWithEmployee[];
  }> {
    const today = new Date().toISOString().split("T")[0];
    const allEmployees = Array.from(this.employeeStore.values());
    const activeEmployees = allEmployees.filter((employee) => employee.isActive !== false);

    const presentEmployeeIds = new Set(
      Array.from(this.attendanceStore.values())
        .filter((attendance) => attendance.date === today)
        .filter((attendance) => attendance.verificationStatus === "ENTRY" || attendance.verificationStatus === "EXIT")
        .map((attendance) => attendance.employeeId)
        .filter((employeeId) => employeeId > 0),
    );

    const recentScans = Array.from(this.attendanceStore.values())
      .sort((a, b) => b.id - a.id)
      .slice(0, 10)
      .map((attendance) => ({
        ...attendance,
        employee: this.employeeStore.get(attendance.employeeId),
      }));

    return {
      totalEmployees: activeEmployees.length,
      presentToday: presentEmployeeIds.size,
      absentToday: Math.max(0, activeEmployees.length - presentEmployeeIds.size),
      recentScans,
    };
  }
}

class RuntimeStorage implements IStorage {
  private readonly memoryStorage = new MemoryStorage();
  private readonly databaseStorage = db ? new DatabaseStorage() : null;

  private getActiveStorage(): IStorage {
    if (this.databaseStorage && shouldUseDatabaseStorage()) {
      return this.databaseStorage;
    }

    return this.memoryStorage;
  }

  private isTransientDatabaseError(error: unknown) {
    if (!error || typeof error !== "object") {
      return false;
    }

    const errorWithCode = error as { code?: string; message?: string; cause?: { message?: string } };
    const message = [
      errorWithCode.message,
      errorWithCode.cause?.message,
    ]
      .filter((value): value is string => typeof value === "string")
      .join(" ")
      .toLowerCase();

    return (
      errorWithCode.code === "ECONNRESET"
      || errorWithCode.code === "EPIPE"
      || errorWithCode.code === "57P01"
      || message.includes("connection terminated due to connection timeout")
      || message.includes("connection terminated unexpectedly")
      || message.includes("terminating connection")
      || message.includes("connection timeout")
    );
  }

  private async runWithRetry<T>(operation: (storage: IStorage) => Promise<T>) {
    const activeStorage = this.getActiveStorage();

    try {
      return await operation(activeStorage);
    } catch (error) {
      if (
        activeStorage !== this.databaseStorage
        || !this.databaseStorage
        || !this.isTransientDatabaseError(error)
      ) {
        throw error;
      }

      console.warn("[storage] Retrying transient PostgreSQL error:", error);
      const recovered = await verifyDatabaseConnection();
      if (!recovered) {
        throw error;
      }

      return await operation(this.databaseStorage);
    }
  }

  getEmployees() {
    return this.runWithRetry((storage) => storage.getEmployees());
  }

  getEmployee(id: number) {
    return this.runWithRetry((storage) => storage.getEmployee(id));
  }

  getEmployeeByRfid(rfidUid: string) {
    return this.runWithRetry((storage) => storage.getEmployeeByRfid(rfidUid));
  }

  createEmployee(employee: InsertEmployee) {
    return this.runWithRetry((storage) => storage.createEmployee(employee));
  }

  updateEmployee(id: number, updates: Partial<InsertEmployee>) {
    return this.runWithRetry((storage) => storage.updateEmployee(id, updates));
  }

  deleteEmployee(id: number) {
    return this.runWithRetry((storage) => storage.deleteEmployee(id));
  }

  getDevice(deviceId: string) {
    return this.runWithRetry((storage) => storage.getDevice(deviceId));
  }

  createDevice(device: InsertDevice) {
    return this.runWithRetry((storage) => storage.createDevice(device));
  }

  getAttendances(filters?: AttendanceFilters) {
    return this.runWithRetry((storage) => storage.getAttendances(filters));
  }

  getOpenAttendance(employeeId: number, date: string) {
    return this.runWithRetry((storage) => storage.getOpenAttendance(employeeId, date));
  }

  createAttendance(attendance: InsertAttendance) {
    return this.runWithRetry((storage) => storage.createAttendance(attendance));
  }

  updateAttendance(id: number, updates: Partial<InsertAttendance>) {
    return this.runWithRetry((storage) => storage.updateAttendance(id, updates));
  }

  getGateEvents(filters?: GateEventFilters) {
    return this.runWithRetry((storage) => storage.getGateEvents(filters));
  }

  createGateEvent(event: InsertGateEvent) {
    return this.runWithRetry((storage) => storage.createGateEvent(event));
  }

  getDashboardStats() {
    return this.runWithRetry((storage) => storage.getDashboardStats());
  }
}

if (!db) {
  console.warn("[storage] DATABASE_URL not set. Using in-memory storage.");
}

export const storage: IStorage = new RuntimeStorage();
