import { db } from "./db";
import { eq, and, desc, gte, sql } from "drizzle-orm";
import {
  employees,
  devices,
  attendances,
  type Employee,
  type InsertEmployee,
  type Device,
  type InsertDevice,
  type Attendance,
  type InsertAttendance,
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
  getAttendances(date?: string, employeeId?: number): Promise<(Attendance & { employee?: Employee })[]>;
  getOpenAttendance(employeeId: number, date: string): Promise<Attendance | undefined>;
  createAttendance(attendance: InsertAttendance): Promise<Attendance>;
  updateAttendance(id: number, updates: Partial<InsertAttendance>): Promise<Attendance | undefined>;
  
  // Stats
  getDashboardStats(): Promise<{ totalEmployees: number; presentToday: number; absentToday: number; recentScans: (Attendance & { employee?: Employee })[] }>;
}

type AttendanceWithEmployee = Attendance & { employee?: Employee };

function requireDb() {
  if (!db) {
    throw new Error("DATABASE_URL is not configured.");
  }

  return db;
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

  async getAttendances(date?: string, employeeId?: number): Promise<(Attendance & { employee?: Employee })[]> {
    let query = requireDb().select({
      attendance: attendances,
      employee: employees
    })
    .from(attendances)
    .leftJoin(employees, eq(attendances.employeeId, employees.id))
    .orderBy(desc(attendances.id));

    const conditions = [];
    if (date) conditions.push(eq(attendances.date, date));
    if (employeeId) conditions.push(eq(attendances.employeeId, employeeId));

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

  async getOpenAttendance(employeeId: number, date: string): Promise<Attendance | undefined> {
    const [attendance] = await requireDb().select()
      .from(attendances)
      .where(
        and(
          eq(attendances.employeeId, employeeId),
          eq(attendances.date, date),
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

  private employeeStore = new Map<number, Employee>();
  private deviceStore = new Map<string, Device>();
  private attendanceStore = new Map<number, Attendance>();

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

  async getAttendances(date?: string, employeeId?: number): Promise<AttendanceWithEmployee[]> {
    return Array.from(this.attendanceStore.values())
      .filter((attendance) => !date || attendance.date === date)
      .filter((attendance) => !employeeId || attendance.employeeId === employeeId)
      .sort((a, b) => b.id - a.id)
      .map((attendance) => ({
        ...attendance,
        employee: this.employeeStore.get(attendance.employeeId),
      }));
  }

  async getOpenAttendance(employeeId: number, date: string): Promise<Attendance | undefined> {
    return Array.from(this.attendanceStore.values())
      .filter((attendance) => {
        return attendance.employeeId === employeeId
          && attendance.date === date
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

if (!db) {
  console.warn("[storage] DATABASE_URL not set. Using in-memory storage.");
}

export const storage: IStorage = db ? new DatabaseStorage() : new MemoryStorage();
