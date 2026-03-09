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

export class DatabaseStorage implements IStorage {
  async getEmployees(): Promise<Employee[]> {
    return await db.select().from(employees).orderBy(employees.id);
  }

  async getEmployee(id: number): Promise<Employee | undefined> {
    const [employee] = await db.select().from(employees).where(eq(employees.id, id));
    return employee;
  }

  async getEmployeeByRfid(rfidUid: string): Promise<Employee | undefined> {
    const [employee] = await db.select().from(employees).where(eq(employees.rfidUid, rfidUid));
    return employee;
  }

  async createEmployee(employee: InsertEmployee): Promise<Employee> {
    const [newEmployee] = await db.insert(employees).values(employee).returning();
    return newEmployee;
  }

  async updateEmployee(id: number, updates: Partial<InsertEmployee>): Promise<Employee | undefined> {
    const [updated] = await db.update(employees).set(updates).where(eq(employees.id, id)).returning();
    return updated;
  }

  async getDevice(deviceId: string): Promise<Device | undefined> {
    const [device] = await db.select().from(devices).where(eq(devices.deviceId, deviceId));
    return device;
  }

  async createDevice(device: InsertDevice): Promise<Device> {
    const [newDevice] = await db.insert(devices).values(device).returning();
    return newDevice;
  }

  async getAttendances(date?: string, employeeId?: number): Promise<(Attendance & { employee?: Employee })[]> {
    let query = db.select({
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
    const [attendance] = await db.select()
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
    const [newAttendance] = await db.insert(attendances).values(attendance).returning();
    return newAttendance;
  }

  async updateAttendance(id: number, updates: Partial<InsertAttendance>): Promise<Attendance | undefined> {
    const [updated] = await db.update(attendances).set(updates).where(eq(attendances.id, id)).returning();
    return updated;
  }

  async getDashboardStats() {
    const today = new Date().toISOString().split('T')[0];
    
    // Total employees
    const [totalEmpResult] = await db.select({ count: sql`count(*)`.mapWith(Number) }).from(employees).where(eq(employees.isActive, true));
    const totalEmployees = totalEmpResult?.count || 0;

    // Present today (has an entry today with verification_status ENTRY or EXIT)
    const presentRows = await db.select({ count: sql`count(distinct ${attendances.employeeId})`.mapWith(Number) })
      .from(attendances)
      .where(and(
        eq(attendances.date, today),
        sql`${attendances.verificationStatus} IN ('ENTRY', 'EXIT')`
      ));
    const presentToday = presentRows[0]?.count || 0;
    
    const absentToday = Math.max(0, totalEmployees - presentToday);

    // Recent scans
    const recentScans = await db.select({
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

export const storage = new DatabaseStorage();