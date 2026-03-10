import { format } from "date-fns";
import { api } from "@shared/routes";
import type { Employee } from "@shared/schema";
import { z } from "zod";

export type AttendanceRecord = z.infer<typeof api.attendances.list.responses[200]>[number];
export type AttendanceStatus = "ENTRY" | "EXIT" | "FAILED_FACE" | "FAILED_DIRECTION" | "UNKNOWN_RFID";

export function formatWorkingDuration(workingHours: number) {
  const totalSeconds = Math.max(0, Math.round(workingHours * 60 * 60));
  const hours = Math.floor(totalSeconds / 3600);
  const remainingSeconds = totalSeconds % 3600;
  const minutes = Math.floor(remainingSeconds / 60);
  const seconds = remainingSeconds % 60;
  return `${hours}h ${minutes}m ${seconds}s`;
}

export function getAttendanceStatusLabel(status: AttendanceStatus | string) {
  switch (status) {
    case "ENTRY":
      return "Verified Entry";
    case "EXIT":
      return "Verified Exit";
    case "FAILED_FACE":
      return "Biometric Mismatch";
    case "FAILED_DIRECTION":
      return "Direction Unclear";
    case "UNKNOWN_RFID":
      return "Unknown Badge";
    default:
      return status;
  }
}

export function calculateAttendanceSummary(records: AttendanceRecord[]) {
  const verifiedEntries = records.filter((record) => record.verificationStatus === "ENTRY").length;
  const verifiedExits = records.filter((record) => record.verificationStatus === "EXIT").length;
  const failedScans = records.filter((record) => {
    return record.verificationStatus === "FAILED_FACE"
      || record.verificationStatus === "FAILED_DIRECTION"
      || record.verificationStatus === "UNKNOWN_RFID";
  }).length;
  const totalHours = Number(
    records.reduce((sum, record) => sum + (record.workingHours ?? 0), 0).toFixed(2),
  );
  const employeesSeen = new Set(records.map((record) => record.employeeId).filter((employeeId) => employeeId > 0));
  const activeDays = new Set(records.map((record) => record.date)).size;

  return {
    totalRecords: records.length,
    verifiedEntries,
    verifiedExits,
    failedScans,
    totalHours,
    activeEmployees: employeesSeen.size,
    activeDays,
    averageHoursPerDay: activeDays ? Number((totalHours / activeDays).toFixed(2)) : 0,
  };
}

export function buildDailyHoursTrend(records: AttendanceRecord[]) {
  const grouped = new Map<string, { date: string; hours: number; verified: number; failed: number }>();

  records.forEach((record) => {
    const existing = grouped.get(record.date) ?? {
      date: record.date,
      hours: 0,
      verified: 0,
      failed: 0,
    };

    existing.hours += record.workingHours ?? 0;
    if (record.verificationStatus === "ENTRY" || record.verificationStatus === "EXIT") {
      existing.verified += 1;
    } else {
      existing.failed += 1;
    }

    grouped.set(record.date, existing);
  });

  return Array.from(grouped.values())
    .sort((a, b) => a.date.localeCompare(b.date))
    .map((row) => ({
      ...row,
      label: format(new Date(`${row.date}T00:00:00`), "MMM d"),
      hours: Number(row.hours.toFixed(2)),
    }));
}

export function buildStatusDistribution(records: AttendanceRecord[]) {
  const counts: Record<AttendanceStatus, number> = {
    ENTRY: 0,
    EXIT: 0,
    FAILED_FACE: 0,
    FAILED_DIRECTION: 0,
    UNKNOWN_RFID: 0,
  };

  records.forEach((record) => {
    const status = record.verificationStatus as AttendanceStatus;
    if (status in counts) {
      counts[status] += 1;
    }
  });

  return (Object.entries(counts) as Array<[AttendanceStatus, number]>)
    .filter(([, count]) => count > 0)
    .map(([status, count]) => ({
      status,
      count,
      label: getAttendanceStatusLabel(status),
    }));
}

export function buildEmployeeRows(records: AttendanceRecord[]) {
  const grouped = new Map<number, {
    employeeId: number;
    employeeName: string;
    employeeCode: string;
    department: string;
    totalHours: number;
    verified: number;
    failed: number;
    lastSeen: string | null;
  }>();

  records.forEach((record) => {
    const employee = record.employee;
    if (!employee) {
      return;
    }

    const existing = grouped.get(employee.id) ?? {
      employeeId: employee.id,
      employeeName: employee.name,
      employeeCode: employee.employeeCode,
      department: employee.department,
      totalHours: 0,
      verified: 0,
      failed: 0,
      lastSeen: null,
    };

    existing.totalHours += record.workingHours ?? 0;
    if (record.verificationStatus === "ENTRY" || record.verificationStatus === "EXIT") {
      existing.verified += 1;
    } else {
      existing.failed += 1;
    }

    const lastSeenCandidate = record.exitTime ?? record.entryTime;
    if (lastSeenCandidate) {
      const lastSeenIso = new Date(lastSeenCandidate).toISOString();
      if (!existing.lastSeen || lastSeenIso > existing.lastSeen) {
        existing.lastSeen = lastSeenIso;
      }
    }

    grouped.set(employee.id, existing);
  });

  return Array.from(grouped.values())
    .sort((a, b) => b.totalHours - a.totalHours)
    .map((row) => ({
      ...row,
      totalHours: Number(row.totalHours.toFixed(2)),
    }));
}

function escapeCsvValue(value: string | number) {
  const stringValue = String(value);
  if (stringValue.includes(",") || stringValue.includes('"') || stringValue.includes("\n")) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }

  return stringValue;
}

export function downloadCsvReport(filename: string, rows: Array<Record<string, string | number>>) {
  if (!rows.length) {
    return;
  }

  const headers = Object.keys(rows[0]);
  const lines = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => escapeCsvValue(row[header] ?? "")).join(",")),
  ];

  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function exportAttendanceRows(records: AttendanceRecord[], filename: string) {
  const rows = records.map((record) => ({
    Date: record.date,
    Employee: record.employee?.name ?? "Unknown",
    EmployeeCode: record.employee?.employeeCode ?? "N/A",
    Department: record.employee?.department ?? "N/A",
    Status: getAttendanceStatusLabel(record.verificationStatus),
    EntryTime: record.entryTime ? format(new Date(record.entryTime), "yyyy-MM-dd HH:mm:ss") : "-",
    ExitTime: record.exitTime ? format(new Date(record.exitTime), "yyyy-MM-dd HH:mm:ss") : "-",
    WorkingHours: record.workingHours ?? 0,
    DeviceId: record.deviceId,
  }));

  downloadCsvReport(filename, rows);
}

export function buildEmployeeDetail(selectedEmployee: Employee | undefined, records: AttendanceRecord[]) {
  if (!selectedEmployee) {
    return null;
  }

  const summary = calculateAttendanceSummary(records);
  const recentRecord = records.find((record) => record.employeeId === selectedEmployee.id);

  return {
    employee: selectedEmployee,
    ...summary,
    lastSeen: recentRecord?.exitTime ?? recentRecord?.entryTime ?? null,
  };
}
