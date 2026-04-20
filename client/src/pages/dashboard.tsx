import { useMemo, useState } from "react";
import { format, startOfMonth, subDays } from "date-fns";
import {
  Activity,
  CalendarRange,
  Download,
  ShieldAlert,
  TrendingUp,
  UserCheck,
  UserMinus,
  Users,
} from "lucide-react";
import {
  Area,
  Bar,
  CartesianGrid,
  Cell,
  ComposedChart,
  Pie,
  PieChart,
  XAxis,
  YAxis,
} from "recharts";
import { useDashboardStats } from "@/hooks/use-stats";
import { useAttendances } from "@/hooks/use-attendances";
import { useGateEvents } from "@/hooks/use-gate-events";
import { useEmployees } from "@/hooks/use-employees";
import {
  buildDailyHoursTrend,
  buildEmployeeDetail,
  buildEmployeeRows,
  calculateGateEventSummary,
  buildStatusDistribution,
  calculateAttendanceSummary,
  exportAttendanceRows,
  exportGateEventRows,
  formatWorkingDuration,
  getGateDecisionLabel,
  getAttendanceStatusLabel,
  getScanTechnologyLabel,
} from "@/lib/reporting";
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

const statusChartConfig = {
  count: {
    label: "Events",
    color: "#2563eb",
  },
  ENTRY: {
    label: "Verified Entry",
    color: "#10b981",
  },
  EXIT: {
    label: "Verified Exit",
    color: "#0ea5e9",
  },
  FAILED_FACE: {
    label: "Biometric Mismatch",
    color: "#f97316",
  },
  FAILED_DIRECTION: {
    label: "Direction Unclear",
    color: "#ef4444",
  },
  UNKNOWN_RFID: {
    label: "Unknown Badge",
    color: "#7c3aed",
  },
} as const;

const hoursChartConfig = {
  hours: {
    label: "Hours",
    color: "#2563eb",
  },
  verified: {
    label: "Verified Events",
    color: "#10b981",
  },
  failed: {
    label: "Failed Events",
    color: "#ef4444",
  },
} as const;

function OverviewCard(props: {
  title: string;
  value: string | number;
  hint: string;
  icon: typeof Users;
  accent?: string;
  isLoading?: boolean;
}) {
  const Icon = props.icon;

  return (
    <Card className="border-border/50 shadow-sm">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <CardTitle className="text-sm font-medium">{props.title}</CardTitle>
        <Icon className={`size-4 ${props.accent ?? "text-muted-foreground"}`} />
      </CardHeader>
      <CardContent>
        {props.isLoading ? (
          <Skeleton className="h-8 w-20" />
        ) : (
          <div className={`text-3xl font-bold ${props.accent ?? "text-foreground"}`}>{props.value}</div>
        )}
        <p className="mt-2 text-xs text-muted-foreground">{props.hint}</p>
      </CardContent>
    </Card>
  );
}

function renderRecentStatus(status: string) {
  switch (status) {
    case "ENTRY":
      return <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white">Entry Validated</Badge>;
    case "EXIT":
      return <Badge variant="secondary" className="bg-blue-100 text-blue-800">Exit Validated</Badge>;
    case "FAILED_FACE":
      return <Badge variant="destructive">Face Mismatch</Badge>;
    case "FAILED_DIRECTION":
      return <Badge variant="destructive">Direction Unclear</Badge>;
    case "UNKNOWN_RFID":
      return <Badge variant="destructive">Invalid RFID</Badge>;
    default:
      return <Badge variant="outline">{status}</Badge>;
  }
}

function renderGateDecision(decision: string) {
  switch (decision) {
    case "ENTRY":
      return <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white">Entry</Badge>;
    case "EXIT":
      return <Badge variant="secondary" className="bg-blue-100 text-blue-800">Exit</Badge>;
    case "REJECTED":
      return <Badge variant="destructive">Rejected</Badge>;
    default:
      return <Badge variant="outline">{getGateDecisionLabel(decision)}</Badge>;
  }
}

export default function Dashboard() {
  const todayString = format(new Date(), "yyyy-MM-dd");
  const [employeeId, setEmployeeId] = useState("all");
  const [departmentFilter, setDepartmentFilter] = useState("all");
  const [deviceFilter, setDeviceFilter] = useState("all");
  const [datePreset, setDatePreset] = useState("today");
  const [dateFrom, setDateFrom] = useState(todayString);
  const [dateTo, setDateTo] = useState(todayString);

  const { data: stats, isLoading, error, dataUpdatedAt: statsUpdatedAt } = useDashboardStats();
  const { data: employees } = useEmployees();
  const departmentOptions = useMemo(() => {
    return Array.from(new Set((employees ?? []).map((employee) => employee.department))).sort((left, right) => {
      return left.localeCompare(right);
    });
  }, [employees]);
  const attendanceFilters = useMemo(() => ({
    employeeId: employeeId !== "all" ? Number(employeeId) : undefined,
    dateFrom: dateFrom || undefined,
    dateTo: dateTo || undefined,
    department: departmentFilter !== "all" ? departmentFilter : undefined,
    deviceId: deviceFilter !== "all" ? deviceFilter : undefined,
  }), [dateFrom, dateTo, departmentFilter, deviceFilter, employeeId]);
  const gateEventFilters = useMemo(() => ({
    employeeId: employeeId !== "all" ? Number(employeeId) : undefined,
    dateFrom: dateFrom || undefined,
    dateTo: dateTo || undefined,
    department: departmentFilter !== "all" ? departmentFilter : undefined,
    deviceId: deviceFilter !== "all" ? deviceFilter : undefined,
  }), [dateFrom, dateTo, departmentFilter, deviceFilter, employeeId]);
  const {
    data: filteredAttendances,
    isLoading: attendanceLoading,
    dataUpdatedAt: attendanceUpdatedAt,
  } = useAttendances(attendanceFilters);
  const {
    data: gateEvents,
    isLoading: gateEventsLoading,
    dataUpdatedAt: gateEventsUpdatedAt,
  } = useGateEvents(gateEventFilters);

  const selectedEmployee = useMemo(() => {
    if (employeeId === "all") {
      return undefined;
    }

    return employees?.find((employee) => employee.id === Number(employeeId));
  }, [employeeId, employees]);

  const filteredSummary = useMemo(() => calculateAttendanceSummary(filteredAttendances ?? []), [filteredAttendances]);
  const gateEventSummary = useMemo(() => calculateGateEventSummary(gateEvents ?? []), [gateEvents]);
  const selectedEmployeeDetail = useMemo(() => {
    return buildEmployeeDetail(selectedEmployee, filteredAttendances ?? []);
  }, [filteredAttendances, selectedEmployee]);
  const dailyTrend = useMemo(() => buildDailyHoursTrend(filteredAttendances ?? []), [filteredAttendances]);
  const statusDistribution = useMemo(() => buildStatusDistribution(filteredAttendances ?? []), [filteredAttendances]);
  const employeeRows = useMemo(() => buildEmployeeRows(filteredAttendances ?? []), [filteredAttendances]);
  const deviceOptions = useMemo(() => {
    return Array.from(new Set([
      ...(filteredAttendances ?? []).map((record) => record.deviceId),
      ...(gateEvents ?? []).map((record) => record.deviceId),
      ...(deviceFilter !== "all" ? [deviceFilter] : []),
    ])).sort((left, right) => left.localeCompare(right));
  }, [deviceFilter, filteredAttendances, gateEvents]);
  const lastUpdatedAt = Math.max(statsUpdatedAt, attendanceUpdatedAt, gateEventsUpdatedAt);

  const handleResetFilters = () => {
    setEmployeeId("all");
    setDepartmentFilter("all");
    setDeviceFilter("all");
    setDatePreset("today");
    setDateFrom(todayString);
    setDateTo(todayString);
  };

  const handleExportFiltered = () => {
    if (!filteredAttendances?.length) {
      return;
    }

    const filename = selectedEmployee
      ? `${selectedEmployee.employeeCode.toLowerCase()}-attendance-report.csv`
      : "all-employees-attendance-report.csv";
    exportAttendanceRows(filteredAttendances, filename);
  };

  const handleExportGateEvents = () => {
    if (!gateEvents?.length) {
      return;
    }

    exportGateEventRows(gateEvents, "gate-events-report.csv");
  };

  const handlePresetChange = (value: string) => {
    setDatePreset(value);

    if (value === "custom") {
      return;
    }

    const today = new Date();
    if (value === "today") {
      const formattedToday = format(today, "yyyy-MM-dd");
      setDateFrom(formattedToday);
      setDateTo(formattedToday);
      return;
    }

    if (value === "last7") {
      setDateFrom(format(subDays(today, 6), "yyyy-MM-dd"));
      setDateTo(format(today, "yyyy-MM-dd"));
      return;
    }

    if (value === "month") {
      setDateFrom(format(startOfMonth(today), "yyyy-MM-dd"));
      setDateTo(format(today, "yyyy-MM-dd"));
    }
  };

  if (error) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-8 text-center">
        <ShieldAlert className="mb-4 size-12 text-destructive" />
        <h2 className="text-2xl font-bold text-foreground">Failed to load dashboard</h2>
        <p className="mt-2 text-muted-foreground">There was a problem connecting to the server.</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 p-6 md:p-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Operations Dashboard</h1>
          <p className="mt-1 text-muted-foreground">
            Track live gate activity, drill into a particular employee, and review both raw gate events and final attendance records.
          </p>
          {lastUpdatedAt > 0 && (
            <p className="mt-2 text-xs text-muted-foreground">
              Live refresh every 5 seconds. Last sync: {format(new Date(lastUpdatedAt), "dd MMM yyyy, hh:mm:ss a")}
            </p>
          )}
        </div>
        <div className="flex flex-col gap-2 sm:flex-row">
          <Button variant="outline" onClick={handleExportGateEvents} disabled={!gateEvents?.length} className="shadow-sm">
            <Download className="mr-2 size-4" />
            Download Gate Events
          </Button>
          <Button onClick={handleExportFiltered} disabled={!filteredAttendances?.length} className="shadow-sm">
            <Download className="mr-2 size-4" />
            {selectedEmployee ? "Download Employee Report" : "Download Attendance Report"}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <OverviewCard
          title="Total Employees"
          value={stats?.totalEmployees || 0}
          hint="Active employee records in the system"
          icon={Users}
          isLoading={isLoading}
        />
        <OverviewCard
          title="Present Today"
          value={stats?.presentToday || 0}
          hint="Employees with verified access events today"
          icon={UserCheck}
          accent="text-emerald-600"
          isLoading={isLoading}
        />
        <OverviewCard
          title="Absent / Unknown"
          value={stats?.absentToday || 0}
          hint="Active employees without a verified event today"
          icon={UserMinus}
          accent="text-rose-600"
          isLoading={isLoading}
        />
      </div>

      <Card className="border-border/50 shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2 text-lg">
            <CalendarRange className="size-4 text-primary" />
            Operations Filters
          </CardTitle>
          <CardDescription>
            Filter by person, department, device, and time window. The charts, gate feed, and exports all follow this scope.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-[1fr_1.4fr_1fr_1fr_auto]">
            <Select value={datePreset} onValueChange={handlePresetChange}>
              <SelectTrigger>
                <SelectValue placeholder="Date preset" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="today">Today</SelectItem>
                <SelectItem value="last7">Last 7 days</SelectItem>
                <SelectItem value="month">This month</SelectItem>
                <SelectItem value="custom">Custom range</SelectItem>
              </SelectContent>
            </Select>
            <Select value={employeeId} onValueChange={setEmployeeId}>
              <SelectTrigger>
                <SelectValue placeholder="All employees" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All employees</SelectItem>
                {employees?.map((employee) => (
                  <SelectItem key={employee.id} value={String(employee.id)}>
                    {employee.name} ({employee.employeeCode})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={departmentFilter} onValueChange={setDepartmentFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All departments" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All departments</SelectItem>
                {departmentOptions.map((department) => (
                  <SelectItem key={department} value={department}>
                    {department}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={deviceFilter} onValueChange={setDeviceFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All devices" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All devices</SelectItem>
                {deviceOptions.map((deviceId) => (
                  <SelectItem key={deviceId} value={deviceId}>
                    {deviceId}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline" onClick={handleResetFilters}>Reset</Button>
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <Input
              type="date"
              value={dateFrom}
              onChange={(event) => {
                setDatePreset("custom");
                setDateFrom(event.target.value);
              }}
            />
            <Input
              type="date"
              value={dateTo}
              onChange={(event) => {
                setDatePreset("custom");
                setDateTo(event.target.value);
              }}
            />
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
            <span className="rounded-full bg-muted px-3 py-1">
              Scope: {selectedEmployee ? `${selectedEmployee.name} only` : "All employees"}
            </span>
            <span className="rounded-full bg-muted px-3 py-1">
              Records: {filteredSummary.totalRecords}
            </span>
            <span className="rounded-full bg-muted px-3 py-1">
              Active days: {filteredSummary.activeDays}
            </span>
            <span className="rounded-full bg-muted px-3 py-1">
              Gate events: {gateEventSummary.totalEvents}
            </span>
            <span className="rounded-full bg-muted px-3 py-1">
              Device scope: {deviceFilter === "all" ? "All devices" : deviceFilter}
            </span>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-4">
        <OverviewCard
          title="Filtered Records"
          value={filteredSummary.totalRecords}
          hint="Attendance rows in the current selection"
          icon={Activity}
          isLoading={attendanceLoading}
        />
        <OverviewCard
          title="Verified Access"
          value={filteredSummary.verifiedEntries + filteredSummary.verifiedExits}
          hint={`${filteredSummary.verifiedEntries} entries and ${filteredSummary.verifiedExits} exits`}
          icon={UserCheck}
          accent="text-emerald-600"
          isLoading={attendanceLoading}
        />
        <OverviewCard
          title="Failed Events"
          value={filteredSummary.failedScans}
          hint="Face, direction, or badge rejection events"
          icon={ShieldAlert}
          accent="text-rose-600"
          isLoading={attendanceLoading}
        />
        <OverviewCard
          title="Total Work Hours"
          value={`${filteredSummary.totalHours.toFixed(2)}h`}
          hint={`${filteredSummary.averageHoursPerDay.toFixed(2)}h average per active day`}
          icon={TrendingUp}
          accent="text-blue-600"
          isLoading={attendanceLoading}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-4">
        <OverviewCard
          title="Gate Events"
          value={gateEventSummary.totalEvents}
          hint="Raw scans and biometric decisions in the current scope"
          icon={Activity}
          isLoading={gateEventsLoading}
        />
        <OverviewCard
          title="Direction Resolved"
          value={gateEventSummary.directionResolved}
          hint="Events where entry or exit direction was classified"
          icon={TrendingUp}
          accent="text-cyan-600"
          isLoading={gateEventsLoading}
        />
        <OverviewCard
          title="Rejected Events"
          value={gateEventSummary.rejectedEvents}
          hint="Rejected because of face, direction, or badge issues"
          icon={ShieldAlert}
          accent="text-rose-600"
          isLoading={gateEventsLoading}
        />
        <OverviewCard
          title="Average Match"
          value={`${(gateEventSummary.averageMatchConfidence * 100).toFixed(1)}%`}
          hint={`${gateEventSummary.trackedEmployees} employees seen in raw gate events`}
          icon={UserCheck}
          accent="text-emerald-600"
          isLoading={gateEventsLoading}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle>{selectedEmployee ? "Daily Workload Trend" : "Daily Attendance Trend"}</CardTitle>
            <CardDescription>
              Date-wise view of hours logged alongside verified and failed events.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {attendanceLoading ? (
              <Skeleton className="h-[320px] w-full" />
            ) : !dailyTrend.length ? (
              <div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
                No attendance data for the selected filters.
              </div>
            ) : (
              <ChartContainer config={hoursChartConfig} className="h-[320px] w-full">
                <ComposedChart data={dailyTrend} margin={{ left: 12, right: 12, top: 10 }}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} />
                  <YAxis tickLine={false} axisLine={false} width={36} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Area
                    type="monotone"
                    dataKey="hours"
                    name="hours"
                    stroke="var(--color-hours)"
                    fill="var(--color-hours)"
                    fillOpacity={0.18}
                    strokeWidth={2}
                  />
                  <Bar dataKey="verified" name="verified" fill="var(--color-verified)" radius={[5, 5, 0, 0]} />
                  <Bar dataKey="failed" name="failed" fill="var(--color-failed)" radius={[5, 5, 0, 0]} />
                </ComposedChart>
              </ChartContainer>
            )}
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle>Status Distribution</CardTitle>
            <CardDescription>
              How the selected attendance events are distributed across successful and failed outcomes.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {attendanceLoading ? (
              <Skeleton className="h-[320px] w-full" />
            ) : !statusDistribution.length ? (
              <div className="flex h-[320px] items-center justify-center text-sm text-muted-foreground">
                No status distribution available yet.
              </div>
            ) : (
              <ChartContainer config={statusChartConfig} className="h-[320px] w-full">
                <PieChart>
                  <ChartTooltip
                    content={(
                      <ChartTooltipContent
                        nameKey="status"
                        formatter={(value, name) => (
                          <div className="flex w-full items-center justify-between gap-3">
                            <span className="text-muted-foreground">{getAttendanceStatusLabel(name as never)}</span>
                            <span className="font-mono font-medium text-foreground">{String(value)}</span>
                          </div>
                        )}
                      />
                    )}
                  />
                  <ChartLegend content={<ChartLegendContent nameKey="status" />} />
                  <Pie
                    data={statusDistribution}
                    dataKey="count"
                    nameKey="status"
                    innerRadius={72}
                    outerRadius={105}
                    paddingAngle={2}
                  >
                    {statusDistribution.map((entry) => (
                      <Cell
                        key={entry.status}
                        fill={statusChartConfig[entry.status as keyof typeof statusChartConfig]?.color ?? "#2563eb"}
                      />
                    ))}
                  </Pie>
                </PieChart>
              </ChartContainer>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-[0.95fr_1.05fr]">
        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle>{selectedEmployee ? "Employee Snapshot" : "Top Employees In Range"}</CardTitle>
            <CardDescription>
              {selectedEmployee
                ? "Focused details for the selected employee using the active date range."
                : "Aggregated attendance performance for employees included in the filtered data set."}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {attendanceLoading ? (
              <div className="space-y-3">
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-20 w-full" />
              </div>
            ) : selectedEmployeeDetail ? (
              <div className="space-y-6">
                <div className="rounded-2xl border border-border/60 bg-muted/20 p-5">
                  <div className="flex flex-col gap-2">
                    <h3 className="text-xl font-semibold text-foreground">{selectedEmployeeDetail.employee.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {selectedEmployeeDetail.employee.employeeCode} | {selectedEmployeeDetail.employee.department}
                    </p>
                  </div>
                  <div className="mt-5 grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Verified events</p>
                      <p className="mt-1 text-2xl font-semibold text-foreground">
                        {selectedEmployeeDetail.verifiedEntries + selectedEmployeeDetail.verifiedExits}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Failed events</p>
                      <p className="mt-1 text-2xl font-semibold text-foreground">{selectedEmployeeDetail.failedScans}</p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Hours logged</p>
                      <p className="mt-1 text-2xl font-semibold text-foreground">
                        {selectedEmployeeDetail.totalHours.toFixed(2)}h
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Last seen</p>
                      <p className="mt-1 text-sm font-medium text-foreground">
                        {selectedEmployeeDetail.lastSeen
                          ? format(new Date(selectedEmployeeDetail.lastSeen), "dd MMM yyyy, hh:mm a")
                          : "No verified scan yet"}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="rounded-xl border border-border/60 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Average hours/day</p>
                    <p className="mt-1 text-xl font-semibold text-foreground">
                      {selectedEmployeeDetail.averageHoursPerDay.toFixed(2)}h
                    </p>
                  </div>
                  <div className="rounded-xl border border-border/60 p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Active days</p>
                    <p className="mt-1 text-xl font-semibold text-foreground">{selectedEmployeeDetail.activeDays}</p>
                  </div>
                </div>
              </div>
            ) : !employeeRows.length ? (
              <div className="py-12 text-center text-sm text-muted-foreground">
                No employee analytics available for the selected filters.
              </div>
            ) : (
              <div className="space-y-3">
                {employeeRows.slice(0, 6).map((row) => (
                  <div
                    key={row.employeeId}
                    className="flex items-center justify-between rounded-2xl border border-border/60 px-4 py-3"
                  >
                    <div>
                      <p className="font-semibold text-foreground">{row.employeeName}</p>
                      <p className="text-sm text-muted-foreground">
                        {row.employeeCode} | {row.department}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-mono text-sm font-semibold text-foreground">{row.totalHours.toFixed(2)}h</p>
                      <p className="text-xs text-muted-foreground">
                        {row.verified} verified / {row.failed} failed
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm overflow-hidden">
          <CardHeader>
            <CardTitle>{selectedEmployee ? "Date-wise Attendance Report" : "Filtered Attendance Report"}</CardTitle>
            <CardDescription>
              {selectedEmployee
                ? "Detailed daily entries and exits for the selected employee."
                : "Recent filtered attendance records across all employees."}
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader className="bg-muted/50">
                <TableRow>
                  <TableHead className="pl-6">Date</TableHead>
                  <TableHead>Employee</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Time Window</TableHead>
                  <TableHead>Total</TableHead>
                  <TableHead className="pr-6">Device</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {attendanceLoading ? (
                  Array.from({ length: 6 }).map((_, index) => (
                    <TableRow key={index}>
                      <TableCell className="pl-6"><Skeleton className="h-5 w-24" /></TableCell>
                      <TableCell><Skeleton className="h-8 w-40" /></TableCell>
                      <TableCell><Skeleton className="h-6 w-24" /></TableCell>
                      <TableCell><Skeleton className="h-5 w-28" /></TableCell>
                      <TableCell><Skeleton className="h-5 w-20" /></TableCell>
                      <TableCell className="pr-6"><Skeleton className="h-5 w-20" /></TableCell>
                    </TableRow>
                  ))
                ) : !filteredAttendances?.length ? (
                  <TableRow>
                    <TableCell colSpan={6} className="py-16 text-center text-sm text-muted-foreground">
                      No attendance rows found for the active dashboard filters.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredAttendances.slice(0, 12).map((record) => {
                    const entryTime = record.entryTime ? format(new Date(record.entryTime), "hh:mm a") : "-";
                    const exitTime = record.exitTime ? format(new Date(record.exitTime), "hh:mm a") : "-";

                    return (
                      <TableRow key={record.id} className="hover:bg-muted/30">
                        <TableCell className="pl-6 font-medium text-foreground">{record.date}</TableCell>
                        <TableCell>
                          {record.employee ? (
                            <div className="flex flex-col">
                              <span className="font-semibold text-foreground">{record.employee.name}</span>
                              <span className="text-xs text-muted-foreground">{record.employee.employeeCode}</span>
                            </div>
                          ) : (
                            <span className="text-muted-foreground italic">Unknown request</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{getAttendanceStatusLabel(record.verificationStatus)}</Badge>
                        </TableCell>
                        <TableCell className="font-mono text-xs text-muted-foreground">
                          {entryTime} - {exitTime}
                        </TableCell>
                        <TableCell className="font-mono text-xs font-medium text-foreground">
                          {record.workingHours != null ? formatWorkingDuration(record.workingHours) : "-"}
                        </TableCell>
                        <TableCell className="pr-6">
                          <Badge variant="outline" className="font-mono text-xs">{record.deviceId}</Badge>
                        </TableCell>
                      </TableRow>
                    );
                  })
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      <Card className="border-border/50 shadow-sm">
        <CardHeader>
          <CardTitle>Live Gate Event Stream</CardTitle>
          <CardDescription>
            Raw UHF RFID reads with direction inference and biometric confidence. This feed reflects the continuous portal-style detection pipeline.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {gateEventsLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map((item) => <Skeleton key={item} className="h-12 w-full" />)}
            </div>
          ) : !gateEvents?.length ? (
            <div className="py-8 text-center text-muted-foreground">No gate events found for the selected filters.</div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Employee / RFID</TableHead>
                  <TableHead>Technology</TableHead>
                  <TableHead>Direction</TableHead>
                  <TableHead>Decision</TableHead>
                  <TableHead>Match</TableHead>
                  <TableHead>Device</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {gateEvents.slice(0, 12).map((event) => {
                  const occurredAt = event.occurredAt ? new Date(event.occurredAt) : null;

                  return (
                    <TableRow key={event.id} className="hover:bg-muted/50 transition-colors">
                      <TableCell className="whitespace-nowrap font-medium">
                        {occurredAt ? format(occurredAt, "dd MMM, hh:mm:ss a") : event.date}
                      </TableCell>
                      <TableCell>
                        {event.employee ? (
                          <div className="flex flex-col">
                            <span className="font-semibold text-foreground">{event.employee.name}</span>
                            <span className="text-xs text-muted-foreground">
                              {event.employee.employeeCode} | {event.employee.department}
                            </span>
                          </div>
                        ) : (
                          <div className="flex flex-col">
                            <span className="italic text-muted-foreground">Unknown employee</span>
                            <span className="text-xs font-mono text-muted-foreground">{event.rfidUid}</span>
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="font-mono text-xs">
                          {getScanTechnologyLabel(event.scanTechnology)}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-col">
                          <span className="font-medium text-foreground">{event.movementDirection ?? "UNKNOWN"}</span>
                          <span className="text-xs text-muted-foreground">
                            {event.movementAxis ?? "none"} | {Math.round((event.movementConfidence ?? 0) * 100)}%
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-col gap-1">
                          {renderGateDecision(event.decision)}
                          <span className="text-xs text-muted-foreground">{event.eventMessage}</span>
                        </div>
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {typeof event.matchConfidence === "number"
                          ? `${(event.matchConfidence * 100).toFixed(1)}%`
                          : "--"}
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-col gap-1">
                          <Badge variant="outline" className="w-fit font-mono text-xs">{event.deviceId}</Badge>
                          <div className="text-xs text-muted-foreground">
                            {renderRecentStatus(event.verificationStatus)}
                          </div>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
