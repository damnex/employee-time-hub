import { useDeferredValue, useMemo, useState } from "react";
import { format } from "date-fns";
import { Download, Filter, RotateCcw, Search, UsersRound } from "lucide-react";
import { useAttendances } from "@/hooks/use-attendances";
import { useEmployees } from "@/hooks/use-employees";
import {
  type AttendanceStatus,
  calculateAttendanceSummary,
  exportAttendanceRows,
  formatWorkingDuration,
  getAttendanceStatusLabel,
} from "@/lib/reporting";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type AttendanceStatusFilter = "all" | AttendanceStatus;

function renderStatusBadge(status: AttendanceStatus | string) {
  switch (status) {
    case "ENTRY":
      return <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white">{getAttendanceStatusLabel(status)}</Badge>;
    case "EXIT":
      return <Badge variant="secondary" className="bg-blue-100 text-blue-800">{getAttendanceStatusLabel(status)}</Badge>;
    case "FAILED_FACE":
    case "FAILED_DIRECTION":
    case "UNKNOWN_RFID":
      return <Badge variant="destructive">{getAttendanceStatusLabel(status)}</Badge>;
    default:
      return <Badge variant="outline">{status}</Badge>;
  }
}

function SummaryCard(props: { label: string; value: string; hint: string }) {
  return (
    <Card className="border-border/50 shadow-sm">
      <CardContent className="p-5">
        <p className="text-sm text-muted-foreground">{props.label}</p>
        <p className="mt-2 text-3xl font-semibold text-foreground">{props.value}</p>
        <p className="mt-1 text-xs text-muted-foreground">{props.hint}</p>
      </CardContent>
    </Card>
  );
}

export default function Attendance() {
  const [search, setSearch] = useState("");
  const [employeeFilter, setEmployeeFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState<AttendanceStatusFilter>("all");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const deferredSearch = useDeferredValue(search.trim());
  const { data: employees } = useEmployees();
  const filters = useMemo(() => {
    return {
      search: deferredSearch || undefined,
      employeeId: employeeFilter !== "all" ? Number(employeeFilter) : undefined,
      status: statusFilter !== "all" ? statusFilter : undefined,
      dateFrom: dateFrom || undefined,
      dateTo: dateTo || undefined,
    };
  }, [dateFrom, dateTo, deferredSearch, employeeFilter, statusFilter]);
  const { data: logs, isLoading } = useAttendances(filters);
  const summary = useMemo(() => calculateAttendanceSummary(logs ?? []), [logs]);

  const handleResetFilters = () => {
    setSearch("");
    setEmployeeFilter("all");
    setStatusFilter("all");
    setDateFrom("");
    setDateTo("");
  };

  const handleExport = () => {
    if (!logs?.length) {
      return;
    }

    exportAttendanceRows(logs, "attendance-report.csv");
  };

  return (
    <div className="p-6 md:p-8 space-y-6 animate-in fade-in duration-500">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Attendance Logs</h1>
          <p className="text-muted-foreground mt-1">
            Search, filter, and export attendance records for all employees or a specific person.
          </p>
        </div>
        <Button onClick={handleExport} disabled={!logs?.length} className="shadow-sm">
          <Download className="mr-2 size-4" />
          Download Filtered Report
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        <SummaryCard
          label="Filtered Records"
          value={String(summary.totalRecords)}
          hint="Current result set after filters"
        />
        <SummaryCard
          label="Verified Access"
          value={String(summary.verifiedEntries + summary.verifiedExits)}
          hint={`${summary.verifiedEntries} entries, ${summary.verifiedExits} exits`}
        />
        <SummaryCard
          label="Failed Events"
          value={String(summary.failedScans)}
          hint="Face, direction, or badge failures"
        />
        <SummaryCard
          label="Work Hours"
          value={`${summary.totalHours.toFixed(2)}h`}
          hint={`${summary.averageHoursPerDay.toFixed(2)}h average per day`}
        />
      </div>

      <Card className="border-border/50 shadow-sm">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Filter className="size-4 text-primary" />
            Search And Filters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-[1.6fr_1fr_1fr_1fr_1fr_auto]">
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                className="pl-9"
                placeholder="Search by employee, code, department, device, or badge"
              />
            </div>
            <Select value={employeeFilter} onValueChange={setEmployeeFilter}>
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
            <Select value={statusFilter} onValueChange={(value) => setStatusFilter(value as AttendanceStatusFilter)}>
              <SelectTrigger>
                <SelectValue placeholder="All statuses" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All statuses</SelectItem>
                <SelectItem value="ENTRY">Verified entry</SelectItem>
                <SelectItem value="EXIT">Verified exit</SelectItem>
                <SelectItem value="FAILED_FACE">Biometric mismatch</SelectItem>
                <SelectItem value="FAILED_DIRECTION">Direction unclear</SelectItem>
                <SelectItem value="UNKNOWN_RFID">Unknown badge</SelectItem>
              </SelectContent>
            </Select>
            <Input type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} />
            <Input type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} />
            <Button variant="outline" onClick={handleResetFilters}>
              <RotateCcw className="mr-2 size-4" />
              Reset
            </Button>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
            <span className="rounded-full bg-muted px-3 py-1">
              Employees in result: {summary.activeEmployees}
            </span>
            <span className="rounded-full bg-muted px-3 py-1">
              Active days: {summary.activeDays}
            </span>
            <span className="rounded-full bg-muted px-3 py-1">
              Search syncs with exports
            </span>
          </div>
        </CardContent>
      </Card>

      <Card className="border-border/50 shadow-sm overflow-hidden">
        <CardContent className="p-0">
          <Table>
            <TableHeader className="bg-muted/50">
              <TableRow>
                <TableHead className="pl-6 w-[130px]">Date</TableHead>
                <TableHead>Employee</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Entry Time</TableHead>
                <TableHead>Exit Time</TableHead>
                <TableHead>Total Time</TableHead>
                <TableHead className="pr-6">Device</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array.from({ length: 8 }).map((_, index) => (
                  <TableRow key={index}>
                    <TableCell className="pl-6"><Skeleton className="h-5 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-8 w-44" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-28" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-16" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-16" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-20" /></TableCell>
                    <TableCell className="pr-6"><Skeleton className="h-5 w-24" /></TableCell>
                  </TableRow>
                ))
              ) : !logs?.length ? (
                <TableRow>
                  <TableCell colSpan={7} className="py-20 text-center">
                    <div className="mx-auto flex max-w-sm flex-col items-center gap-3 text-muted-foreground">
                      <UsersRound className="size-10 opacity-30" />
                      <p className="text-base font-medium text-foreground">No attendance records match these filters.</p>
                      <p className="text-sm">
                        Adjust the employee, date range, or search text and try again.
                      </p>
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                logs.map((log) => {
                  const entryObj = log.entryTime ? new Date(log.entryTime) : null;
                  const exitObj = log.exitTime ? new Date(log.exitTime) : null;

                  return (
                    <TableRow key={log.id} className="hover:bg-muted/30">
                      <TableCell className="pl-6 font-medium text-foreground">{log.date}</TableCell>
                      <TableCell>
                        {log.employee ? (
                          <div className="flex flex-col">
                            <span className="font-semibold text-foreground">{log.employee.name}</span>
                            <span className="text-xs text-muted-foreground">
                              {log.employee.employeeCode} | {log.employee.department}
                            </span>
                          </div>
                        ) : (
                          <span className="text-muted-foreground italic">Unknown request</span>
                        )}
                      </TableCell>
                      <TableCell>{renderStatusBadge(log.verificationStatus)}</TableCell>
                      <TableCell>
                        {entryObj ? format(entryObj, "HH:mm:ss") : <span className="text-muted-foreground">-</span>}
                      </TableCell>
                      <TableCell>
                        {exitObj ? format(exitObj, "HH:mm:ss") : <span className="text-muted-foreground">-</span>}
                      </TableCell>
                      <TableCell>
                        {log.workingHours != null ? (
                          <span className="font-mono font-medium">{formatWorkingDuration(log.workingHours)}</span>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </TableCell>
                      <TableCell className="pr-6">
                        <Badge variant="outline" className="font-mono text-xs">{log.deviceId}</Badge>
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
  );
}
