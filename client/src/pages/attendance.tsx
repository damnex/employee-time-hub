import { useAttendances } from "@/hooks/use-attendances";
import { format } from "date-fns";
import { Card, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Clock3, CalendarDays } from "lucide-react";

export default function Attendance() {
  const { data: logs, isLoading } = useAttendances();

  return (
    <div className="p-6 md:p-8 space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Attendance Logs</h1>
        <p className="text-muted-foreground mt-1">Comprehensive audit trail of all entry and exit events.</p>
      </div>

      <Card className="border-border/50 shadow-sm overflow-hidden">
        <CardContent className="p-0">
          <Table>
            <TableHeader className="bg-muted/50">
              <TableRow>
                <TableHead className="pl-6 w-[150px]">Date</TableHead>
                <TableHead>Employee</TableHead>
                <TableHead>Event Status</TableHead>
                <TableHead>Entry Time</TableHead>
                <TableHead>Exit Time</TableHead>
                <TableHead>Total Hours</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array.from({ length: 6 }).map((_, i) => (
                  <TableRow key={i}>
                    <TableCell className="pl-6"><Skeleton className="h-5 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-9 w-40" /></TableCell>
                    <TableCell><Skeleton className="h-6 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-20" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-20" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-16" /></TableCell>
                  </TableRow>
                ))
              ) : logs?.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-16 text-muted-foreground">
                    <CalendarDays className="size-10 mx-auto mb-3 opacity-20" />
                    No attendance records found.
                  </TableCell>
                </TableRow>
              ) : (
                logs?.map((log) => {
                  const entryObj = log.entryTime ? new Date(log.entryTime) : null;
                  const exitObj = log.exitTime ? new Date(log.exitTime) : null;

                  return (
                    <TableRow key={log.id} className="hover:bg-muted/30">
                      <TableCell className="pl-6 font-medium text-foreground">
                        {log.date}
                      </TableCell>
                      <TableCell>
                        {log.employee ? (
                          <div className="flex flex-col">
                            <span className="font-semibold text-foreground">{log.employee.name}</span>
                            <span className="text-xs text-muted-foreground">{log.employee.employeeCode}</span>
                          </div>
                        ) : (
                          <span className="text-muted-foreground italic">Unknown Request</span>
                        )}
                      </TableCell>
                      <TableCell>
                        {log.verificationStatus === 'ENTRY' && <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white">Verified Entry</Badge>}
                        {log.verificationStatus === 'EXIT' && <Badge variant="secondary" className="bg-blue-100 text-blue-800">Verified Exit</Badge>}
                        {log.verificationStatus === 'FAILED_FACE' && <Badge variant="destructive">Biometric Mismatch</Badge>}
                        {log.verificationStatus === 'UNKNOWN_RFID' && <Badge variant="destructive">Unknown Badge</Badge>}
                      </TableCell>
                      <TableCell>
                        {entryObj ? (
                          <span className="flex items-center text-sm"><Clock3 className="size-3 mr-1.5 text-muted-foreground" /> {format(entryObj, "HH:mm")}</span>
                        ) : <span className="text-muted-foreground">-</span>}
                      </TableCell>
                      <TableCell>
                        {exitObj ? (
                           <span className="flex items-center text-sm"><Clock3 className="size-3 mr-1.5 text-muted-foreground" /> {format(exitObj, "HH:mm")}</span>
                        ) : <span className="text-muted-foreground">-</span>}
                      </TableCell>
                      <TableCell>
                        {log.workingHours != null ? (
                          <span className="font-mono font-medium">{log.workingHours.toFixed(2)}h</span>
                        ) : <span className="text-muted-foreground">-</span>}
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
