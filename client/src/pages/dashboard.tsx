import { useDashboardStats } from "@/hooks/use-stats";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Users, UserCheck, UserMinus, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

export default function Dashboard() {
  const { data: stats, isLoading, error } = useDashboardStats();

  if (error) {
    return (
      <div className="p-8 flex flex-col items-center justify-center h-full text-center">
        <ShieldAlert className="size-12 text-destructive mb-4" />
        <h2 className="text-2xl font-bold text-foreground">Failed to load dashboard</h2>
        <p className="text-muted-foreground mt-2">There was a problem connecting to the server.</p>
      </div>
    );
  }

  return (
    <div className="p-6 md:p-8 space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Today's Overview</h1>
        <p className="text-muted-foreground mt-1">Real-time attendance and facility access metrics.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="shadow-sm border-border/50">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Employees</CardTitle>
            <Users className="size-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <Skeleton className="h-8 w-16 mt-1" />
            ) : (
              <div className="text-3xl font-bold">{stats?.totalEmployees || 0}</div>
            )}
          </CardContent>
        </Card>

        <Card className="shadow-sm border-border/50 relative overflow-hidden">
          <div className="absolute inset-0 bg-emerald-500/5 dark:bg-emerald-500/10 pointer-events-none" />
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium text-emerald-700 dark:text-emerald-400">Present Today</CardTitle>
            <UserCheck className="size-4 text-emerald-600 dark:text-emerald-400" />
          </CardHeader>
          <CardContent className="relative">
            {isLoading ? (
              <Skeleton className="h-8 w-16 mt-1" />
            ) : (
              <div className="text-3xl font-bold text-emerald-700 dark:text-emerald-400">{stats?.presentToday || 0}</div>
            )}
          </CardContent>
        </Card>

        <Card className="shadow-sm border-border/50 relative overflow-hidden">
          <div className="absolute inset-0 bg-rose-500/5 dark:bg-rose-500/10 pointer-events-none" />
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
            <CardTitle className="text-sm font-medium text-rose-700 dark:text-rose-400">Absent / Unknown</CardTitle>
            <UserMinus className="size-4 text-rose-600 dark:text-rose-400" />
          </CardHeader>
          <CardContent className="relative">
            {isLoading ? (
              <Skeleton className="h-8 w-16 mt-1" />
            ) : (
              <div className="text-3xl font-bold text-rose-700 dark:text-rose-400">{stats?.absentToday || 0}</div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="shadow-sm border-border/50 col-span-full">
        <CardHeader>
          <CardTitle>Recent Gate Activity</CardTitle>
          <CardDescription>Live feed of the most recent access logs across all gates.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map(i => <Skeleton key={i} className="h-12 w-full" />)}
            </div>
          ) : !stats?.recentScans || stats.recentScans.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No recent gate activity.
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Time</TableHead>
                  <TableHead>Employee</TableHead>
                  <TableHead>Device</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {stats.recentScans.map((scan) => {
                  // Determine display time based on entry/exit or just current update
                  const timeStr = scan.entryTime ? scan.entryTime : (scan.exitTime || new Date());
                  const dateObj = new Date(timeStr);
                  
                  return (
                    <TableRow key={scan.id} className="hover:bg-muted/50 transition-colors">
                      <TableCell className="font-medium whitespace-nowrap">
                        {format(dateObj, "HH:mm:ss a")}
                      </TableCell>
                      <TableCell>
                        {scan.employee ? (
                          <div className="flex flex-col">
                            <span className="font-semibold text-foreground">{scan.employee.name}</span>
                            <span className="text-xs text-muted-foreground">{scan.employee.department}</span>
                          </div>
                        ) : (
                          <span className="text-muted-foreground italic">Unknown User</span>
                        )}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="font-mono text-xs">{scan.deviceId}</Badge>
                      </TableCell>
                      <TableCell>
                        {scan.verificationStatus === 'ENTRY' && <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white">Entry Validated</Badge>}
                        {scan.verificationStatus === 'EXIT' && <Badge variant="secondary" className="bg-blue-100 text-blue-800 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-100">Exit Validated</Badge>}
                        {scan.verificationStatus === 'FAILED_FACE' && <Badge variant="destructive">Face Mismatch</Badge>}
                        {scan.verificationStatus === 'UNKNOWN_RFID' && <Badge variant="destructive">Invalid RFID</Badge>}
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
