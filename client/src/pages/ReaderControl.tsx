import { useEffect, useMemo, useState } from "react";
import { format } from "date-fns";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Cable,
  Play,
  Radio,
  RefreshCcw,
  ScanLine,
  Settings2,
  SlidersHorizontal,
  Square,
  Tag,
} from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { useToast } from "@/hooks/use-toast";
import {
  fetchRfidActiveTags,
  fetchRfidRegistrationTag,
  fetchRfidTags,
  type RfidMode,
  rfidQueryKeys,
  setRfidMode,
  setRfidPower,
  startRfidReader,
  stopRfidReader,
} from "@/lib/rfid";


function formatReaderTimestamp(value: number | null | undefined) {
  if (!value) {
    return "--";
  }
  return format(new Date(value * 1000), "dd MMM yyyy, hh:mm:ss a");
}

function StatusBadge(props: { active: boolean; activeLabel: string; idleLabel: string }) {
  if (props.active) {
    return <Badge className="bg-emerald-500 hover:bg-emerald-600 text-white">{props.activeLabel}</Badge>;
  }

  return <Badge variant="outline">{props.idleLabel}</Badge>;
}

export default function ReaderControl() {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [port, setPort] = useState("COM3");
  const [baudrate, setBaudrate] = useState("57600");
  const [powerValue, setPowerValue] = useState([30]);
  const [selectedMode, setSelectedMode] = useState<RfidMode>("normal");

  const tagsQuery = useQuery({
    queryKey: rfidQueryKeys.tags,
    queryFn: fetchRfidTags,
    refetchInterval: 2000,
  });

  const activeTagsQuery = useQuery({
    queryKey: rfidQueryKeys.activeTags,
    queryFn: fetchRfidActiveTags,
    enabled: Boolean(tagsQuery.data?.running),
    refetchInterval: 2000,
  });

  const registrationQuery = useQuery({
    queryKey: rfidQueryKeys.registrationTag,
    queryFn: fetchRfidRegistrationTag,
    enabled: Boolean(tagsQuery.data?.running),
    refetchInterval: 1500,
  });

  useEffect(() => {
    if (!tagsQuery.data) {
      return;
    }

    setPort(tagsQuery.data.port);
    setBaudrate(String(tagsQuery.data.baudrate));
    setPowerValue([tagsQuery.data.current_power]);
    setSelectedMode(tagsQuery.data.current_mode);
  }, [
    tagsQuery.data?.port,
    tagsQuery.data?.baudrate,
    tagsQuery.data?.current_power,
    tagsQuery.data?.current_mode,
  ]);

  const refreshRfidQueries = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: rfidQueryKeys.tags }),
      queryClient.invalidateQueries({ queryKey: rfidQueryKeys.activeTags }),
      queryClient.invalidateQueries({ queryKey: rfidQueryKeys.registrationTag }),
    ]);
  };

  const connectMutation = useMutation({
    mutationFn: () => startRfidReader({ port, baudrate: Number(baudrate), debug_raw: false }),
    onSuccess: async () => {
      await refreshRfidQueries();
      toast({
        title: "Reader connected",
        description: `Listening on ${port} at ${baudrate} baud.`,
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to connect reader",
        description: error instanceof Error ? error.message : "RFID service could not start the reader.",
        variant: "destructive",
      });
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: stopRfidReader,
    onSuccess: async () => {
      await refreshRfidQueries();
      toast({
        title: "Reader disconnected",
        description: "RFID reading has been stopped.",
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to stop reader",
        description: error instanceof Error ? error.message : "RFID service could not stop the reader.",
        variant: "destructive",
      });
    },
  });

  const powerMutation = useMutation({
    mutationFn: () => setRfidPower(powerValue[0] ?? 30),
    onSuccess: async () => {
      await refreshRfidQueries();
      toast({
        title: "Power updated",
        description: `Reader power set to ${powerValue[0] ?? 30}.`,
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to set power",
        description: error instanceof Error ? error.message : "Power command failed.",
        variant: "destructive",
      });
    },
  });

  const modeMutation = useMutation({
    mutationFn: () => setRfidMode(selectedMode),
    onSuccess: async () => {
      await refreshRfidQueries();
      toast({
        title: "Mode updated",
        description: `Reader switched to ${selectedMode} mode.`,
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to set mode",
        description: error instanceof Error ? error.message : "Mode change failed.",
        variant: "destructive",
      });
    },
  });

  const status = tagsQuery.data;
  const registration = registrationQuery.data?.registration ?? status?.registration;
  const activeTags = activeTagsQuery.data?.active_tags ?? [];
  const lastTag = status?.last_detected_tag ?? "--";
  const isBusy = useMemo(() => {
    return connectMutation.isPending
      || disconnectMutation.isPending
      || powerMutation.isPending
      || modeMutation.isPending;
  }, [
    connectMutation.isPending,
    disconnectMutation.isPending,
    modeMutation.isPending,
    powerMutation.isPending,
  ]);

  const serviceOffline = tagsQuery.isError;
  const readerRunning = Boolean(status?.running);
  const registrationSelected = selectedMode === "registration" || status?.current_mode === "registration";

  return (
    <div className="space-y-8 p-6 md:p-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">RFID Reader Control Panel</h1>
          <p className="mt-1 text-muted-foreground">
            Connect the UHF reader, tune power and mode profiles, and watch the live EPC stream without leaving the dashboard.
          </p>
        </div>
        <Button
          variant="outline"
          onClick={() => {
            void refreshRfidQueries();
          }}
          disabled={isBusy}
        >
          <RefreshCcw className="mr-2 size-4" />
          Refresh Status
        </Button>
      </div>

      {serviceOffline && (
        <Alert className="border-amber-200 bg-amber-50">
          <AlertTriangle className="size-4 text-amber-700" />
          <AlertTitle>RFID service is not reachable</AlertTitle>
          <AlertDescription>
            Start the FastAPI service from <span className="font-mono">rfid_service/main.py</span>, then refresh this page.
          </AlertDescription>
        </Alert>
      )}

      {registrationSelected && (
        <Alert className="border-sky-200 bg-sky-50">
          <Radio className="size-4 text-sky-700" />
          <AlertTitle>Registration Mode</AlertTitle>
          <AlertDescription>Keep only one tag near the reader.</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Cable className="size-4 text-primary" />
              Connection
            </CardTitle>
            <CardDescription>Choose the serial port and start the background reader thread.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="rfid-port">COM Port</Label>
                <Input
                  id="rfid-port"
                  value={port}
                  onChange={(event) => setPort(event.target.value.toUpperCase())}
                  placeholder="COM3"
                  disabled={readerRunning || isBusy}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rfid-baudrate">Baud Rate</Label>
                <Input
                  id="rfid-baudrate"
                  value={baudrate}
                  onChange={(event) => setBaudrate(event.target.value)}
                  placeholder="57600"
                  disabled={readerRunning || isBusy}
                />
              </div>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row">
              <Button
                className="sm:min-w-44"
                onClick={() => {
                  if (readerRunning) {
                    disconnectMutation.mutate();
                    return;
                  }
                  connectMutation.mutate();
                }}
                disabled={isBusy}
              >
                {readerRunning ? (
                  <>
                    <Square className="mr-2 size-4" />
                    Disconnect
                  </>
                ) : (
                  <>
                    <Play className="mr-2 size-4" />
                    Connect
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                onClick={() => connectMutation.mutate()}
                disabled={readerRunning || isBusy}
              >
                <Play className="mr-2 size-4" />
                Start Reader
              </Button>
              <Button
                variant="outline"
                onClick={() => disconnectMutation.mutate()}
                disabled={!readerRunning || isBusy}
              >
                <Square className="mr-2 size-4" />
                Stop Reader
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <ScanLine className="size-4 text-primary" />
              Reader Status
            </CardTitle>
            <CardDescription>Live state from the Python RFID service.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            {tagsQuery.isLoading ? (
              Array.from({ length: 6 }).map((_, index) => <Skeleton key={index} className="h-20 w-full" />)
            ) : (
              <>
                <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Connection</p>
                  <div className="mt-2">
                    <StatusBadge
                      active={Boolean(status?.connected)}
                      activeLabel="Connected"
                      idleLabel="Disconnected"
                    />
                  </div>
                  <p className="mt-3 text-sm text-muted-foreground">{status?.port ?? port}</p>
                </div>
                <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Current Mode</p>
                  <p className="mt-2 text-lg font-semibold capitalize text-foreground">
                    {status?.current_mode ?? selectedMode}
                  </p>
                  <p className="mt-1 text-sm text-muted-foreground">Continuous scan, registration, or trigger workflow.</p>
                </div>
                <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Current Power</p>
                  <p className="mt-2 text-lg font-semibold text-foreground">{status?.current_power ?? powerValue[0]}</p>
                  <p className="mt-1 text-sm text-muted-foreground">Supported range is 0 to 30.</p>
                </div>
                <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Last Detected Tag</p>
                  <p className="mt-2 break-all font-mono text-sm text-foreground">{lastTag}</p>
                  <p className="mt-1 text-sm text-muted-foreground">{formatReaderTimestamp(status?.last_detected_at)}</p>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_1fr_1.15fr]">
        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <SlidersHorizontal className="size-4 text-primary" />
              Power Control
            </CardTitle>
            <CardDescription>Fine-tune read range and collision behavior.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Power Level</span>
                <span className="font-semibold text-foreground">{powerValue[0]}</span>
              </div>
              <Slider
                min={0}
                max={30}
                step={1}
                value={powerValue}
                onValueChange={setPowerValue}
                disabled={!readerRunning || isBusy}
              />
            </div>
            <Button onClick={() => powerMutation.mutate()} disabled={!readerRunning || isBusy}>
              Set Power
            </Button>
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Settings2 className="size-4 text-primary" />
              Mode Selection
            </CardTitle>
            <CardDescription>Switch between multi-tag scanning and single-tag registration behavior.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="reader-mode">Mode</Label>
              <Select value={selectedMode} onValueChange={(value) => setSelectedMode(value as RfidMode)}>
                <SelectTrigger id="reader-mode">
                  <SelectValue placeholder="Choose mode" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="normal">Normal Mode</SelectItem>
                  <SelectItem value="registration">Registration Mode</SelectItem>
                  <SelectItem value="trigger">Trigger Mode</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button onClick={() => modeMutation.mutate()} disabled={!readerRunning || isBusy}>
              Apply Mode
            </Button>
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Radio className="size-4 text-primary" />
              Registration Capture
            </CardTitle>
            <CardDescription>Stable single-tag detection state for enrollment workflows.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Selected Tag</p>
              <p className="mt-2 break-all font-mono text-sm text-foreground">
                {registration?.selected_tag ?? "Waiting for a stable single-tag read"}
              </p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-2xl border border-border/60 p-4">
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Candidate</p>
                <p className="mt-2 break-all font-mono text-sm text-foreground">{registration?.candidate_tag ?? "--"}</p>
              </div>
              <div className="rounded-2xl border border-border/60 p-4">
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Stability</p>
                <p className="mt-2 text-sm font-semibold text-foreground">
                  {registration ? `${registration.candidate_hits}/${registration.stable_threshold}` : "--"}
                </p>
              </div>
            </div>
            <Alert className="border-border/60 bg-white/80">
              <Tag className="size-4 text-primary" />
              <AlertTitle>Registration state</AlertTitle>
              <AlertDescription>{registration?.message ?? "Registration data is not available yet."}</AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Tag className="size-4 text-primary" />
              Recent EPC Tags
            </CardTitle>
            <CardDescription>Clean EPC values extracted from the live binary stream.</CardDescription>
          </CardHeader>
          <CardContent>
            {!status?.recent_tags?.length ? (
              <div className="rounded-2xl border border-dashed border-border/70 px-5 py-10 text-center text-sm text-muted-foreground">
                No EPC tags have been detected yet.
              </div>
            ) : (
              <div className="space-y-3">
                {status.recent_tags.slice(0, 10).map((tag) => (
                  <div key={`${tag.epc}-${tag.seen_at}`} className="rounded-2xl border border-border/60 p-4">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                      <div className="min-w-0">
                        <p className="break-all font-mono text-sm font-semibold text-foreground">{tag.epc}</p>
                        <p className="mt-1 text-xs text-muted-foreground">{formatReaderTimestamp(tag.seen_at)}</p>
                      </div>
                      <Badge variant="outline">Observed</Badge>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <ScanLine className="size-4 text-primary" />
              Active Tags
            </CardTitle>
            <CardDescription>Tags that are still inside the active ENTRY window.</CardDescription>
          </CardHeader>
          <CardContent>
            {activeTagsQuery.isLoading && readerRunning ? (
              <div className="space-y-3">
                {Array.from({ length: 4 }).map((_, index) => <Skeleton key={index} className="h-20 w-full" />)}
              </div>
            ) : !activeTags.length ? (
              <div className="rounded-2xl border border-dashed border-border/70 px-5 py-10 text-center text-sm text-muted-foreground">
                No active tags right now.
              </div>
            ) : (
              <div className="space-y-3">
                {activeTags.map((tag) => (
                  <div key={tag.epc} className="rounded-2xl border border-border/60 p-4">
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                      <div className="min-w-0">
                        <p className="break-all font-mono text-sm font-semibold text-foreground">{tag.epc}</p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          First seen {formatReaderTimestamp(tag.first_seen_at)}
                        </p>
                      </div>
                      <div className="text-left sm:text-right">
                        <p className="text-sm font-semibold text-foreground">{tag.detections} detections</p>
                        <p className="text-xs text-muted-foreground">Idle {tag.idle_seconds.toFixed(2)}s</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {status?.last_error && (
        <Alert className="border-rose-200 bg-rose-50">
          <AlertTriangle className="size-4 text-rose-700" />
          <AlertTitle>Reader warning</AlertTitle>
          <AlertDescription>{status.last_error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
