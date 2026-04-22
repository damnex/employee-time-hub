import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Cable,
  Play,
  RefreshCcw,
  Settings2,
  SlidersHorizontal,
  Square,
} from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/hooks/use-toast";
import {
  detectRfidPort,
  connectRfidReader,
  disconnectRfidReader,
  fetchRfidStatus,
  type RfidMode,
  type RfidTransportMode,
  rfidQueryKeys,
  setRfidMode,
  setRfidBuzzer,
  setRfidPower,
  setRfidTransportMode,
  startRfidReader,
  stopRfidReader,
} from "@/lib/rfid";

export default function ReaderControl() {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [port, setPort] = useState("COM3");
  const [baudrate, setBaudrate] = useState("57600");
  const [powerValue, setPowerValue] = useState([30]);
  const [selectedMode, setSelectedMode] = useState<RfidMode>("normal");
  const [selectedTransportMode, setSelectedTransportMode] = useState<RfidTransportMode>("scan");
  const [selectedBuzzerEnabled, setSelectedBuzzerEnabled] = useState(false);

  const statusQuery = useQuery({
    queryKey: rfidQueryKeys.status,
    queryFn: fetchRfidStatus,
    refetchInterval: 2000,
  });

  useEffect(() => {
    if (!statusQuery.data) {
      return;
    }

    setPort(statusQuery.data.port);
    setBaudrate(String(statusQuery.data.baudrate));
    setPowerValue([statusQuery.data.current_power]);
    setSelectedMode(statusQuery.data.current_mode);
    setSelectedTransportMode(statusQuery.data.transport_mode);
    setSelectedBuzzerEnabled(statusQuery.data.buzzer_enabled);
  }, [
    statusQuery.data?.baudrate,
    statusQuery.data?.buzzer_enabled,
    statusQuery.data?.current_mode,
    statusQuery.data?.current_power,
    statusQuery.data?.port,
    statusQuery.data?.transport_mode,
  ]);

  const refreshRfidQueries = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: rfidQueryKeys.status }),
      queryClient.invalidateQueries({ queryKey: rfidQueryKeys.tags }),
      queryClient.invalidateQueries({ queryKey: rfidQueryKeys.activeTags }),
      queryClient.invalidateQueries({ queryKey: rfidQueryKeys.registrationTag }),
    ]);
  };

  const connectMutation = useMutation({
    mutationFn: () => connectRfidReader({ port, baudrate: Number(baudrate), debug_raw: false }),
    onSuccess: async (data) => {
      await refreshRfidQueries();
      toast({
        title: "Reader connected",
        description: `Connected on ${data.port} at ${data.baudrate} baud.`,
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to connect reader",
        description: error instanceof Error ? error.message : "Reader connection failed.",
        variant: "destructive",
      });
    },
  });

  const disconnectMutation = useMutation({
    mutationFn: disconnectRfidReader,
    onSuccess: async () => {
      await refreshRfidQueries();
      toast({
        title: "Reader disconnected",
        description: "Serial connection has been closed.",
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to disconnect reader",
        description: error instanceof Error ? error.message : "Disconnect failed.",
        variant: "destructive",
      });
    },
  });

  const detectPortMutation = useMutation({
    mutationFn: () => detectRfidPort({ baudrate: Number(baudrate), debug_raw: false }),
    onSuccess: (data) => {
      setPort(data.detected_port.device);
      toast({
        title: "Port detected",
        description: `Reader found on ${data.detected_port.device}.`,
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to detect reader port",
        description: error instanceof Error ? error.message : "Auto-detect failed.",
        variant: "destructive",
      });
    },
  });

  const startMutation = useMutation({
    mutationFn: () => startRfidReader({ port, baudrate: Number(baudrate), debug_raw: false }),
    onSuccess: async (data) => {
      await refreshRfidQueries();
      toast({
        title: "Reader started",
        description: `Continuous UHF reading is active on ${data.port}.`,
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to start reader",
        description: error instanceof Error ? error.message : "Start failed.",
        variant: "destructive",
      });
    },
  });

  const stopMutation = useMutation({
    mutationFn: stopRfidReader,
    onSuccess: async () => {
      await refreshRfidQueries();
      toast({
        title: "Reader stopped",
        description: "Continuous UHF reading has been paused.",
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to stop reader",
        description: error instanceof Error ? error.message : "Stop failed.",
        variant: "destructive",
      });
    },
  });

  const powerMutation = useMutation({
    mutationFn: () => setRfidPower(powerValue[0] ?? 30),
    onSuccess: async (data) => {
      await refreshRfidQueries();
      toast({
        title: "Power updated",
        description:
          data.current_power === (powerValue[0] ?? 30)
            ? `Reader power set to ${data.current_power}.`
            : `Reader kept power ${data.current_power} after requesting ${powerValue[0] ?? 30}.`,
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to set power",
        description: error instanceof Error ? error.message : "Power update failed.",
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
        description: error instanceof Error ? error.message : "Mode update failed.",
        variant: "destructive",
      });
    },
  });

  const transportModeMutation = useMutation({
    mutationFn: () => setRfidTransportMode(selectedTransportMode),
    onSuccess: async () => {
      await refreshRfidQueries();
      toast({
        title: "Transport updated",
        description: `Reader switched to ${selectedTransportMode} transport.`,
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to set transport",
        description: error instanceof Error ? error.message : "Transport update failed.",
        variant: "destructive",
      });
    },
  });

  const buzzerMutation = useMutation({
    mutationFn: () => setRfidBuzzer(selectedBuzzerEnabled),
    onSuccess: async () => {
      await refreshRfidQueries();
      toast({
        title: "Buzzer updated",
        description: `Reader buzzer ${selectedBuzzerEnabled ? "enabled" : "disabled"}.`,
      });
    },
    onError: (error) => {
      toast({
        title: "Unable to set buzzer",
        description: error instanceof Error ? error.message : "Buzzer update failed.",
        variant: "destructive",
      });
    },
  });

  const status = statusQuery.data;
  const serviceOffline = statusQuery.isError;
  const isConnected = Boolean(status?.connected);
  const isRunning = Boolean(status?.running);
  const effectiveTransportMode = status?.transport_mode ?? selectedTransportMode;
  const buzzerSupported = effectiveTransportMode !== "answer";
  const isBusy = useMemo(() => {
    return detectPortMutation.isPending
      || connectMutation.isPending
      || disconnectMutation.isPending
      || startMutation.isPending
      || stopMutation.isPending
      || powerMutation.isPending
      || modeMutation.isPending
      || buzzerMutation.isPending
      || transportModeMutation.isPending;
  }, [
    buzzerMutation.isPending,
    detectPortMutation.isPending,
    connectMutation.isPending,
    disconnectMutation.isPending,
    modeMutation.isPending,
    powerMutation.isPending,
    startMutation.isPending,
    stopMutation.isPending,
    transportModeMutation.isPending,
  ]);

  return (
    <div className="flex h-full min-h-full min-w-0 flex-col gap-3 overflow-y-auto p-4 md:p-5 animate-in fade-in duration-500">
      <div className="grid gap-3 xl:grid-cols-[minmax(0,420px)_minmax(0,1fr)_auto] xl:items-start">
        <div className="min-w-0 space-y-1">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">UHF Reader Control</h1>
          <p className="max-w-3xl text-sm text-muted-foreground">
            Quick setup for connection, power, mode, transport, and buzzer behavior.
          </p>
        </div>
        <div className="min-w-0 xl:flex xl:justify-center">
          {status?.last_error && (
            <Alert className="w-full max-w-[560px] border-rose-500/35 bg-rose-500/10 py-2.5 text-rose-950 dark:bg-rose-500/12 dark:text-rose-100">
              <AlertTriangle className="size-4 text-rose-700 dark:text-rose-200" />
              <AlertTitle className="text-rose-950 dark:text-rose-100">Reader warning</AlertTitle>
              <AlertDescription className="text-rose-800 dark:text-rose-100/80">{status.last_error}</AlertDescription>
            </Alert>
          )}
        </div>
        <Button
          variant="outline"
          className="h-10 px-4 self-start xl:self-auto"
          onClick={() => {
            void refreshRfidQueries();
          }}
          disabled={isBusy}
        >
          <RefreshCcw className="mr-2 size-4" />
          Refresh
        </Button>
      </div>

      {serviceOffline && (
        <Alert className="shrink-0 border-amber-500/35 bg-amber-500/10 py-2.5 text-amber-950 dark:bg-amber-500/12 dark:text-amber-100">
          <AlertTriangle className="size-4 text-amber-700 dark:text-amber-200" />
          <AlertTitle className="text-amber-950 dark:text-amber-100">RFID service is not reachable</AlertTitle>
          <AlertDescription className="text-amber-800 dark:text-amber-100/80">
            Install <span className="mx-1 font-mono">rfid_service/requirements.txt</span> and restart the server if this stays visible.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
        <div className="rounded-xl border border-border bg-card px-3 py-2.5 shadow-sm">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Connection</p>
          <div className="mt-2">
            <Badge variant={isConnected ? "secondary" : "outline"}>
              {isConnected ? "Connected" : "Disconnected"}
            </Badge>
          </div>
        </div>
        <div className="rounded-xl border border-border bg-card px-3 py-2.5 shadow-sm">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Reader</p>
          <div className="mt-2">
            <Badge variant={isRunning ? "secondary" : "outline"}>
              {isRunning ? "Running" : "Stopped"}
            </Badge>
          </div>
        </div>
        <div className="rounded-xl border border-border bg-card px-3 py-2.5 shadow-sm">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Mode</p>
          <p className="mt-2 text-lg font-semibold capitalize text-foreground">{status?.current_mode ?? selectedMode}</p>
        </div>
        <div className="rounded-xl border border-border bg-card px-3 py-2.5 shadow-sm">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Transport</p>
          <p className="mt-2 text-lg font-semibold capitalize text-foreground">{status?.transport_mode ?? selectedTransportMode}</p>
        </div>
        <div className="rounded-xl border border-border bg-card px-3 py-2.5 shadow-sm">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Power</p>
          <p className="mt-2 text-lg font-semibold text-foreground">{status?.current_power ?? powerValue[0]}</p>
        </div>
        <div className="rounded-xl border border-border bg-card px-3 py-2.5 shadow-sm">
          <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">Buzzer</p>
          <div className="mt-2">
            <Badge
              variant={
                buzzerSupported
                  ? ((status?.buzzer_enabled ?? selectedBuzzerEnabled) ? "secondary" : "outline")
                  : "outline"
              }
            >
              {buzzerSupported
                ? ((status?.buzzer_enabled ?? selectedBuzzerEnabled) ? "Enabled" : "Disabled")
                : "Ignored in Answer"}
            </Badge>
          </div>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 gap-3">
        <div className="grid min-h-0 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] xl:items-stretch">
        <Card className="flex min-w-0 flex-col border-border/50 shadow-sm">
          <CardHeader className="space-y-1 p-4 pb-2.5">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Cable className="size-4 text-primary" />
              Connection
            </CardTitle>
            <CardDescription className="text-[12px] leading-snug">
              COM3 / 57600 is the default baseline for this reader.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-1 flex-col gap-2.5 p-4 pt-0">
            <div className="grid gap-2.5 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] md:items-end">
              <div className="space-y-1.5">
                <Label htmlFor="reader-port">COM Port</Label>
                <Input
                  id="reader-port"
                  className="h-10"
                  value={port}
                  onChange={(event) => setPort(event.target.value.toUpperCase())}
                  placeholder="COM3"
                  disabled={isConnected || isBusy}
                />
              </div>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-10 px-3.5 md:min-w-[122px] md:self-end"
                onClick={() => detectPortMutation.mutate()}
                disabled={isConnected || isBusy}
              >
                Auto Detect
              </Button>
              <div className="space-y-1.5">
                <Label htmlFor="reader-baudrate">Baud Rate</Label>
                <Input
                  id="reader-baudrate"
                  className="h-10"
                  value={baudrate}
                  onChange={(event) => setBaudrate(event.target.value)}
                  placeholder="57600"
                  disabled={isConnected || isBusy}
                />
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              <Button className="h-10" onClick={() => connectMutation.mutate()} disabled={isConnected || isBusy}>
                <Play className="mr-2 size-4" />
                Connect
              </Button>
              <Button variant="outline" className="h-10" onClick={() => disconnectMutation.mutate()} disabled={!isConnected || isBusy}>
                <Square className="mr-2 size-4" />
                Disconnect
              </Button>
              <Button variant="outline" className="h-10" onClick={() => startMutation.mutate()} disabled={!isConnected || isRunning || isBusy}>
                <Play className="mr-2 size-4" />
                Start Reader
              </Button>
              <Button variant="outline" className="h-10" onClick={() => stopMutation.mutate()} disabled={!isRunning || isBusy}>
                <Square className="mr-2 size-4" />
                Stop Reader
              </Button>
            </div>

            <div className="mt-auto grid gap-2 sm:grid-cols-3">
              <div className="rounded-lg border border-border/60 bg-muted/20 px-3 py-2">
                <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Port</p>
                <p className="mt-1 font-semibold text-foreground">{status?.port ?? port}</p>
              </div>
              <div className="rounded-lg border border-border/60 bg-muted/20 px-3 py-2">
                <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Baud</p>
                <p className="mt-1 font-semibold text-foreground">{status?.baudrate ?? Number(baudrate)}</p>
              </div>
              <div className="rounded-lg border border-border/60 bg-muted/20 px-3 py-2">
                <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Transport</p>
                <p className="mt-1 font-semibold capitalize text-foreground">{status?.transport_mode ?? selectedTransportMode}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="flex min-w-0 flex-col overflow-hidden border-border/50 shadow-sm">
          <CardHeader className="space-y-1 p-3.5 pb-2.5">
            <CardTitle className="flex items-center gap-2 text-[1.1rem]">
              <SlidersHorizontal className="size-4 text-primary" />
              Reader Profile
            </CardTitle>
            <CardDescription className="text-[11px] leading-snug">
              Compact controls for mode, transport, and buzzer.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid min-w-0 gap-2 p-3.5 pt-0">
            <div className="grid min-w-0 gap-2 rounded-xl border border-border/60 bg-muted/20 p-2.5 xl:grid-cols-[112px_minmax(0,1fr)_148px] xl:items-end">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-foreground">Mode</p>
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Gate or registration.
                </p>
              </div>
              <div className="min-w-0 space-y-1.5">
                <div className="flex items-center justify-between gap-3">
                  <Label htmlFor="reader-mode">Operating Mode</Label>
                  {selectedMode === "registration" && (
                    <p className="text-[11px] font-medium text-primary/90">
                      P1 / H7
                    </p>
                  )}
                </div>
                <Select value={selectedMode} onValueChange={(value) => setSelectedMode(value as RfidMode)}>
                  <SelectTrigger id="reader-mode" className="h-9 w-full min-w-0">
                    <SelectValue placeholder="Choose mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="normal">Normal Mode</SelectItem>
                    <SelectItem value="registration">Registration Mode</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button className="h-9 w-full" onClick={() => modeMutation.mutate()} disabled={!isConnected || isBusy}>
                Apply Mode
              </Button>
            </div>

            <div className="grid min-w-0 gap-2 rounded-xl border border-border/60 bg-muted/20 p-2.5 xl:grid-cols-[112px_minmax(0,1fr)_148px] xl:items-end">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-foreground">Transport</p>
                <p className="text-[11px] leading-snug text-muted-foreground">
                  Scan for live reads.
                </p>
              </div>
              <div className="min-w-0 space-y-1.5">
                <Label htmlFor="reader-transport-mode">Transport Mode</Label>
                <Select
                  value={selectedTransportMode}
                  onValueChange={(value) => setSelectedTransportMode(value as RfidTransportMode)}
                >
                  <SelectTrigger id="reader-transport-mode" className="h-9 w-full min-w-0">
                    <SelectValue placeholder="Choose transport mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="scan">Scan Transport</SelectItem>
                    <SelectItem value="answer">Answer Transport</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button className="h-9 w-full" onClick={() => transportModeMutation.mutate()} disabled={!isConnected || isBusy}>
                Apply Transport
              </Button>
            </div>

            <div className="grid min-w-0 gap-1.5 rounded-xl border border-border/60 bg-muted/20 p-2.5 xl:grid-cols-[96px_minmax(0,1fr)_148px] xl:items-center">
              <div className="space-y-1">
                <p className="text-sm font-semibold text-foreground">Buzzer</p>
                <p className="text-[11px] leading-snug text-muted-foreground">
                  {buzzerSupported ? "Available in scan transport." : "Switch to scan to control buzzer."}
                </p>
              </div>
              <div className="flex min-w-0 items-center justify-between rounded-lg border border-border/70 bg-background/60 px-3 py-2">
                <div className="space-y-0.5 pr-3">
                  <Label htmlFor="reader-buzzer">Toggle</Label>
                  <p className="text-[11px] text-muted-foreground">
                    {selectedBuzzerEnabled ? "Enabled" : "Disabled"}
                  </p>
                </div>
                <Switch
                  id="reader-buzzer"
                  checked={selectedBuzzerEnabled}
                  onCheckedChange={setSelectedBuzzerEnabled}
                  disabled={!isConnected || isBusy || !buzzerSupported}
                />
              </div>
              <Button
                className="h-9 w-full"
                onClick={() => buzzerMutation.mutate()}
                disabled={!isConnected || isBusy || !buzzerSupported}
              >
                Apply Buzzer
              </Button>
            </div>
          </CardContent>
        </Card>
        </div>

        <Card className="min-w-0 border-border/50 shadow-sm">
          <CardContent className="grid min-w-0 gap-3 p-3.5 xl:grid-cols-[220px_minmax(0,1fr)_160px] xl:items-center">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Settings2 className="size-4 text-primary" />
                <p className="text-base font-semibold text-foreground">Power Control</p>
                <span className="rounded-md bg-background/80 px-2.5 py-0.5 text-base font-semibold text-foreground">{powerValue[0]}</span>
              </div>
              <p className="text-[11px] leading-snug text-muted-foreground">
                30 for normal detection, 1 for close registration.
              </p>
            </div>
            <div className="min-w-0 space-y-2">
              <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
                <span>Power Level</span>
                <span>{powerValue[0]}</span>
              </div>
              <Slider
                className="w-full"
                min={0}
                max={30}
                step={1}
                value={powerValue}
                onValueChange={setPowerValue}
                disabled={!isConnected || isBusy}
              />
            </div>
            <Button className="h-10 w-full xl:w-[160px]" onClick={() => powerMutation.mutate()} disabled={!isConnected || isBusy}>
              Apply Power
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
