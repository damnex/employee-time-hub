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
    <div className="flex h-full min-h-full flex-col gap-4 p-4 md:p-5 animate-in fade-in duration-500">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold tracking-tight text-foreground">UHF Reader Control</h1>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            Connect the reader, set the operating profile, and keep live reading ready without wasting viewport space.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={isConnected ? "secondary" : "outline"} className="h-9 px-3 text-[11px] uppercase tracking-[0.16em]">
            {isConnected ? "Connection Ready" : "Disconnected"}
          </Badge>
          <Badge variant={isRunning ? "secondary" : "outline"} className="h-9 px-3 text-[11px] uppercase tracking-[0.16em]">
            {isRunning ? "Reader Live" : "Reader Stopped"}
          </Badge>
          <Button
            variant="outline"
            className="h-10 px-4"
            onClick={() => {
              void refreshRfidQueries();
            }}
            disabled={isBusy}
          >
            <RefreshCcw className="mr-2 size-4" />
            Refresh
          </Button>
        </div>
      </div>

      {serviceOffline && (
        <Alert className="shrink-0 border-amber-500/30 bg-amber-500/10 py-3 text-amber-100">
          <AlertTriangle className="size-4 text-amber-200" />
          <AlertTitle className="text-amber-100">RFID service is not reachable</AlertTitle>
          <AlertDescription className="text-amber-100/80">
            The backend could not reach the UHF reader service. If this stays visible, install
            <span className="mx-1 font-mono">rfid_service/requirements.txt</span>
            and restart the server.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid min-h-0 flex-1 gap-4 xl:grid-cols-[minmax(0,1.05fr)_minmax(320px,0.95fr)]">
        <Card className="border-border/50 shadow-sm">
          <CardHeader className="space-y-1 p-5 pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Cable className="size-4 text-primary" />
              Connection
            </CardTitle>
            <CardDescription className="text-[13px] leading-snug">
              Use COM3 and 57600 by default, then connect before starting live scan or answer polling.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 p-5 pt-0">
            <div className="grid gap-3 md:grid-cols-2">
              <div className="space-y-1.5">
                <div className="flex items-center justify-between gap-3">
                  <Label htmlFor="reader-port">COM Port</Label>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-8 px-2.5"
                    onClick={() => detectPortMutation.mutate()}
                    disabled={isConnected || isBusy}
                  >
                    Auto Detect
                  </Button>
                </div>
                <Input
                  id="reader-port"
                  className="h-10"
                  value={port}
                  onChange={(event) => setPort(event.target.value.toUpperCase())}
                  placeholder="COM3"
                  disabled={isConnected || isBusy}
                />
              </div>
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

            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
              <Button className="h-10" onClick={() => connectMutation.mutate()} disabled={isConnected || isBusy}>
                <Play className="mr-2 size-4" />
                Connect
              </Button>
              <Button
                variant="outline"
                className="h-10"
                onClick={() => disconnectMutation.mutate()}
                disabled={!isConnected || isBusy}
              >
                <Square className="mr-2 size-4" />
                Disconnect
              </Button>
              <Button
                variant="outline"
                className="h-10"
                onClick={() => startMutation.mutate()}
                disabled={!isConnected || isRunning || isBusy}
              >
                <Play className="mr-2 size-4" />
                Start Reader
              </Button>
              <Button
                variant="outline"
                className="h-10"
                onClick={() => stopMutation.mutate()}
                disabled={!isRunning || isBusy}
              >
                <Square className="mr-2 size-4" />
                Stop Reader
              </Button>
            </div>

            <div className="flex flex-wrap gap-2 rounded-xl border border-border/60 bg-muted/20 p-3">
              <Badge variant="outline" className="px-2.5 py-1">Default Port {status?.port ?? port}</Badge>
              <Badge variant="outline" className="px-2.5 py-1">Baud {status?.baudrate ?? Number(baudrate)}</Badge>
              <Badge variant="outline" className="px-2.5 py-1 capitalize">
                {status?.transport_mode ?? selectedTransportMode} transport
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm">
          <CardHeader className="space-y-1 p-5 pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              <Settings2 className="size-4 text-primary" />
              Current Status
            </CardTitle>
            <CardDescription className="text-[13px] leading-snug">
              Live reader state for connection, registration, and gate operation.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 p-5 pt-0 sm:grid-cols-2">
            <div className="rounded-xl border border-border/60 bg-muted/20 px-3 py-2.5">
              <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Connection</p>
              <div className="mt-2">
                <Badge variant={isConnected ? "secondary" : "outline"}>
                  {isConnected ? "Connected" : "Disconnected"}
                </Badge>
              </div>
            </div>
            <div className="rounded-xl border border-border/60 bg-muted/20 px-3 py-2.5">
              <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Reader</p>
              <div className="mt-2">
                <Badge variant={isRunning ? "secondary" : "outline"}>
                  {isRunning ? "Running" : "Stopped"}
                </Badge>
              </div>
            </div>
            <div className="rounded-xl border border-border/60 bg-muted/20 px-3 py-2.5">
              <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Mode</p>
              <p className="mt-2 text-lg font-semibold capitalize text-foreground">{status?.current_mode ?? selectedMode}</p>
            </div>
            <div className="rounded-xl border border-border/60 bg-muted/20 px-3 py-2.5">
              <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Transport</p>
              <p className="mt-2 text-lg font-semibold capitalize text-foreground">
                {status?.transport_mode ?? selectedTransportMode}
              </p>
            </div>
            <div className="rounded-xl border border-border/60 bg-muted/20 px-3 py-2.5">
              <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Power</p>
              <p className="mt-2 text-lg font-semibold text-foreground">{status?.current_power ?? powerValue[0]}</p>
            </div>
            <div className="rounded-xl border border-border/60 bg-muted/20 px-3 py-2.5">
              <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Buzzer</p>
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
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm xl:col-span-2">
          <CardHeader className="flex flex-col gap-2 border-b border-border/60 p-5 pb-4 md:flex-row md:items-end md:justify-between">
            <div className="space-y-1">
              <CardTitle className="flex items-center gap-2 text-lg">
                <SlidersHorizontal className="size-4 text-primary" />
                Power, Mode & Transport
              </CardTitle>
              <CardDescription className="text-[13px] leading-snug">
                Keep the core tuning controls together so setup, registration, and buzzer behavior stay visible in one pass.
              </CardDescription>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline" className="px-2.5 py-1">Power {powerValue[0]}</Badge>
              <Badge variant="outline" className="px-2.5 py-1 capitalize">{selectedMode}</Badge>
              <Badge variant="outline" className="px-2.5 py-1 capitalize">{selectedTransportMode}</Badge>
            </div>
          </CardHeader>
          <CardContent className="grid gap-4 p-5 pt-4 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1fr)_minmax(0,1fr)]">
            <div className="space-y-3 rounded-2xl border border-border/60 bg-muted/20 p-4">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-foreground">Power Control</span>
                <span className="rounded-md bg-background/80 px-2 py-1 text-base font-semibold text-foreground">
                  {powerValue[0]}
                </span>
              </div>
              <p className="text-[13px] leading-snug text-muted-foreground">
                Use 30 for normal detection and 1 for ultra-close registration mode.
              </p>
              <Slider
                min={0}
                max={30}
                step={1}
                value={powerValue}
                onValueChange={setPowerValue}
                disabled={!isConnected || isBusy}
              />
              <Button className="h-10 w-full" onClick={() => powerMutation.mutate()} disabled={!isConnected || isBusy}>
                Apply Power
              </Button>
            </div>

            <div className="space-y-3 rounded-2xl border border-border/60 bg-muted/20 p-4">
              <div className="space-y-1">
                <p className="text-sm font-medium text-foreground">Operating Mode</p>
                <p className="text-[13px] leading-snug text-muted-foreground">
                  Use normal for open gate reading or registration for close-range single-badge enrollment.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="reader-mode">Mode</Label>
                <Select value={selectedMode} onValueChange={(value) => setSelectedMode(value as RfidMode)}>
                  <SelectTrigger id="reader-mode" className="h-10">
                    <SelectValue placeholder="Choose mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="normal">Normal Mode</SelectItem>
                    <SelectItem value="registration">Registration Mode</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button className="h-10 w-full" onClick={() => modeMutation.mutate()} disabled={!isConnected || isBusy}>
                Apply Mode
              </Button>
              {selectedMode === "registration" && (
                <div className="rounded-xl border border-primary/20 bg-primary/5 p-3 text-sm">
                  <p className="font-medium text-foreground">Registration profile</p>
                  <p className="mt-1 text-[13px] leading-snug text-muted-foreground">
                    Lower power, shorter range, and stable single-tag detection before enrollment.
                  </p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Badge variant="outline">Target Power 1</Badge>
                    <Badge variant="outline">Stable Hits 7</Badge>
                    <Badge variant="outline">Single Tag Only</Badge>
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-3 rounded-2xl border border-border/60 bg-muted/20 p-4">
              <div className="space-y-1">
                <p className="text-sm font-medium text-foreground">Transport & Buzzer</p>
                <p className="text-[13px] leading-snug text-muted-foreground">
                  Scan is best for live reads. Answer transport ignores buzzer settings on this reader.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="reader-transport-mode">Transport Mode</Label>
                <Select
                  value={selectedTransportMode}
                  onValueChange={(value) => setSelectedTransportMode(value as RfidTransportMode)}
                >
                  <SelectTrigger id="reader-transport-mode" className="h-10">
                    <SelectValue placeholder="Choose transport mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="scan">Scan Transport</SelectItem>
                    <SelectItem value="answer">Answer Transport</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <Button className="h-10 w-full" onClick={() => transportModeMutation.mutate()} disabled={!isConnected || isBusy}>
                Apply Transport
              </Button>
              <div className="space-y-3 rounded-xl border border-border/70 bg-background/60 p-3">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <Label htmlFor="reader-buzzer">Buzzer</Label>
                    <p className="mt-1 text-[12px] leading-snug text-muted-foreground">
                      {buzzerSupported
                        ? "Available while using scan transport."
                        : "Switch back to scan transport to manage buzzer behavior."}
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
                  className="h-10 w-full"
                  onClick={() => buzzerMutation.mutate()}
                  disabled={!isConnected || isBusy || !buzzerSupported}
                >
                  Apply Buzzer
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {status?.last_error && (
        <Alert className="shrink-0 border-rose-500/30 bg-rose-500/10 py-3 text-rose-100">
          <AlertTriangle className="size-4 text-rose-200" />
          <AlertTitle className="text-rose-100">Reader warning</AlertTitle>
          <AlertDescription className="text-rose-100/80">{status.last_error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}
