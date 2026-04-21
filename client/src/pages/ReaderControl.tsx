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
    <div className="space-y-6 p-6 md:p-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">UHF Reader Control</h1>
          <p className="mt-1 text-muted-foreground">
            Connect the reader, choose the required mode, transport, and buzzer behavior, then start or stop live reading.
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
          Refresh
        </Button>
      </div>

      {serviceOffline && (
        <Alert className="border-amber-200 bg-amber-50">
          <AlertTriangle className="size-4 text-amber-700" />
          <AlertTitle>RFID service is not reachable</AlertTitle>
          <AlertDescription>
            The backend could not reach the UHF reader service. If this stays visible, install
            <span className="mx-1 font-mono">rfid_service/requirements.txt</span>
            and restart the server.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Cable className="size-4 text-primary" />
              Connection
            </CardTitle>
            <CardDescription>Use COM3 / 57600 by default, then connect before starting live scan or answer polling.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <Label htmlFor="reader-port">COM Port</Label>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => detectPortMutation.mutate()}
                    disabled={isConnected || isBusy}
                  >
                    Auto Detect
                  </Button>
                </div>
                <Input
                  id="reader-port"
                  value={port}
                  onChange={(event) => setPort(event.target.value.toUpperCase())}
                  placeholder="COM3"
                  disabled={isConnected || isBusy}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="reader-baudrate">Baud Rate</Label>
                <Input
                  id="reader-baudrate"
                  value={baudrate}
                  onChange={(event) => setBaudrate(event.target.value)}
                  placeholder="57600"
                  disabled={isConnected || isBusy}
                />
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <Button onClick={() => connectMutation.mutate()} disabled={isConnected || isBusy}>
                <Play className="mr-2 size-4" />
                Connect
              </Button>
              <Button variant="outline" onClick={() => disconnectMutation.mutate()} disabled={!isConnected || isBusy}>
                <Square className="mr-2 size-4" />
                Disconnect
              </Button>
              <Button variant="outline" onClick={() => startMutation.mutate()} disabled={!isConnected || isRunning || isBusy}>
                <Play className="mr-2 size-4" />
                Start Reader
              </Button>
              <Button variant="outline" onClick={() => stopMutation.mutate()} disabled={!isRunning || isBusy}>
                <Square className="mr-2 size-4" />
                Stop Reader
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Settings2 className="size-4 text-primary" />
              Current Status
            </CardTitle>
            <CardDescription>Only the UHF reader lifecycle needed for connection, registration, and gate operation.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Connection</p>
              <div className="mt-2">
                <Badge variant={isConnected ? "secondary" : "outline"}>
                  {isConnected ? "Connected" : "Disconnected"}
                </Badge>
              </div>
            </div>
            <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Reader</p>
              <div className="mt-2">
                <Badge variant={isRunning ? "secondary" : "outline"}>
                  {isRunning ? "Running" : "Stopped"}
                </Badge>
              </div>
            </div>
            <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Mode</p>
              <p className="mt-2 text-lg font-semibold capitalize text-foreground">{status?.current_mode ?? selectedMode}</p>
            </div>
            <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Transport</p>
              <p className="mt-2 text-lg font-semibold capitalize text-foreground">
                {status?.transport_mode ?? selectedTransportMode}
              </p>
            </div>
            <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Power</p>
              <p className="mt-2 text-lg font-semibold text-foreground">{status?.current_power ?? powerValue[0]}</p>
            </div>
            <div className="rounded-2xl border border-border/60 bg-muted/20 p-4">
              <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Buzzer</p>
              <div className="mt-2">
                <Badge variant={(status?.buzzer_enabled ?? selectedBuzzerEnabled) ? "secondary" : "outline"}>
                  {(status?.buzzer_enabled ?? selectedBuzzerEnabled) ? "Enabled" : "Disabled"}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <SlidersHorizontal className="size-4 text-primary" />
              Power Control
            </CardTitle>
            <CardDescription>Use 30 for normal detection and 5-10 for registration mode.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
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
                disabled={!isConnected || isBusy}
              />
            </div>
            <Button onClick={() => powerMutation.mutate()} disabled={!isConnected || isBusy}>
              Apply Power
            </Button>
          </CardContent>
        </Card>

        <Card className="border-border/50 shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <Settings2 className="size-4 text-primary" />
              Mode Selection
            </CardTitle>
            <CardDescription>
              Use normal or registration for business behavior, then choose scan or answer transport and buzzer behavior.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="reader-mode">Mode</Label>
              <Select value={selectedMode} onValueChange={(value) => setSelectedMode(value as RfidMode)}>
                <SelectTrigger id="reader-mode">
                  <SelectValue placeholder="Choose mode" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="normal">Normal Mode</SelectItem>
                  <SelectItem value="registration">Registration Mode</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button onClick={() => modeMutation.mutate()} disabled={!isConnected || isBusy}>
              Apply Mode
            </Button>
            <div className="space-y-2">
              <Label htmlFor="reader-transport-mode">Transport Mode</Label>
              <Select
                value={selectedTransportMode}
                onValueChange={(value) => setSelectedTransportMode(value as RfidTransportMode)}
              >
                <SelectTrigger id="reader-transport-mode">
                  <SelectValue placeholder="Choose transport mode" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="scan">Scan Transport</SelectItem>
                  <SelectItem value="answer">Answer Transport</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button onClick={() => transportModeMutation.mutate()} disabled={!isConnected || isBusy}>
              Apply Transport
            </Button>
            <div className="space-y-3 rounded-2xl border border-border/60 bg-muted/20 p-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <Label htmlFor="reader-buzzer">Buzzer</Label>
                </div>
                <Switch
                  id="reader-buzzer"
                  checked={selectedBuzzerEnabled}
                  onCheckedChange={setSelectedBuzzerEnabled}
                  disabled={!isConnected || isBusy}
                />
              </div>
              <Button onClick={() => buzzerMutation.mutate()} disabled={!isConnected || isBusy}>
                Apply Buzzer
              </Button>
            </div>
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
