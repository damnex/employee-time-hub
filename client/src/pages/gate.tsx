import { useState, useRef, useEffect } from "react";
import { useScanRFID } from "@/hooks/use-gate";
import { useDeviceWS } from "@/hooks/use-device-ws";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Camera, Scan, KeyRound, AlertCircle, CheckCircle2, Wifi, WifiOff } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { averageFaceSamples, captureFaceSample } from "@/lib/biometrics";

const GATE_DEVICE_ID = "GATE-TERMINAL-01";
const GATE_BROWSER_CLIENT_ID = "GATE-TERMINAL-01-BROWSER";
const GATE_FACE_SAMPLE_COUNT = 3;
const GATE_FACE_SAMPLE_DELAY_MS = 140;

export default function GateSimulator() {
  const [rfidUid, setRfidUid] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraRetryToken, setCameraRetryToken] = useState(0);
  const [isSamplingFace, setIsSamplingFace] = useState(false);
  const [readerMessage, setReaderMessage] = useState<string | null>(null);
  const [readerSourceDeviceId, setReaderSourceDeviceId] = useState<string | null>(null);
  const [liveTapUid, setLiveTapUid] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{
    success: boolean;
    message: string;
    employee?: { name: string };
    matchConfidence?: number;
  } | null>(null);
  
  const scanMutation = useScanRFID();
  const { isConnected, lastScanResult, clearResult } = useDeviceWS(GATE_BROWSER_CLIENT_ID, { clientType: "browser" });

  // Initialize real camera
  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    const initCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraActive(false);
        setCameraError("This browser does not support camera access.");
        return;
      }

      setCameraActive(false);
      setCameraError(null);

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
          },
          audio: false
        });

        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        if (!videoRef.current) {
          setCameraError("Camera preview could not be attached.");
          return;
        }

        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraActive(true);
      } catch (error) {
        console.error("Camera access denied:", error);
        setCameraActive(false);
        setCameraError("Allow camera access in the browser to show the live feed.");
      }
    };

    void initCamera();

    return () => {
      cancelled = true;

      if (videoRef.current) {
        videoRef.current.pause();
        videoRef.current.srcObject = null;
      }

      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [cameraRetryToken]);

  const extractFaceDescriptor = (): number[] | null => {
    if (!cameraActive || !videoRef.current || !canvasRef.current) {
      return null;
    }

    try {
      const sample = captureFaceSample(videoRef.current, canvasRef.current);
      return sample?.descriptor ?? null;
    } catch (error) {
      console.error("Error extracting face descriptor:", error);
      return null;
    }
  };

  const captureLiveFaceDescriptor = async () => {
    if (!cameraActive || !videoRef.current || !canvasRef.current) {
      return null;
    }

    const descriptors: number[][] = [];

    for (let sampleIndex = 0; sampleIndex < GATE_FACE_SAMPLE_COUNT; sampleIndex++) {
      const sample = captureFaceSample(videoRef.current, canvasRef.current);
      if (!sample) {
        return null;
      }

      descriptors.push(sample.descriptor);

      if (sampleIndex < GATE_FACE_SAMPLE_COUNT - 1) {
        await new Promise((resolve) => setTimeout(resolve, GATE_FACE_SAMPLE_DELAY_MS));
      }
    }

    return averageFaceSamples(descriptors);
  };

  const authenticateScan = async (
    inputUid: string,
    source: "manual" | "reader" = "manual",
    sourceDeviceId?: string,
  ) => {
    const normalizedUid = inputUid.trim().toUpperCase();
    if (!normalizedUid || scanMutation.isPending || isSamplingFace) return;

    setIsSamplingFace(true);

    try {
      const descriptor = await captureLiveFaceDescriptor();
      if (!descriptor) {
        setLastResult({
          success: false,
          message: "Live camera verification is required before authenticating a badge.",
        });
        return;
      }

      setRfidUid(normalizedUid);
      if (source === "reader") {
        setReaderMessage(`Badge ${normalizedUid} detected. Verifying against live face data...`);
        setLiveTapUid(normalizedUid);
      }

      const data = await scanMutation.mutateAsync({
        rfidUid: normalizedUid,
        deviceId: source === "reader" ? (sourceDeviceId ?? GATE_DEVICE_ID) : GATE_DEVICE_ID,
        faceDescriptor: descriptor
      });

      setLastResult({
        success: data.success,
        message: data.message,
        employee: data.employee,
        matchConfidence: data.matchConfidence,
      });

      if (data.success) {
        setRfidUid("");
      }
    } catch (error) {
      setLastResult({
        success: false,
        message: error instanceof Error ? error.message : "Authentication failed.",
      });
    } finally {
      setIsSamplingFace(false);
    }
  };

  const handleScan = () => {
    void authenticateScan(rfidUid, "manual");
  };

  // Handle WebSocket scan results
  useEffect(() => {
    if (lastScanResult?.type === "rfid_detected") {
      const detectedUid = lastScanResult.rfidUid?.trim().toUpperCase();
      if (!detectedUid) {
        return;
      }

      setRfidUid(detectedUid);
      const sourceDeviceId = lastScanResult.deviceId ?? GATE_DEVICE_ID;
      setReaderSourceDeviceId(sourceDeviceId);
      setReaderMessage(
        sourceDeviceId === GATE_DEVICE_ID
          ? (lastScanResult.message || "Badge detected from physical reader.")
          : `Badge detected from ${sourceDeviceId}. Processing it on this gate terminal.`,
      );
      void authenticateScan(detectedUid, "reader", sourceDeviceId);
      clearResult();
      return;
    }
  }, [lastScanResult, clearResult]);

  return (
    <div className="p-6 md:p-8 h-full flex flex-col items-center justify-center bg-slate-50 dark:bg-slate-950">
      
      <div className="text-center mb-8">
        <h1 className="text-3xl font-display font-bold text-foreground">Gate Terminal</h1>
        <p className="text-muted-foreground mt-2 max-w-md mx-auto">
          Real-time attendance with live camera and RFID scanning
        </p>
      </div>

      <Card className="w-full max-w-md shadow-xl border-border/60 overflow-hidden">
        <div className="bg-primary p-4 text-primary-foreground flex items-center justify-between">
          <div className="flex items-center justify-center gap-2">
            <Scan className="size-5" />
            <span className="font-semibold tracking-wide">GATE-01 TERMINAL</span>
          </div>
          <div className="flex items-center gap-1">
            {isConnected ? (
              <div className="flex items-center gap-1 text-xs bg-green-500/20 px-2 py-1 rounded">
                <Wifi className="size-3" />
                <span>Device</span>
              </div>
            ) : (
              <div className="flex items-center gap-1 text-xs bg-red-500/20 px-2 py-1 rounded">
                <WifiOff className="size-3" />
                <span>API</span>
              </div>
            )}
          </div>
        </div>
        
        <CardContent className="p-8 space-y-8">
          {/* Real/Simulated Webcam View */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider text-center">
              {cameraActive ? 'Live Camera Feed' : cameraError ? 'Camera Permission Required' : 'Starting Camera'}
            </div>
            <div className="aspect-video bg-black rounded-xl border border-border flex flex-col items-center justify-center relative overflow-hidden group">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className={`w-full h-full object-cover transition-opacity duration-300 ${cameraActive ? "opacity-100" : "opacity-0"}`}
              />
              <canvas
                ref={canvasRef}
                width={128}
                height={128}
                className="hidden"
              />
              {!cameraActive && (
                <>
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
                    <Camera className="size-10 text-muted-foreground/40 group-hover:scale-110 transition-transform duration-300" />
                    <p className="text-sm text-muted-foreground">
                      {cameraError ?? "Waiting for browser camera access..."}
                    </p>
                    {cameraError && (
                      <Button
                        type="button"
                        size="sm"
                        variant="secondary"
                        onClick={() => setCameraRetryToken((token) => token + 1)}
                      >
                        Retry Camera
                      </Button>
                    )}
                  </div>
                  <div className="absolute inset-0 border-2 border-primary/20 rounded-xl m-4 pointer-events-none opacity-50" />
                </>
              )}
              {(scanMutation.isPending || isSamplingFace) && (
                <div className="absolute inset-0 bg-primary/10 backdrop-blur-[2px] flex items-center justify-center">
                  <div className="w-16 h-1 bg-primary/80 animate-pulse rounded-full shadow-[0_0_15px_rgba(var(--primary),0.5)]" />
                </div>
              )}
            </div>
          </div>

          {/* Form / Inputs */}
          <div className="space-y-4">
            <div className="rounded-xl border border-border/70 bg-muted/20 p-4 space-y-2">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-muted-foreground">Gate event channel</span>
                <span className={isConnected ? "text-foreground font-medium" : "text-muted-foreground"}>
                  {isConnected ? "Listening for reader taps" : "Browser socket offline"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-muted-foreground">Preferred device</span>
                <span className="font-mono text-foreground">{GATE_DEVICE_ID}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-muted-foreground">Browser client</span>
                <span className="font-mono text-foreground">{GATE_BROWSER_CLIENT_ID}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-muted-foreground">Last physical tap</span>
                <span className="font-mono text-foreground">{liveTapUid ?? "--"}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-muted-foreground">Face samples / scan</span>
                <span className="font-mono text-foreground">{GATE_FACE_SAMPLE_COUNT}</span>
              </div>
              {readerSourceDeviceId && (
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-muted-foreground">Reader source</span>
                  <span className="font-mono text-foreground">{readerSourceDeviceId}</span>
                </div>
              )}
              {readerMessage && (
                <p className="text-sm text-muted-foreground">{readerMessage}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="rfid" className="flex items-center gap-2 text-foreground/80">
                <KeyRound className="size-4" /> RFID Badge
              </Label>
              <Input 
                id="rfid"
                placeholder="Tap on the real reader or enter UID manually" 
                className="font-mono text-lg py-6 text-center tracking-widest bg-background border-2 focus-visible:ring-primary/20"
                value={rfidUid}
                onChange={(e) => setRfidUid(e.target.value.toUpperCase())}
                disabled={scanMutation.isPending}
                onKeyDown={(e) => e.key === 'Enter' && handleScan()}
              />
            </div>
            
            <Button 
              size="lg" 
              className="w-full h-14 text-lg font-semibold shadow-md active:scale-[0.98] transition-transform" 
              onClick={handleScan}
              disabled={scanMutation.isPending || isSamplingFace || !rfidUid.trim() || !cameraActive}
            >
              {isSamplingFace ? "Capturing Face..." : scanMutation.isPending ? "Verifying..." : "Scan & Authenticate"}
            </Button>
          </div>

          {/* Results Area */}
          <div className="min-h-[80px]">
            {lastResult && (
              <Alert variant={lastResult.success ? "default" : "destructive"} className={`border-2 ${lastResult.success ? 'bg-emerald-50 border-emerald-200 dark:bg-emerald-950 dark:border-emerald-900' : ''}`}>
                {lastResult.success ? (
                  <CheckCircle2 className="size-5 text-emerald-600 dark:text-emerald-400" />
                ) : (
                  <AlertCircle className="size-5" />
                )}
                <AlertTitle className={lastResult.success ? "text-emerald-800 dark:text-emerald-300 font-bold" : "font-bold"}>
                  {lastResult.success ? "Access Granted" : "Access Denied"}
                </AlertTitle>
                <AlertDescription className={lastResult.success ? "text-emerald-700 dark:text-emerald-400" : ""}>
                  {lastResult.message}
                  {typeof lastResult.matchConfidence === "number" && (
                    <span className="block mt-1 font-mono">
                      Match confidence: {(lastResult.matchConfidence * 100).toFixed(1)}%
                    </span>
                  )}
                </AlertDescription>
              </Alert>
            )}
          </div>

        </CardContent>
      </Card>
    </div>
  );
}
