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

type FaceAlignmentState =
  | "unsupported"
  | "searching"
  | "aligned"
  | "off-center"
  | "no-face"
  | "multiple";

type DetectedFaceLike = {
  boundingBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

declare global {
  interface Window {
    FaceDetector?: new (options?: { fastMode?: boolean; maxDetectedFaces?: number }) => {
      detect: (source: CanvasImageSource) => Promise<DetectedFaceLike[]>;
    };
  }
}

export default function GateSimulator() {
  const [rfidUid, setRfidUid] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraRetryToken, setCameraRetryToken] = useState(0);
  const [isSamplingFace, setIsSamplingFace] = useState(false);
  const [faceAlignmentState, setFaceAlignmentState] = useState<FaceAlignmentState>("searching");
  const [faceBounds, setFaceBounds] = useState<DetectedFaceLike["boundingBox"] | null>(null);
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
  const faceFrameTone = !cameraActive
    ? "border-slate-300 shadow-none"
    : lastResult?.success
      ? "border-emerald-400 shadow-[0_0_0_1px_rgba(74,222,128,0.85),0_0_30px_rgba(74,222,128,0.32)]"
      : lastResult
        ? "border-rose-400 shadow-[0_0_0_1px_rgba(251,113,133,0.85),0_0_30px_rgba(251,113,133,0.28)]"
        : isSamplingFace || scanMutation.isPending
          ? "border-amber-300 shadow-[0_0_0_1px_rgba(253,224,71,0.75),0_0_24px_rgba(253,224,71,0.22)]"
          : faceAlignmentState === "aligned"
            ? "border-emerald-400/80 shadow-[0_0_0_1px_rgba(74,222,128,0.72),0_0_24px_rgba(74,222,128,0.2)]"
            : "border-rose-400/80 shadow-[0_0_0_1px_rgba(251,113,133,0.72),0_0_24px_rgba(251,113,133,0.18)]";
  const frameStatusLabel = !cameraActive
    ? "CAMERA OFF"
    : lastResult?.success
      ? "MATCHED"
      : lastResult
        ? "RETRY"
        : isSamplingFace || scanMutation.isPending
          ? "SCANNING"
          : faceAlignmentState === "aligned"
            ? "ALIGNMENT OK"
            : "ALIGN FACE";
  const faceGuideMessage = !cameraActive
    ? cameraError ?? "Waiting for browser camera access..."
    : faceAlignmentState === "unsupported"
      ? "Face detector is unavailable in this browser. Matching can still run, but alignment guidance is limited."
      : faceAlignmentState === "multiple"
        ? "Keep only one face inside the frame."
        : faceAlignmentState === "off-center"
          ? "Center your face inside the green guide."
          : faceAlignmentState === "no-face"
            ? "Face not detected. Step into the camera frame."
            : faceAlignmentState === "aligned"
              ? "Face detected and aligned."
              : "Searching for face alignment...";

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

  useEffect(() => {
    if (!cameraActive || !videoRef.current) {
      setFaceAlignmentState((current) => (current === "unsupported" ? current : "searching"));
      setFaceBounds(null);
      return;
    }

    if (!window.FaceDetector) {
      setFaceAlignmentState("unsupported");
      setFaceBounds(null);
      return;
    }

    let cancelled = false;
    const detector = new window.FaceDetector({
      fastMode: true,
      maxDetectedFaces: 2,
    });

    const detectFace = async () => {
      if (!videoRef.current || videoRef.current.readyState < 2) {
        return;
      }

      try {
        const faces = await detector.detect(videoRef.current);
        if (cancelled) {
          return;
        }

        if (faces.length === 0) {
          setFaceBounds(null);
          setFaceAlignmentState("no-face");
          return;
        }

        if (faces.length > 1) {
          setFaceBounds(null);
          setFaceAlignmentState("multiple");
          return;
        }

        const [face] = faces;
        const bounds = face.boundingBox;
        const video = videoRef.current;
        const centerX = bounds.x + bounds.width / 2;
        const centerY = bounds.y + bounds.height / 2;
        const normalizedCenterX = centerX / video.videoWidth;
        const normalizedCenterY = centerY / video.videoHeight;
        const normalizedWidth = bounds.width / video.videoWidth;
        const normalizedHeight = bounds.height / video.videoHeight;
        const centeredEnough =
          normalizedCenterX > 0.34
          && normalizedCenterX < 0.66
          && normalizedCenterY > 0.24
          && normalizedCenterY < 0.76;
        const sizeEnough =
          normalizedWidth > 0.18
          && normalizedWidth < 0.6
          && normalizedHeight > 0.24
          && normalizedHeight < 0.82;

        setFaceBounds(bounds);
        setFaceAlignmentState(centeredEnough && sizeEnough ? "aligned" : "off-center");
      } catch (error) {
        console.error("Face detection failed:", error);
        if (!cancelled) {
          setFaceAlignmentState("unsupported");
          setFaceBounds(null);
        }
      }
    };

    void detectFace();
    const intervalId = window.setInterval(() => {
      void detectFace();
    }, 320);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [cameraActive]);

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
      if (faceAlignmentState !== "aligned" && faceAlignmentState !== "unsupported") {
        setLastResult({
          success: false,
          message: "Center a single face inside the guide before scanning.",
        });
        return;
      }

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
    <div className="min-h-full bg-[radial-gradient(circle_at_top,_#f8fbff_0%,_#eef5ff_42%,_#e5eefb_100%)] text-slate-950">
      <div className="mx-auto flex min-h-full max-w-5xl flex-col items-center justify-center px-6 py-10 md:px-8">
        <div className="mb-8 text-center">
        <h1 className="text-3xl font-display font-bold text-slate-950">Gate Terminal</h1>
        <p className="mt-2 max-w-md mx-auto text-slate-600">
          Real-time attendance with live camera and RFID scanning
        </p>
      </div>

      <Card className="w-full max-w-md overflow-hidden border border-sky-100 bg-white/90 shadow-[0_24px_80px_rgba(30,64,175,0.12)] backdrop-blur">
        <div className="flex items-center justify-between bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 p-4 text-white">
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
              <div className="flex items-center gap-1 rounded bg-white/20 px-2 py-1 text-xs">
                <WifiOff className="size-3" />
                <span>Offline</span>
              </div>
            )}
          </div>
        </div>

        <CardContent className="space-y-8 p-8">
          {/* Real/Simulated Webcam View */}
          <div className="space-y-2">
            <div className="text-center text-xs font-medium uppercase tracking-wider text-slate-500">
              {cameraActive ? 'Live Camera Feed' : cameraError ? 'Camera Permission Required' : 'Starting Camera'}
            </div>
            <div className="group relative aspect-video overflow-hidden rounded-2xl border border-sky-100 bg-slate-950 shadow-inner">
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
                    <Camera className="size-10 text-white/35 transition-transform duration-300 group-hover:scale-110" />
                    <p className="text-sm text-white/70">
                      {faceGuideMessage}
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
              <div className="pointer-events-none absolute inset-0">
                <div className="absolute inset-3">
                  <div className="absolute left-0 top-0 h-10 w-10 rounded-tl-xl border-l-[3px] border-t-[3px] border-rose-400/80" />
                  <div className="absolute right-0 top-0 h-10 w-10 rounded-tr-xl border-r-[3px] border-t-[3px] border-rose-400/80" />
                  <div className="absolute bottom-0 left-0 h-10 w-10 rounded-bl-xl border-b-[3px] border-l-[3px] border-rose-400/80" />
                  <div className="absolute bottom-0 right-0 h-10 w-10 rounded-br-xl border-b-[3px] border-r-[3px] border-rose-400/80" />
                </div>
                <div
                  className={`absolute inset-x-[22%] inset-y-[18%] rounded-[1.75rem] border-[3px] transition-all duration-300 ease-out ${faceFrameTone}`}
                />
                {faceBounds && cameraActive && (
                  <div
                    className="absolute border-2 border-cyan-300/70 transition-all duration-200"
                    style={{
                      left: `${(faceBounds.x / (videoRef.current?.videoWidth || 1)) * 100}%`,
                      top: `${(faceBounds.y / (videoRef.current?.videoHeight || 1)) * 100}%`,
                      width: `${(faceBounds.width / (videoRef.current?.videoWidth || 1)) * 100}%`,
                      height: `${(faceBounds.height / (videoRef.current?.videoHeight || 1)) * 100}%`,
                    }}
                  />
                )}
              </div>
              <div className="pointer-events-none absolute inset-x-[22%] inset-y-[18%] rounded-[1.75rem]">
                <div className="absolute left-1/2 top-3 -translate-x-1/2 rounded-full bg-black/65 px-3 py-1 text-[10px] font-semibold tracking-[0.28em] text-white/90">
                  {frameStatusLabel}
                </div>
                <div className="absolute inset-x-6 top-1/2 h-px -translate-y-1/2 bg-gradient-to-r from-transparent via-emerald-300/70 to-transparent" />
                <div className="absolute left-1/2 inset-y-6 w-px -translate-x-1/2 bg-gradient-to-b from-transparent via-rose-300/55 to-transparent" />
              </div>
              {(scanMutation.isPending || isSamplingFace) && (
                <div className="absolute inset-0 bg-primary/10 backdrop-blur-[2px] flex items-center justify-center">
                  <div className="w-16 h-1 bg-primary/80 animate-pulse rounded-full shadow-[0_0_15px_rgba(var(--primary),0.5)]" />
                </div>
              )}
            </div>
            <p className="text-center text-xs text-slate-500">{faceGuideMessage}</p>
          </div>

          {/* Form / Inputs */}
          <div className="space-y-4">
            <div className="space-y-2 rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-slate-500">Gate event channel</span>
                <span className={isConnected ? "font-medium text-slate-900" : "text-slate-500"}>
                  {isConnected ? "Listening for reader taps" : "Browser socket offline"}
                </span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-slate-500">Preferred device</span>
                <span className="font-mono text-slate-900">{GATE_DEVICE_ID}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-slate-500">Browser client</span>
                <span className="font-mono text-slate-900">{GATE_BROWSER_CLIENT_ID}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-slate-500">Last physical tap</span>
                <span className="font-mono text-slate-900">{liveTapUid ?? "--"}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-slate-500">Face samples / scan</span>
                <span className="font-mono text-slate-900">{GATE_FACE_SAMPLE_COUNT}</span>
              </div>
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="text-slate-500">Face alignment</span>
                <span className={faceAlignmentState === "aligned" ? "font-medium text-emerald-600" : "text-rose-500"}>
                  {faceAlignmentState === "aligned"
                    ? "Ready"
                    : faceAlignmentState === "unsupported"
                      ? "Detector off"
                      : "Adjust"}
                </span>
              </div>
              {readerSourceDeviceId && (
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-500">Reader source</span>
                  <span className="font-mono text-slate-900">{readerSourceDeviceId}</span>
                </div>
              )}
              {readerMessage && (
                <p className="text-sm text-slate-600">{readerMessage}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="rfid" className="flex items-center gap-2 text-slate-700">
                <KeyRound className="size-4" /> RFID Badge
              </Label>
              <Input 
                id="rfid"
                placeholder="Tap on the real reader or enter UID manually" 
                className="bg-white font-mono text-lg tracking-widest text-center py-6 border-2 border-slate-200 focus-visible:ring-primary/20"
                value={rfidUid}
                onChange={(e) => setRfidUid(e.target.value.toUpperCase())}
                disabled={scanMutation.isPending}
                onKeyDown={(e) => e.key === 'Enter' && handleScan()}
              />
            </div>
            
            <Button 
              size="lg" 
              className="h-14 w-full bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 text-lg font-semibold text-white shadow-md transition-transform active:scale-[0.98]" 
              onClick={handleScan}
              disabled={scanMutation.isPending || isSamplingFace || !rfidUid.trim() || !cameraActive}
            >
              {isSamplingFace ? "Capturing Face..." : scanMutation.isPending ? "Verifying..." : "Scan & Authenticate"}
            </Button>
          </div>

          {/* Results Area */}
          <div className="min-h-[80px]">
            {lastResult && (
              <Alert variant={lastResult.success ? "default" : "destructive"} className={`border-2 ${lastResult.success ? 'border-emerald-200 bg-emerald-50' : 'border-rose-200 bg-rose-50 text-rose-900'}`}>
                {lastResult.success ? (
                  <CheckCircle2 className="size-5 text-emerald-600" />
                ) : (
                  <AlertCircle className="size-5" />
                )}
                <AlertTitle className={lastResult.success ? "font-bold text-emerald-800" : "font-bold"}>
                  {lastResult.success ? "Access Granted" : "Access Denied"}
                </AlertTitle>
                <AlertDescription className={lastResult.success ? "text-emerald-700" : ""}>
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
    </div>
  );
}
