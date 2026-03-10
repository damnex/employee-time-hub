import { useState, useRef, useEffect } from "react";
import { useScanRFID } from "@/hooks/use-gate";
import { useDeviceWS } from "@/hooks/use-device-ws";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Camera, Scan, KeyRound, AlertCircle, CheckCircle2, Wifi, WifiOff } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  allowInsecureFaceFallback,
  captureFaceTemplate,
  isFaceDetectorAvailable,
} from "@/lib/biometrics";
import {
  appendMovementSample,
  inferMovementDirection,
  type DirectionInferenceResult,
  type MovementSample,
} from "@/lib/movement";

const GATE_DEVICE_ID = "GATE-TERMINAL-01";
const GATE_BROWSER_CLIENT_ID = "GATE-TERMINAL-01-BROWSER";
const GATE_FACE_SAMPLE_COUNT = 25;
const GATE_FACE_SAMPLE_DELAY_MS = 70;
const GATE_DIRECTION_SAMPLE_WINDOW = 25;
const GATE_DIRECTION_MAX_SAMPLE_AGE_MS = 1800;
const GATE_DIRECTION_CONFIDENCE_THRESHOLD = 0.58;

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
  const motionSamplesRef = useRef<MovementSample[]>([]);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraRetryToken, setCameraRetryToken] = useState(0);
  const [isSamplingFace, setIsSamplingFace] = useState(false);
  const [capturedFaceSamples, setCapturedFaceSamples] = useState(0);
  const [liveFaceQuality, setLiveFaceQuality] = useState<number | null>(null);
  const [liveFaceConsistency, setLiveFaceConsistency] = useState<number | null>(null);
  const [faceAlignmentState, setFaceAlignmentState] = useState<FaceAlignmentState>("searching");
  const [faceBounds, setFaceBounds] = useState<DetectedFaceLike["boundingBox"] | null>(null);
  const [directionState, setDirectionState] = useState<DirectionInferenceResult>({
    direction: "UNKNOWN",
    confidence: 0,
    axis: "none",
    sampleCount: 0,
  });
  const [readerMessage, setReaderMessage] = useState<string | null>(null);
  const [readerSourceDeviceId, setReaderSourceDeviceId] = useState<string | null>(null);
  const [liveTapUid, setLiveTapUid] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<{
    success: boolean;
    message: string;
    employee?: { name: string };
    matchConfidence?: number;
    matchDetails?: {
      primaryConfidence: number;
      anchorAverage: number;
      peakAnchorConfidence: number;
      strongAnchorRatio: number;
      liveConsistency: number;
    };
  } | null>(null);

  const scanMutation = useScanRFID();
  const { isConnected, lastScanResult, clearResult } = useDeviceWS(GATE_BROWSER_CLIENT_ID, { clientType: "browser" });
  const detectorAvailable = isFaceDetectorAvailable();
  const fallbackCaptureAllowed = allowInsecureFaceFallback();
  const supportsDirectionTracking = detectorAvailable;
  const directionReady =
    directionState.direction !== "UNKNOWN"
    && directionState.confidence >= GATE_DIRECTION_CONFIDENCE_THRESHOLD;
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
          : directionReady
            ? directionState.direction === "ENTRY"
              ? "ENTRY PATH"
              : "EXIT PATH"
            : faceAlignmentState === "aligned"
              ? "TRACK MOTION"
              : "ALIGN FACE";
  const faceGuideMessage = (() => {
    if (!cameraActive) {
      return cameraError ?? "Waiting for browser camera access...";
    }

    if (faceAlignmentState === "unsupported") {
      return fallbackCaptureAllowed
        ? "Secure detector is unavailable. Compatibility fallback verification is enabled for this environment, but accuracy is lower."
        : "Secure face detection is unavailable in this browser. Access verification is blocked here. Use Chrome or Edge.";
    }

    if (faceAlignmentState === "multiple") {
      return "Keep only one face inside the frame.";
    }

    if (faceAlignmentState === "off-center") {
      return "Center your face inside the green guide.";
    }

    if (faceAlignmentState === "no-face") {
      return "Face not detected. Step into the camera frame.";
    }

    if (directionReady) {
      return `Movement classified as ${directionState.direction.toLowerCase()} with ${Math.round(directionState.confidence * 100)}% confidence.`;
    }

    if (faceAlignmentState === "aligned") {
      return "Walk through the frame before scanning so the camera can classify entry or exit.";
    }

    return "Searching for face alignment...";
  })();

  const resetMotionEvidence = () => {
    motionSamplesRef.current = [];
    setDirectionState({
      direction: "UNKNOWN",
      confidence: 0,
      axis: "none",
      sampleCount: 0,
    });
  };

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
      resetMotionEvidence();
      return;
    }

    if (!window.FaceDetector) {
      setFaceAlignmentState("unsupported");
      setFaceBounds(null);
      resetMotionEvidence();
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
          resetMotionEvidence();
          return;
        }

        if (faces.length > 1) {
          setFaceBounds(null);
          setFaceAlignmentState("multiple");
          resetMotionEvidence();
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

        const nextSamples = appendMovementSample(
          motionSamplesRef.current,
          {
            timestamp: Date.now(),
            centerX: normalizedCenterX,
            centerY: normalizedCenterY,
            area: normalizedWidth * normalizedHeight,
          },
          GATE_DIRECTION_SAMPLE_WINDOW,
        );
        motionSamplesRef.current = nextSamples;
        setDirectionState(
          inferMovementDirection(nextSamples, {
            maxSamples: GATE_DIRECTION_SAMPLE_WINDOW,
            maxSampleAgeMs: GATE_DIRECTION_MAX_SAMPLE_AGE_MS,
          }),
        );
      } catch (error) {
        console.error("Face detection failed:", error);
        if (!cancelled) {
          setFaceAlignmentState("unsupported");
          setFaceBounds(null);
          resetMotionEvidence();
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

  const captureLiveFaceProfile = async () => {
    if (!cameraActive || !videoRef.current || !canvasRef.current) {
      return null;
    }

    const template = await captureFaceTemplate(videoRef.current, canvasRef.current, {
      sampleCount: GATE_FACE_SAMPLE_COUNT,
      sampleDelayMs: GATE_FACE_SAMPLE_DELAY_MS,
      minQuality: 0.2,
      maxAttempts: GATE_FACE_SAMPLE_COUNT * 2,
      requireDetector: !fallbackCaptureAllowed,
      onProgress: (acceptedSamples) => {
        setCapturedFaceSamples(acceptedSamples);
      },
    });

    setLiveFaceQuality(template.averageQuality);
    setLiveFaceConsistency(template.consistency);
    return template.profile;
  };

  const authenticateScan = async (
    inputUid: string,
    source: "manual" | "reader" = "manual",
    sourceDeviceId?: string,
  ) => {
    const normalizedUid = inputUid.trim().toUpperCase();
    if (!normalizedUid || scanMutation.isPending || isSamplingFace) return;

    setIsSamplingFace(true);
    setCapturedFaceSamples(0);
    setLiveFaceQuality(null);
    setLiveFaceConsistency(null);

    try {
      if (faceAlignmentState === "unsupported" && !fallbackCaptureAllowed) {
        setLastResult({
          success: false,
          message: "Secure face verification is disabled in this browser. Use Chrome or Edge on the gate terminal.",
        });
        return;
      }

      const canUseFallbackAlignment =
        fallbackCaptureAllowed && faceAlignmentState === "unsupported";

      if (faceAlignmentState !== "aligned" && !canUseFallbackAlignment) {
        setLastResult({
          success: false,
          message: "Center a single face inside the guide before scanning.",
        });
        return;
      }

      const latestDirection = supportsDirectionTracking
        ? inferMovementDirection(motionSamplesRef.current, {
            maxSamples: GATE_DIRECTION_SAMPLE_WINDOW,
            maxSampleAgeMs: GATE_DIRECTION_MAX_SAMPLE_AGE_MS,
          })
        : null;

      if (
        supportsDirectionTracking
        && latestDirection
        && (
          latestDirection.direction === "UNKNOWN"
          || latestDirection.confidence < GATE_DIRECTION_CONFIDENCE_THRESHOLD
        )
      ) {
        setLastResult({
          success: false,
          message: "Movement direction is unclear. Walk fully through the frame before scanning again.",
        });
        return;
      }

      const liveFaceProfile = await captureLiveFaceProfile();
      if (!liveFaceProfile) {
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
        faceDescriptor: liveFaceProfile.primaryDescriptor,
        faceAnchorDescriptors: liveFaceProfile.anchorDescriptors,
        faceConsistency: liveFaceProfile.consistency,
        faceCaptureMode: liveFaceProfile.captureMode,
        movementDirection: latestDirection?.direction,
        movementConfidence: latestDirection?.confidence,
      });

      setLastResult({
        success: data.success,
        message: data.message,
        employee: data.employee,
        matchConfidence: data.matchConfidence,
        matchDetails: data.matchDetails,
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
      setCapturedFaceSamples(0);
      resetMotionEvidence();
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
      <div className="mx-auto flex min-h-full max-w-6xl flex-col items-center justify-center px-6 py-10 md:px-8">
        <div className="mb-8 text-center">
        <h1 className="text-3xl font-display font-bold text-slate-950">Gate Terminal</h1>
        <p className="mt-2 max-w-md mx-auto text-slate-600">
          Real-time attendance with live camera and RFID scanning
        </p>
      </div>

      <Card className="w-full max-w-5xl overflow-hidden border border-sky-100 bg-white/90 shadow-[0_24px_80px_rgba(30,64,175,0.12)] backdrop-blur">
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

        <CardContent className="grid gap-6 p-6 md:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.9fr)] md:gap-8 md:p-8 xl:grid-cols-[minmax(0,1.45fr)_minmax(360px,0.85fr)]">
          <div className="space-y-3">
            <div className="text-center text-xs font-medium uppercase tracking-wider text-slate-500 md:text-left">
              {cameraActive ? "Live Camera Feed" : cameraError ? "Camera Permission Required" : "Starting Camera"}
            </div>
            <div className="group relative aspect-[16/9] overflow-hidden rounded-3xl border border-sky-100 bg-slate-950 shadow-inner">
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className={`h-full w-full object-cover transition-opacity duration-300 ${cameraActive ? "opacity-100" : "opacity-0"}`}
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
                  <div className="absolute inset-0 m-4 rounded-xl border-2 border-primary/20 opacity-50 pointer-events-none" />
                </>
              )}
              <div className="pointer-events-none absolute inset-0">
                <div className="absolute inset-5">
                  <div className="absolute left-0 top-0 h-12 w-12 rounded-tl-2xl border-l-[3px] border-t-[3px] border-rose-400/80" />
                  <div className="absolute right-0 top-0 h-12 w-12 rounded-tr-2xl border-r-[3px] border-t-[3px] border-rose-400/80" />
                  <div className="absolute bottom-0 left-0 h-12 w-12 rounded-bl-2xl border-b-[3px] border-l-[3px] border-rose-400/80" />
                  <div className="absolute bottom-0 right-0 h-12 w-12 rounded-br-2xl border-b-[3px] border-r-[3px] border-rose-400/80" />
                </div>
                <div
                  className={`absolute inset-x-[26%] inset-y-[16%] rounded-[2rem] border-[3px] transition-all duration-300 ease-out ${faceFrameTone}`}
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
              <div className="pointer-events-none absolute inset-x-[26%] inset-y-[16%] rounded-[2rem]">
                <div className="absolute left-1/2 top-3 -translate-x-1/2 rounded-full bg-black/65 px-3 py-1 text-[10px] font-semibold tracking-[0.28em] text-white/90">
                  {frameStatusLabel}
                </div>
                <div className="absolute inset-x-6 top-1/2 h-px -translate-y-1/2 bg-gradient-to-r from-transparent via-emerald-300/70 to-transparent" />
                <div className="absolute left-1/2 inset-y-6 w-px -translate-x-1/2 bg-gradient-to-b from-transparent via-rose-300/55 to-transparent" />
              </div>
              {(scanMutation.isPending || isSamplingFace) && (
                <div className="absolute inset-0 flex items-center justify-center bg-primary/10 backdrop-blur-[2px]">
                  {isSamplingFace ? (
                    <div className="rounded-full bg-black/70 px-4 py-2 text-sm font-medium text-white">
                      Capturing sample {Math.min(capturedFaceSamples + 1, GATE_FACE_SAMPLE_COUNT)} / {GATE_FACE_SAMPLE_COUNT}
                    </div>
                  ) : (
                    <div className="h-1 w-16 animate-pulse rounded-full bg-primary/80 shadow-[0_0_15px_rgba(var(--primary),0.5)]" />
                  )}
                </div>
              )}
            </div>
            <p className="text-center text-xs text-slate-500 md:text-left">{faceGuideMessage}</p>
          </div>

          <div className="flex flex-col gap-4 md:justify-center">
            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-900">Gate Status</p>
                  <p className="text-xs text-slate-500">Real-time reader and face verification state.</p>
                </div>
                <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${isConnected ? "bg-emerald-100 text-emerald-700" : "bg-slate-200 text-slate-600"}`}>
                  {isConnected ? "Live" : "Offline"}
                </span>
              </div>
              <div className="space-y-2">
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
                  <span className="text-slate-500">Direction samples</span>
                  <span className="font-mono text-slate-900">{directionState.sampleCount}</span>
                </div>
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-500">Face alignment</span>
                  <span className={faceAlignmentState === "aligned" ? "font-medium text-emerald-600" : "text-rose-500"}>
                      {faceAlignmentState === "aligned"
                        ? "Ready"
                        : faceAlignmentState === "unsupported"
                          ? (fallbackCaptureAllowed ? "Fallback" : "Blocked")
                        : "Adjust"}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-500">Live face quality</span>
                  <span className="font-mono text-slate-900">
                    {liveFaceQuality === null ? "--" : `${Math.round(liveFaceQuality * 100)}%`}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-500">Scan consistency</span>
                  <span className="font-mono text-slate-900">
                    {liveFaceConsistency === null ? "--" : `${Math.round(liveFaceConsistency * 100)}%`}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-500">Motion direction</span>
                  <span
                    className={
                      faceAlignmentState === "unsupported"
                        ? "text-slate-500"
                        : directionReady
                          ? directionState.direction === "ENTRY"
                            ? "font-medium text-emerald-600"
                            : "font-medium text-blue-600"
                          : "text-amber-600"
                    }
                  >
                    {faceAlignmentState === "unsupported"
                      ? "Fallback"
                      : directionReady
                        ? directionState.direction
                        : directionState.sampleCount >= 8
                          ? "Unclear"
                          : "Learning"}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3 text-sm">
                  <span className="text-slate-500">Motion confidence</span>
                  <span className="font-mono text-slate-900">
                    {faceAlignmentState === "unsupported"
                      ? "--"
                      : `${Math.round(directionState.confidence * 100)}%`}
                  </span>
                </div>
                {readerSourceDeviceId && (
                  <div className="flex items-center justify-between gap-3 text-sm">
                    <span className="text-slate-500">Reader source</span>
                    <span className="font-mono text-slate-900">{readerSourceDeviceId}</span>
                  </div>
                )}
              </div>
              {readerMessage && (
                <p className="mt-3 text-sm text-slate-600">{readerMessage}</p>
              )}
              <p className="mt-3 text-xs text-slate-500">
                Entry is inferred from left-to-right or approaching-camera motion. Exit uses the reverse path.
              </p>
            </div>

            <div className="space-y-4 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
              <div className="space-y-2">
                <Label htmlFor="rfid" className="flex items-center gap-2 text-slate-700">
                  <KeyRound className="size-4" /> RFID Badge
                </Label>
                <Input
                  id="rfid"
                  placeholder="Tap on the real reader or enter UID manually"
                  className="border-2 border-slate-200 bg-white py-6 text-center font-mono text-lg tracking-widest focus-visible:ring-primary/20"
                  value={rfidUid}
                  onChange={(e) => setRfidUid(e.target.value.toUpperCase())}
                  disabled={scanMutation.isPending}
                  onKeyDown={(e) => e.key === "Enter" && handleScan()}
                />
              </div>

              <Button
                size="lg"
                className="h-14 w-full bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 text-lg font-semibold text-white shadow-md transition-transform active:scale-[0.98]"
                onClick={handleScan}
                disabled={scanMutation.isPending || isSamplingFace || !rfidUid.trim() || !cameraActive || (faceAlignmentState === "unsupported" && !fallbackCaptureAllowed)}
              >
                {isSamplingFace ? "Capturing Face..." : scanMutation.isPending ? "Verifying..." : "Scan & Authenticate"}
              </Button>
            </div>

            <div className="min-h-[88px]">
              {lastResult && (
                <Alert variant={lastResult.success ? "default" : "destructive"} className={`border-2 ${lastResult.success ? "border-emerald-200 bg-emerald-50" : "border-rose-200 bg-rose-50 text-rose-900"}`}>
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
                      <span className="mt-1 block font-mono">
                        Match confidence: {(lastResult.matchConfidence * 100).toFixed(1)}%
                      </span>
                    )}
                    {lastResult.matchDetails && (
                      <span className="mt-1 block font-mono text-[11px] leading-5">
                        Primary {(lastResult.matchDetails.primaryConfidence * 100).toFixed(1)}% | Anchors {(lastResult.matchDetails.anchorAverage * 100).toFixed(1)}% | Peak {(lastResult.matchDetails.peakAnchorConfidence * 100).toFixed(1)}% | Stable {(lastResult.matchDetails.strongAnchorRatio * 100).toFixed(0)}% | Consistency {(lastResult.matchDetails.liveConsistency * 100).toFixed(0)}%
                      </span>
                    )}
                  </AlertDescription>
                </Alert>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
      </div>
    </div>
  );
}
