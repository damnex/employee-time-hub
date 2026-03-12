import { useState, useRef, useEffect } from "react";
import { useScanRFID } from "@/hooks/use-gate";
import { useDeviceWS } from "@/hooks/use-device-ws";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Camera, Scan, KeyRound, AlertCircle, CheckCircle2, Wifi, WifiOff } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { FacePose } from "@shared/schema";
import {
  captureFaceTemplate,
  describeFacePose,
  getBiometricCameraConstraints,
  getBiometricRuntimeInfo,
  startFaceTracking,
  type FaceTrackingSnapshot,
} from "@/lib/biometrics";
import {
  appendMovementSample,
  inferDirectionFromPose,
  type DirectionInferenceResult,
  type MovementSample,
} from "@/lib/movement";

const GATE_DEVICE_ID = "GATE-TERMINAL-01";
const GATE_BROWSER_CLIENT_ID = "GATE-TERMINAL-01-BROWSER";
const GATE_FACE_SAMPLE_COUNT = 12;
const GATE_FACE_SAMPLE_DELAY_MS = 90;
const GATE_DIRECTION_SAMPLE_WINDOW = 25;
const GATE_DIRECTION_MAX_SAMPLE_AGE_MS = 1800;
const GATE_DIRECTION_CONFIDENCE_THRESHOLD = 0.58;
const ENTRY_SIDE_POSE: "left" | "right" = "left";
const EXIT_SIDE_POSE: "left" | "right" = "right";

type FaceAlignmentState =
  | "unsupported"
  | "searching"
  | "aligned"
  | "off-center"
  | "no-face"
  | "multiple";

function isSidePose(pose: FacePose | null | undefined): pose is "left" | "right" {
  return pose === "left" || pose === "right";
}

function describeSidePose(pose: "left" | "right") {
  return pose === "left" ? "left-side face" : "right-side face";
}

export default function GateSimulator() {
  const [rfidUid, setRfidUid] = useState("");
  const [scanTechnology, setScanTechnology] = useState<"HF_RFID" | "UHF_RFID">("HF_RFID");
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
  const [liveLiveness, setLiveLiveness] = useState<number | null>(null);
  const [liveRealness, setLiveRealness] = useState<number | null>(null);
  const [trackedPose, setTrackedPose] = useState("Front");
  const [faceAlignmentState, setFaceAlignmentState] = useState<FaceAlignmentState>("searching");
  const [trackingSnapshot, setTrackingSnapshot] = useState<FaceTrackingSnapshot | null>(null);
  const [directionState, setDirectionState] = useState<DirectionInferenceResult>({
    direction: "UNKNOWN",
    confidence: 0,
    axis: "none",
    sampleCount: 0,
  });
  const [readerMessage, setReaderMessage] = useState<string | null>(null);
  const [readerSourceDeviceId, setReaderSourceDeviceId] = useState<string | null>(null);
  const [liveTapUid, setLiveTapUid] = useState<string | null>(null);
  const [pendingReaderScan, setPendingReaderScan] = useState<{
    rfidUid: string;
    sourceDeviceId: string;
  } | null>(null);
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
      poseConfidence?: number;
      liveLiveness?: number;
      liveRealness?: number;
    };
  } | null>(null);

  const scanMutation = useScanRFID();
  const { isConnected, lastScanResult, clearResult } = useDeviceWS(GATE_BROWSER_CLIENT_ID, { clientType: "browser" });
  const biometricRuntime = getBiometricRuntimeInfo();
  const detectorAvailable = biometricRuntime.detectorAvailable;
  const compatibilityModeMessage = biometricRuntime.detectorAvailable
    ? biometricRuntime.isIOS || biometricRuntime.isSafari
      ? "Model-backed face tracking is active in the browser camera. This is not Apple Face ID hardware, but it uses a real ML face engine."
      : "Model-backed face tracking is active. The gate auto-detects face box, side pose, liveness, and direction."
    : "This browser cannot run the ML face pipeline reliably.";
  const directionReady =
    directionState.direction !== "UNKNOWN"
    && directionState.confidence >= GATE_DIRECTION_CONFIDENCE_THRESHOLD;
  const showDirectionSource = directionState.axis !== "none";
  const showTrackingSamples = directionState.sampleCount > 0;
  const showLastPhysicalTap = Boolean(liveTapUid);
  const showLiveFaceQuality = liveFaceQuality !== null;
  const showLiveFaceConsistency = liveFaceConsistency !== null;
  const showLiveLiveness = liveLiveness !== null;
  const showLiveRealness = liveRealness !== null;
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
              ? "ENTRY POSE"
              : "EXIT POSE"
            : faceAlignmentState === "aligned"
              ? "ALIGN POSE"
              : "ALIGN FACE";
  const faceGuideMessage = (() => {
    if (!cameraActive) {
      return cameraError ?? "Waiting for browser camera access...";
    }

    if (faceAlignmentState === "unsupported") {
      return compatibilityModeMessage;
    }

    if (faceAlignmentState === "multiple") {
      return "Keep only one face inside the frame.";
    }

    if (faceAlignmentState === "off-center") {
      return trackingSnapshot?.guidance ?? "Move the face fully into the tracked frame.";
    }

    if (faceAlignmentState === "no-face") {
      return "Face not detected. Step into the camera frame.";
    }

    if (directionReady) {
      return `${directionState.direction === "ENTRY" ? "Entry" : "Exit"} direction locked from side pose at ${Math.round(directionState.confidence * 100)}% confidence.`;
    }

    if (faceAlignmentState === "aligned") {
      return `Pose ${trackedPose}. Show the ${describeSidePose(ENTRY_SIDE_POSE)} for entry or the ${describeSidePose(EXIT_SIDE_POSE)} for exit while tapping the badge.`;
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
          video: getBiometricCameraConstraints(),
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
    if (!cameraActive || !videoRef.current || isSamplingFace) {
      setFaceAlignmentState((current) => (current === "unsupported" ? current : "searching"));
      setTrackingSnapshot(null);
      resetMotionEvidence();
      return;
    }

    if (!detectorAvailable) {
      setFaceAlignmentState("unsupported");
      setTrackingSnapshot(null);
      resetMotionEvidence();
      return;
    }

    return startFaceTracking(videoRef.current, (snapshot) => {
      setTrackingSnapshot(snapshot);
      setTrackedPose(describeFacePose(snapshot.pose));

      if (snapshot.status === "unsupported") {
        setFaceAlignmentState("unsupported");
        resetMotionEvidence();
        return;
      }

      if (snapshot.status === "no-face") {
        setFaceAlignmentState("no-face");
        resetMotionEvidence();
        return;
      }

      if (snapshot.status === "multiple") {
        setFaceAlignmentState("multiple");
        resetMotionEvidence();
        return;
      }

      if (!snapshot.bounds || !videoRef.current?.videoWidth || !videoRef.current?.videoHeight) {
        setFaceAlignmentState("searching");
        resetMotionEvidence();
        return;
      }

      const normalizedCenterX =
        (snapshot.bounds.x + snapshot.bounds.width / 2) / videoRef.current.videoWidth;
      const normalizedCenterY =
        (snapshot.bounds.y + snapshot.bounds.height / 2) / videoRef.current.videoHeight;
      const normalizedWidth = snapshot.bounds.width / videoRef.current.videoWidth;
      const normalizedHeight = snapshot.bounds.height / videoRef.current.videoHeight;

      setFaceAlignmentState(snapshot.status === "ready" ? "aligned" : "off-center");

      const nextSamples = appendMovementSample(
        motionSamplesRef.current,
        {
          timestamp: Date.now(),
          centerX: normalizedCenterX,
          centerY: normalizedCenterY,
          area: normalizedWidth * normalizedHeight,
          distance: snapshot.distance,
          yaw: snapshot.yaw,
          pose: snapshot.pose,
        },
        GATE_DIRECTION_SAMPLE_WINDOW,
      );
      motionSamplesRef.current = nextSamples;
      const poseDirection = inferDirectionFromPose(snapshot.pose, {
        entryPose: ENTRY_SIDE_POSE,
        confidence: Math.max(0.72, snapshot.quality),
      });
      setDirectionState({
        ...poseDirection,
        sampleCount: nextSamples.length,
      });
    }, {
      mode: "tracking",
    });
  }, [cameraActive, detectorAvailable, isSamplingFace]);

  const captureLiveFaceProfile = async (expectedPose: "left" | "right") => {
    if (!cameraActive || !videoRef.current || !canvasRef.current) {
      return null;
    }

    const template = await captureFaceTemplate(videoRef.current, canvasRef.current, {
      sampleCount: GATE_FACE_SAMPLE_COUNT,
      sampleDelayMs: GATE_FACE_SAMPLE_DELAY_MS,
      minQuality: 0.45,
      maxAttempts: GATE_FACE_SAMPLE_COUNT * 4,
      poseTargets: {
        front: 0,
        left: expectedPose === "left" ? GATE_FACE_SAMPLE_COUNT : 0,
        right: expectedPose === "right" ? GATE_FACE_SAMPLE_COUNT : 0,
        up: 0,
        down: 0,
        unknown: 0,
      },
      minLiveConfidence: 0.45,
      minRealConfidence: 0.35,
      onProgress: (acceptedSamples) => {
        setCapturedFaceSamples(acceptedSamples);
      },
    });

    setLiveFaceQuality(template.averageQuality);
    setLiveFaceConsistency(template.consistency);
    if ("averageLive" in template.profile && typeof template.profile.averageLive === "number") {
      setLiveLiveness(template.profile.averageLive);
    }
    if ("averageReal" in template.profile && typeof template.profile.averageReal === "number") {
      setLiveRealness(template.profile.averageReal);
    }
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
    setLiveLiveness(null);
    setLiveRealness(null);

    try {
      if (faceAlignmentState === "unsupported") {
        setLastResult({
          success: false,
          message: "Model-backed face verification is unavailable in this browser. Use a supported gate browser.",
        });
        return;
      }

      if (faceAlignmentState !== "aligned") {
        setLastResult({
          success: false,
          message: trackingSnapshot?.guidance ?? "Center a single face inside the tracked frame before scanning.",
        });
        return;
      }

      const trackedSidePose = isSidePose(trackingSnapshot?.pose)
        ? trackingSnapshot.pose
        : null;
      const previewDirection = trackedSidePose
        ? inferDirectionFromPose(trackedSidePose, {
            entryPose: ENTRY_SIDE_POSE,
            confidence: Math.max(0.72, trackingSnapshot?.quality ?? 0.72),
          })
        : null;

      if (
        !trackedSidePose
        || !previewDirection
        || previewDirection.direction === "UNKNOWN"
        || previewDirection.confidence < GATE_DIRECTION_CONFIDENCE_THRESHOLD
      ) {
        setLastResult({
          success: false,
          message: `Direction is unclear. Show the ${describeSidePose(ENTRY_SIDE_POSE)} for entry or the ${describeSidePose(EXIT_SIDE_POSE)} for exit while tapping the badge.`,
        });
        return;
      }

      const liveFaceProfile = await captureLiveFaceProfile(trackedSidePose);
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

      const dominantPose = "poseEmbeddings" in liveFaceProfile && liveFaceProfile.poseEmbeddings[0]
        ? liveFaceProfile.poseEmbeddings[0].pose
        : trackingSnapshot?.pose;
      const resolvedDirection = isSidePose(dominantPose)
        ? inferDirectionFromPose(dominantPose, {
            entryPose: ENTRY_SIDE_POSE,
            confidence: 0.96,
          })
        : previewDirection;

      if (
        !resolvedDirection
        || resolvedDirection.direction === "UNKNOWN"
        || resolvedDirection.confidence < GATE_DIRECTION_CONFIDENCE_THRESHOLD
      ) {
        setLastResult({
          success: false,
          message: `Direction is unclear. Hold the ${describeSidePose(ENTRY_SIDE_POSE)} for entry or the ${describeSidePose(EXIT_SIDE_POSE)} for exit and retry.`,
        });
        return;
      }

      const data = await scanMutation.mutateAsync({
        rfidUid: normalizedUid,
        deviceId: source === "reader" ? (sourceDeviceId ?? GATE_DEVICE_ID) : GATE_DEVICE_ID,
        faceDescriptor: liveFaceProfile.primaryDescriptor,
        faceAnchorDescriptors: liveFaceProfile.anchorDescriptors,
        faceConsistency: liveFaceProfile.consistency,
        faceQuality: liveFaceProfile.averageQuality,
        faceCaptureMode: liveFaceProfile.captureMode,
        facePose: "poseEmbeddings" in liveFaceProfile && liveFaceProfile.poseEmbeddings[0]
          ? liveFaceProfile.poseEmbeddings[0].pose
          : "unknown",
        faceYaw: "poseEmbeddings" in liveFaceProfile && liveFaceProfile.poseEmbeddings[0]
          ? liveFaceProfile.poseEmbeddings[0].yaw
          : undefined,
        facePitch: "poseEmbeddings" in liveFaceProfile && liveFaceProfile.poseEmbeddings[0]
          ? liveFaceProfile.poseEmbeddings[0].pitch
          : undefined,
        faceRoll: "poseEmbeddings" in liveFaceProfile && liveFaceProfile.poseEmbeddings[0]
          ? liveFaceProfile.poseEmbeddings[0].roll
          : undefined,
        faceLiveConfidence: "averageLive" in liveFaceProfile ? liveFaceProfile.averageLive : undefined,
        faceRealConfidence: "averageReal" in liveFaceProfile ? liveFaceProfile.averageReal : undefined,
        scanTechnology,
        movementDirection: resolvedDirection.direction,
        movementAxis: resolvedDirection.axis,
        movementConfidence: resolvedDirection.confidence,
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
          ? `Badge ${detectedUid} detected. Show the ${describeSidePose(ENTRY_SIDE_POSE)} for entry or the ${describeSidePose(EXIT_SIDE_POSE)} for exit.`
          : `Badge ${detectedUid} detected from ${sourceDeviceId}. Show the side-face direction for this tap.`,
      );
      setPendingReaderScan({
        rfidUid: detectedUid,
        sourceDeviceId,
      });
      clearResult();
      return;
    }
  }, [lastScanResult, clearResult]);

  useEffect(() => {
    if (
      !pendingReaderScan
      || !cameraActive
      || faceAlignmentState !== "aligned"
      || !directionReady
      || isSamplingFace
      || scanMutation.isPending
    ) {
      return;
    }

    setReaderMessage(
      `Badge ${pendingReaderScan.rfidUid} detected. ${directionState.direction === "ENTRY" ? "Entry" : "Exit"} direction locked. Verifying live face data...`,
    );
    const nextScan = pendingReaderScan;
    setPendingReaderScan(null);
    void authenticateScan(nextScan.rfidUid, "reader", nextScan.sourceDeviceId);
  }, [
    pendingReaderScan,
    cameraActive,
    faceAlignmentState,
    directionReady,
    isSamplingFace,
    scanMutation.isPending,
    directionState.direction,
  ]);

  return (
    <div className="h-full overflow-hidden bg-[radial-gradient(circle_at_top,_#f8fbff_0%,_#eef5ff_42%,_#e5eefb_100%)] text-slate-950">
      <div className="mx-auto flex h-full w-full max-w-[1500px] flex-col gap-3 px-4 py-3 md:px-5 md:py-4 xl:px-6">
        <div className="shrink-0 text-center">
        <h1 className="text-2xl font-display font-bold text-slate-950 md:text-3xl">Gate Terminal</h1>
        <p className="mt-1 max-w-xl mx-auto text-sm text-slate-600">
          Real-time attendance with live camera and RFID scanning
        </p>
      </div>

      <Card className="flex min-h-0 w-full flex-1 flex-col overflow-hidden border border-sky-100 bg-white/90 shadow-[0_24px_80px_rgba(30,64,175,0.12)] backdrop-blur">
        <div className="flex shrink-0 items-center justify-between bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 px-4 py-3 text-white">
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

        <CardContent className="grid min-h-0 flex-1 gap-3 p-3 pt-3 md:grid-cols-[minmax(0,1.45fr)_minmax(330px,0.85fr)] xl:grid-cols-[minmax(0,1.55fr)_minmax(350px,0.82fr)]">
          <div className="flex min-h-0 flex-col gap-1.5">
            <div className="shrink-0 text-center text-[11px] font-medium uppercase tracking-wider text-slate-500 md:text-left">
              {cameraActive ? "Live Camera Feed" : cameraError ? "Camera Permission Required" : "Starting Camera"}
            </div>
            <div className="group relative aspect-video max-h-[50vh] min-h-0 flex-1 overflow-hidden rounded-[1.75rem] border border-sky-100 bg-slate-950 shadow-inner md:aspect-[16/9] xl:max-h-[54vh]">
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
                {trackingSnapshot?.bounds && cameraActive && (
                  <div
                    className={`absolute rounded-[1.1rem] border-2 transition-all duration-200 ${
                      trackingSnapshot.status === "ready"
                        ? "border-cyan-300/80 shadow-[0_0_0_1px_rgba(103,232,249,0.5),0_0_22px_rgba(6,182,212,0.2)]"
                        : "border-amber-300/80"
                    }`}
                    style={{
                      left: `${(trackingSnapshot.bounds.x / (videoRef.current?.videoWidth || 1)) * 100}%`,
                      top: `${(trackingSnapshot.bounds.y / (videoRef.current?.videoHeight || 1)) * 100}%`,
                      width: `${(trackingSnapshot.bounds.width / (videoRef.current?.videoWidth || 1)) * 100}%`,
                      height: `${(trackingSnapshot.bounds.height / (videoRef.current?.videoHeight || 1)) * 100}%`,
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
            <p className="shrink-0 text-center text-[11px] text-slate-500 md:text-left">{faceGuideMessage}</p>
          </div>

          <div className="flex min-h-0 flex-col gap-2">
            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-2">
              <div className="mb-2 flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-slate-900">Gate Status</p>
                  <p className="text-[11px] text-slate-500">Real-time reader and face verification state.</p>
                </div>
                <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${isConnected ? "bg-emerald-100 text-emerald-700" : "bg-slate-200 text-slate-600"}`}>
                  {isConnected ? "Live" : "Offline"}
                </span>
              </div>
              <div className="space-y-1">
                <div className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="text-slate-500">Gate event channel</span>
                  <span className={isConnected ? "font-medium text-slate-900" : "text-slate-500"}>
                    {isConnected ? "Listening for reader taps" : "Browser socket offline"}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="text-slate-500">Preferred device</span>
                  <span className="font-mono text-slate-900">{GATE_DEVICE_ID}</span>
                </div>
                <div className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="text-slate-500">Browser client</span>
                  <span className="font-mono text-slate-900">{GATE_BROWSER_CLIENT_ID}</span>
                </div>
                <div className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="text-slate-500">RFID mode</span>
                  <span className="font-mono text-slate-900">{scanTechnology === "HF_RFID" ? "HF RFID" : "UHF RFID"}</span>
                </div>
                {showLastPhysicalTap && (
                  <div className="flex items-center justify-between gap-3 text-[13px]">
                    <span className="text-slate-500">Last physical tap</span>
                    <span className="font-mono text-slate-900">{liveTapUid}</span>
                  </div>
                )}
                <div className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="text-slate-500">Face samples / scan</span>
                  <span className="font-mono text-slate-900">{GATE_FACE_SAMPLE_COUNT}</span>
                </div>
                {showDirectionSource && (
                  <div className="flex items-center justify-between gap-3 text-[13px]">
                    <span className="text-slate-500">Direction source</span>
                    <span className="font-mono text-slate-900">
                      {directionState.axis === "horizontal"
                        ? "Horizontal"
                        : directionState.axis === "depth"
                          ? "Depth"
                          : "Face pose"}
                    </span>
                  </div>
                )}
                {showTrackingSamples && (
                  <div className="flex items-center justify-between gap-3 text-[13px]">
                    <span className="text-slate-500">Tracking samples</span>
                    <span className="font-mono text-slate-900">{directionState.sampleCount}</span>
                  </div>
                )}
                <div className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="text-slate-500">Face alignment</span>
                  <span className={faceAlignmentState === "aligned" ? "font-medium text-emerald-600" : "text-rose-500"}>
                      {faceAlignmentState === "aligned"
                        ? "Ready"
                        : faceAlignmentState === "unsupported"
                          ? "Blocked"
                          : "Adjust"}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="text-slate-500">Tracked pose</span>
                  <span className="font-mono text-slate-900">{trackedPose}</span>
                </div>
                {showLiveFaceQuality && (
                  <div className="flex items-center justify-between gap-3 text-[13px]">
                    <span className="text-slate-500">Live face quality</span>
                    <span className="font-mono text-slate-900">{`${Math.round(liveFaceQuality * 100)}%`}</span>
                  </div>
                )}
                {showLiveFaceConsistency && (
                  <div className="flex items-center justify-between gap-3 text-[13px]">
                    <span className="text-slate-500">Scan consistency</span>
                    <span className="font-mono text-slate-900">{`${Math.round(liveFaceConsistency * 100)}%`}</span>
                  </div>
                )}
                {showLiveLiveness && (
                  <div className="flex items-center justify-between gap-3 text-[13px]">
                    <span className="text-slate-500">Liveness</span>
                    <span className="font-mono text-slate-900">{`${Math.round(liveLiveness * 100)}%`}</span>
                  </div>
                )}
                {showLiveRealness && (
                  <div className="flex items-center justify-between gap-3 text-[13px]">
                    <span className="text-slate-500">Anti-spoof</span>
                    <span className="font-mono text-slate-900">{`${Math.round(liveRealness * 100)}%`}</span>
                  </div>
                )}
                <div className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="text-slate-500">Detected direction</span>
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
                      ? "Blocked"
                      : directionReady
                        ? directionState.direction
                        : directionState.axis !== "none"
                          ? "Unclear"
                          : "Learning"}
                  </span>
                </div>
                <div className="flex items-center justify-between gap-3 text-[13px]">
                  <span className="text-slate-500">Direction confidence</span>
                  <span className="font-mono text-slate-900">
                    {faceAlignmentState === "unsupported"
                      ? "--"
                      : `${Math.round(directionState.confidence * 100)}%`}
                  </span>
                </div>
                {readerSourceDeviceId && (
                  <div className="flex items-center justify-between gap-3 text-[13px]">
                    <span className="text-slate-500">Reader source</span>
                    <span className="font-mono text-slate-900">{readerSourceDeviceId}</span>
                  </div>
                )}
              </div>
              {readerMessage && (
                <p className="mt-1.5 text-[11px] leading-4 text-slate-600">{readerMessage}</p>
              )}
              <p className="mt-2 text-[11px] leading-4 text-slate-500">
                Left-side tap = entry. Right-side tap = exit.
              </p>
            </div>

            <div className="space-y-2 rounded-2xl border border-slate-200 bg-white p-2 shadow-sm">
              <div className="space-y-1.5">
                <Label className="text-slate-700">RFID Technology</Label>
                <Select value={scanTechnology} onValueChange={(value) => setScanTechnology(value as "HF_RFID" | "UHF_RFID")}>
                  <SelectTrigger className="h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="HF_RFID">HF RFID</SelectItem>
                    <SelectItem value="UHF_RFID">UHF RFID</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-[11px] leading-4 text-slate-500">
                  Tap badge, then show the correct side face.
                </p>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="rfid" className="flex items-center gap-2 text-slate-700">
                  <KeyRound className="size-4" /> RFID Badge
                </Label>
                <Input
                  id="rfid"
                  placeholder="Tap on the real reader or enter UID manually"
                  className="border-2 border-slate-200 bg-white py-3 text-center font-mono text-sm tracking-[0.22em] focus-visible:ring-primary/20"
                  value={rfidUid}
                  onChange={(e) => setRfidUid(e.target.value.toUpperCase())}
                  disabled={scanMutation.isPending}
                  onKeyDown={(e) => e.key === "Enter" && handleScan()}
                />
              </div>

              <Button
                size="lg"
                className="h-10 w-full bg-gradient-to-r from-blue-600 via-blue-500 to-cyan-500 text-sm font-semibold text-white shadow-md transition-transform active:scale-[0.98]"
                onClick={handleScan}
                disabled={scanMutation.isPending || isSamplingFace || !rfidUid.trim() || !cameraActive || faceAlignmentState === "unsupported"}
              >
                {isSamplingFace ? "Capturing Face..." : scanMutation.isPending ? "Verifying..." : "Scan & Authenticate"}
              </Button>
            </div>

            {lastResult && (
              <div className="shrink-0">
                <Alert variant={lastResult.success ? "default" : "destructive"} className={`border-2 p-3 ${lastResult.success ? "border-emerald-200 bg-emerald-50" : "border-rose-200 bg-rose-50 text-rose-900"}`}>
                  {lastResult.success ? (
                    <CheckCircle2 className="size-5 text-emerald-600" />
                  ) : (
                    <AlertCircle className="size-5" />
                  )}
                  <AlertTitle className={lastResult.success ? "font-bold text-emerald-800" : "font-bold"}>
                    {lastResult.success ? "Access Granted" : "Access Denied"}
                  </AlertTitle>
                  <AlertDescription className={`text-xs leading-4 ${lastResult.success ? "text-emerald-700" : ""}`}>
                    {lastResult.message}
                    {typeof lastResult.matchConfidence === "number" && (
                      <span className="mt-1 block font-mono">
                        Match confidence: {(lastResult.matchConfidence * 100).toFixed(1)}%
                      </span>
                    )}
                    {lastResult.matchDetails && (
                      <span className="mt-1 block font-mono text-[11px] leading-5">
                        Primary {(lastResult.matchDetails.primaryConfidence * 100).toFixed(1)}% | Anchors {(lastResult.matchDetails.anchorAverage * 100).toFixed(1)}% | Peak {(lastResult.matchDetails.peakAnchorConfidence * 100).toFixed(1)}% | Stable {(lastResult.matchDetails.strongAnchorRatio * 100).toFixed(0)}% | Consistency {(lastResult.matchDetails.liveConsistency * 100).toFixed(0)}%
                        {typeof lastResult.matchDetails.poseConfidence === "number" && ` | Pose ${(lastResult.matchDetails.poseConfidence * 100).toFixed(1)}%`}
                        {typeof lastResult.matchDetails.liveLiveness === "number" && ` | Live ${(lastResult.matchDetails.liveLiveness * 100).toFixed(0)}%`}
                        {typeof lastResult.matchDetails.liveRealness === "number" && ` | Real ${(lastResult.matchDetails.liveRealness * 100).toFixed(0)}%`}
                      </span>
                    )}
                  </AlertDescription>
                </Alert>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
      </div>
    </div>
  );
}
