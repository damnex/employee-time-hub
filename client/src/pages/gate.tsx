import { useCallback, useEffect, useRef, useState, type FormEvent, type MutableRefObject } from "react";
import type { Employee } from "@shared/schema";
import { fetchLiveFaceRecognition, useScanRFID } from "@/hooks/use-gate";
import { useEmployees } from "@/hooks/use-employees";
import { useDeviceWS } from "@/hooks/use-device-ws";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { detectLiveTrackingFaces, isFaceDetectorAvailable } from "@/lib/biometrics";
import { cn } from "@/lib/utils";
import {
  AlertCircle,
  ArrowLeftRight,
  Camera,
  CheckCircle2,
  KeyRound,
  Loader2,
  MoveHorizontal,
  RefreshCcw,
  ScanLine,
  ShieldCheck,
  UserCircle2,
  Wifi,
  WifiOff,
} from "lucide-react";


const GATE_DEVICE_ID = "GATE-TERMINAL-01";
const GATE_BROWSER_CLIENT_ID = "GATE-TERMINAL-01-BROWSER";
const GATE_FRAME_COUNT = 3;
const GATE_FRAME_DELAY_MS = 20;
const GATE_MAX_FRAME_WIDTH = 480;
const GATE_FRAME_JPEG_QUALITY = 0.55;
const LIVE_TRACKING_INTERVAL_MS = 120;
const LIVE_TRACKING_BUSY_INTERVAL_MS = 190;
const LIVE_RECOGNITION_INTERVAL_MS = 650;
const LIVE_RECOGNITION_IDLE_DELAY_MS = 260;
const LIVE_RECOGNITION_MAX_FRAME_WIDTH = 640;
const LIVE_RECOGNITION_JPEG_QUALITY = 0.6;
const LIVE_RECOGNITION_MIN_CONFIDENCE = 0.68;
const LIVE_RECOGNITION_STABLE_HITS = 3;
const LIVE_RECOGNITION_TTL_MS = 2200;
const LIVE_RECOGNITION_MATCH_DISTANCE = 0.12;
const SENSOR_ACTIVE_WINDOW_MS = 4500;

type PythonFaceStatus = "training" | "trained" | "failed";

type PythonFaceMeta = {
  status: PythonFaceStatus;
  datasetSampleCount: number;
  trainedAt: string | null;
  lastTrainingMessage: string | null;
} | null;

type MatchDetails = {
  primaryConfidence: number;
  anchorAverage: number;
  peakAnchorConfidence: number;
  strongAnchorRatio: number;
  liveConsistency: number;
  poseConfidence?: number;
  liveLiveness?: number;
  liveRealness?: number;
};

type FaceBox = {
  top: number;
  right: number;
  bottom: number;
  left: number;
};

type RecognitionSource = "python" | "browser";

type ProjectedLiveFace = {
  leftPct: number;
  topPct: number;
  widthPct: number;
  heightPct: number;
  centerX: number;
  centerY: number;
  label: string;
  verified: boolean;
  confidence?: number;
  employeeCode?: string;
  department?: string;
  rfidUid?: string;
  source: RecognitionSource;
};

type LiveTrackedFace = ProjectedLiveFace & {
  trackId: number;
  movement: "LEFT" | "RIGHT" | "STEADY";
  stableHits?: number;
  lastSeenAt?: number;
};

type LiveRecognitionAssignment = {
  trackId: number;
  label: string;
  employeeCode?: string;
  department?: string;
  rfidUid?: string;
  confidence: number;
  verified: true;
  stableHits: number;
  lastSeenAt: number;
};

type GateDisplayResult = {
  success: boolean;
  ignored?: boolean;
  message: string;
  employee?: Employee;
  badgeOwner?: Employee;
  action?: "ENTRY" | "EXIT";
  verifiedAt: string;
  latencyMs: number;
  previewImage: string | null;
  source: "manual" | "reader";
  matchConfidence?: number;
  matchDetails?: MatchDetails;
  movementDirection?: "ENTRY" | "EXIT" | "UNKNOWN";
  movementConfidence?: number;
  detectedFaceLabel?: string;
  detectedFaceBox?: FaceBox | null;
  previewFrameSize: {
    width: number;
    height: number;
  } | null;
};

function sleep(durationMs: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs);
  });
}

function getCameraConstraints(): MediaTrackConstraints {
  return {
    facingMode: "user",
    width: { ideal: 1280, min: 640 },
    height: { ideal: 720, min: 480 },
    frameRate: { ideal: 30, max: 30 },
  };
}

function getEmployeeInitials(employee?: Employee) {
  return employee?.name
    ?.split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((segment) => segment[0]?.toUpperCase())
    .join("") || "ID";
}

function formatPercent(value?: number) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function getPythonFaceMeta(faceDescriptor: unknown): PythonFaceMeta {
  if (!faceDescriptor || typeof faceDescriptor !== "object") {
    return null;
  }

  const candidate = faceDescriptor as Record<string, unknown>;
  if (candidate.provider !== "python-opencv-lbph") {
    return null;
  }

  const status = candidate.status;
  if (status !== "training" && status !== "trained" && status !== "failed") {
    return null;
  }

  return {
    status,
    datasetSampleCount:
      typeof candidate.datasetSampleCount === "number" ? candidate.datasetSampleCount : 0,
    trainedAt: typeof candidate.trainedAt === "string" ? candidate.trainedAt : null,
    lastTrainingMessage:
      typeof candidate.lastTrainingMessage === "string" ? candidate.lastTrainingMessage : null,
  };
}

function captureGateFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  options: {
    maxWidth?: number;
    quality?: number;
  } = {},
) {
  const context = canvas.getContext("2d");
  if (!context || !video.videoWidth || !video.videoHeight) {
    return null;
  }

  const aspectRatio = video.videoHeight / video.videoWidth;
  const targetWidth = Math.min(video.videoWidth, options.maxWidth ?? GATE_MAX_FRAME_WIDTH);
  const targetHeight = Math.round(targetWidth * aspectRatio);

  canvas.width = targetWidth;
  canvas.height = targetHeight;
  context.clearRect(0, 0, targetWidth, targetHeight);
  context.drawImage(video, 0, 0, targetWidth, targetHeight);

  return {
    dataUrl: canvas.toDataURL("image/jpeg", options.quality ?? GATE_FRAME_JPEG_QUALITY),
    width: targetWidth,
    height: targetHeight,
  };
}

function getCaptureMessage(progress: number) {
  if (progress <= 2) {
    return "Hold the face centered while the badge tap starts the burst.";
  }

  if (progress <= 5) {
    return "Move in your entry or exit direction so Python can read motion.";
  }

  return "Hold steady for the final verification frames.";
}

function getFaceBoxStyle(
  faceBox: FaceBox | null | undefined,
  frameSize: { width: number; height: number } | null | undefined,
) {
  if (!faceBox || !frameSize || !frameSize.width || !frameSize.height) {
    return null;
  }

  const width = faceBox.right - faceBox.left;
  const height = faceBox.bottom - faceBox.top;
  return {
    left: `${(faceBox.left / frameSize.width) * 100}%`,
    top: `${(faceBox.top / frameSize.height) * 100}%`,
    width: `${(width / frameSize.width) * 100}%`,
    height: `${(height / frameSize.height) * 100}%`,
  };
}

function mapRectToViewport(
  left: number,
  top: number,
  width: number,
  height: number,
  sourceWidth: number,
  sourceHeight: number,
  viewport: HTMLDivElement,
) {
  const viewportWidth = viewport.clientWidth;
  const viewportHeight = viewport.clientHeight;

  if (!viewportWidth || !viewportHeight || !sourceWidth || !sourceHeight) {
    return null;
  }

  const scale = Math.max(viewportWidth / sourceWidth, viewportHeight / sourceHeight);
  const displayWidth = sourceWidth * scale;
  const displayHeight = sourceHeight * scale;
  const offsetX = (viewportWidth - displayWidth) / 2;
  const offsetY = (viewportHeight - displayHeight) / 2;

  const scaledLeft = left * scale + offsetX;
  const scaledTop = top * scale + offsetY;
  const scaledWidth = width * scale;
  const scaledHeight = height * scale;

  const clampedLeft = Math.max(0, Math.min(viewportWidth, scaledLeft));
  const clampedTop = Math.max(0, Math.min(viewportHeight, scaledTop));
  const clampedRight = Math.max(0, Math.min(viewportWidth, scaledLeft + scaledWidth));
  const clampedBottom = Math.max(0, Math.min(viewportHeight, scaledTop + scaledHeight));
  const clampedWidth = Math.max(0, clampedRight - clampedLeft);
  const clampedHeight = Math.max(0, clampedBottom - clampedTop);

  if (!clampedWidth || !clampedHeight) {
    return null;
  }

  return {
    leftPct: (clampedLeft / viewportWidth) * 100,
    topPct: (clampedTop / viewportHeight) * 100,
    widthPct: (clampedWidth / viewportWidth) * 100,
    heightPct: (clampedHeight / viewportHeight) * 100,
    centerX: (clampedLeft + clampedWidth / 2) / viewportWidth,
    centerY: (clampedTop + clampedHeight / 2) / viewportHeight,
  };
}

function mapBoundingBoxToViewport(
  boundingBox: DOMRectReadOnly,
  video: HTMLVideoElement,
  viewport: HTMLDivElement,
) {
  return mapRectToViewport(
    boundingBox.x,
    boundingBox.y,
    boundingBox.width,
    boundingBox.height,
    video.videoWidth,
    video.videoHeight,
    viewport,
  );
}

function mapFaceBoxToViewport(
  faceBox: FaceBox,
  frameSize: { width: number; height: number },
  viewport: HTMLDivElement,
) {
  return mapRectToViewport(
    faceBox.left,
    faceBox.top,
    faceBox.right - faceBox.left,
    faceBox.bottom - faceBox.top,
    frameSize.width,
    frameSize.height,
    viewport,
  );
}

function buildTrackedFaces(
  projectedFaces: ProjectedLiveFace[],
  previousTracksRef: MutableRefObject<Array<{ trackId: number; centerX: number; centerY: number }>>,
  nextTrackIdRef: MutableRefObject<number>,
): LiveTrackedFace[] {
  const previousTracks = [...previousTracksRef.current];
  const usedTrackIds = new Set<number>();
  const nextTrackedFaces = projectedFaces.map((face) => {
    let matchedTrack = previousTracks.find((track) => {
      if (usedTrackIds.has(track.trackId)) {
        return false;
      }

      const dx = track.centerX - face.centerX;
      const dy = track.centerY - face.centerY;
      return Math.sqrt(dx * dx + dy * dy) <= 0.18;
    });

    if (!matchedTrack) {
      matchedTrack = {
        trackId: nextTrackIdRef.current++,
        centerX: face.centerX,
        centerY: face.centerY,
      };
    }

    usedTrackIds.add(matchedTrack.trackId);
    const deltaX = face.centerX - matchedTrack.centerX;
    const movement = deltaX >= 0.015 ? "RIGHT" : deltaX <= -0.015 ? "LEFT" : "STEADY";

    return {
      ...face,
      trackId: matchedTrack.trackId,
      movement,
    } satisfies LiveTrackedFace;
  });

  previousTracksRef.current = nextTrackedFaces.map((face) => ({
    trackId: face.trackId,
    centerX: face.centerX,
    centerY: face.centerY,
  }));

  return nextTrackedFaces;
}

function getProjectedFaceDistance(leftFace: ProjectedLiveFace, rightFace: ProjectedLiveFace) {
  const dx = leftFace.centerX - rightFace.centerX;
  const dy = leftFace.centerY - rightFace.centerY;
  return Math.sqrt(dx * dx + dy * dy);
}

function getProjectedFaceOverlap(leftFace: ProjectedLiveFace, rightFace: ProjectedLiveFace) {
  const left = Math.max(leftFace.leftPct, rightFace.leftPct);
  const top = Math.max(leftFace.topPct, rightFace.topPct);
  const right = Math.min(leftFace.leftPct + leftFace.widthPct, rightFace.leftPct + rightFace.widthPct);
  const bottom = Math.min(leftFace.topPct + leftFace.heightPct, rightFace.topPct + rightFace.heightPct);
  const width = Math.max(0, right - left);
  const height = Math.max(0, bottom - top);
  const intersection = width * height;
  const leftArea = leftFace.widthPct * leftFace.heightPct;
  const rightArea = rightFace.widthPct * rightFace.heightPct;
  return intersection / Math.max(1, Math.min(leftArea, rightArea));
}

function matchRecognizedFacesToTrackedFaces(
  recognizedFaces: ProjectedLiveFace[],
  trackedFaces: LiveTrackedFace[],
) {
  const usedTrackIds = new Set<number>();

  return [...recognizedFaces]
    .sort((left, right) => {
      return (right.confidence ?? 0) - (left.confidence ?? 0);
    })
    .reduce<Array<{ trackId: number; face: ProjectedLiveFace }>>((matches, face) => {
      let bestTrack: LiveTrackedFace | null = null;
      let bestScore = Number.NEGATIVE_INFINITY;

      for (const trackedFace of trackedFaces) {
        if (usedTrackIds.has(trackedFace.trackId)) {
          continue;
        }

        const overlap = getProjectedFaceOverlap(face, trackedFace);
        const distance = getProjectedFaceDistance(face, trackedFace);
        if (overlap < 0.18 && distance > LIVE_RECOGNITION_MATCH_DISTANCE) {
          continue;
        }

        const score = overlap * 3 - distance;
        if (score > bestScore) {
          bestScore = score;
          bestTrack = trackedFace;
        }
      }

      if (!bestTrack) {
        return matches;
      }

      usedTrackIds.add(bestTrack.trackId);
      matches.push({
        trackId: bestTrack.trackId,
        face,
      });
      return matches;
    }, []);
}

export default function GateTerminal() {
  const { data: employees } = useEmployees();
  const scanMutation = useScanRFID();
  const {
    isConnected,
    deviceOnline,
    lastScanResult,
    clearResult,
  } = useDeviceWS(GATE_BROWSER_CLIENT_ID, { clientType: "browser" });
  const cameraViewportRef = useRef<HTMLDivElement>(null);
  const pythonPreviousLiveTracksRef = useRef<Array<{ trackId: number; centerX: number; centerY: number }>>([]);
  const pythonNextTrackIdRef = useRef(1);
  const pythonMissCountRef = useRef(0);
  const browserPreviousLiveTracksRef = useRef<Array<{ trackId: number; centerX: number; centerY: number }>>([]);
  const browserNextTrackIdRef = useRef(1);
  const browserMissCountRef = useRef(0);
  const browserTrackedFacesRef = useRef<LiveTrackedFace[]>([]);
  const sensorWindowTimerRef = useRef<number | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rfidUid, setRfidUid] = useState("");
  const [scanTechnology, setScanTechnology] = useState<"HF_RFID" | "UHF_RFID">("HF_RFID");
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraRetryToken, setCameraRetryToken] = useState(0);
  const [isCapturingFrames, setIsCapturingFrames] = useState(false);
  const [captureProgress, setCaptureProgress] = useState(0);
  const [readerMessage, setReaderMessage] = useState<string | null>(null);
  const [readerSourceDeviceId, setReaderSourceDeviceId] = useState<string | null>(null);
  const [liveTapUid, setLiveTapUid] = useState<string | null>(null);
  const [pythonTrackedFaces, setPythonTrackedFaces] = useState<LiveTrackedFace[]>([]);
  const [browserTrackedFaces, setBrowserTrackedFaces] = useState<LiveTrackedFace[]>([]);
  const [liveRecognitionAssignments, setLiveRecognitionAssignments] = useState<LiveRecognitionAssignment[]>([]);
  const [liveTrackerAvailable, setLiveTrackerAvailable] = useState<boolean | null>(null);
  const [liveRecognitionMessage, setLiveRecognitionMessage] = useState<string | null>(null);
  const [sensorWindowOpen, setSensorWindowOpen] = useState(false);
  const [pendingReaderScan, setPendingReaderScan] = useState<{
    rfidUid: string;
    sourceDeviceId: string;
  } | null>(null);
  const [lastResult, setLastResult] = useState<GateDisplayResult | null>(null);

  const busy = isCapturingFrames || scanMutation.isPending;
  const sensorWindowActive = sensorWindowOpen || busy;
  const busyRef = useRef(busy);
  const normalizedRfidUid = rfidUid.trim().toUpperCase();
  const selectedBadgeOwner = employees?.find((employee) => {
    return employee.rfidUid.toUpperCase() === normalizedRfidUid;
  });
  const latestEmployee = lastResult?.employee ?? lastResult?.badgeOwner ?? selectedBadgeOwner;
  const [hasProfilePhoto, setHasProfilePhoto] = useState<boolean | null>(null);
  const [profileNonce, setProfileNonce] = useState<number>(0);
  const latestProfileImage = latestEmployee
    ? `/api/employees/${latestEmployee.id}/photo${profileNonce ? `?t=${profileNonce}` : ""}`
    : lastResult?.previewImage ?? null;
  const latestFaceMeta = latestEmployee ? getPythonFaceMeta(latestEmployee.faceDescriptor) : null;
  const pythonRosterCount = (employees ?? []).filter((employee) => {
    return Boolean(getPythonFaceMeta(employee.faceDescriptor));
  }).length;
  const pythonTrainedCount = (employees ?? []).filter((employee) => {
    return getPythonFaceMeta(employee.faceDescriptor)?.status === "trained";
  }).length;
  const lastTapDisplay = liveTapUid ?? (normalizedRfidUid || "--");
  const detectedFaceBoxStyle = getFaceBoxStyle(lastResult?.detectedFaceBox, lastResult?.previewFrameSize);
  const stableRecognitionAssignments = liveRecognitionAssignments.filter((assignment) => {
    return Date.now() - assignment.lastSeenAt <= LIVE_RECOGNITION_TTL_MS
      && assignment.stableHits >= LIVE_RECOGNITION_STABLE_HITS;
  });
  const recognitionByTrack = new Map<number, LiveRecognitionAssignment>(
    stableRecognitionAssignments.map((assignment) => [assignment.trackId, assignment]),
  );
  const fusedBrowserTrackedFaces = browserTrackedFaces.map((face) => {
    const recognition = recognitionByTrack.get(face.trackId);
    if (!recognition) {
      return face;
    }

    return {
      ...face,
      label: recognition.label,
      verified: true,
      confidence: recognition.confidence,
      employeeCode: recognition.employeeCode,
      department: recognition.department,
      rfidUid: recognition.rfidUid,
      source: "python" as const,
      stableHits: recognition.stableHits,
      lastSeenAt: recognition.lastSeenAt,
    } satisfies LiveTrackedFace;
  });
  const liveTrackedFaces = liveTrackerAvailable === false ? pythonTrackedFaces : fusedBrowserTrackedFaces;
  const liveMatchedCount = liveTrackerAvailable === false
    ? pythonTrackedFaces.filter((face) => face.verified).length
    : fusedBrowserTrackedFaces.filter((face) => face.verified || face.source === "python").length;
  const liveRecognitionMode: RecognitionSource | "none" = liveTrackerAvailable === false
    ? (pythonTrackedFaces.length ? "python" : "none")
    : liveMatchedCount > 0
      ? "python"
      : browserTrackedFaces.length > 0
        ? "browser"
        : "none";

  useEffect(() => {
    setLiveTrackerAvailable(isFaceDetectorAvailable());
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!latestEmployee?.id) {
      setHasProfilePhoto(null);
      return;
    }

    const loadMeta = async () => {
      try {
        const res = await fetch(`/api/employees/${latestEmployee.id}/photo/meta`, { credentials: "include" });
        if (!res.ok) {
          throw new Error("meta fetch failed");
        }
        const data = await res.json();
        if (!cancelled) {
          setHasProfilePhoto(Boolean(data?.hasProfilePhoto));
          setProfileNonce(Date.now());
        }
      } catch {
        if (!cancelled) {
          setHasProfilePhoto(false);
          setProfileNonce(Date.now());
        }
      }
    };

    void loadMeta();

    return () => {
      cancelled = true;
    };
  }, [latestEmployee?.id]);

  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  useEffect(() => {
    browserTrackedFacesRef.current = browserTrackedFaces;
  }, [browserTrackedFaces]);

  const armSensorWindow = useCallback((durationMs = SENSOR_ACTIVE_WINDOW_MS) => {
    setSensorWindowOpen(true);

    if (sensorWindowTimerRef.current !== null) {
      window.clearTimeout(sensorWindowTimerRef.current);
    }

    sensorWindowTimerRef.current = window.setTimeout(() => {
      sensorWindowTimerRef.current = null;
      setSensorWindowOpen(false);
    }, durationMs);
  }, []);

  useEffect(() => {
    return () => {
      if (sensorWindowTimerRef.current !== null) {
        window.clearTimeout(sensorWindowTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;

    const initCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraActive(false);
        setCameraError("This browser does not support webcam access.");
        return;
      }

      setCameraActive(false);
      setCameraError(null);

      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: getCameraConstraints(),
          audio: false,
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
        console.error("Gate camera failed:", error);
        setCameraActive(false);
        setCameraError("Allow camera access so the gate can capture live verification frames.");
      }
    };

    void initCamera();

    return () => {
      cancelled = true;
      setCameraActive(false);

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
    if (
      !cameraActive
      || !videoRef.current
      || !canvasRef.current
      || !cameraViewportRef.current
      || !sensorWindowActive
    ) {
      setPythonTrackedFaces([]);
      setLiveRecognitionAssignments([]);
      pythonPreviousLiveTracksRef.current = [];
      pythonMissCountRef.current = 0;
      setLiveRecognitionMessage(
        cameraActive ? "Waiting for an RFID trigger before running face recognition." : null,
      );
      return;
    }

    let cancelled = false;
    let timerId: number | null = null;

    const scheduleNext = (
      delayMs = busyRef.current ? LIVE_RECOGNITION_INTERVAL_MS + 180 : LIVE_RECOGNITION_INTERVAL_MS,
    ) => {
      if (cancelled) {
        return;
      }

      timerId = window.setTimeout(() => {
        void recognizeFaces();
      }, delayMs);
    };

    const pruneAssignments = () => {
      const now = Date.now();
      setLiveRecognitionAssignments((previous) => {
        return previous.filter((assignment) => now - assignment.lastSeenAt <= LIVE_RECOGNITION_TTL_MS);
      });
    };

    const recognizeFaces = async () => {
      const hasBrowserTracker = liveTrackerAvailable !== false;
      const trackedFacesBeforeRequest = hasBrowserTracker ? browserTrackedFacesRef.current : [];

      if (
        cancelled
        || !videoRef.current
        || !canvasRef.current
        || !cameraViewportRef.current
        || videoRef.current.readyState < 2
        || !videoRef.current.videoWidth
        || !videoRef.current.videoHeight
      ) {
        scheduleNext(LIVE_RECOGNITION_IDLE_DELAY_MS);
        return;
      }

      if (busyRef.current) {
        scheduleNext(LIVE_RECOGNITION_INTERVAL_MS + 180);
        return;
      }

      if (hasBrowserTracker && !trackedFacesBeforeRequest.length) {
        pruneAssignments();
        scheduleNext(LIVE_RECOGNITION_IDLE_DELAY_MS);
        return;
      }

      const frame = captureGateFrame(videoRef.current, canvasRef.current, {
        maxWidth: LIVE_RECOGNITION_MAX_FRAME_WIDTH,
        quality: LIVE_RECOGNITION_JPEG_QUALITY,
      });
      if (!frame) {
        scheduleNext(LIVE_RECOGNITION_IDLE_DELAY_MS);
        return;
      }

      try {
        const response = await fetchLiveFaceRecognition({
          deviceId: GATE_BROWSER_CLIENT_ID,
          frame: frame.dataUrl,
          maxFaces: hasBrowserTracker
            ? Math.min(Math.max(trackedFacesBeforeRequest.length + 2, 4), 50)
            : 50,
        });
        if (cancelled || !cameraViewportRef.current) {
          return;
        }

        setLiveRecognitionMessage(response.message);

        if (!response.success) {
          pythonMissCountRef.current += 1;
          pruneAssignments();
          if (liveTrackerAvailable === false && pythonMissCountRef.current >= 3) {
            setPythonTrackedFaces([]);
            pythonPreviousLiveTracksRef.current = [];
          }
          scheduleNext(1200);
          return;
        }

        const frameSize = {
          width: response.frameWidth ?? frame.width,
          height: response.frameHeight ?? frame.height,
        };
        const projectedFaces = response.faces.reduce<ProjectedLiveFace[]>((result, face) => {
          if (
            !face.verified
            || typeof face.confidence !== "number"
            || face.confidence < LIVE_RECOGNITION_MIN_CONFIDENCE
          ) {
            return result;
          }

          const projectedFace = mapFaceBoxToViewport(face.box, frameSize, cameraViewportRef.current!);
          if (!projectedFace) {
            return result;
          }

          result.push({
            ...projectedFace,
            label: face.label,
            verified: true,
            confidence: face.confidence,
            employeeCode: face.employeeCode ?? undefined,
            department: face.department ?? undefined,
            rfidUid: face.rfidUid ?? undefined,
            source: "python",
          });
          return result;
        }, []).sort((left, right) => (right.confidence ?? 0) - (left.confidence ?? 0));

        if (!hasBrowserTracker) {
          if (!projectedFaces.length) {
            pythonMissCountRef.current += 1;
            if (pythonMissCountRef.current >= 4) {
              setPythonTrackedFaces([]);
              pythonPreviousLiveTracksRef.current = [];
            }
            scheduleNext(LIVE_RECOGNITION_IDLE_DELAY_MS);
            return;
          }

          pythonMissCountRef.current = 0;
          setPythonTrackedFaces(
            buildTrackedFaces(
              projectedFaces,
              pythonPreviousLiveTracksRef,
              pythonNextTrackIdRef,
            ),
          );
          scheduleNext();
          return;
        }

        pythonMissCountRef.current = projectedFaces.length ? 0 : pythonMissCountRef.current + 1;

        const trackedFacesForMatching = browserTrackedFacesRef.current;
        setLiveRecognitionAssignments((previous) => {
          const now = Date.now();
          const previousByTrack = new Map(previous.map((assignment) => [assignment.trackId, assignment]));
          const matchedFaces = matchRecognizedFacesToTrackedFaces(projectedFaces, trackedFacesForMatching);
          const nextAssignments = matchedFaces.map((match) => {
            const prior = previousByTrack.get(match.trackId);
            const sameEmployee = prior?.label === match.face.label && prior?.employeeCode === match.face.employeeCode;
            return {
              trackId: match.trackId,
              label: match.face.label,
              employeeCode: match.face.employeeCode,
              department: match.face.department,
              rfidUid: match.face.rfidUid,
              confidence: match.face.confidence ?? 0,
              verified: true as const,
              stableHits: sameEmployee ? Math.min((prior?.stableHits ?? 0) + 1, LIVE_RECOGNITION_STABLE_HITS + 4) : 1,
              lastSeenAt: now,
            } satisfies LiveRecognitionAssignment;
          });
          const matchedTrackIds = new Set(nextAssignments.map((assignment) => assignment.trackId));
          const preservedAssignments = previous.filter((assignment) => {
            return !matchedTrackIds.has(assignment.trackId)
              && now - assignment.lastSeenAt <= LIVE_RECOGNITION_TTL_MS;
          });
          return [...nextAssignments, ...preservedAssignments];
        });

        scheduleNext(projectedFaces.length ? LIVE_RECOGNITION_INTERVAL_MS : LIVE_RECOGNITION_IDLE_DELAY_MS);
      } catch (error) {
        console.warn("Live Python recognition failed:", error);
        pythonMissCountRef.current += 1;
        pruneAssignments();
        if (liveTrackerAvailable === false && pythonMissCountRef.current >= 3) {
          setPythonTrackedFaces([]);
          pythonPreviousLiveTracksRef.current = [];
        }
        setLiveRecognitionMessage(
          error instanceof Error
            ? error.message
            : "Live recognition is temporarily unavailable.",
        );
        scheduleNext(1100);
      }
    };

    if (liveTrackerAvailable !== false) {
      setPythonTrackedFaces([]);
    }

    void recognizeFaces();

    return () => {
      cancelled = true;
      if (timerId !== null) {
        window.clearTimeout(timerId);
      }
      if (liveTrackerAvailable === false) {
        setPythonTrackedFaces([]);
        pythonPreviousLiveTracksRef.current = [];
        pythonMissCountRef.current = 0;
      }
    };
  }, [cameraActive, liveTrackerAvailable, sensorWindowActive]);

  useEffect(() => {
    if (
      !cameraActive
      || !videoRef.current
      || !cameraViewportRef.current
      || liveTrackerAvailable === false
      || !sensorWindowActive
    ) {
      setBrowserTrackedFaces([]);
      browserTrackedFacesRef.current = [];
      browserPreviousLiveTracksRef.current = [];
      browserMissCountRef.current = 0;
      return;
    }

    let cancelled = false;
    let timerId: number | null = null;

    const scheduleNext = (
      delayMs = busyRef.current ? LIVE_TRACKING_BUSY_INTERVAL_MS : LIVE_TRACKING_INTERVAL_MS,
    ) => {
      if (cancelled) {
        return;
      }

      timerId = window.setTimeout(() => {
        void detectFaces();
      }, delayMs);
    };

    const detectFaces = async () => {
      if (
        cancelled
        || !videoRef.current
        || !cameraViewportRef.current
        || videoRef.current.readyState < 2
        || !videoRef.current.videoWidth
        || !videoRef.current.videoHeight
      ) {
        scheduleNext(LIVE_TRACKING_INTERVAL_MS);
        return;
      }

      try {
        const detections = await detectLiveTrackingFaces(videoRef.current, {
          maxDetected: Math.min(Math.max(browserTrackedFacesRef.current.length + 6, 12), 50),
        });
        if (cancelled || !videoRef.current || !cameraViewportRef.current) {
          return;
        }

        const projectedFaces = detections.reduce<ProjectedLiveFace[]>((result, detection) => {
          const projectedFace = mapRectToViewport(
            detection.bounds.x,
            detection.bounds.y,
            detection.bounds.width,
            detection.bounds.height,
            videoRef.current!.videoWidth,
            videoRef.current!.videoHeight,
            cameraViewportRef.current!,
          );
          if (!projectedFace) {
            return result;
          }

          result.push({
            ...projectedFace,
            label: "Tracking Face",
            verified: false,
            confidence: detection.faceScore,
            source: "browser",
          });
          return result;
        }, []).sort((left, right) => left.centerX - right.centerX);

        if (!projectedFaces.length) {
          browserMissCountRef.current += 1;
          if (browserMissCountRef.current >= 6) {
            setBrowserTrackedFaces([]);
            browserTrackedFacesRef.current = [];
            browserPreviousLiveTracksRef.current = [];
          }
          scheduleNext();
          return;
        }

        browserMissCountRef.current = 0;
        const nextTrackedFaces = buildTrackedFaces(
          projectedFaces,
          browserPreviousLiveTracksRef,
          browserNextTrackIdRef,
        );
        browserTrackedFacesRef.current = nextTrackedFaces;
        setBrowserTrackedFaces(nextTrackedFaces);
      } catch (error) {
        console.warn("Live motion tracker failed:", error);
        browserMissCountRef.current += 1;
        if (browserMissCountRef.current >= 4) {
          setBrowserTrackedFaces([]);
          browserTrackedFacesRef.current = [];
          browserPreviousLiveTracksRef.current = [];
        }
      }

      scheduleNext();
    };

    void detectFaces();

    return () => {
      cancelled = true;
      if (timerId !== null) {
        window.clearTimeout(timerId);
      }
      setBrowserTrackedFaces([]);
      browserTrackedFacesRef.current = [];
      browserPreviousLiveTracksRef.current = [];
      browserMissCountRef.current = 0;
    };
  }, [cameraActive, liveTrackerAvailable, sensorWindowActive]);

  const captureFrameBurst = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      throw new Error("Camera preview is not ready yet.");
    }

    const frames: string[] = [];
    let previewImage: string | null = null;
    let previewFrameSize: { width: number; height: number } | null = null;

    for (let index = 0; index < GATE_FRAME_COUNT; index += 1) {
      if (index > 0) {
        await sleep(GATE_FRAME_DELAY_MS);
      }

      const frame = captureGateFrame(videoRef.current, canvasRef.current);
      if (!frame) {
        throw new Error("Frame capture failed. Keep one face visible and retry.");
      }

      frames.push(frame.dataUrl);
      if (index === Math.floor(GATE_FRAME_COUNT / 2)) {
        previewImage = frame.dataUrl;
        previewFrameSize = {
          width: frame.width,
          height: frame.height,
        };
      }
      setCaptureProgress(index + 1);
    }

    return {
      frames,
      previewImage: previewImage ?? frames[frames.length - 1] ?? null,
      previewFrameSize,
    };
  }, []);

  const handleScan = useCallback(async (
    rawUid: string,
    source: "manual" | "reader" = "manual",
    sourceDeviceId?: string,
  ) => {
    const normalizedUid = rawUid.trim().toUpperCase();
    if (!normalizedUid || busy) {
      return;
    }

    armSensorWindow();
    setRfidUid(normalizedUid);
    setLiveTapUid(normalizedUid);

    const badgeOwner = employees?.find((employee) => {
      return employee.rfidUid.toUpperCase() === normalizedUid;
    });
    const requestDeviceId = sourceDeviceId ?? readerSourceDeviceId ?? GATE_DEVICE_ID;
    const scanStartedAt = performance.now();
    const visibleFaceCount = liveTrackerAvailable === false
      ? pythonTrackedFaces.length
      : browserTrackedFacesRef.current.length;

    if (!cameraActive) {
      setLastResult({
        success: false,
        message: cameraError ?? "Camera is not ready. Retry the webcam before scanning.",
        employee: badgeOwner,
        badgeOwner,
        verifiedAt: new Date().toLocaleTimeString(),
        latencyMs: Math.round(performance.now() - scanStartedAt),
        previewImage: null,
        source,
        previewFrameSize: null,
      });
      return;
    }

    if (visibleFaceCount > 1) {
      setLastResult({
        success: false,
        message: "Multiple people are in view. Keep only one face visible before scanning again.",
        employee: badgeOwner,
        badgeOwner,
        verifiedAt: new Date().toLocaleTimeString(),
        latencyMs: Math.round(performance.now() - scanStartedAt),
        previewImage: null,
        source,
        previewFrameSize: null,
      });
      return;
    }

    setIsCapturingFrames(true);
    setCaptureProgress(0);

    try {
      const { frames, previewImage, previewFrameSize } = await captureFrameBurst();
      const response = await scanMutation.mutateAsync({
        rfidUid: normalizedUid,
        deviceId: requestDeviceId,
        scanTechnology,
        faceFrames: frames,
      });

      setReaderSourceDeviceId(requestDeviceId);
      setReaderMessage(
        source === "reader"
          ? `Badge tap received from ${requestDeviceId}. Python verification completed.`
          : `Manual verification sent through ${requestDeviceId}.`,
      );
      setLastResult({
        ...response,
        badgeOwner,
        verifiedAt: new Date().toLocaleTimeString(),
        latencyMs: Math.round(performance.now() - scanStartedAt),
        previewImage,
        source,
        previewFrameSize,
      });
    } catch (error) {
      setLastResult({
        success: false,
        message: error instanceof Error ? error.message : "Gate verification failed unexpectedly.",
        employee: badgeOwner,
        badgeOwner,
        verifiedAt: new Date().toLocaleTimeString(),
        latencyMs: Math.round(performance.now() - scanStartedAt),
        previewImage: null,
        source,
        previewFrameSize: null,
      });
    } finally {
      setIsCapturingFrames(false);
      setCaptureProgress(0);
    }
  }, [
    armSensorWindow,
    busy,
    cameraActive,
    cameraError,
    captureFrameBurst,
    employees,
    liveTrackerAvailable,
    pythonTrackedFaces,
    readerSourceDeviceId,
    scanMutation,
    scanTechnology,
  ]);

  useEffect(() => {
    if (lastScanResult?.type !== "rfid_detected" || !lastScanResult.rfidUid) {
      return;
    }

    const tappedUid = lastScanResult.rfidUid.trim().toUpperCase();
    const sourceDeviceId = lastScanResult.deviceId ?? GATE_DEVICE_ID;

    armSensorWindow();
    setRfidUid(tappedUid);
    setLiveTapUid(tappedUid);
    setReaderMessage(lastScanResult.message);
    setReaderSourceDeviceId(sourceDeviceId);
    clearResult();

    if (busy) {
      setPendingReaderScan({
        rfidUid: tappedUid,
        sourceDeviceId,
      });
      return;
    }

    void handleScan(tappedUid, "reader", sourceDeviceId);
  }, [armSensorWindow, busy, clearResult, handleScan, lastScanResult]);

  useEffect(() => {
    if (lastScanResult?.type !== "scan_result" || !lastScanResult.rfidUid) {
      return;
    }

    const scannedUid = lastScanResult.rfidUid.trim().toUpperCase();
    const badgeOwner = employees?.find((employee) => {
      return employee.rfidUid.toUpperCase() === scannedUid;
    });
    const matchedEmployee = lastScanResult.employee?.id
      ? employees?.find((employee) => employee.id === lastScanResult.employee?.id)
      : undefined;

    setRfidUid(scannedUid);
    setLiveTapUid(scannedUid);
    setReaderMessage(lastScanResult.message);
    setReaderSourceDeviceId(lastScanResult.deviceId ?? GATE_DEVICE_ID);
    setLastResult({
      success: Boolean(lastScanResult.success),
      ignored: lastScanResult.ignored,
      message: lastScanResult.message,
      employee: matchedEmployee,
      badgeOwner,
      action: lastScanResult.action,
      verifiedAt: new Date().toLocaleTimeString(),
      latencyMs: 0,
      previewImage: null,
      source: "reader",
      matchConfidence: lastScanResult.matchConfidence,
      matchDetails: lastScanResult.matchDetails,
      movementDirection: lastScanResult.movementDirection,
      movementConfidence: lastScanResult.movementConfidence,
      detectedFaceLabel: lastScanResult.detectedFaceLabel,
      detectedFaceBox: lastScanResult.detectedFaceBox ?? null,
      previewFrameSize: null,
    });
    clearResult();
  }, [clearResult, employees, lastScanResult]);

  useEffect(() => {
    if (!pendingReaderScan || busy) {
      return;
    }

    const queuedScan = pendingReaderScan;
    setPendingReaderScan(null);
    void handleScan(queuedScan.rfidUid, "reader", queuedScan.sourceDeviceId);
  }, [busy, handleScan, pendingReaderScan]);

  const handleManualSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void handleScan(normalizedRfidUid, "manual");
  };

  const cameraFrameTone = !cameraActive
    ? "border-slate-300 shadow-none"
    : lastResult?.ignored
      ? "border-amber-300 shadow-[0_0_0_1px_rgba(253,224,71,0.82),0_0_28px_rgba(253,224,71,0.24)]"
    : lastResult?.success
      ? "border-emerald-400 shadow-[0_0_0_1px_rgba(74,222,128,0.82),0_0_28px_rgba(74,222,128,0.28)]"
      : lastResult
        ? "border-rose-400 shadow-[0_0_0_1px_rgba(251,113,133,0.82),0_0_28px_rgba(251,113,133,0.24)]"
        : busy
          ? "border-amber-300 shadow-[0_0_0_1px_rgba(253,224,71,0.78),0_0_24px_rgba(253,224,71,0.22)]"
          : "border-cyan-300 shadow-[0_0_0_1px_rgba(34,211,238,0.78),0_0_24px_rgba(34,211,238,0.22)]";
  const lastResultTone = !lastResult
    ? "border-slate-200 bg-white/90"
    : lastResult.ignored
      ? "border-amber-200 bg-amber-50/90"
    : lastResult.success
      ? "border-emerald-200 bg-emerald-50/90"
      : "border-rose-200 bg-rose-50/90";

  return (
    <div className="flex h-[calc(100vh-60px)] flex-col gap-2 px-6 md:px-8 lg:px-10 xl:px-12 py-3 animate-in fade-in duration-300 overflow-hidden">
      <Card className="overflow-hidden border-border/60 bg-white/90 shadow-sm">
        <CardContent className="flex flex-wrap items-center justify-between gap-2 p-3">
          <div className="flex items-baseline gap-2">
            <h1 className="text-2xl font-semibold tracking-tight text-foreground">Gate Monitor</h1>
            <p className="text-sm text-muted-foreground">RFID tap + live face verification in one view.</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge
              variant={deviceOnline ? "secondary" : "outline"}
              className={cn(
                deviceOnline
                  ? "bg-emerald-100 text-emerald-800 border-emerald-200"
                  : "border-slate-300 text-slate-600",
              )}
            >
              {deviceOnline ? "Reader Online" : "Reader Offline"}
            </Badge>
            <Badge
              variant={cameraActive ? "secondary" : "outline"}
              className={cn(
                cameraActive ? "bg-cyan-100 text-cyan-800" : "border-slate-300 text-slate-600",
              )}
            >
              {cameraActive ? "Camera Ready" : "Camera Blocked"}
            </Badge>
            <Badge variant="outline" className="border-slate-200 text-slate-700">
              {pythonTrainedCount} trained / {pythonRosterCount} enrolled
            </Badge>
          </div>
        </CardContent>
      </Card>

      <div className="grid flex-1 min-h-0 gap-2 xl:grid-cols-[1.65fr_0.85fr]">
        <Card className="h-full overflow-hidden border-border/60 shadow-sm">
          <CardContent className="flex h-full flex-col gap-2 p-3">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <h2 className="text-xl font-semibold text-foreground">Live gate camera</h2>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                {isConnected ? <Wifi className="size-4 text-emerald-600" /> : <WifiOff className="size-4" />}
                <span>{readerSourceDeviceId ?? GATE_DEVICE_ID}</span>
              </div>
            </div>

            <div className="relative flex-1 overflow-hidden rounded-[1.6rem] border border-border/60 bg-slate-950 min-h-[320px]">
              <div ref={cameraViewportRef} className="h-full w-full overflow-hidden">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  className={`h-full w-full object-cover transition-opacity duration-300 ${
                    cameraActive ? "opacity-100" : "opacity-0"
                  }`}
                />
                {!cameraActive && (
                  <div className="absolute inset-0 flex h-full items-center justify-center px-6 text-center text-white/80">
                    <div className="space-y-3">
                      <Camera className="mx-auto size-8 text-white/65" />
                      <p>{cameraError ?? "Starting the webcam preview..."}</p>
                      {cameraError && (
                        <Button type="button" variant="secondary" onClick={() => setCameraRetryToken((value) => value + 1)}>
                          <RefreshCcw className="mr-2 size-4" />
                          Retry Camera
                        </Button>
                      )}
                    </div>
                  </div>
                )}
                <canvas ref={canvasRef} className="hidden" />
              </div>

              <div className="pointer-events-none absolute inset-0">
                {liveTrackedFaces.map((face) => (
                  <div
                    key={`track-${face.trackId}`}
                    className={cn(
                      "absolute rounded-[1.35rem] border-[3px] shadow-[0_0_0_1px_rgba(255,255,255,0.18),0_0_22px_rgba(15,23,42,0.14)] transition-all duration-150",
                      face.verified || face.source === "python"
                        ? "border-emerald-300 shadow-[0_0_0_1px_rgba(110,231,183,0.5),0_0_24px_rgba(16,185,129,0.2)]"
                        : "border-cyan-300 shadow-[0_0_0_1px_rgba(34,211,238,0.45),0_0_22px_rgba(34,211,238,0.18)]",
                    )}
                    style={{
                      left: `${face.leftPct}%`,
                      top: `${face.topPct}%`,
                      width: `${face.widthPct}%`,
                      height: `${face.heightPct}%`,
                    }}
                  >
                    <div className="absolute left-0 top-0 max-w-[calc(100%+7rem)] -translate-y-[calc(100%+0.42rem)] rounded-full bg-black/80 px-3 py-1 text-[10px] font-semibold tracking-[0.12em] text-white shadow-lg">
                      {face.verified || face.source === "python"
                        ? `${face.label}${face.employeeCode ? ` ${face.employeeCode}` : ""} ${face.movement}${typeof face.confidence === "number" ? ` ${formatPercent(face.confidence)}` : ""}`
                        : `TRACK ${String(face.trackId).padStart(2, "0")} ${face.movement}`}
                    </div>
                  </div>
                ))}
                <div className="absolute left-1/2 top-4 -translate-x-1/2 rounded-full bg-black/70 px-3 py-1 text-[10px] font-semibold tracking-[0.28em] text-white/90">
                  {liveRecognitionMode === "python"
                    ? "FAST TRACKER + PYTHON NAMES"
                    : liveTrackerAvailable === false
                      ? "PYTHON LIVE RECOGNITION"
                      : "FAST LIVE FACE TRACKER"}
                </div>
              </div>

              {busy && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-[2px]">
                  <div className="rounded-2xl bg-black/70 px-5 py-4 text-center text-sm text-white shadow-lg">
                    <div className="flex items-center justify-center gap-2">
                      <Loader2 className="size-4 animate-spin" />
                      <span>
                        {isCapturingFrames
                          ? `Capturing frame ${captureProgress} / ${GATE_FRAME_COUNT}`
                          : "Sending frames to Python verifier"}
                      </span>
                    </div>
                    {isCapturingFrames && (
                      <p className="mt-2 text-xs text-white/70">{getCaptureMessage(captureProgress)}</p>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-2 text-xs text-muted-foreground overflow-auto">
              <div className="flex items-center justify-between">
                <span>Scan progress</span>
                <span>{isCapturingFrames ? `${captureProgress}/${GATE_FRAME_COUNT}` : "Idle"}</span>
              </div>
              <Progress value={isCapturingFrames ? (captureProgress / GATE_FRAME_COUNT) * 100 : 0} />
              <p>
                {liveTrackerAvailable === false
                  ? `${liveRecognitionMessage ?? "Python live recognition is active."} Faces: ${liveTrackedFaces.length}, names: ${liveMatchedCount}.`
                  : liveRecognitionMode === "python"
                    ? `${liveRecognitionMessage ?? "Tracking with Python names."} Tracks: ${browserTrackedFaces.length}, names: ${liveMatchedCount}.`
                    : `${liveRecognitionMessage ?? "Tracking locally while Python samples frames."} Tracks: ${browserTrackedFaces.length}, names: ${liveMatchedCount}.`}
              </p>
            </div>

            <div className="grid gap-2 sm:grid-cols-3">
              <div className="rounded-2xl border border-border/60 bg-muted/20 px-3 py-2">
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Last Tap</p>
                <p className="mt-1 font-mono text-sm text-foreground">{lastTapDisplay}</p>
              </div>
              <div className="rounded-2xl border border-border/60 bg-muted/20 px-3 py-2">
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Queue</p>
                <p className="mt-1 text-sm text-foreground">
                  {pendingReaderScan ? `Waiting ${pendingReaderScan.rfidUid}` : "No queued badge"}
                </p>
              </div>
              <div className="rounded-2xl border border-border/60 bg-muted/20 px-3 py-2">
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Reader Channel</p>
                <p className="mt-1 text-sm text-foreground">{isConnected ? "Live" : "Offline"}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-3 w-full xl:max-w-[360px] xl:justify-self-end min-h-0 overflow-auto hide-scrollbar">
          <Card className={cn("border shadow-sm", lastResultTone)}>
            <CardContent className="space-y-2 p-3">
              <div className="flex items-start gap-4 md:gap-5">
                <Avatar className="size-28 rounded-[2rem] border border-white/60 shadow-sm">
                  <AvatarImage src={latestProfileImage ?? undefined} alt={latestEmployee?.name ?? "Latest gate preview"} />
                  <AvatarFallback className="rounded-[2rem] bg-slate-200 text-lg font-semibold text-slate-700">
                    {getEmployeeInitials(latestEmployee)}
                  </AvatarFallback>
                </Avatar>

                <div className="min-w-0 flex-1">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                        Latest Verification
                      </p>
                      <h2 className="truncate text-lg font-semibold text-foreground leading-tight">
                        {latestEmployee?.name ?? "Waiting for next badge tap"}
                      </h2>
                      <p className="truncate text-sm text-muted-foreground">
                        {latestEmployee
                          ? `${latestEmployee.employeeCode} - ${latestEmployee.department}`
                          : "Badge tap or enter a UID manually to start the Python face scan."}
                      </p>
                    </div>
                    <Badge
                      variant="outline"
                      className={cn(
                        "shrink-0",
                        !lastResult
                          ? "border-slate-300 text-slate-600"
                          : lastResult.ignored
                            ? "border-amber-300 bg-amber-100 text-amber-700"
                          : lastResult.success
                            ? "border-emerald-300 bg-emerald-100 text-emerald-700"
                            : "border-rose-300 bg-rose-100 text-rose-700",
                      )}
                    >
                      {lastResult
                        ? lastResult.ignored
                          ? "IGNORED"
                          : lastResult.success
                          ? lastResult.action ?? "MATCHED"
                          : "REJECTED"
                        : "IDLE"}
                    </Badge>
                  </div>

                  <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                    <div className="rounded-xl bg-white/80 p-2">
                      <p className="uppercase tracking-[0.18em] text-muted-foreground">Badge</p>
                      <p className="mt-1 font-mono text-foreground">
                        {(lastResult?.badgeOwner?.rfidUid ?? normalizedRfidUid) || "--"}
                      </p>
                    </div>
                    <div className="rounded-xl bg-white/80 p-2">
                      <p className="uppercase tracking-[0.18em] text-muted-foreground">Latency</p>
                      <p className="mt-1 font-mono text-foreground">
                        {lastResult ? `${lastResult.latencyMs} ms` : "--"}
                      </p>
                    </div>
                    <div className="rounded-xl bg-white/80 p-2">
                      <p className="uppercase tracking-[0.18em] text-muted-foreground">Match</p>
                      <p className="mt-1 font-mono text-foreground">{formatPercent(lastResult?.matchConfidence)}</p>
                    </div>
                    <div className="rounded-xl bg-white/80 p-2">
                      <p className="uppercase tracking-[0.18em] text-muted-foreground">Direction</p>
                      <p className="mt-1 font-mono text-foreground">
                        {lastResult?.movementDirection ?? "--"} {lastResult ? `(${formatPercent(lastResult.movementConfidence)})` : ""}
                      </p>
                    </div>
                    <div className="rounded-xl bg-white/80 p-2">
                      <p className="uppercase tracking-[0.18em] text-muted-foreground">Verified At</p>
                      <p className="mt-1 font-mono text-foreground">{lastResult?.verifiedAt ?? "--"}</p>
                    </div>
                    <div className="rounded-xl bg-white/80 p-2">
                      <p className="uppercase tracking-[0.18em] text-muted-foreground">Source</p>
                      <p className="mt-1 font-mono text-foreground">{lastResult?.source ?? "--"}</p>
                    </div>
                  </div>
                </div>
              </div>

              <Alert className={cn(
                !lastResult
                  ? "border-slate-200 bg-white/75"
                  : lastResult.ignored
                    ? "border-amber-200 bg-amber-50"
                  : lastResult.success
                    ? "border-emerald-200 bg-emerald-50"
                    : "border-rose-200 bg-rose-50",
              )}>
                {lastResult ? (
                  lastResult.ignored ? (
                    <AlertCircle className="size-4 text-amber-700" />
                  ) : lastResult.success ? (
                    <CheckCircle2 className="size-4 text-emerald-600" />
                  ) : (
                    <AlertCircle className="size-4 text-rose-600" />
                  )
                ) : (
                  <ShieldCheck className="size-4 text-slate-500" />
                )}
                <AlertTitle>
                  {lastResult?.ignored
                    ? "Duplicate ignored"
                    : lastResult?.success
                      ? "Access granted"
                      : lastResult
                        ? "Access denied"
                        : "Ready for next employee"}
                </AlertTitle>
                {lastResult?.message && (
                  <AlertDescription>{lastResult.message}</AlertDescription>
                )}
              </Alert>

              {lastResult?.matchDetails && (
                <div className="rounded-2xl border border-border/60 bg-white/75 px-4 py-3 font-mono text-[11px] leading-6 text-muted-foreground">
                  Primary {formatPercent(lastResult.matchDetails.primaryConfidence)} | Anchor avg {formatPercent(lastResult.matchDetails.anchorAverage)} | Stable {formatPercent(lastResult.matchDetails.liveConsistency)}
                </div>
              )}

              {latestFaceMeta && (
                <div className="rounded-2xl border border-border/60 bg-white/70 px-4 py-3 text-sm">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-medium text-foreground">Roster profile</span>
                    <Badge
                      variant="outline"
                      className={cn(
                        latestFaceMeta.status === "trained"
                          ? "border-emerald-300 bg-emerald-100 text-emerald-700"
                          : latestFaceMeta.status === "training"
                            ? "border-sky-300 bg-sky-100 text-sky-700"
                            : "border-amber-300 bg-amber-100 text-amber-700",
                      )}
                    >
                      {latestFaceMeta.status}
                    </Badge>
                  </div>
                  <p className="mt-2 text-muted-foreground">
                    {latestFaceMeta.datasetSampleCount} dataset photos enrolled for this employee.
                  </p>
                  {latestFaceMeta.lastTrainingMessage && (
                    <p className="mt-1 text-xs text-muted-foreground">{latestFaceMeta.lastTrainingMessage}</p>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="border-border/60 shadow-sm">
            <CardContent className="space-y-3 p-4">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                  Manual Trigger
                </p>
                <h2 className="mt-1 text-xl font-semibold text-foreground">Badge + face burst</h2>
              </div>

              <form className="space-y-4" onSubmit={handleManualSubmit}>
                <div className="space-y-2">
                  <Label htmlFor="gate-scan-technology">RFID Technology</Label>
                  <Select value={scanTechnology} onValueChange={(value) => setScanTechnology(value as "HF_RFID" | "UHF_RFID")}>
                    <SelectTrigger id="gate-scan-technology">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="HF_RFID">HF RFID</SelectItem>
                      <SelectItem value="UHF_RFID">UHF RFID</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="gate-rfid" className="flex items-center gap-2">
                    <KeyRound className="size-4" />
                    RFID UID
                  </Label>
                  <Input
                    id="gate-rfid"
                    placeholder="Tap a card or type A1B2C3D4"
                    className="font-mono uppercase tracking-[0.2em]"
                    value={rfidUid}
                    onChange={(event) => setRfidUid(event.target.value.toUpperCase())}
                    disabled={busy}
                  />
                </div>

                {!!readerMessage && (
                  <div className="rounded-2xl border border-dashed border-border/70 bg-muted/20 px-4 py-3 text-sm text-muted-foreground">
                    <div className="flex items-start gap-2">
                      <ScanLine className="mt-0.5 size-4 shrink-0" />
                      <div className="space-y-1">
                        <p>{readerMessage}</p>
                        <p className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
                          Source: {readerSourceDeviceId ?? GATE_DEVICE_ID}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <Button
                  type="submit"
                  size="lg"
                  className="w-full"
                  disabled={busy || !normalizedRfidUid || !cameraActive}
                >
                  {isCapturingFrames ? (
                    <>
                      <Loader2 className="mr-2 size-4 animate-spin" />
                      Capturing Frames...
                    </>
                  ) : scanMutation.isPending ? (
                    <>
                      <Loader2 className="mr-2 size-4 animate-spin" />
                      Verifying in Python...
                    </>
                  ) : (
                    <>
                      <ArrowLeftRight className="mr-2 size-4" />
                      Scan Badge and Verify Face
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>

          {!cameraActive && cameraError && (
            <Alert className="border-amber-200 bg-amber-50">
              <AlertCircle className="size-4 text-amber-700" />
              <AlertTitle>Camera access is required</AlertTitle>
              <AlertDescription>{cameraError}</AlertDescription>
            </Alert>
          )}
        </div>
      </div>
    </div>
  );
}






