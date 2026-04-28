import { useEffect, useRef, useState, type ChangeEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useDeleteEmployee, useEmployees, usePythonEnrollEmployee, useUpdateEmployee } from "@/hooks/use-employees";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Camera,
  CheckCircle2,
  Database,
  Loader2,
  Plus,
  Pencil,
  RefreshCcw,
  ScanLine,
  ShieldCheck,
  Trash2,
  UserCircle,
  Volume2,
  VolumeX,
} from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { z } from "zod";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { insertEmployeeSchema, type Employee } from "@shared/schema";
import {
  describeFacePose,
  startFaceTracking,
  type FaceCropBounds,
  type FaceTrackingSnapshot,
} from "@/lib/biometrics";
import {
  connectRfidReader,
  fetchRfidRegistrationTag,
  fetchRfidTags,
  rfidQueryKeys,
  setRfidMode,
  startRfidReader,
  stopRfidReader,
} from "@/lib/rfid";

const REGISTRATION_PORT = "COM3";
const REGISTRATION_BAUDRATE = 57600;
const FIXED_DATASET_TARGET = 100;
const MIN_DATASET_SAMPLES = FIXED_DATASET_TARGET;
const DEFAULT_DATASET_SAMPLES = FIXED_DATASET_TARGET;
const MAX_DATASET_SAMPLES = FIXED_DATASET_TARGET;
const DATASET_CAPTURE_DELAY_MS = 180;
const DATASET_CAPTURE_SIZE = 360;
const DATASET_TRACKING_INTERVAL_MS = 140;
const DATASET_MIN_QUALITY = 0.62;
const DATASET_MIN_FACE_SCORE = 0.68;
const DATASET_MIN_BOX_SCORE = 0.67;
const DATASET_MIN_LIVE_CONFIDENCE = 0.42;
const DATASET_MIN_REAL_CONFIDENCE = 0.36;
const DATASET_REQUIRED_STABLE_HITS = 4;
const DATASET_MAX_ATTEMPTS_MULTIPLIER = 14;
const DATASET_MAX_ABS_ROLL = 18;
const VOICE_GUIDANCE_REPEAT_MS = 2600;

type DatasetPoseKey = "front" | "left" | "right" | "up" | "down";
type DatasetPoseCoverage = Record<DatasetPoseKey, number>;

const DATASET_POSE_SEQUENCE: DatasetPoseKey[] = ["front", "left", "right", "up", "down"];

const defaultFormValues = {
  employeeCode: "",
  name: "",
  department: "",
  phone: "",
  email: "",
  rfidUid: "",
  isActive: true,
};

const formSchema = insertEmployeeSchema
  .omit({ faceDescriptor: true })
  .extend({
    employeeCode: z.string().trim().min(1, "Employee code is required."),
    name: z.string().trim().min(1, "Employee name is required."),
    department: z.string().trim().min(1, "Department is required."),
    rfidUid: z.string().trim().min(1, "RFID badge is required."),
    phone: z.string().trim().optional(),
    email: z.string().trim().optional(),
  });

type FormValues = z.infer<typeof formSchema>;

function sleep(durationMs: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, durationMs);
  });
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

function getCameraConstraints(): MediaTrackConstraints {
  return {
    facingMode: "user",
    width: { ideal: 1280, min: 640 },
    height: { ideal: 720, min: 480 },
    frameRate: { ideal: 30, max: 30 },
  };
}

function createEmptyDatasetPoseCoverage(): DatasetPoseCoverage {
  return {
    front: 0,
    left: 0,
    right: 0,
    up: 0,
    down: 0,
  };
}

function buildDatasetPoseTargets(sampleCount: number): DatasetPoseCoverage {
  if (sampleCount === FIXED_DATASET_TARGET) {
    return {
      front: 30,
      left: 20,
      right: 20,
      up: 15,
      down: 15,
    };
  }

  const front = Math.max(10, Math.round(sampleCount * 0.3));
  const left = Math.max(6, Math.round(sampleCount * 0.2));
  const right = Math.max(6, Math.round(sampleCount * 0.2));
  const up = Math.max(4, Math.round(sampleCount * 0.15));
  const down = Math.max(4, sampleCount - front - left - right - up);

  return { front, left, right, up, down };
}

function isDatasetPoseKey(pose: string | null | undefined): pose is DatasetPoseKey {
  return pose === "front" || pose === "left" || pose === "right" || pose === "up" || pose === "down";
}

function getNextRequiredDatasetPose(
  captured: DatasetPoseCoverage,
  targets: DatasetPoseCoverage,
) {
  return DATASET_POSE_SEQUENCE.find((pose) => captured[pose] < targets[pose]) ?? null;
}

function summarizeMissingDatasetCoverage(
  captured: DatasetPoseCoverage,
  targets: DatasetPoseCoverage,
) {
  return DATASET_POSE_SEQUENCE
    .filter((pose) => captured[pose] < targets[pose])
    .map((pose) => `${describeFacePose(pose)} ${captured[pose]}/${targets[pose]}`)
    .join(", ");
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
  };
}

function getDatasetCaptureBlocker(
  snapshot: FaceTrackingSnapshot | null,
  stableHits: number,
) {
  if (!snapshot) {
    return "Starting the live face detector.";
  }

  if (snapshot.status === "unsupported") {
    return snapshot.guidance || "This browser cannot run the assisted enrollment detector.";
  }

  if (snapshot.faceCount > 1 || snapshot.status === "multiple") {
    return "Keep only one employee face in the camera.";
  }

  if (!snapshot.bounds || snapshot.faceCount === 0 || snapshot.status === "no-face" || snapshot.status === "loading") {
    return snapshot.guidance || "Step into the camera so the face detector can lock on.";
  }

  if (snapshot.status === "off-center" || snapshot.status === "low-quality") {
    return snapshot.guidance;
  }

  if (snapshot.quality < DATASET_MIN_QUALITY) {
    return "Face quality is still low. Improve lighting and keep the face sharper.";
  }

  if (snapshot.faceScore < DATASET_MIN_FACE_SCORE || snapshot.boxScore < DATASET_MIN_BOX_SCORE) {
    return "The detector does not trust this face strongly yet. Hold steady inside the live box.";
  }

  if (snapshot.liveConfidence < DATASET_MIN_LIVE_CONFIDENCE || snapshot.realConfidence < DATASET_MIN_REAL_CONFIDENCE) {
    return "Liveness confidence is weak. Avoid glare and keep a real face fully visible.";
  }

  if (Math.abs(snapshot.roll) > DATASET_MAX_ABS_ROLL) {
    return "Keep the head level and avoid tilting while the detector locks.";
  }

  if (stableHits < DATASET_REQUIRED_STABLE_HITS) {
    return `Hold steady for a stronger lock (${stableHits}/${DATASET_REQUIRED_STABLE_HITS}).`;
  }

  return null;
}

function buildVoiceGuidanceMessage(args: {
  cameraActive: boolean;
  trackingSnapshot: FaceTrackingSnapshot | null;
  blocker: string | null;
  nextRequiredPose: DatasetPoseKey | null;
  isCapturingDataset: boolean;
  datasetReady: boolean;
}) {
  if (!args.cameraActive) {
    return null;
  }

  if (args.datasetReady) {
    return "Dataset capture complete. Save the employee when ready.";
  }

  if (args.blocker) {
    if (args.trackingSnapshot?.status === "multiple" || (args.trackingSnapshot?.faceCount ?? 0) > 1) {
      return "Only one employee should stay in the frame.";
    }
    if (args.trackingSnapshot?.status === "no-face") {
      return "Please stand in front of the camera.";
    }
    if (args.trackingSnapshot?.status === "off-center") {
      return "Move your face inside the live frame.";
    }
    if (args.trackingSnapshot?.status === "low-quality") {
      return "Move closer and improve lighting.";
    }
    return args.blocker;
  }

  if (args.isCapturingDataset && args.nextRequiredPose) {
    return `Hold ${describeFacePose(args.nextRequiredPose)} pose.`;
  }

  if (args.nextRequiredPose) {
    return `Face lock ready. Hold ${describeFacePose(args.nextRequiredPose)} pose to begin.`;
  }

  return "Face lock ready.";
}

function captureDatasetFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  faceBounds: FaceCropBounds,
) {
  const context = canvas.getContext("2d");
  if (!context || !video.videoWidth || !video.videoHeight) {
    return null;
  }

  const faceCenterX = faceBounds.x + (faceBounds.width / 2);
  const faceCenterY = faceBounds.y + (faceBounds.height / 2) + (faceBounds.height * 0.08);
  const sourceSize = Math.min(
    Math.min(video.videoWidth, video.videoHeight),
    Math.max(faceBounds.width * 1.85, faceBounds.height * 1.6),
  );
  const sourceX = clamp(faceCenterX - (sourceSize / 2), 0, Math.max(0, video.videoWidth - sourceSize));
  const sourceY = clamp(faceCenterY - (sourceSize / 2), 0, Math.max(0, video.videoHeight - sourceSize));

  canvas.width = DATASET_CAPTURE_SIZE;
  canvas.height = DATASET_CAPTURE_SIZE;
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.clearRect(0, 0, DATASET_CAPTURE_SIZE, DATASET_CAPTURE_SIZE);
  context.drawImage(
    video,
    sourceX,
    sourceY,
    sourceSize,
    sourceSize,
    0,
    0,
    DATASET_CAPTURE_SIZE,
    DATASET_CAPTURE_SIZE,
  );

  return canvas.toDataURL("image/jpeg", 0.78);
}

function getPythonFaceStatus(faceDescriptor: unknown) {
  if (!faceDescriptor || typeof faceDescriptor !== "object") {
    return null;
  }

  const candidate = faceDescriptor as Record<string, unknown>;
  if (candidate.provider !== "python-opencv-lbph") {
    return null;
  }

  return {
    status: typeof candidate.status === "string" ? candidate.status : "failed",
    datasetSampleCount:
      typeof candidate.datasetSampleCount === "number" ? candidate.datasetSampleCount : 0,
    lastTrainingMessage:
      typeof candidate.lastTrainingMessage === "string" ? candidate.lastTrainingMessage : null,
  };
}

export default function Employees() {
  const queryClient = useQueryClient();
  const { data: employees, isLoading } = useEmployees();
  const deleteEmployee = useDeleteEmployee();
  const pythonEnrollEmployee = usePythonEnrollEmployee();
  const updateEmployee = useUpdateEmployee();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [editingEmployee, setEditingEmployee] = useState<Employee | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraRetryToken, setCameraRetryToken] = useState(0);
  const [datasetSamplesTarget, setDatasetSamplesTarget] = useState(DEFAULT_DATASET_SAMPLES);
  const [datasetPhotos, setDatasetPhotos] = useState<string[]>([]);
  const [profilePhoto, setProfilePhoto] = useState<string | null>(null);
  const [isCapturingDataset, setIsCapturingDataset] = useState(false);
  const [captureProgress, setCaptureProgress] = useState(0);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [rfidReaderMessage, setRfidReaderMessage] = useState<string | null>(null);
  const [rfidSourceDeviceId, setRfidSourceDeviceId] = useState<string | null>(null);
  const [registrationModeEnabled, setRegistrationModeEnabled] = useState(false);
  const [editProfilePreview, setEditProfilePreview] = useState<string | null>(null);
  const [editProfilePhoto, setEditProfilePhoto] = useState<string | null>(null);
  const [trackingSnapshot, setTrackingSnapshot] = useState<FaceTrackingSnapshot | null>(null);
  const [trackingStableHits, setTrackingStableHits] = useState(0);
  const [datasetPoseCoverage, setDatasetPoseCoverage] = useState<DatasetPoseCoverage>(createEmptyDatasetPoseCoverage());
  const [datasetRuntimeMessage, setDatasetRuntimeMessage] = useState<string | null>(null);
  const [lastAcceptedPose, setLastAcceptedPose] = useState<DatasetPoseKey | null>(null);
  const [voiceAssistantEnabled, setVoiceAssistantEnabled] = useState(true);
  const [saveTrainingElapsedMs, setSaveTrainingElapsedMs] = useState(0);
  const cameraViewportRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const autoRegistrationAttemptedRef = useRef(false);
  const registrationReaderStartedRef = useRef(false);
  const trackingSnapshotRef = useRef<FaceTrackingSnapshot | null>(null);
  const trackingStableHitsRef = useRef(0);
  const trackingAnchorRef = useRef<FaceCropBounds | null>(null);
  const lastVoiceMessageRef = useRef("");
  const lastVoiceAtRef = useRef(0);

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: defaultFormValues,
  });

  const editForm = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: defaultFormValues,
  });

  const watchedRfidUid = form.watch("rfidUid");
  const normalizedRfidUid = watchedRfidUid.trim().toUpperCase();
  const mappedBadgeOwner = employees?.find((employee) => {
    return employee.rfidUid.toUpperCase() === normalizedRfidUid;
  });
  const rfidReady = Boolean(normalizedRfidUid) && !mappedBadgeOwner;
  const datasetReady = datasetPhotos.length >= MIN_DATASET_SAMPLES;
  const datasetPoseTargets = buildDatasetPoseTargets(datasetSamplesTarget);
  const nextRequiredDatasetPose = getNextRequiredDatasetPose(datasetPoseCoverage, datasetPoseTargets);
  const datasetCaptureBlocker = getDatasetCaptureBlocker(trackingSnapshot, trackingStableHits);
  const saveTrainingStageIndex = saveTrainingElapsedMs >= 4800 ? 2 : saveTrainingElapsedMs >= 1600 ? 1 : 0;
  const saveTrainingSteps = [
    {
      title: "Saving employee",
      description: "Writing the employee record, RFID mapping, and profile settings.",
    },
    {
      title: "Writing dataset",
      description: `Storing ${datasetPhotos.length || FIXED_DATASET_TARGET} guided dataset images for Python training.`,
    },
    {
      title: "Refreshing model",
      description: "Rebuilding the Python face model so the new employee is production-ready.",
    },
  ] as const;
  const voiceGuidanceMessage = buildVoiceGuidanceMessage({
    cameraActive,
    trackingSnapshot,
    blocker: datasetCaptureBlocker,
    nextRequiredPose: nextRequiredDatasetPose,
    isCapturingDataset,
    datasetReady,
  });

  const readerStatusQuery = useQuery({
    queryKey: rfidQueryKeys.tags,
    queryFn: fetchRfidTags,
    enabled: isDialogOpen,
    refetchInterval: isDialogOpen ? 2000 : false,
  });

  const registrationTagQuery = useQuery({
    queryKey: rfidQueryKeys.registrationTag,
    queryFn: fetchRfidRegistrationTag,
    enabled: isDialogOpen && registrationModeEnabled,
    refetchInterval: isDialogOpen && registrationModeEnabled ? 1200 : false,
  });
  const registrationState = registrationTagQuery.data?.registration;
  const registrationPower = registrationTagQuery.data?.current_power ?? readerStatusQuery.data?.current_power ?? null;
  const registrationProgress = registrationState
    ? Math.min(100, (registrationState.candidate_hits / Math.max(1, registrationState.stable_threshold)) * 100)
    : 0;

  const enableRegistrationModeMutation = useMutation({
    mutationFn: async () => {
      const readerPort = readerStatusQuery.data?.port ?? REGISTRATION_PORT;
      const readerBaudrate = readerStatusQuery.data?.baudrate ?? REGISTRATION_BAUDRATE;
      registrationReaderStartedRef.current = !readerStatusQuery.data?.running;

      await connectRfidReader({
        port: readerPort,
        baudrate: readerBaudrate,
        debug_raw: false,
      });
      await setRfidMode("registration");
      return startRfidReader({
        port: readerPort,
        baudrate: readerBaudrate,
        debug_raw: false,
      });
    },
    onSuccess: async () => {
      setRegistrationModeEnabled(true);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: rfidQueryKeys.tags }),
        queryClient.invalidateQueries({ queryKey: rfidQueryKeys.registrationTag }),
      ]);
    },
    onError: (error) => {
      registrationReaderStartedRef.current = false;
      setRfidReaderMessage(
        error instanceof Error
          ? error.message
          : "Unable to enable UHF registration mode.",
      );
      setRfidSourceDeviceId("RFID Service");
    },
  });

  const enrollmentReaderOnline = Boolean(readerStatusQuery.data?.connected && readerStatusQuery.data?.running);

  useEffect(() => {
    if (!isDialogOpen) {
      return;
    }

    let stream: MediaStream | null = null;
    let cancelled = false;

    const initCamera = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraActive(false);
        setCameraError("This browser does not support camera enrollment.");
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
        console.error("Camera enrollment failed:", error);
        setCameraActive(false);
        setCameraError("Allow camera access to capture the employee dataset.");
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
  }, [cameraRetryToken, isDialogOpen]);

  useEffect(() => {
    if (!isDialogOpen || !cameraActive || !videoRef.current) {
      trackingSnapshotRef.current = null;
      trackingAnchorRef.current = null;
      trackingStableHitsRef.current = 0;
      setTrackingSnapshot(null);
      setTrackingStableHits(0);
      return;
    }

    const stopTracking = startFaceTracking(
      videoRef.current,
      (snapshot) => {
        trackingSnapshotRef.current = snapshot;
        setTrackingSnapshot(snapshot);

        if (snapshot.status === "ready" && snapshot.bounds) {
          const previousBounds = trackingAnchorRef.current;
          const currentCenterX = snapshot.bounds.x + (snapshot.bounds.width / 2);
          const currentCenterY = snapshot.bounds.y + (snapshot.bounds.height / 2);
          const previousCenterX = previousBounds ? previousBounds.x + (previousBounds.width / 2) : currentCenterX;
          const previousCenterY = previousBounds ? previousBounds.y + (previousBounds.height / 2) : currentCenterY;
          const stable =
            previousBounds
            && Math.abs(currentCenterX - previousCenterX) <= snapshot.bounds.width * 0.12
            && Math.abs(currentCenterY - previousCenterY) <= snapshot.bounds.height * 0.12
            && Math.abs(snapshot.bounds.width - previousBounds.width) <= snapshot.bounds.width * 0.18
            && Math.abs(snapshot.bounds.height - previousBounds.height) <= snapshot.bounds.height * 0.18;

          const nextStableHits = stable ? Math.min(trackingStableHitsRef.current + 1, 12) : 1;
          trackingAnchorRef.current = snapshot.bounds;
          trackingStableHitsRef.current = nextStableHits;
          setTrackingStableHits(nextStableHits);
          return;
        }

        trackingAnchorRef.current = null;
        trackingStableHitsRef.current = 0;
        setTrackingStableHits(0);
      },
      {
        intervalMs: DATASET_TRACKING_INTERVAL_MS,
        mode: "capture",
      },
    );

    return () => {
      stopTracking();
      trackingSnapshotRef.current = null;
      trackingAnchorRef.current = null;
      trackingStableHitsRef.current = 0;
      setTrackingStableHits(0);
    };
  }, [cameraActive, isDialogOpen]);

  useEffect(() => {
    if (!isDialogOpen || !voiceAssistantEnabled || !voiceGuidanceMessage) {
      return;
    }

    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }

    const now = Date.now();
    if (voiceGuidanceMessage === lastVoiceMessageRef.current) {
      return;
    }

    if ((now - lastVoiceAtRef.current) < VOICE_GUIDANCE_REPEAT_MS) {
      return;
    }

    const utterance = new SpeechSynthesisUtterance(voiceGuidanceMessage);
    utterance.rate = 1;
    utterance.pitch = 1;
    utterance.volume = 1;

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    lastVoiceMessageRef.current = voiceGuidanceMessage;
    lastVoiceAtRef.current = now;
  }, [isDialogOpen, voiceAssistantEnabled, voiceGuidanceMessage]);

  useEffect(() => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }

    return () => {
      window.speechSynthesis.cancel();
    };
  }, []);

  useEffect(() => {
    if (!pythonEnrollEmployee.isPending) {
      setSaveTrainingElapsedMs(0);
      return;
    }

    const startedAt = Date.now();
    setSaveTrainingElapsedMs(0);
    const timer = window.setInterval(() => {
      setSaveTrainingElapsedMs(Date.now() - startedAt);
    }, 120);

    return () => {
      window.clearInterval(timer);
    };
  }, [pythonEnrollEmployee.isPending]);

  useEffect(() => {
    if (!isDialogOpen) {
      return;
    }

    if (readerStatusQuery.data?.current_mode === "registration" && readerStatusQuery.data?.running) {
      setRegistrationModeEnabled(true);
    }
  }, [isDialogOpen, readerStatusQuery.data?.current_mode]);

  useEffect(() => {
    if (!isDialogOpen || autoRegistrationAttemptedRef.current || enableRegistrationModeMutation.isPending) {
      return;
    }

    if (readerStatusQuery.data?.current_mode === "registration" && readerStatusQuery.data?.running) {
      autoRegistrationAttemptedRef.current = true;
      setRegistrationModeEnabled(true);
      return;
    }

    autoRegistrationAttemptedRef.current = true;
    enableRegistrationModeMutation.mutate();
  }, [
    enableRegistrationModeMutation,
    enableRegistrationModeMutation.isPending,
    isDialogOpen,
    readerStatusQuery.data?.current_mode,
    readerStatusQuery.data?.running,
  ]);

  useEffect(() => {
    if (!isDialogOpen) {
      return;
    }

    const registration = registrationTagQuery.data?.registration;
    if (!registrationModeEnabled || !registration) {
      return;
    }

    setRfidReaderMessage(registration.message);
    setRfidSourceDeviceId(readerStatusQuery.data?.port ?? "RFID Service");

    if (registration.multiple_tags_detected) {
      form.setError("rfidUid", {
        type: "manual",
        message: "Multiple UHF tags detected. Keep only one tag near the reader.",
      });
      return;
    }

    if (registration.selected_tag) {
      form.setValue("rfidUid", registration.selected_tag, {
        shouldDirty: true,
        shouldTouch: true,
        shouldValidate: true,
      });
      form.clearErrors("rfidUid");
    }
  }, [form, isDialogOpen, readerStatusQuery.data?.port, registrationModeEnabled, registrationTagQuery.data?.registration]);

  useEffect(() => {
    if (!isDialogOpen) {
      return;
    }

    if (!normalizedRfidUid) {
      if (!registrationModeEnabled) {
        setRfidReaderMessage(null);
        setRfidSourceDeviceId(null);
      }
      return;
    }

    if (mappedBadgeOwner) {
      setRfidReaderMessage(`RFID badge already mapped to ${mappedBadgeOwner.name}.`);
      form.setError("rfidUid", {
        type: "manual",
        message: `RFID badge already mapped to ${mappedBadgeOwner.name}.`,
      });
      return;
    }

    form.clearErrors("rfidUid");
  }, [form, isDialogOpen, mappedBadgeOwner, normalizedRfidUid, registrationModeEnabled]);

  const resetEnrollment = () => {
    setDatasetPhotos([]);
    setProfilePhoto(null);
    setCaptureProgress(0);
    setDatasetError(null);
    setDatasetRuntimeMessage(null);
    setDatasetPoseCoverage(createEmptyDatasetPoseCoverage());
    setLastAcceptedPose(null);
    setDatasetSamplesTarget(DEFAULT_DATASET_SAMPLES);
    setRfidReaderMessage(null);
    setRfidSourceDeviceId(null);
    setRegistrationModeEnabled(false);
    setCameraError(null);
    setIsCapturingDataset(false);
    lastVoiceMessageRef.current = "";
    lastVoiceAtRef.current = 0;
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    form.reset(defaultFormValues);
  };

  const handleDialogChange = (open: boolean) => {
    if (!open && pythonEnrollEmployee.isPending) {
      return;
    }

    setIsDialogOpen(open);

    if (open) {
      autoRegistrationAttemptedRef.current = false;
      return;
    }

    if (!open) {
      if (registrationModeEnabled) {
        void setRfidMode("normal").catch(() => undefined);
        if (registrationReaderStartedRef.current) {
          void stopRfidReader().catch(() => undefined);
        }
      }
      registrationReaderStartedRef.current = false;
      resetEnrollment();
    }
  };

  const handleCaptureDataset = async () => {
    if (!cameraActive || !videoRef.current || !canvasRef.current) {
      setDatasetError("Camera is not ready. Allow access, then retry capture.");
      return;
    }

    if (datasetCaptureBlocker) {
      setDatasetError(datasetCaptureBlocker);
      return;
    }

    setIsCapturingDataset(true);
    setCaptureProgress(0);
    setDatasetError(null);
    setDatasetRuntimeMessage("Locking the live face and capturing pose-balanced samples.");
    setDatasetPhotos([]);
    setDatasetPoseCoverage(createEmptyDatasetPoseCoverage());
    setLastAcceptedPose(null);

    try {
      const capturedPhotos: string[] = [];
      const capturedCoverage = createEmptyDatasetPoseCoverage();
      const maxAttempts = datasetSamplesTarget * DATASET_MAX_ATTEMPTS_MULTIPLIER;

      for (let attempt = 1; attempt <= maxAttempts && capturedPhotos.length < datasetSamplesTarget; attempt += 1) {
        const snapshot = trackingSnapshotRef.current;
        const stableHits = trackingStableHitsRef.current;
        const captureBlocker = getDatasetCaptureBlocker(snapshot, stableHits);

        if (captureBlocker) {
          setDatasetRuntimeMessage(captureBlocker);
          await sleep(DATASET_CAPTURE_DELAY_MS);
          continue;
        }

        if (!snapshot?.bounds) {
          setDatasetRuntimeMessage("Waiting for the detector to produce a valid face box.");
          await sleep(DATASET_CAPTURE_DELAY_MS);
          continue;
        }

        const requiredPose = getNextRequiredDatasetPose(capturedCoverage, datasetPoseTargets);
        const detectedPose = isDatasetPoseKey(snapshot.pose) ? snapshot.pose : null;

        if (requiredPose && detectedPose !== requiredPose) {
          setDatasetRuntimeMessage(`Hold ${describeFacePose(requiredPose)} pose inside the live face box.`);
          await sleep(DATASET_CAPTURE_DELAY_MS);
          continue;
        }

        const frame = captureDatasetFrame(videoRef.current, canvasRef.current, snapshot.bounds);
        if (!frame) {
          throw new Error("Dataset frame capture failed. Retry with one face held inside the live detection box.");
        }

        capturedPhotos.push(frame);
        if (detectedPose) {
          capturedCoverage[detectedPose] += 1;
          setLastAcceptedPose(detectedPose);
        }

        setDatasetPhotos([...capturedPhotos]);
        setDatasetPoseCoverage({ ...capturedCoverage });
        setCaptureProgress(capturedPhotos.length);
        setDatasetRuntimeMessage(
          `${detectedPose ? describeFacePose(detectedPose) : "Face"} sample accepted (${capturedPhotos.length}/${datasetSamplesTarget}).`,
        );
        await sleep(DATASET_CAPTURE_DELAY_MS);
      }

      if (capturedPhotos.length < datasetSamplesTarget) {
        const missingCoverage = summarizeMissingDatasetCoverage(capturedCoverage, datasetPoseTargets);
        throw new Error(
          missingCoverage
            ? `Dataset capture stopped before meeting the training plan. Missing: ${missingCoverage}.`
            : "Dataset capture stopped before meeting the training plan. Keep one well-lit face inside the live detection box and retry.",
        );
      }

      setDatasetRuntimeMessage("Dataset capture complete. Review the sample count and save the employee.");
    } catch (error) {
      setDatasetError(
        error instanceof Error
          ? error.message
          : "Unable to capture the dataset photos.",
      );
    } finally {
      setIsCapturingDataset(false);
    }
  };

  const handleClearDataset = () => {
    setDatasetPhotos([]);
    setCaptureProgress(0);
    setDatasetError(null);
    setDatasetRuntimeMessage(null);
    setDatasetPoseCoverage(createEmptyDatasetPoseCoverage());
    setLastAcceptedPose(null);
  };

  const handleProfilePhotoChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      setProfilePhoto(null);
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result === "string") {
        setProfilePhoto(result);
      }
    };
    reader.readAsDataURL(file);
  };

  const onSubmit = (values: FormValues) => {
    if (!datasetReady) {
      setDatasetError(`Capture at least ${MIN_DATASET_SAMPLES} dataset photos before saving.`);
      return;
    }

    pythonEnrollEmployee.mutate({
      ...values,
      rfidUid: values.rfidUid.trim().toUpperCase(),
      isActive: values.isActive ?? true,
      datasetPhotos,
      profilePhoto: profilePhoto ?? undefined,
    }, {
      onSuccess: () => {
        setIsDialogOpen(false);
        resetEnrollment();
      },
    });
  };

  const handleDeleteEmployee = (employeeId: number, employeeName: string) => {
    const shouldDelete = window.confirm(
      `Delete ${employeeName} and all related attendance logs? This will also remove the Python dataset for that employee.`,
    );

    if (!shouldDelete) {
      return;
    }

    deleteEmployee.mutate(employeeId);
  };

  const handleEditDialogChange = (open: boolean) => {
    setIsEditDialogOpen(open);
    if (!open) {
      setEditingEmployee(null);
      editForm.reset(defaultFormValues);
      setEditProfilePreview(null);
      setEditProfilePhoto(null);
    }
  };

  const handleEditEmployee = (employee: Employee) => {
    setEditingEmployee(employee);
    editForm.reset({
      employeeCode: employee.employeeCode,
      name: employee.name,
      department: employee.department,
      phone: employee.phone ?? "",
      email: employee.email ?? "",
      rfidUid: employee.rfidUid,
      isActive: employee.isActive,
    });
    setIsEditDialogOpen(true);
    setEditProfilePhoto(null);
    setEditProfilePreview(null);
    void (async () => {
      try {
        const metaRes = await fetch(`/api/employees/${employee.id}/photo/meta`, { credentials: "include" });
        const meta = await metaRes.json();
        if (meta?.hasProfilePhoto) {
          setEditProfilePreview(`/api/employees/${employee.id}/photo?t=${Date.now()}`);
        }
      } catch {
        // ignore preview fetch failures
      }
    })();
  };

  const handleEditSubmit = (values: FormValues) => {
    if (!editingEmployee) {
      return;
    }

    updateEmployee.mutate(
      {
        id: editingEmployee.id,
        data: {
          ...values,
          rfidUid: values.rfidUid.trim().toUpperCase(),
          ...(editProfilePhoto ? { profilePhoto: editProfilePhoto } : {}),
        },
      },
      {
        onSuccess: () => {
          setIsEditDialogOpen(false);
          setEditingEmployee(null);
          setEditProfilePhoto(null);
          setEditProfilePreview(null);
        },
      },
    );
  };

  const enrollmentFaceOverlay = trackingSnapshot?.bounds
    && videoRef.current
    && cameraViewportRef.current
    ? mapRectToViewport(
      trackingSnapshot.bounds.x,
      trackingSnapshot.bounds.y,
      trackingSnapshot.bounds.width,
      trackingSnapshot.bounds.height,
      videoRef.current.videoWidth,
      videoRef.current.videoHeight,
      cameraViewportRef.current,
    )
    : null;

  const enrollmentCaptureGuidance = datasetRuntimeMessage
    ?? datasetCaptureBlocker
    ?? (nextRequiredDatasetPose
      ? `Hold ${describeFacePose(nextRequiredDatasetPose)} pose inside the live face box.`
      : "Face lock is ready. Start dataset capture.");

  return (
    <div className="space-y-6 p-6 md:p-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Directory</h1>
          <p className="mt-1 text-muted-foreground">
            Register employees, assign cards, and capture Python training datasets.
          </p>
        </div>

        <Dialog open={isDialogOpen} onOpenChange={handleDialogChange}>
          <DialogTrigger asChild>
            <Button className="shadow-sm hover:-translate-y-0.5 transition-transform">
              <Plus className="mr-2 size-4" /> Add Employee
            </Button>
          </DialogTrigger>
          <DialogContent className="fixed inset-0 h-[100dvh] w-screen max-w-none translate-x-0 translate-y-0 gap-0 overflow-hidden rounded-none border-0 p-0 data-[state=closed]:slide-out-to-left-0 data-[state=closed]:slide-out-to-top-0 data-[state=closed]:zoom-out-100 data-[state=open]:slide-in-from-left-0 data-[state=open]:slide-in-from-top-0 data-[state=open]:zoom-in-100 sm:rounded-none">
            <DialogHeader className="shrink-0 border-b border-border/70 px-4 py-2.5 text-left sm:px-5">
              <DialogTitle className="text-base sm:text-lg">Register New Employee</DialogTitle>
              <DialogDescription className="text-xs leading-snug sm:text-sm">
                Fill the employee details, register the RFID badge, and capture a dataset for Python training.
              </DialogDescription>
            </DialogHeader>

            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="flex h-full min-h-0 flex-col overflow-hidden">
                <div className="flex-1 overflow-hidden px-4 py-2.5 sm:px-5">
                  <div className="grid h-full min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-2.5">
                    <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
                  <FormField
                    control={form.control}
                    name="name"
                    render={({ field }) => (
                      <FormItem className="space-y-1">
                        <FormLabel>Full Name</FormLabel>
                        <FormControl>
                          <Input className="h-9" placeholder="Jane Doe" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="employeeCode"
                    render={({ field }) => (
                      <FormItem className="space-y-1">
                        <FormLabel>Employee Code</FormLabel>
                        <FormControl>
                          <Input className="h-9" placeholder="EMP-1042" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="department"
                    render={({ field }) => (
                      <FormItem className="space-y-1">
                        <FormLabel>Department</FormLabel>
                        <FormControl>
                          <Input className="h-9" placeholder="Operations" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="email"
                    render={({ field }) => (
                      <FormItem className="space-y-1">
                        <FormLabel>Email (Optional)</FormLabel>
                        <FormControl>
                          <Input className="h-9" placeholder="jane@company.com" {...field} value={field.value || ""} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="grid min-h-0 gap-2 border-t pt-2.5">
                  <div className="space-y-0">
                    <h4 className="text-sm font-semibold tracking-wide">Access Credentials</h4>
                    <p className="text-[12px] leading-snug text-muted-foreground">
                      Register the card and capture many dataset images so Python can train the face model well.
                    </p>
                  </div>

                  <div className="grid min-h-0 gap-2.5 xl:grid-cols-[340px_minmax(0,1fr)]">
                    <div className="space-y-2">
                      <FormField
                        control={form.control}
                        name="rfidUid"
                        render={({ field }) => (
                          <FormItem className="space-y-1">
                            <div className="flex items-center justify-between gap-3">
                              <FormLabel>RFID UID</FormLabel>
                              <Badge
                                variant={enrollmentReaderOnline ? "secondary" : "outline"}
                                className={cn(
                                  enrollmentReaderOnline
                                    ? "bg-emerald-100 text-emerald-800 border-emerald-200"
                                    : "border-slate-300 text-slate-600",
                                )}
                              >
                                {enrollmentReaderOnline ? "Reader Online" : "Reader Offline"}
                              </Badge>
                            </div>
                            <FormControl>
                              <Input
                                placeholder="Present one UHF tag or type the EPC..."
                                className="h-9 font-mono uppercase tracking-[0.14em]"
                                {...field}
                                onChange={(event) => {
                                  field.onChange(event.target.value.toUpperCase());
                                }}
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <div className="flex flex-wrap items-center gap-1.5">
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => enableRegistrationModeMutation.mutate()}
                          disabled={enableRegistrationModeMutation.isPending}
                        >
                          {enableRegistrationModeMutation.isPending ? (
                            <>
                              <Loader2 className="mr-2 size-4 animate-spin" />
                              Starting Registration Mode
                            </>
                          ) : (
                            <>
                              <ScanLine className="mr-2 size-4" />
                              Retry Registration Mode
                            </>
                          )}
                        </Button>
                        <Badge
                          variant={registrationModeEnabled ? "secondary" : "outline"}
                          className={cn(
                            registrationModeEnabled
                              ? "bg-sky-100 text-sky-800 border-sky-200"
                              : "border-slate-300 text-slate-600",
                          )}
                        >
                          {registrationModeEnabled ? "Registration Active" : "Registration Inactive"}
                        </Badge>
                      </div>

                      <div className="rounded-xl border border-dashed border-border/70 bg-muted/20 px-2.5 py-1.5 text-sm">
                        <div className="flex items-start gap-2 text-muted-foreground">
                          <ShieldCheck className="mt-0.5 size-4 shrink-0" />
                          <div className="space-y-1">
                            <p>
                              {rfidReaderMessage ?? "Registration mode starts automatically. Keep one UHF tag very close to the reader or type the EPC manually."}
                            </p>
                            <p className="text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
                              Source: {rfidSourceDeviceId ?? "RFID Service"}
                            </p>
                          </div>
                        </div>
                      </div>

                      {registrationModeEnabled && registrationState && (
                        <div className="rounded-xl border border-primary/20 bg-primary/5 px-2.5 py-1.5">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-primary/80">
                                Registration Lock
                              </p>
                              <p className="mt-1 text-sm text-foreground">
                                Close-range single-badge enrollment only.
                              </p>
                            </div>
                            <div className="flex flex-wrap items-center justify-end gap-2">
                              <Badge variant={registrationState.selected_tag ? "secondary" : "outline"}>
                                {registrationState.multiple_tags_detected
                                  ? "Multiple Tags"
                                  : registrationState.selected_tag
                                    ? "Locked"
                                    : "Stabilizing"}
                              </Badge>
                              {registrationPower !== null && (
                                <Badge variant="outline">Power {registrationPower}</Badge>
                              )}
                            </div>
                          </div>
                          <div className="mt-2.5 flex items-center justify-between text-[11px] text-muted-foreground">
                            <span>
                              Stability {registrationState.candidate_hits}/{registrationState.stable_threshold}
                            </span>
                            <span>Hold one tag very close</span>
                          </div>
                          <Progress className="mt-2 h-2" value={registrationProgress} />
                        </div>
                      )}

                      <Button
                        type="button"
                        variant="outline"
                        className="h-10 w-full justify-between rounded-xl border-dashed border-border/70 bg-muted/20 px-3"
                        onClick={() => {
                          if (voiceAssistantEnabled && typeof window !== "undefined" && "speechSynthesis" in window) {
                            window.speechSynthesis.cancel();
                          }

                          if (!voiceAssistantEnabled) {
                            lastVoiceMessageRef.current = "";
                            lastVoiceAtRef.current = 0;
                          }

                          setVoiceAssistantEnabled((enabled) => !enabled);
                        }}
                      >
                        <span className="flex items-center gap-2 text-sm">
                          {voiceAssistantEnabled ? (
                            <Volume2 className="size-4 text-primary" />
                          ) : (
                            <VolumeX className="size-4 text-muted-foreground" />
                          )}
                          {voiceAssistantEnabled ? "Voice Guidance On" : "Voice Guidance Off"}
                        </span>
                        <Badge variant="outline" className="ml-3">
                          {voiceAssistantEnabled ? "Mute" : "Enable"}
                        </Badge>
                      </Button>

                    </div>

                    <div className="grid min-h-0 gap-2 xl:grid-cols-[minmax(0,1fr)_260px] xl:grid-rows-[auto_auto_auto] xl:items-start">
                      <div className="relative min-h-0 overflow-hidden rounded-[1.25rem] border border-border/70 bg-black xl:row-span-3">
                        <div ref={cameraViewportRef} className="relative aspect-[16/9] overflow-hidden xl:h-[396px] xl:aspect-auto 2xl:h-[440px]">
                          <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className={`h-full w-full object-cover transition-opacity duration-300 ${
                              cameraActive ? "opacity-100" : "opacity-0"
                            }`}
                          />
                          {!cameraActive && (
                            <div className="absolute inset-0 flex items-center justify-center bg-slate-950 text-center text-sm text-white/80">
                              <div className="space-y-2 px-6">
                                <Camera className="mx-auto size-7 text-white/70" />
                                <p>{cameraError ?? "Waiting for camera preview..."}</p>
                              </div>
                            </div>
                          )}
                          <canvas ref={canvasRef} className="hidden" />
                          <div className="pointer-events-none absolute inset-0">
                            <div className="absolute inset-3.5 rounded-[1.15rem] border border-white/20" />
                            {enrollmentFaceOverlay && (
                              <div
                                className={cn(
                                  "absolute rounded-[1.2rem] border-[3px] transition-all duration-150",
                                  datasetCaptureBlocker
                                    ? "border-amber-300 shadow-[0_0_0_1px_rgba(253,224,71,0.35),0_0_18px_rgba(245,158,11,0.16)]"
                                    : "border-emerald-300 shadow-[0_0_0_1px_rgba(110,231,183,0.35),0_0_18px_rgba(16,185,129,0.16)]",
                                )}
                                style={{
                                  left: `${enrollmentFaceOverlay.leftPct}%`,
                                  top: `${enrollmentFaceOverlay.topPct}%`,
                                  width: `${enrollmentFaceOverlay.widthPct}%`,
                                  height: `${enrollmentFaceOverlay.heightPct}%`,
                                }}
                              >
                                <div className="absolute left-0 top-0 max-w-[calc(100%+8rem)] -translate-y-[calc(100%+0.35rem)] rounded-full bg-black/70 px-2.5 py-1 text-[10px] font-semibold tracking-[0.16em] text-white/90 shadow-lg">
                                  {trackingSnapshot && isDatasetPoseKey(trackingSnapshot.pose)
                                    ? `${describeFacePose(trackingSnapshot.pose)} Q${Math.round(trackingSnapshot.quality * 100)}`
                                    : "FACE LOCK"}
                                </div>
                              </div>
                            )}
                            <div className="absolute left-1/2 top-2.5 -translate-x-1/2 rounded-full bg-black/65 px-2.5 py-1 text-[10px] font-semibold tracking-[0.22em] text-white/90">
                              PYTHON DATASET
                            </div>
                            <div className="absolute bottom-2.5 left-2.5 right-2.5 rounded-2xl bg-black/60 px-3 py-2.5 text-[11px] text-white/90 shadow-xl backdrop-blur-sm">
                              <div className="flex items-center justify-between gap-3">
                                <span className="font-semibold tracking-[0.12em] text-white/95">
                                  {datasetCaptureBlocker ? "Awaiting Face Lock" : "Face Lock Ready"}
                                </span>
                                <span className="rounded-full bg-white/10 px-2.5 py-1 text-[10px] font-semibold tracking-[0.16em]">
                                  Stable {trackingStableHits}/{DATASET_REQUIRED_STABLE_HITS}
                                </span>
                              </div>
                              <p className="mt-1.5 leading-snug text-white/80">{enrollmentCaptureGuidance}</p>
                            </div>
                          </div>
                          {isCapturingDataset && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/40 backdrop-blur-[2px]">
                              <div className="rounded-2xl bg-black/80 px-4 py-2.5 text-center text-sm font-medium text-white shadow-xl backdrop-blur-md">
                                Capturing sample {captureProgress} / {datasetSamplesTarget}
                              </div>
                              <div className="rounded-2xl bg-primary/95 px-4 py-2.5 text-center text-base font-bold text-primary-foreground shadow-2xl animate-in zoom-in duration-300">
                                {nextRequiredDatasetPose
                                  ? `Hold ${describeFacePose(nextRequiredDatasetPose)} pose`
                                  : "Coverage complete"}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="space-y-1 rounded-xl border border-dashed border-border/70 bg-muted/20 p-2 xl:col-start-2">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Profile photo (optional)</p>
                            <p className="text-sm text-foreground">Add a cover photo for badges and dashboards. If empty, we’ll use a dataset sample.</p>
                          </div>
                          {profilePhoto && (
                            <img
                              src={profilePhoto}
                              alt="Profile preview"
                              className="h-9 w-9 rounded-xl object-cover border border-border/70 shadow-sm"
                            />
                          )}
                        </div>
                        <Input className="h-8 text-xs" type="file" accept="image/*" onChange={handleProfilePhotoChange} />
                      </div>

                      <div className="space-y-1.5 rounded-xl bg-muted/30 p-2 xl:col-start-2">
                        <div className="grid grid-cols-2 gap-1">
                          <div className="rounded-lg bg-background/60 px-2 py-1">
                            <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Captured</p>
                            <p className="text-sm font-semibold text-foreground">{datasetPhotos.length}</p>
                          </div>
                          <div className="rounded-lg bg-background/60 px-2 py-1">
                            <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Minimum</p>
                            <p className="text-sm font-semibold text-foreground">{MIN_DATASET_SAMPLES}</p>
                          </div>
                          <div className="rounded-lg bg-background/60 px-2 py-1">
                            <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Training</p>
                            <p className="text-sm font-semibold text-foreground">{datasetReady ? "Ready" : "Pending"}</p>
                          </div>
                          <div className="rounded-lg bg-background/60 px-2 py-1">
                            <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Face Quality</p>
                            <p className="text-sm font-semibold text-foreground">
                              {trackingSnapshot ? `${Math.round(trackingSnapshot.quality * 100)}%` : "--"}
                            </p>
                          </div>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-background/70">
                          <div
                            className="h-full rounded-full bg-primary transition-all duration-300"
                            style={{ width: `${(datasetPhotos.length / Math.max(1, datasetSamplesTarget)) * 100}%` }}
                          />
                        </div>
                        <div className="rounded-lg border border-dashed border-border/70 bg-background/40 px-2.5 py-2">
                          <div className="flex items-center justify-between gap-3 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                            <span>Pose Plan</span>
                            <span>{nextRequiredDatasetPose ? `Next ${describeFacePose(nextRequiredDatasetPose)}` : "Complete"}</span>
                          </div>
                          <div className="mt-1.5 grid grid-cols-5 gap-1">
                            {DATASET_POSE_SEQUENCE.map((pose) => (
                              <div key={pose} className="rounded-md bg-background/70 px-1.5 py-1 text-center">
                                <p className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">{describeFacePose(pose)}</p>
                                <p className="text-xs font-semibold text-foreground">
                                  {datasetPoseCoverage[pose]}/{datasetPoseTargets[pose]}
                                </p>
                              </div>
                            ))}
                          </div>
                          <div className="mt-2 flex items-center justify-between gap-3 text-xs text-muted-foreground">
                            <span>Current pose: {trackingSnapshot ? describeFacePose(trackingSnapshot.pose) : "--"}</span>
                            <span>Last accepted: {lastAcceptedPose ? describeFacePose(lastAcceptedPose) : "--"}</span>
                          </div>
                        </div>
                      </div>

                      <div className="grid gap-1.5 sm:grid-cols-2 xl:col-start-2 xl:grid-cols-1">
                        <Button
                          type="button"
                          className="h-9 justify-center px-3 text-sm"
                          onClick={handleCaptureDataset}
                          disabled={!cameraActive || isCapturingDataset}
                        >
                          {isCapturingDataset ? (
                            <>
                              <Loader2 className="mr-2 size-4 animate-spin" />
                              Capturing Dataset...
                            </>
                          ) : datasetReady ? (
                            <>
                              <Database className="mr-2 size-4" />
                              Re-Capture Dataset
                            </>
                          ) : (
                            <>
                              <ScanLine className="mr-2 size-4" />
                              Capture Dataset
                            </>
                          )}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          className="h-9 justify-center px-3 text-sm"
                          onClick={datasetPhotos.length ? handleClearDataset : () => setCameraRetryToken((value) => value + 1)}
                          disabled={isCapturingDataset}
                        >
                          {datasetPhotos.length ? (
                            <>
                              <RefreshCcw className="mr-2 size-4" />
                              Clear Dataset
                            </>
                          ) : (
                            <>
                              <RefreshCcw className="mr-2 size-4" />
                              Retry Camera
                            </>
                          )}
                        </Button>
                      </div>

                      {datasetError && (
                        <p className="text-xs font-medium text-destructive xl:col-start-2">{datasetError}</p>
                      )}
                    </div>
                  </div>
                </div>

                  </div>
                </div>

                <DialogFooter className="shrink-0 border-t border-border/70 bg-background/95 px-4 py-2 sm:px-5">
                  <Button className="h-10 px-4" type="button" variant="ghost" onClick={() => handleDialogChange(false)}>
                    Cancel
                  </Button>
                  <Button
                    className="h-10 px-4"
                    type="submit"
                    disabled={pythonEnrollEmployee.isPending || isCapturingDataset || !datasetReady || !rfidReady}
                  >
                    Save Employee
                  </Button>
                </DialogFooter>
              </form>
            </Form>

            {pythonEnrollEmployee.isPending && (
              <div className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/82 backdrop-blur-md">
                <div className="mx-4 w-full max-w-2xl rounded-[2rem] border border-white/10 bg-slate-950/92 p-6 shadow-[0_24px_80px_rgba(0,0,0,0.45)]">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-primary/80">
                        Saving And Training
                      </p>
                      <h3 className="mt-2 text-2xl font-semibold text-white">
                        Building the employee profile for production use
                      </h3>
                      <p className="mt-2 max-w-xl text-sm leading-relaxed text-white/70">
                        Please wait while we save the employee, write the 100-sample dataset, and refresh the Python face model.
                      </p>
                    </div>
                    <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-primary/25 bg-primary/10 text-primary">
                      <Loader2 className="size-7 animate-spin" />
                    </div>
                  </div>

                  <div className="mt-6 h-2 overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{ width: `${Math.min(92, 26 + (saveTrainingStageIndex * 28) + ((saveTrainingElapsedMs % 1400) / 1400) * 18)}%` }}
                    />
                  </div>

                  <div className="mt-6 grid gap-3">
                    {saveTrainingSteps.map((step, index) => {
                      const state = index < saveTrainingStageIndex ? "warm" : index === saveTrainingStageIndex ? "active" : "queued";
                      return (
                        <div
                          key={step.title}
                          className={cn(
                            "rounded-2xl border px-4 py-3 transition-colors",
                            state === "active"
                              ? "border-primary/35 bg-primary/10"
                              : state === "warm"
                                ? "border-emerald-400/20 bg-emerald-400/8"
                                : "border-white/10 bg-white/5",
                          )}
                        >
                          <div className="flex items-start gap-3">
                            <div
                              className={cn(
                                "mt-0.5 flex h-9 w-9 items-center justify-center rounded-xl border",
                                state === "active"
                                  ? "border-primary/35 bg-primary/15 text-primary"
                                  : state === "warm"
                                    ? "border-emerald-400/25 bg-emerald-400/10 text-emerald-300"
                                    : "border-white/10 bg-white/5 text-white/60",
                              )}
                            >
                              {state === "active" ? (
                                <Loader2 className="size-4 animate-spin" />
                              ) : state === "warm" ? (
                                <CheckCircle2 className="size-4" />
                              ) : (
                                <Database className="size-4" />
                              )}
                            </div>
                            <div className="min-w-0">
                              <div className="flex items-center gap-2">
                                <p className="font-medium text-white">{step.title}</p>
                                <span
                                  className={cn(
                                    "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em]",
                                    state === "active"
                                      ? "bg-primary/15 text-primary"
                                      : state === "warm"
                                        ? "bg-emerald-400/12 text-emerald-300"
                                        : "bg-white/8 text-white/55",
                                  )}
                                >
                                  {state === "active" ? "In Progress" : state === "warm" ? "Pipeline" : "Queued"}
                                </span>
                              </div>
                              <p className="mt-1 text-sm text-white/65">{step.description}</p>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="mt-5 flex items-center justify-between gap-4 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white/65">
                    <span>Do not close this dialog while the Python model is refreshing.</span>
                    <span className="font-medium text-white/80">This can take a few seconds.</span>
                  </div>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>

      <Dialog open={isEditDialogOpen} onOpenChange={handleEditDialogChange}>
        <DialogContent className="sm:max-w-[540px]">
          <DialogHeader>
            <DialogTitle>Edit Employee</DialogTitle>
            <DialogDescription>Update badge details without re-capturing the dataset.</DialogDescription>
          </DialogHeader>
          <Form {...editForm}>
            <form onSubmit={editForm.handleSubmit(handleEditSubmit)} className="space-y-4">
              <div className="flex items-center gap-4 rounded-lg border border-border/70 bg-muted/30 p-3">
                <div className="h-16 w-16 overflow-hidden rounded-full border border-border/70 bg-background shadow-sm">
                  {editProfilePreview ? (
                    <img src={editProfilePreview} alt="Profile" className="h-full w-full object-cover" />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-xs text-muted-foreground">No photo</div>
                  )}
                </div>
                <div className="flex flex-col gap-2">
                  <Input
                    type="file"
                    accept="image/*"
                    onChange={(event) => {
                      const file = event.target.files?.[0];
                      if (!file) {
                        return;
                      }
                      const reader = new FileReader();
                      reader.onload = () => {
                        const result = reader.result;
                        if (typeof result === "string") {
                          setEditProfilePhoto(result);
                          setEditProfilePreview(result);
                        }
                      };
                      reader.readAsDataURL(file);
                    }}
                  />
                  <div className="flex gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setEditProfilePhoto(null);
                        if (editingEmployee) {
                          setEditProfilePreview(`/api/employees/${editingEmployee.id}/photo?t=${Date.now()}`);
                        } else {
                          setEditProfilePreview(null);
                        }
                      }}
                    >
                      Reset
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        setEditProfilePhoto(null);
                        setEditProfilePreview(null);
                      }}
                    >
                      Clear
                    </Button>
                  </div>
                  <p className="text-[11px] text-muted-foreground">Profile photo shows on badges and dashboards.</p>
                </div>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <FormField
                  control={editForm.control}
                  name="name"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Name</FormLabel>
                      <FormControl>
                        <Input placeholder="Employee name" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={editForm.control}
                  name="employeeCode"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Employee Code</FormLabel>
                      <FormControl>
                        <Input placeholder="EMP001" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <FormField
                  control={editForm.control}
                  name="department"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Department</FormLabel>
                      <FormControl>
                        <Input placeholder="Department" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={editForm.control}
                  name="rfidUid"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>RFID Badge</FormLabel>
                      <FormControl>
                        <Input placeholder="A2BE752A" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <FormField
                  control={editForm.control}
                  name="phone"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Phone</FormLabel>
                      <FormControl>
                        <Input placeholder="Phone (optional)" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={editForm.control}
                  name="email"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Email</FormLabel>
                      <FormControl>
                        <Input placeholder="Email (optional)" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <FormField
                control={editForm.control}
                name="isActive"
                render={({ field }) => (
                  <FormItem className="flex items-center justify-between rounded-lg border border-border/60 px-4 py-3">
                    <div>
                      <FormLabel className="text-sm">Active status</FormLabel>
                      <p className="text-xs text-muted-foreground">Controls whether this profile is usable at the gate.</p>
                    </div>
                    <FormControl>
                      <Switch checked={Boolean(field.value)} onCheckedChange={field.onChange} />
                    </FormControl>
                  </FormItem>
                )}
              />

              <DialogFooter>
                <Button type="button" variant="ghost" onClick={() => handleEditDialogChange(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={updateEmployee.isPending || !editingEmployee}>
                  {updateEmployee.isPending ? "Saving..." : "Save Changes"}
                </Button>
              </DialogFooter>
            </form>
          </Form>
        </DialogContent>
      </Dialog>

      <Card className="border-border/50 shadow-sm overflow-hidden">
        <CardContent className="p-0">
          <Table>
            <TableHeader className="bg-muted/50">
              <TableRow>
                <TableHead className="pl-6">Employee</TableHead>
                <TableHead>Code</TableHead>
                <TableHead>Department</TableHead>
                <TableHead>RFID Badge</TableHead>
                <TableHead>Python Face</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="pr-6 text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array.from({ length: 5 }).map((_, index) => (
                  <TableRow key={index}>
                    <TableCell className="pl-6"><Skeleton className="h-5 w-32" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-20" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-24" /></TableCell>
                    <TableCell><Skeleton className="h-5 w-16" /></TableCell>
                    <TableCell className="pr-6"><Skeleton className="ml-auto h-9 w-20" /></TableCell>
                  </TableRow>
                ))
              ) : employees?.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="py-12 text-center text-muted-foreground">
                    <UserCircle className="mx-auto mb-3 size-12 opacity-20" />
                    No employee records found. Add a real employee to start Python training.
                  </TableCell>
                </TableRow>
              ) : (
                employees?.map((employee) => {
                  const pythonStatus = getPythonFaceStatus(employee.faceDescriptor);
                  return (
                    <TableRow key={employee.id} className="hover:bg-muted/30">
                      <TableCell className="pl-6 font-medium text-foreground">{employee.name}</TableCell>
                      <TableCell className="text-muted-foreground">{employee.employeeCode}</TableCell>
                      <TableCell>{employee.department}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className="font-mono bg-background">
                          {employee.rfidUid}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {pythonStatus ? (
                          <div className="space-y-1">
                            <Badge
                              variant="outline"
                              className={
                                pythonStatus.status === "trained"
                                  ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                                  : pythonStatus.status === "training"
                                    ? "border-sky-300 bg-sky-50 text-sky-700"
                                    : "border-amber-300 bg-amber-50 text-amber-700"
                              }
                            >
                              {pythonStatus.status === "trained"
                                ? "Python Trained"
                                : pythonStatus.status === "training"
                                  ? "Training"
                                  : "Needs Re-Capture"}
                            </Badge>
                            <p className="text-[11px] text-muted-foreground">
                              {pythonStatus.datasetSampleCount} dataset photos
                            </p>
                          </div>
                        ) : (
                          <span className="text-xs text-muted-foreground">Pending</span>
                        )}
                      </TableCell>
                      <TableCell>
                        {employee.isActive ? (
                          <div className="flex items-center gap-2">
                            <div className="h-2 w-2 rounded-full bg-emerald-500" />
                            <span className="text-sm">Active</span>
                          </div>
                        ) : (
                          <div className="flex items-center gap-2">
                            <div className="h-2 w-2 rounded-full bg-muted-foreground" />
                            <span className="text-sm text-muted-foreground">Inactive</span>
                          </div>
                        )}
                      </TableCell>
                      <TableCell className="pr-6">
                        <div className="flex justify-end gap-2">
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            disabled={updateEmployee.isPending}
                            onClick={() => handleEditEmployee(employee)}
                          >
                            <Pencil className="mr-2 size-4" />
                            Edit
                          </Button>
                          <Button
                            type="button"
                            variant="outline"
                            size="sm"
                            className="text-destructive hover:text-destructive"
                            disabled={deleteEmployee.isPending}
                            onClick={() => handleDeleteEmployee(employee.id, employee.name)}
                          >
                            <Trash2 className="mr-2 size-4" />
                            Delete
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
