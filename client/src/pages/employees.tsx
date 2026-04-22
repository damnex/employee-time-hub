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
  AlertCircle,
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
const MIN_DATASET_SAMPLES = 20;
const DEFAULT_DATASET_SAMPLES = 60;
const MAX_DATASET_SAMPLES = 100;
const DATASET_CAPTURE_DELAY_MS = 160;
const DATASET_CAPTURE_SIZE = 360;

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

function captureDatasetFrame(video: HTMLVideoElement, canvas: HTMLCanvasElement) {
  const context = canvas.getContext("2d");
  if (!context || !video.videoWidth || !video.videoHeight) {
    return null;
  }

  const sourceSize = Math.min(video.videoWidth, video.videoHeight);
  const sourceX = Math.max(0, (video.videoWidth - sourceSize) / 2);
  const sourceY = Math.max(0, (video.videoHeight - sourceSize) / 2);

  canvas.width = DATASET_CAPTURE_SIZE;
  canvas.height = DATASET_CAPTURE_SIZE;
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
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const autoRegistrationAttemptedRef = useRef(false);
  const registrationReaderStartedRef = useRef(false);

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
    setDatasetSamplesTarget(DEFAULT_DATASET_SAMPLES);
    setRfidReaderMessage(null);
    setRfidSourceDeviceId(null);
    setRegistrationModeEnabled(false);
    setCameraError(null);
    setIsCapturingDataset(false);
    form.reset(defaultFormValues);
  };

  const handleDialogChange = (open: boolean) => {
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

    setIsCapturingDataset(true);
    setCaptureProgress(0);
    setDatasetError(null);
    setDatasetPhotos([]);

    try {
      const capturedPhotos: string[] = [];
      for (let index = 0; index < datasetSamplesTarget; index += 1) {
        const frame = captureDatasetFrame(videoRef.current, canvasRef.current);
        if (!frame) {
          throw new Error("Dataset frame capture failed. Retry with the employee centered in the camera.");
        }
        capturedPhotos.push(frame);
        setDatasetPhotos([...capturedPhotos]);
        setCaptureProgress(index + 1);
        await sleep(DATASET_CAPTURE_DELAY_MS);
      }
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
          <DialogContent className="left-0 top-0 h-[100dvh] w-screen max-w-none translate-x-0 translate-y-0 gap-0 overflow-hidden rounded-none border-0 p-0 sm:rounded-none">
            <DialogHeader className="shrink-0 border-b border-border/70 px-4 py-3 text-left sm:px-5">
              <DialogTitle>Register New Employee</DialogTitle>
              <DialogDescription className="text-sm">
                Fill the employee details, register the RFID badge, and capture a dataset for Python training.
              </DialogDescription>
            </DialogHeader>

            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="flex h-full min-h-0 flex-col">
                <div className="flex-1 overflow-y-auto px-4 py-3 sm:px-5">
                  <div className="space-y-3">
                    <div className="grid gap-2.5 md:grid-cols-2 xl:grid-cols-4">
                  <FormField
                    control={form.control}
                    name="name"
                    render={({ field }) => (
                      <FormItem className="space-y-1.5">
                        <FormLabel>Full Name</FormLabel>
                        <FormControl>
                          <Input placeholder="Jane Doe" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="employeeCode"
                    render={({ field }) => (
                      <FormItem className="space-y-1.5">
                        <FormLabel>Employee Code</FormLabel>
                        <FormControl>
                          <Input placeholder="EMP-1042" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="department"
                    render={({ field }) => (
                      <FormItem className="space-y-1.5">
                        <FormLabel>Department</FormLabel>
                        <FormControl>
                          <Input placeholder="Operations" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="email"
                    render={({ field }) => (
                      <FormItem className="space-y-1.5">
                        <FormLabel>Email (Optional)</FormLabel>
                        <FormControl>
                          <Input placeholder="jane@company.com" {...field} value={field.value || ""} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="space-y-3 border-t pt-3">
                  <div className="space-y-0.5">
                    <h4 className="text-sm font-semibold tracking-wide">Access Credentials</h4>
                    <p className="text-[13px] leading-snug text-muted-foreground">
                      Register the card and capture many dataset images so Python can train the face model well.
                    </p>
                  </div>

                  <div className="grid gap-3 lg:grid-cols-[340px_minmax(0,1fr)]">
                    <div className="space-y-2.5">
                      <FormField
                        control={form.control}
                        name="rfidUid"
                        render={({ field }) => (
                          <FormItem className="space-y-1.5">
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
                                className="font-mono uppercase tracking-[0.14em]"
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

                      <div className="rounded-xl border border-dashed border-border/70 bg-muted/20 px-2.5 py-2 text-sm">
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
                        <div className="rounded-xl border border-primary/20 bg-primary/5 px-2.5 py-2">
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

                      <div className="space-y-2 rounded-xl bg-muted/30 p-2.5">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Dataset Target</p>
                            <p className="text-[13px] leading-snug text-foreground">Capture more samples for stronger Python training coverage.</p>
                          </div>
                          <Input
                            type="number"
                            min={MIN_DATASET_SAMPLES}
                            max={MAX_DATASET_SAMPLES}
                            value={datasetSamplesTarget}
                            onChange={(event) => {
                              setDatasetSamplesTarget(clamp(Number(event.target.value) || DEFAULT_DATASET_SAMPLES, MIN_DATASET_SAMPLES, MAX_DATASET_SAMPLES));
                            }}
                            className="h-8 w-[72px] px-2 text-center text-sm"
                            disabled={isCapturingDataset}
                          />
                        </div>
                        <p className="text-[11px] leading-relaxed text-muted-foreground">
                          Ask the employee to look front, slightly left, slightly right, and move naturally while the dataset is being captured.
                        </p>
                      </div>
                    </div>

                    <div className="grid gap-2.5 xl:grid-cols-[minmax(0,1fr)_260px] xl:items-start">
                      <div className="relative overflow-hidden rounded-[1.5rem] border border-border/70 bg-black xl:row-span-5">
                        <div className="aspect-[16/9] overflow-hidden xl:aspect-[2/1]">
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
                            <div className="absolute inset-4 rounded-[1.35rem] border border-white/20" />
                            <div className="absolute inset-x-[24%] inset-y-[14%] rounded-[1.75rem] border-[3px] border-emerald-300/70 shadow-[0_0_0_1px_rgba(110,231,183,0.35),0_0_18px_rgba(16,185,129,0.16)]" />
                            <div className="absolute left-1/2 top-3 -translate-x-1/2 rounded-full bg-black/65 px-3 py-1 text-[10px] font-semibold tracking-[0.24em] text-white/90">
                              PYTHON DATASET
                            </div>
                          </div>
                          {isCapturingDataset && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-black/40 backdrop-blur-[2px]">
                              <div className="rounded-2xl bg-black/80 px-4 py-3 text-center text-sm font-medium text-white shadow-xl backdrop-blur-md">
                                Capturing sample {captureProgress} / {datasetSamplesTarget}
                              </div>
                              <div className="rounded-2xl bg-primary/95 px-5 py-3 text-center text-lg font-bold text-primary-foreground shadow-2xl animate-in zoom-in duration-300">
                                {captureProgress / datasetSamplesTarget < 0.2 && "Look straight at the camera"}
                                {captureProgress / datasetSamplesTarget >= 0.2 && captureProgress / datasetSamplesTarget < 0.4 && "Turn head slightly left"}
                                {captureProgress / datasetSamplesTarget >= 0.4 && captureProgress / datasetSamplesTarget < 0.6 && "Turn head slightly right"}
                                {captureProgress / datasetSamplesTarget >= 0.6 && captureProgress / datasetSamplesTarget < 0.8 && "Tilt head slightly up"}
                                {captureProgress / datasetSamplesTarget >= 0.8 && "Tilt head slightly down"}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="space-y-1.5 rounded-xl border border-dashed border-border/70 bg-muted/20 p-2.5 xl:col-start-2">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Profile photo (optional)</p>
                            <p className="text-sm text-foreground">Add a cover photo for badges and dashboards. If empty, we’ll use a dataset sample.</p>
                          </div>
                          {profilePhoto && (
                            <img
                              src={profilePhoto}
                              alt="Profile preview"
                              className="h-10 w-10 rounded-xl object-cover border border-border/70 shadow-sm"
                            />
                          )}
                        </div>
                        <Input className="h-9 text-sm" type="file" accept="image/*" onChange={handleProfilePhotoChange} />
                      </div>

                      <div className="space-y-2 rounded-xl bg-muted/30 p-2.5 xl:col-start-2">
                        <div className="grid grid-cols-3 gap-1.5">
                          <div className="rounded-lg bg-background/60 px-2 py-1.5">
                            <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Captured</p>
                            <p className="text-sm font-semibold text-foreground">{datasetPhotos.length}</p>
                          </div>
                          <div className="rounded-lg bg-background/60 px-2 py-1.5">
                            <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Minimum</p>
                            <p className="text-sm font-semibold text-foreground">{MIN_DATASET_SAMPLES}</p>
                          </div>
                          <div className="rounded-lg bg-background/60 px-2 py-1.5">
                            <p className="text-[10px] uppercase tracking-[0.16em] text-muted-foreground">Training</p>
                            <p className="text-sm font-semibold text-foreground">{datasetReady ? "Ready" : "Pending"}</p>
                          </div>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-background/70">
                          <div
                            className="h-full rounded-full bg-primary transition-all duration-300"
                            style={{ width: `${(datasetPhotos.length / Math.max(1, datasetSamplesTarget)) * 100}%` }}
                          />
                        </div>
                      </div>

                      <div className="flex flex-col gap-1.5 sm:flex-row xl:col-start-2 xl:flex-col">
                        <Button
                          type="button"
                          className="h-10 flex-1 text-sm"
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
                              Capture Dataset Photos
                            </>
                          )}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          className="h-10 text-sm"
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

                      <div className="rounded-xl border border-dashed border-border/80 bg-muted/20 px-2.5 py-2 text-[13px] leading-snug xl:col-start-2">
                        {datasetReady ? (
                          <div className="flex items-start gap-2 text-emerald-700">
                            <CheckCircle2 className="mt-0.5 size-4 shrink-0" />
                            <p>
                              Dataset captured. Saving will store the employee, write the dataset images, and refresh the Python face model.
                              Aim for 60 to 100 samples for the strongest roster quality.
                            </p>
                          </div>
                        ) : (
                          <div className="flex items-start gap-2 text-muted-foreground">
                            <AlertCircle className="mt-0.5 size-4 shrink-0" />
                            <p>
                              Capture at least {MIN_DATASET_SAMPLES} photos. For better accuracy, aim for 60 or more with good lighting and natural face turns.
                            </p>
                          </div>
                        )}
                      </div>

                      {datasetError && (
                        <p className="text-xs font-medium text-destructive xl:col-start-2">{datasetError}</p>
                      )}
                    </div>
                  </div>
                </div>

                  </div>
                </div>

                <DialogFooter className="shrink-0 border-t border-border/70 bg-background/95 px-4 py-2.5 sm:px-5">
                  <Button className="h-10 px-4" type="button" variant="ghost" onClick={() => handleDialogChange(false)}>
                    Cancel
                  </Button>
                  <Button
                    className="h-10 px-4"
                    type="submit"
                    disabled={pythonEnrollEmployee.isPending || isCapturingDataset || !datasetReady || !rfidReady}
                  >
                    {pythonEnrollEmployee.isPending ? "Saving & Training..." : "Save Employee"}
                  </Button>
                </DialogFooter>
              </form>
            </Form>
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
