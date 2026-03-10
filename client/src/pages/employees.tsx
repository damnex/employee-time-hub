import { useEffect, useRef, useState } from "react";
import { useEmployees, useCreateEmployee, useDeleteEmployee } from "@/hooks/use-employees";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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
  Loader2,
  Plus,
  RefreshCcw,
  ScanFace,
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
import { insertEmployeeSchema } from "@shared/schema";
import {
  captureFaceTemplate,
} from "@/lib/biometrics";
import { useDeviceWS } from "@/hooks/use-device-ws";

const ENROLLMENT_SAMPLE_COUNT = 25;
const ENROLLMENT_SAMPLE_DELAY_MS = 90;

const defaultFormValues = {
  employeeCode: "",
  name: "",
  department: "",
  phone: "",
  email: "",
  rfidUid: "",
  isActive: true,
  faceDescriptor: null,
};

const formSchema = insertEmployeeSchema
  .extend({
    faceDescriptor: z.array(z.number()).length(128).nullable().optional(),
  })
  .superRefine((values, ctx) => {
    if (!values.faceDescriptor?.length) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["faceDescriptor"],
        message: "Capture a live biometric profile before saving.",
      });
    }
  });

export default function Employees() {
  const { data: employees, isLoading } = useEmployees();
  const createEmployee = useCreateEmployee();
  const deleteEmployee = useDeleteEmployee();
  const {
    isConnected: enrollmentSocketConnected,
    lastScanResult: lastDeviceMessage,
    clearResult: clearDeviceMessage,
  } = useDeviceWS("ENROLLMENT-CONSOLE-01", { clientType: "browser" });
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraRetryToken, setCameraRetryToken] = useState(0);
  const [isCapturingFace, setIsCapturingFace] = useState(false);
  const [capturedSamples, setCapturedSamples] = useState(0);
  const [captureQuality, setCaptureQuality] = useState<number | null>(null);
  const [rfidReaderMessage, setRfidReaderMessage] = useState<string | null>(null);
  const [rfidSourceDeviceId, setRfidSourceDeviceId] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: defaultFormValues,
  });

  const faceDescriptor = form.watch("faceDescriptor");
  const watchedRfidUid = form.watch("rfidUid");
  const normalizedRfidUid = watchedRfidUid.trim().toUpperCase();
  const mappedBadgeOwner = employees?.find((employee) => {
    return employee.rfidUid.toUpperCase() === normalizedRfidUid;
  });
  const rfidReady = Boolean(normalizedRfidUid) && !mappedBadgeOwner;
  const faceProfileReady = Boolean(faceDescriptor?.length);
  const faceError = form.formState.errors.faceDescriptor?.message;

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
          video: {
            width: { ideal: 960 },
            height: { ideal: 540 },
            facingMode: "user",
          },
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
        setCameraError("Allow camera access to enroll the employee face profile.");
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
  }, [isDialogOpen, cameraRetryToken]);

  useEffect(() => {
    if (!isDialogOpen) {
      return;
    }

    if (lastDeviceMessage?.type !== "rfid_detected" || !lastDeviceMessage.rfidUid) {
      return;
    }

    const scannedUid = lastDeviceMessage.rfidUid.trim().toUpperCase();
    form.setValue("rfidUid", scannedUid, {
      shouldDirty: true,
      shouldTouch: true,
      shouldValidate: true,
    });

    setRfidReaderMessage(lastDeviceMessage.message);
    setRfidSourceDeviceId(lastDeviceMessage.deviceId ?? null);

    if (lastDeviceMessage.available === false && lastDeviceMessage.employee) {
      form.setError("rfidUid", {
        type: "manual",
        message: `RFID badge already mapped to ${lastDeviceMessage.employee.name}.`,
      });
    } else {
      form.clearErrors("rfidUid");
    }

    clearDeviceMessage();
  }, [clearDeviceMessage, form, isDialogOpen, lastDeviceMessage]);

  useEffect(() => {
    if (!isDialogOpen) {
      return;
    }

    if (!normalizedRfidUid) {
      setRfidReaderMessage(null);
      setRfidSourceDeviceId(null);
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

    setRfidReaderMessage((currentMessage) => {
      return currentMessage && currentMessage.includes("already mapped")
        ? "Badge ready to assign."
        : currentMessage;
    });
    form.clearErrors("rfidUid");
  }, [form, isDialogOpen, mappedBadgeOwner, normalizedRfidUid]);

  const resetEnrollment = () => {
    setIsCapturingFace(false);
    setCapturedSamples(0);
    setCaptureQuality(null);
    setCameraError(null);
    setRfidReaderMessage(null);
    setRfidSourceDeviceId(null);
    form.reset(defaultFormValues);
  };

  const handleDialogChange = (open: boolean) => {
    setIsDialogOpen(open);

    if (!open) {
      resetEnrollment();
    }
  };

  const handleCaptureFace = async () => {
    if (!videoRef.current || !canvasRef.current || !cameraActive) {
      form.setError("faceDescriptor", {
        type: "manual",
        message: "Camera is not ready. Allow access, then retry capture.",
      });
      return;
    }

    setIsCapturingFace(true);
    setCapturedSamples(0);
    setCaptureQuality(null);
    form.clearErrors("faceDescriptor");

    try {
      const template = await captureFaceTemplate(videoRef.current, canvasRef.current, {
        sampleCount: ENROLLMENT_SAMPLE_COUNT,
        sampleDelayMs: ENROLLMENT_SAMPLE_DELAY_MS,
        minQuality: 0.24,
        maxAttempts: ENROLLMENT_SAMPLE_COUNT * 3,
        onProgress: (acceptedSamples) => {
          setCapturedSamples(acceptedSamples);
        },
      });

      form.setValue("faceDescriptor", template.descriptor, {
        shouldDirty: true,
        shouldTouch: true,
        shouldValidate: true,
      });
      setCaptureQuality(template.averageQuality);
    } catch (error) {
      form.setError("faceDescriptor", {
        type: "manual",
        message:
          error instanceof Error
            ? error.message
            : "Unable to capture the biometric profile.",
      });
    } finally {
      setIsCapturingFace(false);
    }
  };

  const handleClearEnrollment = () => {
    setCapturedSamples(0);
    setCaptureQuality(null);
    form.setValue("faceDescriptor", null, {
      shouldDirty: true,
      shouldTouch: true,
      shouldValidate: true,
    });
  };

  const onSubmit = (values: z.infer<typeof formSchema>) => {
    createEmployee.mutate(values as z.infer<typeof insertEmployeeSchema>, {
      onSuccess: () => {
        setIsDialogOpen(false);
        resetEnrollment();
      },
    });
  };

  const handleDeleteEmployee = (employeeId: number, employeeName: string) => {
    const shouldDelete = window.confirm(
      `Delete ${employeeName} and all related attendance logs? This cannot be undone.`,
    );

    if (!shouldDelete) {
      return;
    }

    deleteEmployee.mutate(employeeId);
  };

  return (
    <div className="p-6 md:p-8 space-y-6 animate-in fade-in duration-500">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Directory</h1>
          <p className="text-muted-foreground mt-1">
            Manage personnel, credentials, and biometric data.
          </p>
        </div>

        <Dialog open={isDialogOpen} onOpenChange={handleDialogChange}>
          <DialogTrigger asChild>
            <Button className="shadow-sm hover:-translate-y-0.5 transition-transform">
              <Plus className="size-4 mr-2" /> Add Employee
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[760px] max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Register New Employee</DialogTitle>
              <DialogDescription>
                Capture live credentials and biometric data in one enrollment session.
              </DialogDescription>
            </DialogHeader>

            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6 py-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <FormField
                    control={form.control}
                    name="name"
                    render={({ field }) => (
                      <FormItem>
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
                      <FormItem>
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
                      <FormItem>
                        <FormLabel>Department</FormLabel>
                        <FormControl>
                          <Input placeholder="Engineering" {...field} />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="email"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Email (Optional)</FormLabel>
                        <FormControl>
                          <Input
                            type="email"
                            placeholder="jane@company.com"
                            {...field}
                            value={field.value || ""}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>

                <div className="border-t pt-5 space-y-5">
                  <div className="space-y-1">
                    <h4 className="text-sm font-semibold tracking-wide">
                      Access Credentials
                    </h4>
                    <p className="text-sm text-muted-foreground">
                      Enroll the RFID badge and live face profile before saving.
                    </p>
                  </div>

                  <div className="grid gap-5 lg:grid-cols-[0.8fr_1.2fr]">
                    <div className="space-y-4">
                      <FormField
                        control={form.control}
                        name="rfidUid"
                        render={({ field }) => (
                          <FormItem>
                            <div className="flex items-center justify-between gap-3">
                              <FormLabel>RFID UID</FormLabel>
                              <Badge variant={enrollmentSocketConnected ? "secondary" : "outline"}>
                                {enrollmentSocketConnected ? "Socket Listening" : "Socket Offline"}
                              </Badge>
                            </div>
                            <FormControl>
                              <Input
                                placeholder="Tap a real badge or type A1B2C3D4"
                                className="font-mono uppercase tracking-[0.2em]"
                                {...field}
                                onChange={(event) =>
                                  field.onChange(event.target.value.toUpperCase())
                                }
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <div className="rounded-2xl border border-border/70 bg-background p-4 space-y-3 shadow-sm">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <p className="text-sm font-semibold">Real RFID Reader</p>
                            <p className="text-xs text-muted-foreground">
                              Tap a physical badge on the connected reader. The UID will auto-fill and be checked against the existing employee map.
                            </p>
                          </div>
                          <Badge variant={rfidReady ? "secondary" : "outline"}>
                            {rfidReady ? "Ready" : normalizedRfidUid ? "Check" : "Waiting"}
                          </Badge>
                        </div>

                        <div className="rounded-xl border border-dashed border-border/80 bg-muted/20 px-3 py-3 text-sm">
                          <div className="flex items-center justify-between gap-3">
                            <span className="text-muted-foreground">Last badge read</span>
                            <span className="font-mono text-foreground">
                              {normalizedRfidUid || "--"}
                            </span>
                          </div>
                          <div className="mt-2 flex items-center justify-between gap-3">
                            <span className="text-muted-foreground">Mapping status</span>
                            <span className={mappedBadgeOwner ? "text-destructive font-medium" : normalizedRfidUid ? "text-emerald-600 font-medium" : ""}>
                              {mappedBadgeOwner
                                ? `Mapped to ${mappedBadgeOwner.name}`
                                : normalizedRfidUid
                                  ? "Available"
                                  : "Waiting for scan"}
                            </span>
                          </div>
                          {rfidSourceDeviceId && (
                            <div className="mt-2 flex items-center justify-between gap-3">
                              <span className="text-muted-foreground">Source device</span>
                              <span className="font-mono text-foreground">{rfidSourceDeviceId}</span>
                            </div>
                          )}
                          {rfidReaderMessage && (
                            <p className={`mt-3 text-sm ${mappedBadgeOwner ? "text-destructive" : "text-muted-foreground"}`}>
                              {rfidReaderMessage}
                            </p>
                          )}
                        </div>
                      </div>

                      <div className="rounded-2xl border border-border/70 bg-muted/20 p-4 space-y-3">
                        <div className="flex items-center gap-2">
                          <ShieldCheck className="size-4 text-primary" />
                          <p className="text-sm font-medium">
                            Enrollment Checklist
                          </p>
                        </div>
                        <div className="space-y-2 text-sm text-muted-foreground">
                          <div className="flex items-center justify-between">
                            <span>Badge assigned</span>
                            <span className={rfidReady ? "text-foreground" : mappedBadgeOwner ? "text-destructive" : ""}>
                              {mappedBadgeOwner
                                ? "Conflict"
                                : rfidReady
                                  ? "Ready"
                                  : "Waiting"}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span>Face profile captured</span>
                            <span className={faceProfileReady ? "text-emerald-600" : ""}>
                              {faceProfileReady ? "Ready" : "Required"}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span>Capture quality</span>
                            <span>
                              {captureQuality !== null
                                ? `${Math.round(captureQuality * 100)}%`
                                : "--"}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-border/70 bg-background shadow-sm p-4 space-y-4">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div>
                          <p className="text-sm font-semibold">Biometric Profile</p>
                          <p className="text-xs text-muted-foreground">
                            Twenty-five clear live samples are merged into one enrollment template.
                          </p>
                        </div>
                        {faceProfileReady ? (
                          <Badge className="bg-emerald-50 text-emerald-700 border-transparent hover:bg-emerald-50">
                            Enrolled
                          </Badge>
                        ) : (
                          <Badge variant="outline">Pending</Badge>
                        )}
                      </div>

                      <div className="aspect-video rounded-2xl border border-border overflow-hidden bg-black relative">
                        <video
                          ref={videoRef}
                          autoPlay
                          muted
                          playsInline
                          className={`h-full w-full object-cover transition-opacity duration-300 ${cameraActive ? "opacity-100" : "opacity-0"}`}
                        />
                        <canvas ref={canvasRef} className="hidden" />

                        {cameraActive && (
                          <>
                            <div className="absolute inset-5 rounded-[1.25rem] border border-primary/40" />
                            <div className="absolute left-4 top-4 rounded-full bg-black/55 px-3 py-1 text-[11px] font-medium text-white">
                              Live enrollment camera
                            </div>
                          </>
                        )}

                        {!cameraActive && (
                          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
                            <Camera className="size-10 text-white/40" />
                            <p className="max-w-xs text-sm text-white/70">
                              {cameraError ?? "Starting live camera feed..."}
                            </p>
                          </div>
                        )}

                        {isCapturingFace && (
                          <div className="absolute inset-0 bg-primary/15 backdrop-blur-[2px] flex items-center justify-center">
                            <div className="rounded-full bg-black/70 px-4 py-2 text-sm font-medium text-white">
                              Capturing sample {capturedSamples + 1} / {ENROLLMENT_SAMPLE_COUNT}
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="grid gap-3 sm:grid-cols-2">
                        <div className="rounded-xl bg-muted/30 px-3 py-2">
                          <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                            Samples
                          </p>
                          <p className="text-lg font-semibold text-foreground">
                            {capturedSamples} / {ENROLLMENT_SAMPLE_COUNT}
                          </p>
                        </div>
                        <div className="rounded-xl bg-muted/30 px-3 py-2">
                          <p className="text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
                            Quality
                          </p>
                          <p className="text-lg font-semibold text-foreground">
                            {captureQuality !== null
                              ? `${Math.round(captureQuality * 100)}%`
                              : "--"}
                          </p>
                        </div>
                      </div>

                      <div className="h-2 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-primary transition-all duration-300"
                          style={{
                            width: `${(capturedSamples / ENROLLMENT_SAMPLE_COUNT) * 100}%`,
                          }}
                        />
                      </div>

                      <div className="flex flex-col gap-2 sm:flex-row">
                        <Button
                          type="button"
                          className="flex-1"
                          onClick={handleCaptureFace}
                          disabled={!cameraActive || isCapturingFace}
                        >
                          {isCapturingFace ? (
                            <>
                              <Loader2 className="mr-2 size-4 animate-spin" />
                              Capturing...
                            </>
                          ) : faceProfileReady ? (
                            <>
                              <ScanFace className="mr-2 size-4" />
                              Re-Capture Face
                            </>
                          ) : (
                            <>
                              <ScanFace className="mr-2 size-4" />
                              Capture Face Data
                            </>
                          )}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          onClick={
                            faceProfileReady
                              ? handleClearEnrollment
                              : () => setCameraRetryToken((value) => value + 1)
                          }
                          disabled={isCapturingFace}
                        >
                          {faceProfileReady ? (
                            <>
                              <RefreshCcw className="mr-2 size-4" />
                              Clear
                            </>
                          ) : (
                            <>
                              <RefreshCcw className="mr-2 size-4" />
                              Retry Camera
                            </>
                          )}
                        </Button>
                      </div>

                      <div className="rounded-xl border border-dashed border-border/80 bg-muted/20 px-4 py-3 text-sm">
                        {faceProfileReady ? (
                          <div className="flex items-start gap-2 text-emerald-700">
                            <CheckCircle2 className="mt-0.5 size-4 shrink-0" />
                            <p>
                              Live facial template captured. This employee can now use
                              gate verification on the same camera setup.
                            </p>
                          </div>
                        ) : (
                          <div className="flex items-start gap-2 text-muted-foreground">
                            <AlertCircle className="mt-0.5 size-4 shrink-0" />
                            <p>
                              Center the face in frame, keep still for one second, then
                              capture all 25 samples.
                            </p>
                          </div>
                        )}
                      </div>

                      {faceError && (
                        <p className="text-sm font-medium text-destructive">
                          {faceError}
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                <DialogFooter className="pt-2">
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => handleDialogChange(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    disabled={
                      createEmployee.isPending
                      || isCapturingFace
                      || !faceProfileReady
                      || !rfidReady
                    }
                  >
                    {createEmployee.isPending ? "Saving..." : "Save Employee"}
                  </Button>
                </DialogFooter>
              </form>
            </Form>
          </DialogContent>
        </Dialog>
      </div>

      <Card className="border-border/50 shadow-sm overflow-hidden">
        <CardContent className="p-0">
          <Table>
            <TableHeader className="bg-muted/50">
              <TableRow>
                <TableHead className="pl-6">Employee</TableHead>
                <TableHead>Code</TableHead>
                <TableHead>Department</TableHead>
                <TableHead>RFID Badge</TableHead>
                <TableHead>Biometrics</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right pr-6">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <TableRow key={i}>
                    <TableCell className="pl-6">
                      <Skeleton className="h-5 w-32" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-5 w-20" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-5 w-24" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-5 w-24" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-5 w-16" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-5 w-16" />
                    </TableCell>
                    <TableCell className="pr-6">
                      <Skeleton className="ml-auto h-9 w-20" />
                    </TableCell>
                  </TableRow>
                ))
              ) : employees?.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={7}
                    className="text-center py-12 text-muted-foreground"
                  >
                    <UserCircle className="size-12 mx-auto mb-3 opacity-20" />
                    No employee records found. Add a real employee to get started.
                  </TableCell>
                </TableRow>
              ) : (
                employees?.map((emp) => (
                  <TableRow key={emp.id} className="hover:bg-muted/30">
                    <TableCell className="pl-6 font-medium text-foreground">
                      {emp.name}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {emp.employeeCode}
                    </TableCell>
                    <TableCell>{emp.department}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="font-mono bg-background">
                        {emp.rfidUid}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      {emp.faceDescriptor ? (
                        <Badge className="bg-emerald-50 text-emerald-700 hover:bg-emerald-100 border-none">
                          Enrolled
                        </Badge>
                      ) : (
                        <span className="text-xs text-muted-foreground">Pending</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {emp.isActive ? (
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-emerald-500" />
                          <span className="text-sm">Active</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full bg-muted-foreground" />
                          <span className="text-sm text-muted-foreground">
                            Inactive
                          </span>
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="pr-6">
                      <div className="flex justify-end">
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="text-destructive hover:text-destructive"
                          disabled={deleteEmployee.isPending}
                          onClick={() => handleDeleteEmployee(emp.id, emp.name)}
                        >
                          <Trash2 className="mr-2 size-4" />
                          Delete
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
