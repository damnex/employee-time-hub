import { useState, useRef, useEffect } from "react";
import { useScanRFID } from "@/hooks/use-gate";
import { useDeviceWS } from "@/hooks/use-device-ws";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Camera, Scan, KeyRound, AlertCircle, CheckCircle2, Wifi, WifiOff } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function GateSimulator() {
  const [rfidUid, setRfidUid] = useState("");
  const [useRealCamera, setUseRealCamera] = useState(true);
  const [useRealDevice, setUseRealDevice] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [lastResult, setLastResult] = useState<{success: boolean, message: string, employee?: {name: string}} | null>(null);
  
  const scanMutation = useScanRFID();
  const { isConnected, lastScanResult, sendRFIDScan, clearResult } = useDeviceWS("GATE-TERMINAL-01");

  // Initialize real camera
  useEffect(() => {
    if (!useRealCamera) return;

    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setCameraActive(true);
        }
      } catch (error) {
        console.error("Camera access denied:", error);
        setUseRealCamera(false);
      }
    };

    initCamera();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, [useRealCamera]);

  // Extract face descriptor from camera frame (simulated ML)
  const extractFaceDescriptor = (): number[] => {
    if (!cameraActive || !videoRef.current || !canvasRef.current) {
      return Array.from({ length: 128 }, () => Number(Math.random().toFixed(4)));
    }

    try {
      const ctx = canvasRef.current.getContext('2d');
      if (!ctx) return Array.from({ length: 128 }, () => Number(Math.random().toFixed(4)));

      ctx.drawImage(videoRef.current, 0, 0, 128, 128);
      const imageData = ctx.getImageData(0, 0, 128, 128);
      
      // Simple: convert pixels to normalized vector (128 elements)
      const descriptor = [];
      for (let i = 0; i < imageData.data.length; i += 4) {
        descriptor.push(imageData.data[i] / 255);
        if (descriptor.length >= 128) break;
      }
      
      while (descriptor.length < 128) {
        descriptor.push(Math.random());
      }

      return descriptor.slice(0, 128);
    } catch (error) {
      console.error("Error extracting face descriptor:", error);
      return Array.from({ length: 128 }, () => Number(Math.random().toFixed(4)));
    }
  };

  const handleScan = () => {
    if (!rfidUid.trim()) return;

    if (useRealDevice && isConnected) {
      // Send to real ESP8266 device via WebSocket
      const descriptor = extractFaceDescriptor();
      sendRFIDScan(rfidUid.trim(), descriptor);
      setRfidUid("");
    } else {
      // Fallback to HTTP API
      const faceDescriptor = extractFaceDescriptor();
      scanMutation.mutate({
        rfidUid: rfidUid.trim(),
        deviceId: "GATE-TERMINAL-01",
        faceDescriptor
      }, {
        onSuccess: (data) => {
          setLastResult({ success: data.success, message: data.message, employee: data.employee });
          if (data.success) {
            setRfidUid("");
          }
        },
        onError: (error) => {
          setLastResult({ success: false, message: error.message });
        }
      });
    }
  };

  // Handle WebSocket scan results
  useEffect(() => {
    if (lastScanResult?.type === 'scan_result') {
      setLastResult({
        success: lastScanResult.success ?? false,
        message: lastScanResult.message,
        employee: lastScanResult.employee
      });
      
      if (lastScanResult.success) {
        setRfidUid("");
      }
      
      // Clear result after 5 seconds
      const timer = setTimeout(() => clearResult(), 5000);
      return () => clearTimeout(timer);
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
              {cameraActive ? 'Live Camera Feed' : 'Camera Feed (Simulated)'}
            </div>
            <div className="aspect-video bg-black rounded-xl border border-border flex flex-col items-center justify-center relative overflow-hidden group">
              {cameraActive ? (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full h-full object-cover"
                  />
                  <canvas
                    ref={canvasRef}
                    width={128}
                    height={128}
                    className="hidden"
                  />
                </>
              ) : (
                <>
                  <Camera className="size-10 text-muted-foreground/40 group-hover:scale-110 transition-transform duration-300" />
                  <div className="absolute inset-0 border-2 border-primary/20 rounded-xl m-4 pointer-events-none opacity-50" />
                </>
              )}
              {(scanMutation.isPending || (useRealDevice && !isConnected && rfidUid)) && (
                <div className="absolute inset-0 bg-primary/10 backdrop-blur-[2px] flex items-center justify-center">
                  <div className="w-16 h-1 bg-primary/80 animate-pulse rounded-full shadow-[0_0_15px_rgba(var(--primary),0.5)]" />
                </div>
              )}
            </div>
          </div>

          {/* Form / Inputs */}
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="rfid" className="flex items-center gap-2 text-foreground/80">
                <KeyRound className="size-4" /> Tap Badge (RFID UID)
              </Label>
              <Input 
                id="rfid"
                placeholder="e.g. A1B2C3D4" 
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
              disabled={scanMutation.isPending || !rfidUid.trim()}
            >
              {scanMutation.isPending ? "Verifying..." : "Scan & Authenticate"}
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
                </AlertDescription>
              </Alert>
            )}
          </div>

        </CardContent>
      </Card>
    </div>
  );
}
