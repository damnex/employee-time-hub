import { useState } from "react";
import { useScanRFID } from "@/hooks/use-gate";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Camera, Scan, KeyRound, AlertCircle, CheckCircle2 } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export default function GateSimulator() {
  const [rfidUid, setRfidUid] = useState("");
  const [lastResult, setLastResult] = useState<{success: boolean, message: string} | null>(null);
  const scanMutation = useScanRFID();

  const handleScan = () => {
    if (!rfidUid.trim()) return;

    // Simulate webcam facial recognition array extraction
    const fakeDescriptor = Array.from({ length: 128 }, () => Number(Math.random().toFixed(4)));

    scanMutation.mutate({
      rfidUid: rfidUid.trim(),
      deviceId: "MAIN-GATE-01",
      faceDescriptor: fakeDescriptor
    }, {
      onSuccess: (data) => {
        setLastResult({ success: data.success, message: data.message });
        if (data.success) {
          setRfidUid(""); // clear input on success
        }
      },
      onError: (error) => {
        setLastResult({ success: false, message: error.message });
      }
    });
  };

  return (
    <div className="p-6 md:p-8 h-full flex flex-col items-center justify-center bg-slate-50 dark:bg-slate-950">
      
      <div className="text-center mb-8">
        <h1 className="text-3xl font-display font-bold text-foreground">Terminal Kiosk Simulator</h1>
        <p className="text-muted-foreground mt-2 max-w-md mx-auto">
          Test the physical terminal flow. Enter an enrolled RFID badge and the system will simulate a biometric match attempt.
        </p>
      </div>

      <Card className="w-full max-w-md shadow-xl border-border/60 overflow-hidden">
        <div className="bg-primary p-4 text-primary-foreground flex items-center justify-center gap-2">
          <Scan className="size-5" />
          <span className="font-semibold tracking-wide">GATE-01 TERMINAL</span>
        </div>
        
        <CardContent className="p-8 space-y-8">
          {/* Simulated Webcam View */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider text-center">Camera Feed</div>
            <div className="aspect-video bg-black/5 dark:bg-white/5 rounded-xl border border-dashed border-border flex flex-col items-center justify-center relative overflow-hidden group">
              <Camera className="size-10 text-muted-foreground/40 group-hover:scale-110 transition-transform duration-300" />
              <div className="absolute inset-0 border-2 border-primary/20 rounded-xl m-4 pointer-events-none opacity-50" />
              {scanMutation.isPending && (
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
