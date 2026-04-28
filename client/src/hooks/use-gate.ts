import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@shared/routes";
import { useToast } from "@/hooks/use-toast";
import { z } from "zod";

type ScanRequest = z.infer<typeof api.scan.rfid.input>;
export type LiveRecognitionRequest = z.infer<typeof api.scan.liveFaces.input>;
export type LiveRecognitionResponse = z.infer<typeof api.scan.liveFaces.responses[200]>;

export async function fetchLiveFaceRecognition(data: LiveRecognitionRequest): Promise<LiveRecognitionResponse> {
  const validated = api.scan.liveFaces.input.parse(data);
  const res = await fetch(api.scan.liveFaces.path, {
    method: api.scan.liveFaces.method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(validated),
    credentials: "include",
  });

  const resData = await res.json();

  if (!res.ok) {
    if (res.status === 400) {
      const error = api.scan.liveFaces.responses[400].parse(resData);
      throw new Error(error.message);
    }

    throw new Error(
      typeof resData?.message === "string" && resData.message.trim()
        ? resData.message
        : "Live face recognition failed",
    );
  }

  return api.scan.liveFaces.responses[200].parse(resData);
}

export function useScanRFID() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: async (data: ScanRequest) => {
      const validated = api.scan.rfid.input.parse(data);
      const res = await fetch(api.scan.rfid.path, {
        method: api.scan.rfid.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validated),
        credentials: "include",
      });

      const resData = await res.json();

      if (!res.ok) {
        if (res.status === 400 || res.status === 404) {
          const error = api.scan.rfid.responses[res.status as 400 | 404].parse(resData);
          throw new Error(error.message);
        }
        throw new Error(
          typeof resData?.message === "string" && resData.message.trim()
            ? resData.message
            : "Scan failed",
        );
      }
      return api.scan.rfid.responses[200].parse(resData);
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: [api.attendances.list.path] });
      queryClient.invalidateQueries({ queryKey: [api.gateEvents.list.path] });
      queryClient.invalidateQueries({ queryKey: [api.stats.dashboard.path] });

      if (data.ignored) {
        toast({
          title: "Scan Ignored",
          description: data.message,
          className: "bg-amber-50 text-amber-900 border-amber-200",
        });
      } else if (data.success) {
        toast({
          title: "Access Granted",
          description: data.message,
          className: "bg-emerald-50 text-emerald-900 border-emerald-200"
        });
      } else {
        toast({
          title: "Access Denied",
          description: data.message,
          variant: "destructive"
        });
      }
    },
    onError: (error: Error) => {
      toast({ title: "System Error", description: error.message, variant: "destructive" });
    }
  });
}
