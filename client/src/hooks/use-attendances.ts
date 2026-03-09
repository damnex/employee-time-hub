import { useQuery } from "@tanstack/react-query";
import { api } from "@shared/routes";

export function useAttendances() {
  return useQuery({
    queryKey: [api.attendances.list.path],
    queryFn: async () => {
      const res = await fetch(api.attendances.list.path, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch attendances");
      const data = await res.json();
      return api.attendances.list.responses[200].parse(data);
    },
    refetchInterval: 3000,
    refetchIntervalInBackground: true,
  });
}
