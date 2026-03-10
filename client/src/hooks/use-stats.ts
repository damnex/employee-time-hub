import { useQuery } from "@tanstack/react-query";
import { api } from "@shared/routes";

export function useDashboardStats() {
  return useQuery({
    queryKey: [api.stats.dashboard.path],
    queryFn: async () => {
      const res = await fetch(api.stats.dashboard.path, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch dashboard stats");
      const data = await res.json();
      return api.stats.dashboard.responses[200].parse(data);
    },
    refetchInterval: 5000,
    refetchIntervalInBackground: false,
  });
}
