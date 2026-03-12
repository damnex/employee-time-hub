import { useQuery } from "@tanstack/react-query";
import { api } from "@shared/routes";
import { z } from "zod";

export type GateEventFilters = z.infer<typeof api.gateEvents.list.input>;

function normalizeFilters(filters?: GateEventFilters) {
  return Object.fromEntries(
    Object.entries(filters ?? {}).filter(([, value]) => {
      return value !== undefined && value !== null && value !== "";
    }),
  ) as Exclude<GateEventFilters, undefined>;
}

function buildGateEventsUrl(filters?: GateEventFilters) {
  const normalizedFilters = normalizeFilters(filters);
  const searchParams = new URLSearchParams();

  Object.entries(normalizedFilters).forEach(([key, value]) => {
    searchParams.set(key, String(value));
  });

  const queryString = searchParams.toString();
  return queryString
    ? `${api.gateEvents.list.path}?${queryString}`
    : api.gateEvents.list.path;
}

export function useGateEvents(filters?: GateEventFilters) {
  const requestUrl = buildGateEventsUrl(filters);
  const normalizedFilters = normalizeFilters(filters);

  return useQuery({
    queryKey: [api.gateEvents.list.path, normalizedFilters],
    queryFn: async () => {
      const res = await fetch(requestUrl, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch gate events");
      const data = await res.json();
      return api.gateEvents.list.responses[200].parse(data);
    },
    refetchInterval: 5000,
    refetchIntervalInBackground: false,
  });
}
