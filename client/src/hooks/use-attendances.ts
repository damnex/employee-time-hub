import { useQuery } from "@tanstack/react-query";
import { api } from "@shared/routes";
import { z } from "zod";

export type AttendanceFilters = z.infer<typeof api.attendances.list.input>;

function normalizeFilters(filters?: AttendanceFilters) {
  return Object.fromEntries(
    Object.entries(filters ?? {}).filter(([, value]) => {
      return value !== undefined && value !== null && value !== "";
    }),
  ) as Exclude<AttendanceFilters, undefined>;
}

function buildAttendancesUrl(filters?: AttendanceFilters) {
  const normalizedFilters = normalizeFilters(filters);
  const searchParams = new URLSearchParams();

  Object.entries(normalizedFilters).forEach(([key, value]) => {
    searchParams.set(key, String(value));
  });

  const queryString = searchParams.toString();
  return queryString
    ? `${api.attendances.list.path}?${queryString}`
    : api.attendances.list.path;
}

export function useAttendances(filters?: AttendanceFilters) {
  const requestUrl = buildAttendancesUrl(filters);
  const normalizedFilters = normalizeFilters(filters);

  return useQuery({
    queryKey: [api.attendances.list.path, normalizedFilters],
    queryFn: async () => {
      const res = await fetch(requestUrl, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch attendances");
      const data = await res.json();
      return api.attendances.list.responses[200].parse(data);
    },
    refetchInterval: 5000,
    refetchIntervalInBackground: false,
  });
}
