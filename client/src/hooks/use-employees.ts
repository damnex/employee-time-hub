import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, buildUrl } from "@shared/routes";
import { type InsertEmployee } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";

export function useEmployees() {
  return useQuery({
    queryKey: [api.employees.list.path],
    queryFn: async () => {
      const res = await fetch(api.employees.list.path, { credentials: "include" });
      if (!res.ok) throw new Error("Failed to fetch employees");
      const data = await res.json();
      return api.employees.list.responses[200].parse(data);
    },
    staleTime: 15000,
    refetchOnWindowFocus: true,
  });
}

export function useCreateEmployee() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: async (data: InsertEmployee) => {
      const validated = api.employees.create.input.parse(data);
      const res = await fetch(api.employees.create.path, {
        method: api.employees.create.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validated),
        credentials: "include",
      });
      
      const resData = await res.json();
      
      if (!res.ok) {
        if (res.status === 400) {
          const error = api.employees.create.responses[400].parse(resData);
          throw new Error(error.message || "Validation failed");
        }
        throw new Error("Failed to create employee");
      }
      return api.employees.create.responses[201].parse(resData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [api.employees.list.path] });
      queryClient.invalidateQueries({ queryKey: [api.stats.dashboard.path] });
      toast({ title: "Success", description: "Employee created successfully." });
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    }
  });
}

export function useDeleteEmployee() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: async (employeeId: number) => {
      const res = await fetch(buildUrl(api.employees.delete.path, { id: employeeId }), {
        method: api.employees.delete.method,
        credentials: "include",
      });

      const resData = await res.json();

      if (!res.ok) {
        if (res.status === 404) {
          const error = api.employees.delete.responses[404].parse(resData);
          throw new Error(error.message || "Employee not found");
        }

        throw new Error("Failed to delete employee");
      }

      return api.employees.delete.responses[200].parse(resData);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [api.employees.list.path] });
      queryClient.invalidateQueries({ queryKey: [api.attendances.list.path] });
      queryClient.invalidateQueries({ queryKey: [api.stats.dashboard.path] });
      toast({ title: "Success", description: "Employee deleted successfully." });
    },
    onError: (error: Error) => {
      toast({ title: "Error", description: error.message, variant: "destructive" });
    },
  });
}
