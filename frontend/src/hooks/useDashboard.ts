import { createContext, useContext } from "react";
import { useQuery, type UseQueryResult } from "@tanstack/react-query";
import { api, ApiError } from "@/services/api";
import type { DashboardSummary } from "@/types";

/**
 * Dashboard-wide data. Every page reads from this single query so a federated
 * round started in one tab is reflected everywhere after invalidation.
 */
export function useDashboard(): UseQueryResult<DashboardSummary, ApiError> {
  return useQuery({
    queryKey: ["dashboard"],
    queryFn: api.dashboard,
    refetchInterval: 5000,
  });
}

/** Bumped after any mutation that can change dashboard aggregates. */
export const dashboardKeys = {
  all: ["dashboard"] as const,
};

export interface SystemStatusContextValue {
  demoMode: boolean;
  realModelLoaded: boolean;
  datasetLoaded: boolean;
  datasetName: string | null;
  totalSamples: number;
}

export const SystemStatusContext = createContext<SystemStatusContextValue>({
  demoMode: true,
  realModelLoaded: false,
  datasetLoaded: false,
  datasetName: null,
  totalSamples: 0,
});

export function useSystemStatus(): SystemStatusContextValue {
  return useContext(SystemStatusContext);
}