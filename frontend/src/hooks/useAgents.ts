import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/services/api";
import type { HospitalAgent } from "@/types";

/**
 * Agent roster plus the three per-agent controls.
 *
 * Every mutation invalidates the dashboard summary as well as the agent cache,
 * because starting a round on one site changes the aggregates shown on the
 * dashboard and the federated training view in the same breath.
 */
export function useAgents() {
  return useQuery({
    queryKey: ["agents"],
    queryFn: api.listAgents,
    retry: 1,
  });
}

export function useAgent(agentId: number) {
  return useQuery({
    queryKey: ["agents", agentId],
    queryFn: () => api.getAgent(agentId),
    enabled: Number.isFinite(agentId),
    retry: 1,
  });
}

function useAgentMutation(fn: (id: number) => Promise<HospitalAgent>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: (updated) => {
      // Patch the single agent in the cached roster rather than refetching the
      // whole list, then let the dashboard refetch on its own 5s interval.
      qc.setQueryData<HospitalAgent[]>(["agents"], (prev) =>
        prev?.map((a) => (a.id === updated.id ? updated : a)) ?? prev,
      );
      qc.setQueryData<HospitalAgent>(["agents", updated.id], updated);
      void qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });
}

export function useStartAgentTraining() {
  return useAgentMutation(api.startAgentTraining);
}

export function usePauseAgentTraining() {
  return useAgentMutation(api.pauseAgentTraining);
}

export function useSyncAgent() {
  return useAgentMutation(api.syncAgent);
}

export type { ApiError };