import { useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";

import { api } from "@/services/api";
import type { FederatedState, RoundDetail, TrainingRound } from "@/types";

/**
 * Federated training data layer.
 *
 * `/state` is polled fast only while a run is in flight — the round queue lives
 * in the backend worker thread, so polling is how the UI observes it. History
 * (`/rounds`) is database-backed and changes only when a round completes.
 */
export const federatedKeys = {
  state: ["federated", "state"] as const,
  detail: (round: number) => ["federated", "round", round] as const,
  /** Shared with the Dashboard's `api.listRounds` query. */
  rounds: ["rounds"] as const,
};

export function useFederatedState() {
  return useQuery<FederatedState>({
    queryKey: federatedKeys.state,
    queryFn: api.currentRoundState,
    // 1s while rounds are moving, 4s idle: no point hammering a stopped worker.
    refetchInterval: (query) => (query.state.data?.status === "running" ? 1000 : 4000),
    retry: 1,
  });
}

export function useTrainingRounds() {
  return useQuery<TrainingRound[]>({
    queryKey: federatedKeys.rounds,
    queryFn: api.listRounds,
    refetchInterval: 4000,
    retry: 1,
  });
}

export function useRoundDetail(round: number | null) {
  return useQuery<RoundDetail>({
    queryKey: federatedKeys.detail(round ?? -1),
    queryFn: () => api.roundDetail(round as number),
    enabled: round !== null,
    retry: 1,
  });
}

/**
 * What the backend can honestly run right now.
 *
 * Reuses `GET /api/detection/status`, which reports the same `registry.caches_ready()`
 * map the federated `readiness()` gate reads. Without a single train cache a
 * round is refused with a 503, so this lets the UI show the instruction panel
 * up front instead of failing after a click.
 */
export function useCacheReadiness() {
  const q = useQuery({ queryKey: ["detection-status"], queryFn: api.detectionStatus });
  const caches = q.data?.caches_ready ?? {};
  const trainKeys = Object.keys(caches).filter((k) => k.startsWith("train_"));
  const present = trainKeys.filter((k) => caches[k]);
  return {
    isLoading: q.isLoading,
    isError: q.isError,
    refetch: q.refetch,
    ready: present.length > 0,
    valReady: Boolean(caches.val),
    presentTrain: present,
    missingTrain: trainKeys.filter((k) => !caches[k]),
    caches,
  };
}

function refresh(qc: QueryClient) {
  void qc.invalidateQueries({ queryKey: ["federated"] });
  void qc.invalidateQueries({ queryKey: federatedKeys.rounds });
  void qc.invalidateQueries({ queryKey: ["dashboard"] });
  void qc.invalidateQueries({ queryKey: ["agents"] });
}

function useAction<TData>(fn: () => Promise<TData>) {
  const qc = useQueryClient();
  return useMutation({ mutationFn: fn, onSuccess: () => refresh(qc) });
}

function useActionWith<TVariables, TData>(fn: (vars: TVariables) => Promise<TData>) {
  const qc = useQueryClient();
  return useMutation({ mutationFn: fn, onSuccess: () => refresh(qc) });
}

/** The five round-control endpoints, each invalidating the shared caches. */
export function useFederatedControls() {
  return {
    start: useAction(api.startTraining),
    pause: useAction(api.pauseTraining),
    nextRound: useAction(api.nextRound),
    runRounds: useActionWith<number, { ok: boolean; rounds: number }>(api.runRounds),
    reset: useAction(api.resetFederated),
  };
}
