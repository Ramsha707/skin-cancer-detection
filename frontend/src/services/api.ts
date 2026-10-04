/**
 * Thin typed fetch wrapper around the FastAPI backend.
 *
 * Every method returns the same shape as `backend/app/schemas/`, so the UI can
 * never silently drift from the API contract. Non-2xx responses throw an
 * `ApiError` carrying the backend's `detail` message.
 */

import type {
  AgentRoundState,
  AuditLog,
  DashboardSummary,
  DetectionResult,
  Experiment,
  HospitalAgent,
  ModelVersion,
  PretrainedModelInfo,
  SystemStatus,
  TrainingRound,
} from "@/types";

export const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;
  readonly path?: string;

  constructor(message: string, status: number, path?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.path = path;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  let res: Response;
  try {
    res = await fetch(url, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
    });
  } catch {
    throw new ApiError(
      `Cannot reach the backend at ${API_BASE}. Is the FastAPI server running?`,
      0,
      path,
    );
  }

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new ApiError(detail, res.status, path);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const post = <T>(path: string, body?: unknown) =>
  request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

const del = <T>(path: string) => request<T>(path, { method: "DELETE" });

function qs(params: Record<string, unknown>): string {
  const sp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") sp.append(k, String(v));
  });
  const s = sp.toString();
  return s ? `?${s}` : "";
}

export const api = {
  /* ---------------- system ---------------- */
  health: () => request<SystemStatus>("/api/system/health"),
  dashboard: () => request<DashboardSummary>("/api/dashboard/summary"),

  /* ---------------- agents ---------------- */
  listAgents: () => request<HospitalAgent[]>("/api/agents"),
  getAgent: (id: number) => request<HospitalAgent>(`/api/agents/${id}`),
  agentClassDistribution: (id: number) =>
    request<Record<string, number>>(`/api/agents/${id}/class-distribution`),
  startAgentTraining: (id: number) => post<HospitalAgent>(`/api/agents/${id}/train`),
  pauseAgentTraining: (id: number) => post<HospitalAgent>(`/api/agents/${id}/pause`),
  syncAgent: (id: number) => post<HospitalAgent>(`/api/agents/${id}/sync`),

  /* ---------------- federated ---------------- */
  listRounds: () => request<TrainingRound[]>("/api/federated/rounds"),
  currentRoundState: () =>
    request<{
      status: string;
      current_round: number;
      global_model_version: string | null;
      aggregation_status: string;
      agents: AgentRoundState[];
      training_status: string;
    }>("/api/federated/state"),
  startTraining: () => post<{ ok: boolean; round: number }>("/api/federated/start"),
  pauseTraining: () => post<{ ok: boolean }>("/api/federated/pause"),
  nextRound: () => post<{ ok: boolean; round: number }>("/api/federated/next-round"),
  runRounds: (rounds: number) =>
    post<{ ok: boolean; rounds: number }>("/api/federated/run", { rounds }),
  resetFederated: () => post<{ ok: boolean }>("/api/federated/reset"),
  roundDetail: (round: number) =>
    request<{ round: TrainingRound; agents: AgentRoundState[] }>(
      `/api/federated/rounds/${round}`,
    ),

  /* ---------------- models ---------------- */
  pretrainedInfo: () => request<PretrainedModelInfo>("/api/models/pretrained"),
  listModels: () => request<ModelVersion[]>("/api/models"),
  getModel: (id: number) => request<ModelVersion>(`/api/models/${id}`),
  recommendModel: () => post<ModelVersion>("/api/models/recommend"),

  /* ---------------- experiments ---------------- */
  listExperiments: () => request<Experiment[]>("/api/experiments"),
  runExperiment: (name: string) => post<Experiment>("/api/experiments/run", { name }),
  deleteExperiment: (id: number) => del<{ ok: boolean }>(`/api/experiments/${id}`),

  /* ---------------- detection ---------------- */
  detect: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<DetectionResult>("/api/detection/predict", { method: "POST", body: fd });
  },
  recentDetections: (limit = 20) =>
    request<DetectionResult[]>(`/api/detection/recent${qs({ limit })}`),

  /* ---------------- audit ---------------- */
  listAuditLogs: (limit = 100) => request<AuditLog[]>(`/api/audit${qs({ limit })}`),
};

/** WebSocket URL for the live training event bus. */
export function trainingSocketUrl(): string {
  const base = API_BASE.replace(/^http/, "ws");
  return `${base}/ws/training`;
}

export type {
  AgentRoundState,
  AuditLog,
  DashboardSummary,
  DetectionResult,
  Experiment,
  HospitalAgent,
  ModelVersion,
  PretrainedModelInfo,
  SystemStatus,
  TrainingRound,
};