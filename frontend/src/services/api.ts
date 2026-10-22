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
  CancerClass,
  IntegrationStatus,
  LiveEventsOut,
  DashboardSummary,
  DetectionClasses,
  DetectionResult,
  DetectionStatus,
  EvaluationSummary,
  Experiment,
  ExplainabilityCapability,
  ExplainabilityResult,
  FederatedState,
  HospitalAgent,
  ModelVersion,
  PretrainedModelInfo,
  PrivacySummaryOut,
  RoundDetail,
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
  integration: () => request<IntegrationStatus>("/api/system/integration"),
  systemEvents: (limit = 50) => request<LiveEventsOut>(`/api/system/events${qs({ limit })}`),
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
  currentRoundState: () => request<FederatedState>("/api/federated/state"),
  startTraining: () => post<{ ok: boolean; round: number }>("/api/federated/start"),
  pauseTraining: () => post<{ ok: boolean }>("/api/federated/pause"),
  nextRound: () => post<{ ok: boolean; round: number }>("/api/federated/next-round"),
  runRounds: (rounds: number) =>
    post<{ ok: boolean; rounds: number }>("/api/federated/run", { rounds }),
  resetFederated: () => post<{ ok: boolean }>("/api/federated/reset"),
  roundDetail: (round: number) =>
    request<RoundDetail>(`/api/federated/rounds/${round}`),

  /* ---------------- models ---------------- */
  pretrainedInfo: () => request<PretrainedModelInfo>("/api/models/pretrained"),
  listModels: () => request<ModelVersion[]>("/api/models"),
  getModel: (id: number) => request<ModelVersion>(`/api/models/${id}`),
  recommendModel: () => post<ModelVersion>("/api/models/recommend"),

  /* ---------------- experiments ---------------- */
  listExperiments: () => request<Experiment[]>("/api/experiments"),
  runExperiment: (name: string) => post<Experiment>("/api/experiments/run", { name }),
  deleteExperiment: (id: number) => del<{ ok: boolean }>(`/api/experiments/${id}`),

  /* ---------------- evaluation ---------------- */
  evaluationSummary: () => request<EvaluationSummary>("/api/evaluation/summary"),

  /* ---------------- detection ---------------- */
  detectionStatus: () => request<DetectionStatus>("/api/detection/status"),
  detectionClasses: () => request<DetectionClasses>("/api/detection/classes"),
  listDetections: () => request<DetectionResult[]>("/api/detection"),
  detect: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<DetectionResult>("/api/detection/predict", { method: "POST", body: fd });
  },
  recentDetections: (limit = 20) =>
    request<DetectionResult[]>(`/api/detection/recent${qs({ limit })}`),

  /* ---------------- audit ---------------- */
  listAuditLogs: (
    limit = 100,
    filters: { event_type?: string; round_number?: number } = {},
  ) => request<AuditLog[]>(`/api/audit${qs({ limit, ...filters })}`),

  /* ---------------- explainability (week 10) ---------------- */
  explainabilityStatus: () =>
    request<ExplainabilityCapability>("/api/explainability/status"),
  explainImage: (file: File, targetClass?: CancerClass) => {
    const fd = new FormData();
    fd.append("file", file);
    if (targetClass) fd.append("target_class", targetClass);
    return request<ExplainabilityResult>("/api/explainability/generate", {
      method: "POST",
      body: fd,
    });
  },

  /* ---------------- privacy (week 10) ---------------- */
  privacySummary: (ledgerLimit = 50) =>
    request<PrivacySummaryOut>(`/api/privacy/summary${qs({ ledger_limit: ledgerLimit })}`),
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
  IntegrationStatus,
  LiveEventsOut,
  DetectionClasses,
  DetectionResult,
  DetectionStatus,
  EvaluationSummary,
  Experiment,
  ExplainabilityCapability,
  ExplainabilityResult,
  FederatedState,
  HospitalAgent,
  ModelVersion,
  PretrainedModelInfo,
  PrivacySummaryOut,
  RoundDetail,
  SystemStatus,
  TrainingRound,
};