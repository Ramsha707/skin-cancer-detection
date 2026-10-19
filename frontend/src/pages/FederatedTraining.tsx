/**
 * Week 8 — federated training dashboard.
 *
 * Drives the week-7 engine end to end: queue rounds, watch the round worker
 * live, and read the audit-backed history. The topology shows *where* model
 * deltas travel (hospital → aggregator → hospital); images never leave a node.
 *
 * There is no simulation path. If the embedding caches are missing the backend
 * refuses the start with a 503 and this page renders the exact command that
 * fixes it, rather than a fabricated round.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  ArrowRight,
  Ban,
  Building2,
  CheckCircle2,
  Clock,
  Gauge,
  Layers,
  Loader2,
  Network,
  Pause,
  Play,
  Radio,
  RefreshCw,
  RotateCcw,
  Server,
  ShieldCheck,
  StepForward,
  Zap,
} from "lucide-react";

import { PageHeader } from "@/components/layout/AppLayout";
import { EmptyState, Section, StatCard } from "@/components/common";
import { Badge, StatusDot } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  useCacheReadiness,
  useFederatedControls,
  useFederatedState,
  useRoundDetail,
  useTrainingRounds,
} from "@/hooks/useFederated";
import { ApiError } from "@/services/api";
import type {
  AgentRoundState,
  RoundStatus,
  TrainingRound,
  TrainingStatus,
} from "@/types";
import { RESEARCH_DISCLAIMER } from "@/types";
import { cn, formatDateTime, formatDuration, formatMetric, formatNumber, formatPercent } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/* Metadata tables                                                     */
/* ------------------------------------------------------------------ */

type Variant = "neutral" | "info" | "warning" | "success" | "destructive";

const TRAINING_STATUS: Record<TrainingStatus, { label: string; variant: Variant; tone: string }> = {
  idle: { label: "Idle", variant: "neutral", tone: "slate" },
  running: { label: "Running", variant: "info", tone: "sky" },
  paused: { label: "Paused", variant: "warning", tone: "amber" },
  completed: { label: "Completed", variant: "success", tone: "emerald" },
  error: { label: "Error", variant: "destructive", tone: "red" },
};

const ROUND_STATUS: Record<RoundStatus, { label: string; variant: Variant }> = {
  pending: { label: "Pending", variant: "neutral" },
  running: { label: "Running", variant: "info" },
  aggregating: { label: "Aggregating", variant: "info" },
  completed: { label: "Completed", variant: "success" },
  failed: { label: "Failed", variant: "destructive" },
};

const PHASE: Record<string, { label: string; tone: string }> = {
  training: { label: "Local training", tone: "sky" },
  aggregating: { label: "Aggregating", tone: "accent" },
  synced: { label: "Synced", tone: "emerald" },
  completed: { label: "Completed", tone: "emerald" },
};

function phaseMeta(phase: string | null): { label: string; tone: string } {
  if (phase && phase in PHASE) return PHASE[phase];
  return { label: phase ?? "—", tone: "slate" };
}

/** Node status colour: live phase while running, otherwise the agent's own status. */
function agentTone(agent: AgentRoundState, running: boolean) {
  if (running) return phaseMeta(agent.phase).tone;
  switch (agent.status) {
    case "training":
      return "sky";
    case "synced":
      return "emerald";
    case "paused":
      return "amber";
    case "error":
      return "red";
    default:
      return "slate";
  }
}

/* ------------------------------------------------------------------ */
/* Element size probe (for the SVG connector geometry)                 */
/* ------------------------------------------------------------------ */

function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (rect) setSize({ w: rect.width, h: rect.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { ref, w: size.w, h: size.h };
}

/* ------------------------------------------------------------------ */
/* Topology                                                            */
/* ------------------------------------------------------------------ */

/** Node centres as fractions of the board, in render order. */
const NODE_POS: [number, number][] = [
  [0.5, 0.17],
  [0.88, 0.5],
  [0.5, 0.83],
  [0.12, 0.5],
];
const AGG = { fx: 0.5, fy: 0.5 };

function HospitalNode({
  agent,
  running,
  selected,
  onSelect,
}: {
  agent: AgentRoundState;
  running: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const meta = phaseMeta(agent.phase);
  return (
    <button
      onClick={onSelect}
      className={cn(
        "w-[160px] rounded-xl border bg-white/95 p-2.5 text-left shadow-soft transition-all",
        selected ? "border-accent-400 ring-2 ring-accent-300/60" : "border-ink-950/10 hover:border-accent-300",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-xs font-semibold text-ink-950">{agent.agent_name}</span>
        <StatusDot tone={agentTone(agent, running)} pulse={running && agent.phase !== "synced"} />
      </div>
      <p className="mt-0.5 font-mono text-[10px] text-ink-500">{agent.agent_id}</p>
      <div className="mt-1.5 flex items-center gap-1">
        <Badge variant="neutral" className="px-1.5 py-0 text-[9px]">
          {meta.label}
        </Badge>
      </div>
      <dl className="mt-2 grid grid-cols-3 gap-1 text-center">
        <div>
          <dt className="text-[8px] uppercase tracking-wider text-ink-500">N</dt>
          <dd className="text-[11px] font-semibold tabular-nums text-ink-900">
            {formatNumber(agent.samples)}
          </dd>
        </div>
        <div>
          <dt className="text-[8px] uppercase tracking-wider text-ink-500">Acc</dt>
          <dd className="text-[11px] font-semibold tabular-nums text-ink-900">
            {agent.local_accuracy === null ? "--" : formatPercent(agent.local_accuracy, 0)}
          </dd>
        </div>
        <div>
          <dt className="text-[8px] uppercase tracking-wider text-ink-500">Δ KB</dt>
          <dd className="text-[11px] font-semibold tabular-nums text-ink-900">
            {agent.update_size_kb === null ? "--" : agent.update_size_kb.toFixed(1)}
          </dd>
        </div>
      </dl>
      <p className="mt-1.5 flex items-center gap-1 text-[9px] text-emerald-700">
        <ShieldCheck className="h-2.5 w-2.5" /> images out: 0
      </p>
    </button>
  );
}

function Topology({
  agents,
  status,
  currentRound,
  globalVersion,
  selected,
  onSelect,
}: {
  agents: AgentRoundState[];
  status: TrainingStatus;
  currentRound: number;
  globalVersion: string | null;
  selected: string | null;
  onSelect: (agentId: string) => void;
}) {
  const { ref, w, h } = useElementSize<HTMLDivElement>();
  const running = status === "running";
  const center = useMemo(() => ({ x: w * AGG.fx, y: h * AGG.fy }), [w, h]);

  return (
    <Section
      title="Aggregator topology"
      description="Only weight-update tensors travel the wire. Each hospital trains locally and uploads a delta; the aggregator emails nothing back but the averaged global head."
      actions={
        <div className="flex items-center gap-3 text-[10px] text-ink-600">
          <span className="flex items-center gap-1">
            <span className="h-2 w-4 rounded-full bg-sky-400" /> upload delta
          </span>
          <span className="flex items-center gap-1">
            <span className="h-2 w-4 rounded-full bg-emerald-400" /> broadcast global
          </span>
        </div>
      }
    >
      <div className="overflow-x-auto">
        <div ref={ref} className="relative h-[420px] min-w-[720px]">
          {w > 0 && h > 0 && (
            <svg
              className="pointer-events-none absolute inset-0"
              width={w}
              height={h}
              viewBox={`0 0 ${w} ${h}`}
            >
              {agents.slice(0, NODE_POS.length).map((agent, i) => {
                const [fx, fy] = NODE_POS[i];
                const node = { x: fx * w, y: fy * h };
                const uploading = running && agent.phase !== "synced";
                const downloading = running && agent.phase === "synced";
                const stroke = uploading ? "#38bdf8" : downloading ? "#34d399" : "#c7c3dd";
                return (
                  <g key={agent.agent_id}>
                    <line
                      x1={node.x}
                      y1={node.y}
                      x2={center.x}
                      y2={center.y}
                      stroke={stroke}
                      strokeWidth={1.5}
                      strokeDasharray="5 5"
                      className={cn(running && "animate-dash-flow")}
                      opacity={running ? 0.9 : 0.5}
                    />
                    {uploading &&
                      [0, 1].map((k) => (
                        <motion.circle
                          key={k}
                          r={2.6}
                          fill="#38bdf8"
                          animate={{ cx: [node.x, center.x], cy: [node.y, center.y] }}
                          transition={{
                            duration: 1.3,
                            repeat: Infinity,
                            ease: "linear",
                            delay: k * 0.6,
                          }}
                        />
                      ))}
                    {downloading &&
                      [0, 1].map((k) => (
                        <motion.circle
                          key={k}
                          r={2.6}
                          fill="#34d399"
                          animate={{ cx: [center.x, node.x], cy: [center.y, node.y] }}
                          transition={{
                            duration: 1.3,
                            repeat: Infinity,
                            ease: "linear",
                            delay: k * 0.6,
                          }}
                        />
                      ))}
                  </g>
                );
              })}
            </svg>
          )}

          {/* Aggregator */}
          <div
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${AGG.fx * 100}%`, top: `${AGG.fy * 100}%` }}
          >
            <div className="w-[200px] rounded-2xl border border-accent-200 bg-gradient-to-br from-blush-50 via-white to-blue-50 p-3 text-center shadow-soft">
              <div className="flex justify-center">
                <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-blush-400 to-blue-500 text-white">
                  <Server className="h-4 w-4" />
                </div>
              </div>
              <p className="mt-1.5 text-sm font-semibold text-ink-950">Central Aggregator</p>
              <p className="font-mono text-[10px] text-ink-600">
                global {globalVersion ?? "—"}
              </p>
              <p className="text-[9px] uppercase tracking-wider text-ink-400">
                round {currentRound}
              </p>
              <Badge variant="info" className="mt-1">
                FedAvg · sample-weighted
              </Badge>
            </div>
          </div>

          {/* Hospitals */}
          {agents.slice(0, NODE_POS.length).map((agent, i) => {
            const [fx, fy] = NODE_POS[i];
            return (
              <div
                key={agent.agent_id}
                className="absolute -translate-x-1/2 -translate-y-1/2"
                style={{ left: `${fx * 100}%`, top: `${fy * 100}%` }}
              >
                <HospitalNode
                  agent={agent}
                  running={running}
                  selected={selected === agent.agent_id}
                  onSelect={() => onSelect(agent.agent_id)}
                />
              </div>
            );
          })}
        </div>
      </div>

      <p className="mt-3 border-t border-ink-950/10 pt-3 text-[10px] leading-relaxed text-ink-600">
        Each hospital keeps its images. The arrow carries a flat float32 weight delta for the
        4-class head (4,612 parameters); the return arrow carries the averaged global head. Both
        directions are written to the audit ledger with payload size and round number.
      </p>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Round controls                                                      */
/* ------------------------------------------------------------------ */

function RoundControls({
  status,
  ready,
  controls,
}: {
  status: TrainingStatus;
  ready: boolean;
  controls: ReturnType<typeof useFederatedControls>;
}) {
  const [count, setCount] = useState(5);
  const pending =
    controls.start.isPending ||
    controls.pause.isPending ||
    controls.nextRound.isPending ||
    controls.runRounds.isPending ||
    controls.reset.isPending;

  const actions: { label: string; fn: () => void; disabled: boolean; icon: typeof Play; variant?: "default" | "outline"; pending: boolean }[] = [
    {
      label: "Start",
      fn: () => controls.start.mutate(),
      disabled: !ready || status === "running",
      icon: Play,
      variant: "default",
      pending: controls.start.isPending,
    },
    {
      label: "Next round",
      fn: () => controls.nextRound.mutate(),
      disabled: !ready,
      icon: StepForward,
      variant: "outline",
      pending: controls.nextRound.isPending,
    },
    {
      label: "Pause",
      fn: () => controls.pause.mutate(),
      disabled: status !== "running",
      icon: Pause,
      variant: "outline",
      pending: controls.pause.isPending,
    },
    {
      label: "Reset",
      fn: () => controls.reset.mutate(),
      disabled: false,
      icon: RotateCcw,
      variant: "outline",
      pending: controls.reset.isPending,
    },
  ];

  const refused = [
    { name: "Start", m: controls.start },
    { name: "Next round", m: controls.nextRound },
    { name: "Run rounds", m: controls.runRounds },
    { name: "Pause", m: controls.pause },
    { name: "Reset", m: controls.reset },
  ].find((e) => e.m.isError);

  return (
    <Section
      title="Round control"
      description="Rounds run on a background worker. Pause drops queued rounds; the round in flight always finishes."
      actions={
        <Badge variant={TRAINING_STATUS[status].variant}>
          {status === "running" && <Radio className="h-3 w-3" />}
          {TRAINING_STATUS[status].label}
        </Badge>
      }
    >
      <div className="flex flex-wrap items-center gap-2">
        {actions.map((a) => (
          <Button
            key={a.label}
            size="sm"
            variant={a.variant}
            disabled={pending || a.disabled}
            onClick={a.fn}
          >
            {a.pending ? <Loader2 className="animate-spin" /> : <a.icon />}
            {a.label}
          </Button>
        ))}

        <div className="ml-auto flex items-center gap-2">
          <label className="text-[11px] text-ink-600" htmlFor="round-count">
            Run
          </label>
          <input
            id="round-count"
            type="number"
            min={1}
            max={50}
            value={count}
            onChange={(e) => setCount(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
            className="h-8 w-16 rounded-md border border-ink-950/15 bg-white px-2 text-sm tabular-nums text-ink-900 outline-none focus:border-accent-400"
          />
          <Button
            size="sm"
            variant="outline"
            disabled={pending || !ready}
            onClick={() => controls.runRounds.mutate(count)}
          >
            {controls.runRounds.isPending ? <Loader2 className="animate-spin" /> : <Layers />}
            rounds
          </Button>
        </div>
      </div>

      {refused && (
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 p-3">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-500" />
          <div className="min-w-0">
            <p className="text-xs font-medium text-red-700">{refused.name} refused</p>
            <p className="mt-0.5 break-words text-[11px] leading-relaxed text-red-600">
              {refused.m.error instanceof ApiError
                ? refused.m.error.message
                : "The request failed — check the backend log."}
            </p>
          </div>
        </div>
      )}
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Live agent table                                                    */
/* ------------------------------------------------------------------ */

function AgentLiveTable({
  agents,
  selected,
  onSelect,
}: {
  agents: AgentRoundState[];
  selected: string | null;
  onSelect: (agentId: string) => void;
}) {
  const totalKb = agents.reduce((s, a) => s + (a.update_size_kb ?? 0), 0);

  return (
    <Section
      title="Per-hospital state"
      description="Live local metrics during a round, falling back to the last completed round's values."
      actions={<Badge variant="neutral">{formatNumber(totalKb, 1)} KB uploaded</Badge>}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-lilac-50 text-[10px] uppercase tracking-wider text-ink-500">
            <tr>
              <th className="px-3 py-2 font-medium">Hospital</th>
              <th className="px-3 py-2 font-medium">Phase</th>
              <th className="px-3 py-2 text-right font-medium">Samples</th>
              <th className="px-3 py-2 text-right font-medium">Loss</th>
              <th className="px-3 py-2 text-right font-medium">Acc</th>
              <th className="px-3 py-2 text-right font-medium">Recall</th>
              <th className="px-3 py-2 text-right font-medium">F1</th>
              <th className="px-3 py-2 text-right font-medium">Δ norm</th>
              <th className="px-3 py-2 text-right font-medium">Update</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-950/10">
            {agents.map((a) => {
              const meta = phaseMeta(a.phase);
              return (
                <tr
                  key={a.agent_id}
                  onClick={() => onSelect(a.agent_id)}
                  className={cn(
                    "cursor-pointer transition-colors",
                    selected === a.agent_id ? "bg-accent-50" : "bg-white hover:bg-lilac-50/60",
                  )}
                >
                  <td className="px-3 py-2">
                    <p className="font-medium text-ink-950">{a.agent_name}</p>
                    <p className="font-mono text-[10px] text-ink-500">{a.agent_id}</p>
                  </td>
                  <td className="px-3 py-2">
                    <span className="inline-flex items-center gap-1.5 text-xs text-ink-700">
                      <StatusDot tone={meta.tone} />
                      {meta.label}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                    {formatNumber(a.samples)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                    {formatMetric(a.local_loss)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                    {formatPercent(a.local_accuracy)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                    {formatPercent(a.local_recall)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                    {formatPercent(a.local_f1)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                    {formatMetric(a.weight_delta_norm)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                    {a.update_size_kb === null ? "--" : `${a.update_size_kb.toFixed(2)} KB`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Round history + detail                                              */
/* ------------------------------------------------------------------ */

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-ink-950/10 bg-white p-2.5">
      <p className="text-[9px] uppercase tracking-wider text-ink-500">{label}</p>
      <p className="mt-0.5 text-sm font-semibold tabular-nums text-ink-950">{value}</p>
    </div>
  );
}

function RoundHistory({
  rounds,
  selectedRound,
  onSelect,
}: {
  rounds: TrainingRound[];
  selectedRound: number | null;
  onSelect: (round: number) => void;
}) {
  const detail = useRoundDetail(selectedRound);

  return (
    <>
      <Section
        title="Round history"
        description="One row per completed aggregation, written to the database by the round worker."
        actions={<Badge variant="neutral">{rounds.length} recorded</Badge>}
      >
        {rounds.length === 0 ? (
          <EmptyState
            icon={Network}
            title="No federated rounds yet"
            description="Start a round above. Each completed round is recorded here with its global metrics and per-hospital contributions."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-lilac-50 text-[10px] uppercase tracking-wider text-ink-500">
                <tr>
                  <th className="px-3 py-2 font-medium">#</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 text-right font-medium">Hosts</th>
                  <th className="px-3 py-2 font-medium">Global model</th>
                  <th className="px-3 py-2 text-right font-medium">Acc</th>
                  <th className="px-3 py-2 text-right font-medium">Prec</th>
                  <th className="px-3 py-2 text-right font-medium">Recall</th>
                  <th className="px-3 py-2 text-right font-medium">F1</th>
                  <th className="px-3 py-2 text-right font-medium">AUC</th>
                  <th className="px-3 py-2 text-right font-medium">Duration</th>
                  <th className="px-3 py-2 text-right font-medium">Samples</th>
                  <th className="px-3 py-2 font-medium">Completed</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-950/10">
                {rounds.map((r) => (
                  <tr
                    key={r.id}
                    onClick={() => onSelect(r.round_number)}
                    className={cn(
                      "cursor-pointer transition-colors",
                      selectedRound === r.round_number
                        ? "bg-accent-50"
                        : "bg-white hover:bg-lilac-50/60",
                    )}
                  >
                    <td className="px-3 py-2 font-mono text-xs font-medium text-ink-900">
                      {r.round_number}
                    </td>
                    <td className="px-3 py-2">
                      <Badge variant={ROUND_STATUS[r.status].variant}>
                        {ROUND_STATUS[r.status].label}
                      </Badge>
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                      {r.participating_agents}
                    </td>
                    <td className="px-3 py-2 font-mono text-xs text-ink-700">
                      {r.global_model_version ?? "--"}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                      {formatPercent(r.global_accuracy)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                      {formatPercent(r.global_precision)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                      {formatPercent(r.global_recall)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                      {formatPercent(r.global_f1)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                      {formatMetric(r.global_auc)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                      {formatDuration(r.duration_seconds)}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                      {formatNumber(r.samples_processed)}
                    </td>
                    <td className="px-3 py-2 text-xs text-ink-500">
                      {formatDateTime(r.completed_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <p className="mt-3 border-t border-ink-950/10 pt-3 text-[10px] leading-relaxed text-ink-600">
          Global metrics are scored once on the untouched validation split. A blank cell means the
          value was never produced — for example no validation cache exists yet — and is left
          empty rather than filled with a stand-in number.
        </p>
      </Section>

      {selectedRound !== null && (
        <Section
          title={`Round ${selectedRound} detail`}
          description="Per-hospital local metrics and the aggregate they produced, read from the persisted round row."
          actions={
            <Button size="sm" variant="ghost" onClick={() => onSelect(selectedRound)}>
              <RefreshCw /> Refresh
            </Button>
          }
        >
          {detail.isLoading ? (
            <EmptyState icon={Loader2} title="Loading round detail…" />
          ) : detail.isError || !detail.data ? (
            <EmptyState
              variant="error"
              icon={AlertTriangle}
              title="Could not load round detail"
              description={
                detail.error instanceof Error ? detail.error.message : "Unexpected payload."
              }
            />
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
                <Metric label="Accuracy" value={formatPercent(detail.data.round.global_accuracy)} />
                <Metric label="Precision" value={formatPercent(detail.data.round.global_precision)} />
                <Metric label="Recall" value={formatPercent(detail.data.round.global_recall)} />
                <Metric label="F1" value={formatPercent(detail.data.round.global_f1)} />
                <Metric label="AUC" value={formatMetric(detail.data.round.global_auc)} />
                <Metric label="Global loss" value={formatMetric(detail.data.round.global_loss)} />
              </div>

              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-lilac-50 text-[10px] uppercase tracking-wider text-ink-500">
                    <tr>
                      <th className="px-3 py-2 font-medium">Hospital</th>
                      <th className="px-3 py-2 text-right font-medium">Samples</th>
                      <th className="px-3 py-2 text-right font-medium">Loss</th>
                      <th className="px-3 py-2 text-right font-medium">Acc</th>
                      <th className="px-3 py-2 text-right font-medium">Recall</th>
                      <th className="px-3 py-2 text-right font-medium">F1</th>
                      <th className="px-3 py-2 text-right font-medium">Δ norm</th>
                      <th className="px-3 py-2 text-right font-medium">Update</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-ink-950/10">
                    {detail.data.agents.map((a) => (
                      <tr key={a.agent_id} className="bg-white">
                        <td className="px-3 py-2">
                          <span className="font-medium text-ink-950">{a.agent_name}</span>{" "}
                          <span className="font-mono text-[10px] text-ink-500">{a.agent_id}</span>
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                          {formatNumber(a.samples)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                          {formatMetric(a.local_loss)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                          {formatPercent(a.local_accuracy)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                          {formatPercent(a.local_recall)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                          {formatPercent(a.local_f1)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                          {formatMetric(a.weight_delta_norm)}
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums text-ink-800">
                          {a.update_size_kb === null ? "--" : `${a.update_size_kb.toFixed(2)} KB`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </Section>
      )}
    </>
  );
}

/* ------------------------------------------------------------------ */
/* Readiness gate                                                      */
/* ------------------------------------------------------------------ */

function ReadinessPanel({
  readiness,
  onRetry,
}: {
  readiness: ReturnType<typeof useCacheReadiness>;
  onRetry: () => void;
}) {
  if (readiness.isLoading) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-ink-950/10 bg-white p-3 text-[11px] text-ink-600">
        <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking for real training caches…
      </div>
    );
  }

  if (readiness.isError) {
    return (
      <div className="flex items-start gap-2 rounded-xl border border-red-300 bg-red-50 p-3">
        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-500" />
        <div className="flex-1">
          <p className="text-xs font-medium text-red-700">Could not read backend readiness</p>
          <p className="mt-0.5 text-[11px] text-red-600">
            The API status endpoint is unreachable.
          </p>
        </div>
        <Button size="sm" variant="outline" onClick={onRetry}>
          <RefreshCw /> Retry
        </Button>
      </div>
    );
  }

  if (readiness.ready) {
    return (
      <div className="flex items-start gap-2 rounded-xl border border-emerald-300/70 bg-emerald-50 p-3">
        <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
        <div className="text-[11px] leading-relaxed text-emerald-800">
          <p className="font-medium text-emerald-900">
            Ready — {readiness.presentTrain.length}/4 hospital train caches present
            {readiness.valReady ? ", validation split present" : ", no validation split"}
          </p>
          <p className="mt-0.5">
            Rounds train on real embeddings.{" "}
            {readiness.valReady
              ? "Global metrics are scored on the held-out validation split."
              : "Without a validation cache global metrics stay blank."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-2 rounded-xl border border-amber-300 bg-amber-50 p-3">
      <Ban className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-amber-900">
          No agent embedding caches found — nothing real to train on
        </p>
        <p className="mt-0.5 text-[11px] leading-relaxed text-amber-800">
          A round would be refused with a 503 rather than simulated. Build the partitions and
          extract the embeddings first:
        </p>
        <pre className="mt-2 overflow-x-auto rounded-lg bg-ink-950/90 p-2.5 font-mono text-[10px] leading-relaxed text-ink-50">
          {".\\backend-venv\\Scripts\\python.exe scripts/build_splits.py\n" +
            ".\\backend-venv\\Scripts\\python.exe scripts/extract_embeddings.py --split train"}
        </pre>
        {readiness.missingTrain.length > 0 && (
          <p className="mt-2 font-mono text-[10px] text-amber-700">
            missing: {readiness.missingTrain.join(", ")}
          </p>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function FederatedTraining() {
  const state = useFederatedState();
  const rounds = useTrainingRounds();
  const readiness = useCacheReadiness();
  const controls = useFederatedControls();
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [selectedRound, setSelectedRound] = useState<number | null>(null);

  if (state.isError || !state.data) {
    if (state.isLoading) {
      return (
        <div className="space-y-5">
          <PageHeader
            title="Federated Training"
            description="Loading live round state…"
            icon={Network}
          />
          <div className="glass-panel h-64 animate-pulse" />
        </div>
      );
    }
    return (
      <div className="space-y-5">
        <PageHeader title="Federated Training" icon={Network} />
        <EmptyState
          variant="error"
          icon={AlertTriangle}
          title="Backend is not reachable"
          description={
            state.error instanceof Error
              ? state.error.message
              : "Start the FastAPI server with 'python -m uvicorn app.main:app --app-dir backend --reload' from the repository root."
          }
          action={
            <Button variant="outline" onClick={() => void state.refetch()}>
              <RefreshCw /> Retry
            </Button>
          }
        />
      </div>
    );
  }

  const s = state.data;
  const meta = TRAINING_STATUS[s.status];
  const roundsData = rounds.data ?? [];
  const totalKb = s.agents.reduce((acc, a) => acc + (a.update_size_kb ?? 0), 0);

  return (
    <div className="space-y-5">
      <PageHeader
        title="Federated Training"
        description="Queue FedAvg rounds across four hospital agents and watch them run. Hospital images never move; only sample-weighted head deltas are aggregated."
        icon={Network}
        actions={
          <>
            <Badge variant={meta.variant}>
              {s.status === "running" && <Radio className="h-3 w-3" />}
              {meta.label}
            </Badge>
            <Badge variant="neutral">Round {s.current_round}</Badge>
          </>
        }
      />

      <ReadinessPanel readiness={readiness} onRetry={() => void readiness.refetch()} />

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Current round"
          value={s.current_round}
          hint={`${roundsData.length} recorded in the database`}
          icon={Network}
          tone="accent"
        />
        <StatCard
          label="Global model"
          value={s.global_model_version ?? "—"}
          hint="latest aggregated head"
          icon={Server}
          tone="emerald"
        />
        <StatCard
          label="Queue state"
          value={meta.label}
          hint={`aggregation: ${s.aggregation_status}`}
          icon={Gauge}
          tone={s.status === "error" ? "red" : s.status === "running" ? "violet" : "slate"}
        />
        <StatCard
          label="Updates uploaded"
          value={`${formatNumber(totalKb, 1)} KB`}
          hint="weight deltas only — no images"
          icon={Zap}
          tone="violet"
        />
      </div>

      <RoundControls status={s.status} ready={readiness.ready} controls={controls} />

      <Topology
        agents={s.agents}
        status={s.status}
        currentRound={s.current_round}
        globalVersion={s.global_model_version}
        selected={selectedAgent}
        onSelect={(id) => setSelectedAgent((cur) => (cur === id ? null : id))}
      />

      <AgentLiveTable
        agents={s.agents}
        selected={selectedAgent}
        onSelect={(id) => setSelectedAgent((cur) => (cur === id ? null : id))}
      />

      <RoundHistory
        rounds={roundsData}
        selectedRound={selectedRound}
        onSelect={(n) => setSelectedRound((cur) => (cur === n ? null : n))}
      />

      <Section
        title="What this dashboard never shows"
        description="The honesty rules the federated engine is built around."
        tone="dark"
        actions={<Badge variant="success">images shared: 0</Badge>}
      >
        <ul className="space-y-2 text-xs text-ink-700">
          {[
            "No simulated round. Missing caches refuse the start with a 503 and the fix command.",
            "No fabricated metric. Global scores stay blank until a real validation pass produces them.",
            "No raw image or patient identifier in any payload — only the 4-class head's weight delta.",
            "No rewritten history. Round rows and the audit ledger are append-only; reset drops live state only.",
          ].map((t) => (
            <li key={t} className="flex items-start gap-2">
              <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
              <span>{t}</span>
            </li>
          ))}
        </ul>
        <p className="mt-3 flex items-center gap-1.5 border-t border-ink-950/10 pt-3 text-[10px] text-ink-500">
          <Building2 className="h-3 w-3" /> Four isolated hospital partitions ·
          <ArrowRight className="h-3 w-3" /> aggregator
          <Clock className="ml-auto" /> polled every {s.status === "running" ? "1s" : "4s"}
        </p>
        <p className="mt-2 text-[10px] leading-relaxed text-ink-500">{RESEARCH_DISCLAIMER}</p>
      </Section>
    </div>
  );
}
