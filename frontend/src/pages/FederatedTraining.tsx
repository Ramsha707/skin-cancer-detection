/**
 * Week 7/8 — federated training dashboard.
 *
 * The topology diagram is the story: one aggregator, four hospitals, and a single
 * arrow label between each pair carrying `param_kb`. The page renders that shape
 * and states the invariant in words, because "no images leave the hospital" is a
 * claim that deserves to be visible rather than buried in a readme.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  Building2,
  CheckCircle2,
  Database,
  History,
  Loader2,
  Network,
  PlayCircle,
  Radio,
  Server,
  ShieldCheck,
  Users,
  Weight,
} from "lucide-react";

import { api } from "@/services/api";
import {
  CANCER_CLASSES,
  type FederatedStatus,
  type HospitalAgent,
  type TrainingRound,
} from "@/types";
import { EmptyState, Section, StatCard } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  cn,
  formatDateTime,
  formatDuration,
  formatNumber,
  formatPercent,
} from "@/lib/utils";

const AGENT_TONES: Record<string, string> = {
  agent_a: "#38bdf8",
  agent_b: "#a78bfa",
  agent_c: "#fbbf24",
  agent_d: "#34d399",
};

export default function FederatedTraining() {
  const qc = useQueryClient();
  const status = useQuery({ queryKey: ["federated-status"], queryFn: api.federatedStatus });
  const agents = useQuery({ queryKey: ["agents"], queryFn: api.listAgents });
  const rounds = useQuery({ queryKey: ["rounds"], queryFn: api.listRounds });
  const audit = useQuery({ queryKey: ["audit"], queryFn: () => api.listAuditLogs(40) });

  const runRound = useMutation({
    mutationFn: api.runRound,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["rounds"] });
      void qc.invalidateQueries({ queryKey: ["federated-status"] });
      void qc.invalidateQueries({ queryKey: ["agents"] });
      void qc.invalidateQueries({ queryKey: ["models"] });
      void qc.invalidateQueries({ queryKey: ["audit"] });
      void qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });

  const st = status.data;
  const ready = st?.ready ?? false;
  const list = agents.data ?? [];
  const roundList = rounds.data ?? [];
  const completedRounds = roundList.filter((r) => r.status === "completed");
  const latest = completedRounds[0] ?? roundList[0] ?? null;
  const latestModel = roundList.find((r) => r.global_model_version) ?? null;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-white">
            Federated Training Dashboard
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-navy-400">
            FedAvg over four hospital agents. Each round: every site trains its 4,612-parameter
            head on its own cached embeddings, uploads only that weight delta, and the
            aggregator takes a sample-count weighted mean.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={ready ? "success" : "warning"}>
            {ready ? (
              <>
                <CheckCircle2 className="h-3 w-3" /> Caches ready
              </>
            ) : (
              <>
                <AlertTriangle className="h-3 w-3" /> Not ready
              </>
            )}
          </Badge>
          <Button onClick={() => runRound.mutate()} disabled={runRound.isPending || !ready}>
            {runRound.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Running round
              </>
            ) : (
              <>
                <PlayCircle className="h-4 w-4" /> Run one round
              </>
            )}
          </Button>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Rounds completed"
          value={completedRounds.length}
          hint={latest ? `last: round ${latest.round_number}` : "none yet"}
          icon={Network}
          tone="accent"
        />
        <StatCard
          label="Payload per agent"
          value={st ? `${st.param_kb} KB` : "--"}
          hint={`${formatNumber(st?.param_count)} float32s, nothing else`}
          icon={Weight}
          tone="violet"
        />
        <StatCard
          label="Global accuracy"
          value={latest ? formatPercent(latest.global_accuracy) : "--"}
          hint={latestModel?.global_model_version ?? "no model version yet"}
          icon={Radio}
          tone={latest ? "emerald" : "slate"}
        />
        <StatCard
          label="Agents participating"
          value={`${latest?.participating_agents ?? 0}/${list.length || 4}`}
          hint="last round"
          icon={Users}
          tone="emerald"
        />
      </div>

      {!ready && st?.instructions && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/25 bg-amber-500/10 p-4">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-amber-200">
              A round cannot run on this checkout yet
            </p>
            <p className="mt-1 break-words text-xs leading-relaxed text-amber-200/70">
              {st.instructions}
            </p>
          </div>
        </div>
      )}

      {runRound.isError && (
        <div className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-4">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
          <p className="text-xs leading-relaxed text-red-200">
            The round failed: {runRound.error.message}
          </p>
        </div>
      )}

      {/* ------------------------------------------------ topology */}
      <Section
        title="Aggregator topology"
        description="Only parameter tensors traverse the dashed links. No pixel array, path or patient identifier has a code path across them."
      >
        <Topology agents={list} status={st} latest={latest} />
      </Section>

      {/* ------------------------------------------------ agent roster */}
      <Section
        title="Hospital agents"
        description="Each agent holds a disjoint patient-level partition. Distributions are read from the backend, not generated here."
      >
        {agents.isLoading ? (
          <p className="text-xs text-navy-500">Loading agents...</p>
        ) : agents.isError ? (
          <EmptyState
            variant="error"
            icon={AlertTriangle}
            title="Could not load agents"
            description={agents.error.message}
          />
        ) : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {list.map((a) => (
              <AgentCard key={a.id} agent={a} />
            ))}
          </div>
        )}
      </Section>

      {/* ------------------------------------------------ rounds + audit */}
      <div className="grid gap-5 xl:grid-cols-2">
        <Section
          title="Round history"
          description="Newest first. Global metrics come from the aggregated head evaluated on the validation split."
        >
          {rounds.isLoading ? (
            <p className="text-xs text-navy-500">Loading rounds...</p>
          ) : roundList.length === 0 ? (
            <EmptyState
              icon={History}
              variant="awaiting"
              title="No rounds recorded"
              description="Run one round above. At this head size a round is seconds of numpy work, not hours of GPU time."
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[520px] text-left text-xs">
                <thead>
                  <tr className="border-b border-white/10 text-[10px] uppercase tracking-wider text-navy-500">
                    <th className="pb-2 pr-3 font-medium">Round</th>
                    <th className="pb-2 pr-3 font-medium">Status</th>
                    <th className="pb-2 pr-3 text-right font-medium">Acc</th>
                    <th className="pb-2 pr-3 text-right font-medium">Macro F1</th>
                    <th className="pb-2 pr-3 text-right font-medium">Loss</th>
                    <th className="pb-2 text-right font-medium">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {roundList.map((r) => (
                    <tr key={r.id} className="border-b border-white/5">
                      <td className="py-2.5 pr-3">
                        <p className="font-medium text-white">#{r.round_number}</p>
                        <p className="text-[10px] text-navy-600">
                          {r.global_model_version ?? formatDateTime(r.completed_at ?? r.started_at)}
                        </p>
                      </td>
                      <td className="py-2.5 pr-3">
                        <Badge
                          variant={
                            r.status === "completed"
                              ? "success"
                              : r.status === "failed"
                                ? "destructive"
                                : "neutral"
                          }
                        >
                          {r.status}
                        </Badge>
                      </td>
                      <td className="py-2.5 pr-3 text-right tabular-nums text-navy-200">
                        {r.global_accuracy === null ? "—" : formatPercent(r.global_accuracy)}
                      </td>
                      <td className="py-2.5 pr-3 text-right tabular-nums text-navy-200">
                        {r.global_f1 === null ? "—" : formatPercent(r.global_f1)}
                      </td>
                      <td className="py-2.5 pr-3 text-right tabular-nums text-navy-400">
                        {r.global_loss === null ? "—" : r.global_loss.toFixed(4)}
                      </td>
                      <td className="py-2.5 text-right tabular-nums text-navy-400">
                        {r.duration_seconds ? formatDuration(r.duration_seconds) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        <Section
          title="Transmission ledger"
          description="Every message that crossed a hospital boundary, with direction and payload size."
        >
          {audit.isLoading ? (
            <p className="text-xs text-navy-500">Loading ledger...</p>
          ) : (audit.data?.length ?? 0) === 0 ? (
            <EmptyState
              icon={ShieldCheck}
              variant="awaiting"
              title="Ledger empty"
              description="Entries appear the moment a round starts. This is the audit trail the privacy counters are derived from."
            />
          ) : (
            <ul className="max-h-[420px] space-y-1.5 overflow-y-auto pr-1">
              {audit.data!.map((log) => (
                <li
                  key={log.id}
                  className="flex items-start gap-3 rounded-lg border border-white/10 bg-navy-950/40 px-3 py-2"
                >
                  <span
                    className={cn(
                      "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full",
                      log.status === "success" ? "bg-emerald-400" : "bg-amber-400",
                    )}
                    aria-hidden
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-[11px] text-navy-200">{log.message}</p>
                    <p className="mt-0.5 flex flex-wrap gap-x-2 text-[10px] text-navy-600">
                      <span className="font-mono">{log.event_type}</span>
                      {log.agent_id && <span>{log.agent_id}</span>}
                      {log.round_number && <span>round {log.round_number}</span>}
                      <span>{formatDateTime(log.timestamp)}</span>
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Section>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Topology                                                            */
/* ------------------------------------------------------------------ */

function Topology({
  agents,
  status,
  latest,
}: {
  agents: HospitalAgent[];
  status: FederatedStatus | undefined;
  latest: TrainingRound | null;
}) {
  const list = agents.length > 0 ? agents : null;

  return (
    <div className="relative">
      <svg className="hidden h-2 w-full" aria-hidden>
        <defs>
          <linearGradient id="link" x1="0" x2="1">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.15" />
          </linearGradient>
        </defs>
      </svg>

      <div className="flex flex-col items-center gap-6">
        {/* aggregator */}
        <div className="w-full max-w-md">
          <div className="rounded-xl border border-accent-400/30 bg-accent-500/10 p-4 text-center">
            <Server className="mx-auto h-5 w-5 text-accent-300" />
            <p className="mt-1.5 text-sm font-semibold text-white">
              Central Aggregator
            </p>
            <p className="mt-0.5 text-[11px] text-navy-400">
              Holds one global head, {status?.param_count ? formatNumber(status.param_count) : "4,612"} params.
              Never holds an image.
            </p>
            {latest && (
              <Badge variant="info" className="mt-2">
                {latest.global_model_version ?? "no version yet"}
              </Badge>
            )}
          </div>
        </div>

        {/* links */}
        <div className="relative w-full" style={{ height: 44 }}>
          <div className="absolute inset-x-[12%] top-0 h-px border-t border-dashed border-accent-400/25" />
          <div className="absolute inset-x-[12%] top-0 h-8 w-px bg-gradient-to-b from-accent-400/30 to-accent-400/10" />
          <div className="absolute inset-x-[12.5%] top-8 h-px bg-accent-400/20" />
          {list && (
            <p className="absolute left-1/2 top-3 -translate-x-1/2 rounded-full border border-accent-400/25 bg-navy-950 px-2.5 py-0.5 text-[9px] uppercase tracking-wider text-accent-300">
              {status?.param_kb ?? 18.016} KB each · weights only
            </p>
          )}
        </div>

        {/* agents */}
        <div className="grid w-full gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {list
            ? list.map((a) => (
                <div
                  key={a.id}
                  className="relative rounded-xl border border-white/10 bg-navy-950/50 p-3"
                >
                  <div
                    className="absolute inset-x-0 top-0 h-0.5 rounded-t-xl"
                    style={{ background: AGENT_TONES[a.agent_id] ?? "#64748b" }}
                  />
                  <div className="flex items-center gap-2">
                    <Building2 className="h-3.5 w-3.5 text-navy-400" />
                    <p className="truncate text-xs font-semibold text-white">{a.name}</p>
                  </div>
                  <p className="mt-0.5 truncate text-[10px] text-navy-500">{a.location}</p>
                  <div className="mt-2 flex items-center justify-between text-[10px]">
                    <span className="text-navy-500">{formatNumber(a.dataset_size)} images</span>
                    <Badge variant={a.privacy_status === "protected" ? "success" : "destructive"}>
                      {a.privacy_status}
                    </Badge>
                  </div>
                </div>
              ))
            : Array.from({ length: 4 }).map((_, i) => (
                <div
                  key={i}
                  className="rounded-xl border border-dashed border-white/10 bg-navy-950/30 p-3"
                >
                  <div className="h-2.5 w-20 animate-pulse rounded bg-white/10" />
                  <div className="mt-2 h-2 w-28 animate-pulse rounded bg-white/5" />
                </div>
              ))}
        </div>
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-3">
        <Invariant
          icon={ShieldCheck}
          label="Raw images shared"
          value="0"
          note="The aggregator calls fit(), update() and load() on each agent. All three exchange tensors, so no image can reach the server."
        />
        <Invariant
          icon={Weight}
          label="Payload per agent per round"
          value={status ? `${status.param_kb} KB` : "--"}
          note={`${formatNumber(status?.param_count)} float32 values: one 1,152-wide weight row and bias per class.`}
        />
        <Invariant
          icon={Database}
          label="Aggregation rule"
          value={status?.aggregation ?? "FedAvg"}
          note="Weighted by each agent's sample count, so a 2,049-image site counts for more than a 1,413-image site."
        />
      </div>
    </div>
  );
}

function Invariant({
  icon: Icon,
  label,
  value,
  note,
}: {
  icon: typeof ShieldCheck;
  label: string;
  value: string;
  note: string;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-navy-950/40 p-3">
      <div className="flex items-center gap-2">
        <Icon className="h-3.5 w-3.5 text-emerald-400" />
        <p className="text-[10px] uppercase tracking-wider text-navy-500">{label}</p>
      </div>
      <p className="mt-1 text-sm font-semibold text-white">{value}</p>
      <p className="mt-1 text-[10px] leading-relaxed text-navy-500">{note}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Agent card                                                          */
/* ------------------------------------------------------------------ */

function AgentCard({ agent }: { agent: HospitalAgent }) {
  const total = Object.values(agent.class_distribution).reduce((a, b) => a + b, 0);
  const tone = AGENT_TONES[agent.agent_id] ?? "#64748b";

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative overflow-hidden rounded-xl border border-white/10 bg-navy-950/50 p-4"
    >
      <div className="absolute inset-x-0 top-0 h-0.5" style={{ background: tone }} />
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-white">{agent.name}</p>
          <p className="truncate text-[10px] text-navy-500">{agent.location}</p>
        </div>
        <Badge
          variant={
            agent.status === "training"
              ? "info"
              : agent.status === "error"
                ? "destructive"
                : "neutral"
          }
        >
          {agent.status}
        </Badge>
      </div>

      <div className="mt-3 space-y-1.5">
        {CANCER_CLASSES.map((c) => {
          const n = agent.class_distribution[c] ?? 0;
          const frac = total > 0 ? (n / total) * 100 : 0;
          return (
            <div key={c}>
              <div className="mb-0.5 flex justify-between text-[10px]">
                <span className="text-navy-400">{c.replace(/_/g, " ")}</span>
                <span className="tabular-nums text-navy-500">{n}</span>
              </div>
              <div className="h-1 w-full overflow-hidden rounded-full bg-white/10">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{ width: `${frac}%`, background: tone }}
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-3 flex items-center justify-between border-t border-white/10 pt-2 text-[10px] text-navy-500">
        <span>{formatNumber(agent.dataset_size)} images</span>
        <span>
          {agent.rounds_participated} rounds · {agent.local_epochs} local epochs
        </span>
      </div>
    </motion.div>
  );
}