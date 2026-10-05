/**
 * Week 6 — fine-tuning experiments.
 *
 * The comparison is the point of this page, and so is the honesty about it. Three
 * strategies are always named; the ones that did not run say why. A skipped arm
 * renders as skipped with its reason attached, never as a zero and never as a
 * missing row that looks like a bug.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Cpu,
  FlaskConical,
  Gauge,
  Layers,
  Loader2,
  MemoryStick,
  PlayCircle,
  SkipForward,
  Zap,
} from "lucide-react";

import { api } from "@/services/api";
import {
  type Experiment,
  type StrategyKind,
  RESEARCH_DISCLAIMER,
} from "@/types";
import { EmptyState, Section, StatCard } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, formatNumber, formatPercent } from "@/lib/utils";

const STRATEGY_META: Record<
  StrategyKind,
  { label: string; tone: string; blurb: string }
> = {
  frozen: {
    label: "Frozen backbone",
    tone: "sky",
    blurb:
      "Entire 878M-parameter vision tower frozen. Only the 4-class head trains. Cheapest per round and the baseline every other arm must beat.",
  },
  selective: {
    label: "Selective unfreeze",
    tone: "violet",
    blurb:
      "Last vision blocks unfrozen at a 100x smaller learning rate. The early blocks encode the edge and texture operators dermoscopy still depends on, so they stay put.",
  },
  full: {
    label: "Full fine-tune",
    tone: "red",
    blurb:
      "All 878M parameters trainable. Needs roughly 17.5 GB of optimiser state and is refused outright without a CUDA device.",
  },
};

const METRICS = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "specificity", label: "Specificity" },
  { key: "f1", label: "Macro F1" },
  { key: "auc", label: "AUC" },
] as const;

export default function Experiments() {
  const qc = useQueryClient();
  const comparison = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const status = useQuery({ queryKey: ["detection-status"], queryFn: api.detectionStatus });

  const run = useMutation({
    mutationFn: api.runExperiments,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["experiments"] });
      void qc.invalidateQueries({ queryKey: ["models"] });
      void qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });

  const budget = comparison.data?.budget;
  const caches = status.data?.caches_ready ?? comparison.data?.caches_ready ?? {};
  const haveTest = caches.test === true;
  const haveTrain = Object.entries(caches)
    .filter(([k]) => k.startsWith("train_"))
    .some(([, v]) => v);

  const data = comparison.data;
  const results = data?.experiments ?? [];
  const completed = results.filter((e) => e.status === "completed");
  const best = pickBest(completed);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-white">
            Fine-Tuning &amp; Experiments
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-navy-400">
            Frozen linear probe against selective unfreezing against a full fine-tune.
            Every arm reports accuracy, precision, recall, specificity, macro-F1, AUC,
            trainable parameters and wall-clock cost — and every arm that did not run
            says why.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={budget?.cuda_available ? "success" : "warning"}>
            {budget?.cuda_available ? "CUDA available" : "CPU only"}
          </Badge>
          <Button
            onClick={() => run.mutate()}
            disabled={run.isPending || !haveTrain || !haveTest}
          >
            {run.isPending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" /> Running
              </>
            ) : (
              <>
                <PlayCircle className="h-4 w-4" /> Run comparison
              </>
            )}
          </Button>
        </div>
      </div>

      {/* ------------------------------------------------ host budget */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="CUDA device"
          value={budget?.cuda_available ? "Yes" : "No"}
          hint={budget?.cuda_available ? "Full fine-tune allowed" : "Full fine-tune refused"}
          icon={Zap}
          tone={budget?.cuda_available ? "emerald" : "amber"}
        />
        <StatCard
          label="System RAM"
          value={budget ? `${budget.total_ram_gb} GB` : "--"}
          hint="measured, not assumed"
          icon={MemoryStick}
          tone="accent"
        />
        <StatCard
          label="Completed arms"
          value={`${completed.length}/${3}`}
          hint={data?.status === "not_run" ? "nothing run yet" : "from persisted results"}
          icon={CheckCircle2}
          tone={completed.length === 3 ? "emerald" : "amber"}
        />
        <StatCard
          label="Best macro-F1"
          value={best ? formatPercent(best.f1) : "--"}
          hint={best ? best.strategy ?? "strategy" : "awaiting a run"}
          icon={BarChart3}
          tone={best ? "emerald" : "slate"}
        />
      </div>

      {!haveTest && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/25 bg-amber-500/10 p-4">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-amber-200">
              The comparison cannot run yet
            </p>
            <p className="mt-1 text-xs leading-relaxed text-amber-200/70">
              Both arms are real measurements, so the run button stays disabled until the
              train and test embedding caches exist. Extract them with{" "}
              <code className="text-amber-100">
                .\.venv-ov\Scripts\python.exe scripts/extract_embeddings.py --split train
              </code>{" "}
              and the same command with{" "}
              <code className="text-amber-100">--split test</code>. Nothing below is
              simulated in the meantime.
            </p>
          </div>
        </div>
      )}

      {run.isError && (
        <div className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-4">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
          <p className="text-xs leading-relaxed text-red-200">
            The run failed: {run.error.message}
          </p>
        </div>
      )}

      {/* ------------------------------------------------ results or plan */}
      {results.length > 0 ? (
        <Section
          title="Strategy comparison"
          description={
            data?.status === "complete"
              ? "Measured on the held-out test split. Nulls mean the arm was skipped, not zero."
              : undefined
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[820px] text-left text-xs">
              <thead>
                <tr className="border-b border-white/10 text-[10px] uppercase tracking-wider text-navy-500">
                  <th className="pb-2 pr-3 font-medium">Strategy</th>
                  {METRICS.map((m) => (
                    <th key={m.key} className="pb-2 pr-3 text-right font-medium">
                      {m.label}
                    </th>
                  ))}
                  <th className="pb-2 pr-3 text-right font-medium">Params</th>
                  <th className="pb-2 pr-3 text-right font-medium">Time</th>
                  <th className="pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {results.map((e) => (
                  <ExperimentRow key={e.id} experiment={e} />
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 border-t border-white/10 pt-3 text-[11px] leading-relaxed text-navy-500">
            {RESEARCH_DISCLAIMER}
          </p>
        </Section>
      ) : (
        <Section
          title="Planned arms"
          description="Nothing has run on this checkout. These are the arms and their reasons, not results."
        >
          {comparison.isLoading ? (
            <p className="text-xs text-navy-500">Loading plan...</p>
          ) : (data?.planned.length ?? 0) === 0 ? (
            <EmptyState
              icon={FlaskConical}
              variant="empty"
              title="No arms registered"
              description="The backend returned an empty plan."
            />
          ) : (
            <ul className="grid gap-3 lg:grid-cols-2">
              {data!.planned.map((p) => {
                const meta = STRATEGY_META[p.strategy];
                return (
                  <li
                    key={p.name}
                    className="rounded-xl border border-white/10 bg-navy-950/40 p-4"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-xs font-semibold text-white">{p.name}</p>
                      <Badge variant={p.status === "unavailable" ? "warning" : "neutral"}>
                        {p.status === "unavailable" ? (
                          <>
                            <SkipForward className="h-3 w-3" /> unavailable
                          </>
                        ) : (
                          p.status
                        )}
                      </Badge>
                    </div>
                    <p className="mt-1.5 text-[11px] leading-relaxed text-navy-400">
                      {p.note}
                    </p>
                    <p className="mt-2 border-t border-white/10 pt-2 text-[10px] text-navy-500">
                      {meta.blurb}
                    </p>
                  </li>
                );
              })}
            </ul>
          )}
        </Section>
      )}

      {/* ------------------------------------------------ rationale */}
      <div className="grid gap-5 xl:grid-cols-3">
        {(Object.keys(STRATEGY_META) as StrategyKind[]).map((k) => {
          const meta = STRATEGY_META[k];
          const Icon = k === "frozen" ? Cpu : k === "selective" ? Layers : Gauge;
          const unavailable = !budget?.supports_full_finetune && k === "full";
          return (
            <Section key={k} title={meta.label}>
              <div
                className={cn(
                  "rounded-lg p-3",
                  unavailable && "border border-red-500/25 bg-red-500/5",
                )}
              >
                <Icon
                  className={cn(
                    "h-4 w-4",
                    unavailable ? "text-red-400" : "text-accent-400",
                  )}
                />
                <p className="mt-2 text-[11px] leading-relaxed text-navy-400">{meta.blurb}</p>
                {unavailable && (
                  <p className="mt-2 text-[10px] leading-relaxed text-red-300">
                    Refused on this host: no CUDA device, so the optimiser state for 878M
                    trainable parameters cannot be allocated. Recorded as skipped rather
                    than estimated.
                  </p>
                )}
              </div>
            </Section>
          );
        })}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Table row                                                           */
/* ------------------------------------------------------------------ */

function ExperimentRow({ experiment: e }: { experiment: Experiment }) {
  const strategy = (e.strategy ?? "frozen") as StrategyKind;
  const meta = STRATEGY_META[strategy];
  const skipped = e.status === "skipped";

  return (
    <tr
      className={cn(
        "border-b border-white/5 align-top",
        skipped && "opacity-70",
      )}
    >
      <td className="py-3 pr-3">
        <p className="font-medium text-white">{meta.label}</p>
        <p className="mt-0.5 max-w-[22ch] text-[10px] leading-snug text-navy-500">
          {e.trainable_layers ?? "—"}
        </p>
      </td>
      {METRICS.map((m) => {
        const v = e[m.key];
        return (
          <td key={m.key} className="py-3 pr-3 text-right tabular-nums">
            {v === null || v === undefined ? (
              <span className="text-navy-600">—</span>
            ) : (
              <span className="text-navy-200">{formatPercent(v)}</span>
            )}
          </td>
        );
      })}
      <td className="py-3 pr-3 text-right tabular-nums text-navy-400">
        {e.trainable_params ? formatNumber(e.trainable_params) : "—"}
      </td>
      <td className="py-3 pr-3 text-right tabular-nums text-navy-400">
        {e.train_seconds ? `${e.train_seconds.toFixed(1)} s` : "—"}
      </td>
      <td className="py-3">
        <Badge
          variant={
            skipped ? "warning" : e.status === "completed" ? "success" : "neutral"
          }
        >
          {skipped && <SkipForward className="h-3 w-3" />}
          {e.status}
        </Badge>
        {!e.is_real_result && !skipped && (
          <p className="mt-1 text-[9px] uppercase tracking-wide text-amber-400/80">
            not a measurement
          </p>
        )}
      </td>
    </tr>
  );
}

/* ------------------------------------------------------------------ */
/* Best-arm selection                                                  */
/* ------------------------------------------------------------------ */

/**
 * Highest macro-F1 among completed, real arms. Ties break on malignant-class AUC,
 * then recall — recall is the clinically costly one to lose on this task.
 */
function pickBest(experiments: Experiment[]): Experiment | null {
  const real = experiments.filter((e) => e.is_real_result && e.f1 !== null);
  if (real.length === 0) return null;
  return real.reduce((a, b) => {
    const f1 = (a.f1 ?? 0) - (b.f1 ?? 0);
    if (Math.abs(f1) > 1e-9) return f1 > 0 ? a : b;
    const auc = (a.auc ?? 0) - (b.auc ?? 0);
    if (Math.abs(auc) > 1e-9) return auc > 0 ? a : b;
    return (a.recall ?? 0) >= (b.recall ?? 0) ? a : b;
  });
}