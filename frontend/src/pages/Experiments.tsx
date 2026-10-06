/**
 * Week 6 — fine-tuning experiments.
 *
 * Every metric on this page is written by `POST /api/experiments/run`, which
 * trains the frozen head on this machine and scores it on the untouched
 * validation split. Strategies the host cannot honestly execute are refused
 * by the backend feasibility gate with a reason, and render as
 * "Awaiting Experiment" rather than a simulated number.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  Clock,
  Cpu,
  FlaskConical,
  Gauge,
  Loader2,
  Play,
  RefreshCw,
  SlidersHorizontal,
} from "lucide-react";

import { api, ApiError } from "@/services/api";
import { RESEARCH_DISCLAIMER, type ExperimentStatus } from "@/types";
import { EmptyState, Section, StatCard } from "@/components/common";
import { PageHeader } from "@/components/layout/AppLayout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, formatDuration, formatNumber, formatPercent } from "@/lib/utils";

const STATUS_LABEL: Record<ExperimentStatus, string> = {
  awaiting: "Awaiting experiment",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
};

const STATUS_VARIANT: Record<ExperimentStatus, "warning" | "info" | "success" | "destructive"> = {
  awaiting: "warning",
  running: "info",
  completed: "success",
  failed: "destructive",
};

export default function Experiments() {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["experiments"], queryFn: api.listExperiments });

  const run = useMutation({
    mutationFn: api.runExperiment,
    onSuccess: () => void qc.invalidateQueries({ queryKey: ["experiments"] }),
  });

  if (q.isLoading) {
    return (
      <div className="space-y-5">
        <PageHeader
          title="Fine-Tuning & Experiments"
          description="Frozen backbone vs selective vs full fine-tuning."
        />
        <EmptyState
          title="Loading registered experiments…"
          description="Reading strategy definitions, feasibility gates and any recorded metrics from the backend."
          icon={Loader2}
        />
      </div>
    );
  }

  if (q.isError || !q.data) {
    return (
      <div className="space-y-5">
        <PageHeader
          title="Fine-Tuning & Experiments"
          description="Frozen backbone vs selective vs full fine-tuning."
        />
        <EmptyState
          variant="error"
          title="Could not load experiments"
          description={q.error instanceof Error ? q.error.message : "Unexpected payload."}
          icon={AlertTriangle}
          action={
            <Button variant="outline" onClick={() => void q.refetch()}>
              <RefreshCw className="h-3.5 w-3.5" /> Retry
            </Button>
          }
        />
      </div>
    );
  }

  const rows = q.data;
  const completed = rows.filter((r) => r.status === "completed" && r.is_real_result);
  const runnable = rows.filter((r) => r.feasible);
  const accuracies = completed
    .map((r) => r.accuracy)
    .filter((v): v is number => typeof v === "number");
  const bestAccuracy = accuracies.length ? Math.max(...accuracies) : null;

  return (
    <div className="space-y-5">
      <PageHeader
        title="Fine-Tuning & Experiments"
        description="Three transfer-learning strategies over the MedSigLIP-448 vision tower, compared on the held-out validation split. Metrics appear only when a strategy actually runs on this host; the page never substitutes a simulated figure."
        actions={
          <>
            <Badge variant={completed.length ? "success" : "warning"}>
              {completed.length ? (
                <>
                  <CheckCircle2 className="h-3 w-3" /> {completed.length} real result
                  {completed.length > 1 ? "s" : ""}
                </>
              ) : (
                <>
                  <Clock className="h-3 w-3" /> Awaiting experiment
                </>
              )}
            </Badge>
            <Badge variant="neutral">
              <Cpu className="h-3 w-3" /> {runnable.length}/{rows.length} runnable here
            </Badge>
          </>
        }
      />

      {/* ------------------------------------------------ headline stats */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Strategies"
          value={rows.length}
          hint="registered at seed time"
          icon={FlaskConical}
          tone="accent"
        />
        <StatCard
          label="Runnable on this host"
          value={`${runnable.length}/${rows.length}`}
          hint="backend feasibility gate"
          icon={Cpu}
          tone={runnable.length === rows.length ? "emerald" : "amber"}
        />
        <StatCard
          label="Real results"
          value={completed.length}
          hint="written by actual training runs"
          icon={CheckCircle2}
          tone={completed.length ? "emerald" : "slate"}
        />
        <StatCard
          label="Best val accuracy"
          value={bestAccuracy === null ? "--" : formatPercent(bestAccuracy)}
          hint={bestAccuracy === null ? "no completed run yet" : "held-out val split"}
          icon={Gauge}
          tone="violet"
        />
      </div>

      {/* ------------------------------------------------ results table */}
      <Section
        title="Results"
        description="Scored once on the untouched validation split by the real evaluation path. Trainable counts are floats in the optimiser, wall-clock covers head training plus scoring."
        actions={<Badge variant="neutral">{completed.length}/{rows.length} completed</Badge>}
      >
        {completed.length === 0 && (
          <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-300 bg-amber-50 p-3">
            <Clock className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
            <p className="text-[11px] leading-relaxed text-amber-800">
              No experiment has been executed on this checkout yet. Every metric below reads
              <strong> Awaiting experiment </strong> until a strategy actually runs — nothing on
              this page is simulated.
            </p>
          </div>
        )}

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-lilac-50 text-[10px] uppercase tracking-wider text-ink-500">
              <tr>
                <th className="px-4 py-2 font-medium">Strategy</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium">Trainable layers</th>
                <th className="px-4 py-2 text-right font-medium">Accuracy</th>
                <th className="px-4 py-2 text-right font-medium">Precision</th>
                <th className="px-4 py-2 text-right font-medium">Recall</th>
                <th className="px-4 py-2 text-right font-medium">Specificity</th>
                <th className="px-4 py-2 text-right font-medium">F1</th>
                <th className="px-4 py-2 text-right font-medium">AUC</th>
                <th className="px-4 py-2 text-right font-medium">Trainable</th>
                <th className="px-4 py-2 text-right font-medium">Wall-clock</th>
                <th className="px-4 py-2 text-right font-medium">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-950/10">
              {rows.map((r) => {
                const attempted = run.variables === r.name;
                return (
                  <tr key={r.id} className="bg-white">
                    <td className="px-4 py-2.5">
                      <p className="font-medium text-ink-950">{r.name}</p>
                      <p className="text-[11px] text-ink-600">
                        {r.learning_rate != null ? `lr ${r.learning_rate.toExponential(0)}` : "lr --"}
                        {" · "}
                        {r.epochs ?? "--"} epochs
                      </p>
                    </td>
                    <td className="px-4 py-2.5">
                      <Badge variant={STATUS_VARIANT[r.status]}>{STATUS_LABEL[r.status]}</Badge>
                    </td>
                    <td className="max-w-[16rem] px-4 py-2.5 text-xs text-ink-700">
                      {r.trainable_layers ?? "--"}
                    </td>
                    {([r.accuracy, r.precision, r.recall, r.specificity, r.f1, r.auc] as const).map(
                      (v, i) => (
                        <td key={i} className="px-4 py-2.5 text-right tabular-nums text-ink-800">
                          {typeof v === "number" ? formatPercent(v) : <span className="text-ink-400">--</span>}
                        </td>
                      ),
                    )}
                    <td className="px-4 py-2.5 text-right tabular-nums text-ink-800">
                      {r.trainable_params ? formatNumber(r.trainable_params) : "--"}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-ink-800">
                      {r.train_seconds != null ? formatDuration(r.train_seconds) : "--"}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      {r.feasible ? (
                        <Button
                          size="sm"
                          variant={r.status === "completed" ? "outline" : "default"}
                          onClick={() => run.mutate(r.name)}
                          disabled={run.isPending}
                        >
                          {run.isPending && attempted ? (
                            <>
                              <Loader2 className="h-3.5 w-3.5 animate-spin" /> Running
                            </>
                          ) : (
                            <>
                              <Play className="h-3.5 w-3.5" />
                              {r.status === "completed" ? "Re-run" : "Run"}
                            </>
                          )}
                        </Button>
                      ) : (
                        <span
                          className="inline-flex items-center gap-1 text-[11px] text-ink-500"
                          title={r.feasibility_note ?? undefined}
                        >
                          <Ban className="h-3 w-3" /> Not runnable here
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {run.isError && (
          <div className="mt-4 flex items-start gap-2 rounded-lg border border-red-300 bg-red-50 p-3">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-500" />
            <div className="min-w-0">
              <p className="text-xs font-medium text-red-700">
                {run.variables ?? "Experiment"} refused
              </p>
              <p className="mt-0.5 break-words text-[11px] leading-relaxed text-red-600">
                {run.error instanceof ApiError
                  ? run.error.message
                  : "The run failed unexpectedly — check the backend log."}
              </p>
            </div>
          </div>
        )}

        <p className="mt-4 border-t border-ink-950/10 pt-3 text-[10px] leading-relaxed text-ink-500">
          {RESEARCH_DISCLAIMER}
        </p>
      </Section>

      {/* ------------------------------------------------ design & feasibility */}
      <Section
        title="Strategy design & feasibility"
        description="Why each strategy exists, what it would train, and the backend's refusal reason when this host cannot run it."
        actions={
          <Badge variant="neutral">
            <SlidersHorizontal className="h-3 w-3" /> Adam · holdout early stopping · class weights
          </Badge>
        }
      >
        <div className="space-y-3">
          {rows.map((r) => (
            <div
              key={r.id}
              className={cn(
                "rounded-xl border p-4",
                r.feasible ? "border-ink-950/10 bg-white" : "border-amber-200 bg-amber-50/60",
              )}
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-semibold text-ink-950">{r.name}</span>
                  <Badge variant={r.feasible ? "success" : "warning"}>
                    {r.feasible ? "Runnable here" : "Not runnable here"}
                  </Badge>
                  {r.status === "completed" && (
                    <Badge variant="info">
                      <CheckCircle2 className="h-3 w-3" /> Real result recorded
                    </Badge>
                  )}
                </div>
                <span className="font-mono text-[11px] text-ink-600">
                  {r.learning_rate != null ? `lr ${r.learning_rate.toExponential(0)}` : "lr --"}
                  {" · "}
                  {r.epochs ?? "--"} epochs · {r.trainable_layers}
                </span>
              </div>
              <p className="mt-2 whitespace-pre-line text-xs leading-relaxed text-ink-700">
                {r.notes}
              </p>
              {!r.feasible && r.feasibility_note && (
                <p className="mt-2 flex items-start gap-1.5 text-[11px] leading-relaxed text-amber-700">
                  <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
                  {r.feasibility_note}
                </p>
              )}
            </div>
          ))}
        </div>

        <p className="mt-4 border-t border-ink-950/10 pt-3 text-[10px] leading-relaxed text-ink-600">
          The frozen head trains full-batch on the concatenated per-hospital train caches with a
          10% holdout carved from train (never from val) for early stopping at patience 20,
          inverse-frequency class weights against the 80% benign majority, L2 1e-4 and row
          L2-normalisation matching <code className="font-mono">CancerHead.logits</code>. Val
          stays untouched until the single final scoring pass.
        </p>
        <p className="mt-2 text-[10px] leading-relaxed text-ink-500">{RESEARCH_DISCLAIMER}</p>
      </Section>
    </div>
  );
}
