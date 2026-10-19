/**
 * Week 9 — model performance & evaluation.
 *
 * Everything on this page is one payload from `GET /api/evaluation/summary`,
 * produced by running the aggregated head over the untouched validation split.
 * The page deliberately leads with macro F1, malignant AUC and recall rather
 * than accuracy: on HAM10000 the benign majority makes accuracy alone look far
 * better than triage performance really is.
 *
 * When no checkpoint or no validation embeddings exist, the backend returns
 * `ready: false` with instructions, and this page renders that instead of a
 * chart. There is no simulated fallback.
 */
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Crosshair,
  Gauge,
  Loader2,
  RefreshCw,
  ShieldAlert,
  Target,
  TrendingUp,
} from "lucide-react";

import { api } from "@/services/api";
import {
  RESEARCH_DISCLAIMER,
  type EvaluationSummary,
  type LearningCurvePoint,
  type PerClassMetric,
  type RocPoint,
} from "@/types";
import { EmptyState, Section, StatCard } from "@/components/common";
import { PageHeader } from "@/components/layout/AppLayout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, formatDateTime, formatMetric, formatNumber, formatPercent } from "@/lib/utils";

const PAGE_TITLE = "Model Performance & Evaluation";
const PAGE_DESCRIPTION =
  "The aggregated head scored once on the held-out validation split: four-class metrics, malignant-vs-benign ROC, per-class recall and the confusion matrix. This split is never trained on, by any agent.";

const CLASS_COLORS: Record<string, string> = {
  melanoma: "#ea580c",
  basal_cell_carcinoma: "#7c3aed",
  actinic_keratosis: "#0d9488",
  benign_lesion: "#2563eb",
};
const MALIGNANT_COLOR = "#e11d48";
const FALLBACK_COLORS = ["#db2777", "#0891b2", "#ca8a04", "#4f46e5"];

function humanize(name: string): string {
  return name.replace(/_/g, " ");
}

function classColor(name: string, index: number): string {
  return CLASS_COLORS[name] ?? FALLBACK_COLORS[index % FALLBACK_COLORS.length];
}

/** Trapezoidal area under a monotonically increasing ROC sweep. */
function aucFromPoints(points: RocPoint[]): number | null {
  if (points.length < 2) return null;
  let area = 0;
  for (let i = 1; i < points.length; i += 1) {
    const dx = points[i].fpr - points[i - 1].fpr;
    area += (dx * (points[i].tpr + points[i - 1].tpr)) / 2;
  }
  return area;
}

/* ------------------------------------------------------------------ */
/* ROC chart                                                           */
/* ------------------------------------------------------------------ */

const ROC_SIZE = 340;
const ROC_PAD = 36;
const ROC_PLOT = ROC_SIZE - ROC_PAD * 2;

function rocX(fpr: number): number {
  return ROC_PAD + fpr * ROC_PLOT;
}

function rocY(tpr: number): number {
  return ROC_PAD + (1 - tpr) * ROC_PLOT;
}

function rocPath(points: RocPoint[]): string {
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"}${rocX(p.fpr).toFixed(2)},${rocY(p.tpr).toFixed(2)}`)
    .join(" ");
}

function RocChart({ summary }: { summary: EvaluationSummary }) {
  const curves = Object.entries(summary.roc.per_class);
  const malignant = summary.roc.malignant;

  if (curves.length === 0 && !malignant) {
    return (
      <EmptyState
        variant="empty"
        title="No ROC curve available"
        description="The validation split contains a single class, so a one-vs-rest curve cannot be computed."
        icon={Activity}
      />
    );
  }

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-center">
      <svg
        viewBox={`0 0 ${ROC_SIZE} ${ROC_SIZE}`}
        role="img"
        aria-label="Receiver operating characteristic curves"
        className="mx-auto w-full max-w-[340px]"
      >
        <rect
          x={ROC_PAD}
          y={ROC_PAD}
          width={ROC_PLOT}
          height={ROC_PLOT}
          fill="none"
          className="stroke-ink-950/10"
        />
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={rocX(t)}
              y1={ROC_PAD}
              x2={rocX(t)}
              y2={ROC_PAD + ROC_PLOT}
              className="stroke-ink-950/[0.06]"
            />
            <line
              x1={ROC_PAD}
              y1={rocY(t)}
              x2={ROC_PAD + ROC_PLOT}
              y2={rocY(t)}
              className="stroke-ink-950/[0.06]"
            />
            <text
              x={rocX(t)}
              y={ROC_SIZE - ROC_PAD + 14}
              textAnchor="middle"
              className="fill-ink-500 text-[9px]"
            >
              {t.toFixed(2)}
            </text>
            <text
              x={ROC_PAD - 8}
              y={rocY(t) + 3}
              textAnchor="end"
              className="fill-ink-500 text-[9px]"
            >
              {t.toFixed(2)}
            </text>
          </g>
        ))}

        <line
          x1={rocX(0)}
          y1={rocY(0)}
          x2={rocX(1)}
          y2={rocY(1)}
          strokeDasharray="4 4"
          className="stroke-ink-400"
          strokeWidth={1.2}
        />

        {curves.map(([name, points], index) => (
          <path
            key={name}
            d={rocPath(points)}
            fill="none"
            stroke={classColor(name, index)}
            strokeWidth={1.8}
            strokeLinejoin="round"
            opacity={0.9}
          />
        ))}

        {malignant && (
          <path
            d={rocPath(malignant)}
            fill="none"
            stroke={MALIGNANT_COLOR}
            strokeWidth={3}
            strokeLinejoin="round"
          />
        )}

        <text
          x={ROC_PAD + ROC_PLOT / 2}
          y={ROC_SIZE - 4}
          textAnchor="middle"
          className="fill-ink-600 text-[9px] font-medium uppercase tracking-wider"
        >
          False positive rate
        </text>
        <text
          x={10}
          y={ROC_PAD + ROC_PLOT / 2}
          textAnchor="middle"
          transform={`rotate(-90 10 ${ROC_PAD + ROC_PLOT / 2})`}
          className="fill-ink-600 text-[9px] font-medium uppercase tracking-wider"
        >
          True positive rate
        </text>
      </svg>

      <div className="space-y-1.5">
        {curves.map(([name, points], index) => {
          const auc = aucFromPoints(points);
          return (
            <div key={name} className="flex items-center justify-between gap-2 text-[11px]">
              <span className="flex min-w-0 items-center gap-2 text-ink-700">
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: classColor(name, index) }}
                />
                <span className="truncate capitalize">{humanize(name)}</span>
              </span>
              <span className="tabular-nums text-ink-600">
                AUC {auc === null ? "--" : formatMetric(auc)}
              </span>
            </div>
          );
        })}
        {malignant && (
          <div className="mt-2 flex items-center justify-between gap-2 border-t border-ink-950/10 pt-2 text-[11px]">
            <span className="flex items-center gap-2 font-semibold text-ink-900">
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: MALIGNANT_COLOR }}
              />
              Malignant vs benign
            </span>
            <span className="tabular-nums font-semibold text-ink-900">
              AUC{" "}
              {formatMetric(
                summary.metrics?.malignant_auc ?? aucFromPoints(malignant),
              )}
            </span>
          </div>
        )}
        <p className="pt-2 text-[10px] leading-relaxed text-ink-500">
          Curves are the threshold sweep of the real predicted probabilities. Per-class AUC is
          integrated from the plotted points; the malignant curve sums the three malignant class
          probabilities.
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Confusion matrix                                                    */
/* ------------------------------------------------------------------ */

function ConfusionMatrix({ summary }: { summary: EvaluationSummary }) {
  const { classes, confusion_matrix: matrix } = summary;
  if (matrix.length === 0) {
    return (
      <EmptyState
        variant="empty"
        title="No confusion matrix available"
        description="The evaluation did not run, so there are no cells to show."
        icon={Target}
      />
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-separate border-spacing-1 text-center text-xs">
        <thead>
          <tr>
            <th className="px-1 py-1 text-left text-[10px] font-medium uppercase tracking-wider text-ink-500">
              actual \ predicted
            </th>
            {classes.map((name) => (
              <th
                key={name}
                className="min-w-[4.5rem] px-1 py-1 text-[10px] font-medium uppercase tracking-wide text-ink-500"
              >
                {humanize(name)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, actual) => {
            const rowTotal = row.reduce((sum, value) => sum + value, 0);
            return (
              <tr key={classes[actual]}>
                <th className="whitespace-nowrap px-1 py-1 text-right text-[10px] font-medium uppercase tracking-wide text-ink-500">
                  {humanize(classes[actual])}
                </th>
                {row.map((count, predicted) => {
                  const share = rowTotal ? count / rowTotal : 0;
                  const correct = actual === predicted;
                  const background =
                    count === 0
                      ? "transparent"
                      : correct
                        ? `rgba(16, 185, 129, ${0.12 + 0.68 * share})`
                        : `rgba(244, 63, 94, ${0.1 + 0.5 * share})`;
                  return (
                    <td
                      key={predicted}
                      title={`${count} samples: actual ${humanize(
                        classes[actual],
                      )} predicted as ${humanize(classes[predicted])}`}
                      className={cn(
                        "rounded-md px-1 py-2 text-sm font-semibold tabular-nums",
                        count === 0 ? "text-ink-300" : "text-ink-950",
                      )}
                      style={{ backgroundColor: background }}
                    >
                      {count}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-2 text-[10px] leading-relaxed text-ink-500">
        Rows are the true class, columns the prediction. Diagonal cells (green) are correct;
        off-diagonal cells (red) are errors, shaded by their share of that true class. Read a row
        to see per-class recall.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Per-class table                                                     */
/* ------------------------------------------------------------------ */

const PER_CLASS_COLUMNS: { key: keyof PerClassMetric; label: string; percent?: boolean }[] = [
  { key: "precision", label: "Precision", percent: true },
  { key: "recall", label: "Recall", percent: true },
  { key: "specificity", label: "Specificity", percent: true },
  { key: "f1", label: "F1", percent: true },
];

function PerClassTable({ summary }: { summary: EvaluationSummary }) {
  const names = summary.classes.filter((name) => summary.per_class[name]);
  if (names.length === 0) {
    return (
      <EmptyState
        variant="empty"
        title="No per-class metrics"
        description="Metrics appear only after a real evaluation over the validation split."
        icon={Gauge}
      />
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="bg-lilac-50 text-[10px] uppercase tracking-wider text-ink-500">
          <tr>
            <th className="px-4 py-2 font-medium">Class</th>
            <th className="px-4 py-2 text-right font-medium">Support</th>
            {PER_CLASS_COLUMNS.map((column) => (
              <th key={column.key} className="px-4 py-2 text-right font-medium">
                {column.label}
              </th>
            ))}
            <th className="px-4 py-2 text-right font-medium">TP / FP / FN</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-ink-950/10">
          {names.map((name, index) => {
            const metric = summary.per_class[name];
            const isMalignant = summary.malignant_classes.includes(name);
            return (
              <tr key={name} className="bg-white">
                <td className="px-4 py-2.5">
                  <span className="flex items-center gap-2">
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ backgroundColor: classColor(name, index) }}
                    />
                    <span className="font-medium capitalize text-ink-950">{humanize(name)}</span>
                    {isMalignant && <Badge variant="warning">malignant</Badge>}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-ink-700">
                  {formatNumber(metric.support)}
                </td>
                {PER_CLASS_COLUMNS.map((column) => {
                  const value = metric[column.key];
                  return (
                    <td
                      key={column.key}
                      className={cn(
                        "px-4 py-2.5 text-right tabular-nums",
                        column.key === "recall" && value < 0.5
                          ? "font-semibold text-red-600"
                          : "text-ink-800",
                      )}
                    >
                      {typeof value === "number" ? formatPercent(value) : "--"}
                    </td>
                  );
                })}
                <td className="px-4 py-2.5 text-right font-mono text-[11px] tabular-nums text-ink-600">
                  {metric.tp} / {metric.fp} / {metric.fn}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Learning curves                                                     */
/* ------------------------------------------------------------------ */

const CURVE_WIDTH = 520;
const CURVE_HEIGHT = 220;
const CURVE_PAD = { left: 36, right: 14, top: 12, bottom: 26 };
const CURVE_PLOT_W = CURVE_WIDTH - CURVE_PAD.left - CURVE_PAD.right;
const CURVE_PLOT_H = CURVE_HEIGHT - CURVE_PAD.top - CURVE_PAD.bottom;

type CurveKey = "accuracy" | "f1" | "auc";

const CURVE_SERIES: { key: CurveKey; label: string; color: string }[] = [
  { key: "accuracy", label: "Accuracy", color: "#2563eb" },
  { key: "f1", label: "Malignant F1", color: "#ea580c" },
  { key: "auc", label: "Malignant AUC", color: "#7c3aed" },
];

function curveSegments(
  points: LearningCurvePoint[],
  key: CurveKey,
  xFor: (round: number) => number,
): { x: number; y: number }[][] {
  const segments: { x: number; y: number }[][] = [];
  let current: { x: number; y: number }[] = [];
  for (const point of points) {
    const value = point[key];
    if (!point.evaluated || value === null) {
      if (current.length) segments.push(current);
      current = [];
      continue;
    }
    current.push({ x: xFor(point.round_number), y: CURVE_PAD.top + (1 - value) * CURVE_PLOT_H });
  }
  if (current.length) segments.push(current);
  return segments;
}

function LearningCurves({ points }: { points: LearningCurvePoint[] }) {
  const evaluated = points.filter((point) => point.evaluated);
  if (evaluated.length === 0) {
    return (
      <EmptyState
        variant="empty"
        title="No completed rounds yet"
        description="The learning curve fills in as federated rounds finish and record real global metrics."
        icon={TrendingUp}
      />
    );
  }

  const rounds = points.map((point) => point.round_number);
  const minRound = Math.min(...rounds);
  const maxRound = Math.max(...rounds);
  const span = Math.max(1, maxRound - minRound);
  const xFor = (round: number) => CURVE_PAD.left + ((round - minRound) / span) * CURVE_PLOT_W;
  const yTicks = [0, 0.25, 0.5, 0.75, 1];
  const xTicks =
    points.length <= 8
      ? points.map((point) => point.round_number)
      : points.filter((_, index) => index % Math.ceil(points.length / 8) === 0).map((p) => p.round_number);

  return (
    <div className="space-y-3">
      <svg
        viewBox={`0 0 ${CURVE_WIDTH} ${CURVE_HEIGHT}`}
        role="img"
        aria-label="Global metrics across federated rounds"
        className="w-full"
      >
        {yTicks.map((tick) => {
          const y = CURVE_PAD.top + (1 - tick) * CURVE_PLOT_H;
          return (
            <g key={tick}>
              <line
                x1={CURVE_PAD.left}
                y1={y}
                x2={CURVE_WIDTH - CURVE_PAD.right}
                y2={y}
                className="stroke-ink-950/[0.07]"
              />
              <text x={CURVE_PAD.left - 7} y={y + 3} textAnchor="end" className="fill-ink-500 text-[9px]">
                {tick.toFixed(2)}
              </text>
            </g>
          );
        })}
        {xTicks.map((round) => (
          <text
            key={round}
            x={xFor(round)}
            y={CURVE_HEIGHT - 8}
            textAnchor="middle"
            className="fill-ink-500 text-[9px]"
          >
            R{round}
          </text>
        ))}

        {CURVE_SERIES.map((series) => {
          const segments = curveSegments(points, series.key, xFor);
          return (
            <g key={series.key}>
              {segments.map((segment, index) =>
                segment.length > 1 ? (
                  <path
                    key={index}
                    d={segment
                      .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`)
                      .join(" ")}
                    fill="none"
                    stroke={series.color}
                    strokeWidth={2}
                    strokeLinejoin="round"
                  />
                ) : (
                  segment.map((p, i) => (
                    <circle key={`${index}-${i}`} cx={p.x} cy={p.y} r={2.5} fill={series.color} />
                  ))
                ),
              )}
            </g>
          );
        })}
      </svg>

      <div className="flex flex-wrap items-center gap-4">
        {CURVE_SERIES.map((series) => (
          <span key={series.key} className="flex items-center gap-1.5 text-[11px] text-ink-600">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: series.color }} />
            {series.label}
          </span>
        ))}
        {evaluated.length < 2 && (
          <span className="text-[11px] text-ink-500">
            One completed round is a point, not a trend — run more rounds to see the curve.
          </span>
        )}
      </div>
      <p className="text-[10px] leading-relaxed text-ink-500">
        Each point is the aggregated head&apos;s score on the validation split immediately after that
        round. The line breaks at rounds that are still running or have no recorded metrics.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Model history                                                       */
/* ------------------------------------------------------------------ */

function ModelHistoryTable({ summary }: { summary: EvaluationSummary }) {
  if (summary.model_history.length === 0) {
    return (
      <EmptyState
        variant="empty"
        title="No model versions yet"
        description="A row appears here each time a federated round produces a new global checkpoint."
        icon={Clock}
      />
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="bg-lilac-50 text-[10px] uppercase tracking-wider text-ink-500">
          <tr>
            <th className="px-4 py-2 font-medium">Version</th>
            <th className="px-4 py-2 font-medium">Round</th>
            <th className="px-4 py-2 font-medium">Strategy</th>
            <th className="px-4 py-2 text-right font-medium">Accuracy</th>
            <th className="px-4 py-2 text-right font-medium">Recall</th>
            <th className="px-4 py-2 text-right font-medium">F1</th>
            <th className="px-4 py-2 text-right font-medium">AUC</th>
            <th className="px-4 py-2 text-right font-medium">Loss</th>
            <th className="px-4 py-2 font-medium">Status</th>
            <th className="px-4 py-2 font-medium">Created</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-ink-950/10">
          {summary.model_history.map((model) => (
            <tr key={model.version} className="bg-white">
              <td className="px-4 py-2.5">
                <span className="flex items-center gap-1.5 font-mono text-xs text-ink-900">
                  {model.version}
                  {model.is_recommended && <Badge variant="success">recommended</Badge>}
                </span>
              </td>
              <td className="px-4 py-2.5 tabular-nums text-ink-700">
                {model.training_round ?? "--"}
              </td>
              <td className="px-4 py-2.5 capitalize text-ink-700">{model.strategy ?? "--"}</td>
              {([model.accuracy, model.recall, model.f1, model.auc] as const).map((value, index) => (
                <td key={index} className="px-4 py-2.5 text-right tabular-nums text-ink-800">
                  {typeof value === "number" ? formatPercent(value) : "--"}
                </td>
              ))}
              <td className="px-4 py-2.5 text-right tabular-nums text-ink-800">
                {formatMetric(model.loss)}
              </td>
              <td className="px-4 py-2.5">
                <Badge
                  variant={
                    model.status === "active"
                      ? "success"
                      : model.status === "archived"
                        ? "neutral"
                        : "info"
                  }
                >
                  {model.status}
                </Badge>
              </td>
              <td className="whitespace-nowrap px-4 py-2.5 text-[11px] text-ink-500">
                {formatDateTime(model.created_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function Performance() {
  const q = useQuery({ queryKey: ["evaluation-summary"], queryFn: api.evaluationSummary });

  if (q.isLoading) {
    return (
      <div className="space-y-5">
        <PageHeader title={PAGE_TITLE} description={PAGE_DESCRIPTION} icon={Activity} />
        <EmptyState
          title="Scoring the global model…"
          description="Loading the checkpoint and the held-out validation embeddings, then computing the metric suite."
          icon={Loader2}
        />
      </div>
    );
  }

  if (q.isError || !q.data) {
    return (
      <div className="space-y-5">
        <PageHeader title={PAGE_TITLE} description={PAGE_DESCRIPTION} icon={Activity} />
        <EmptyState
          variant="error"
          title="Could not load the evaluation"
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

  const data = q.data;
  const metrics = data.metrics;
  const malignantAuc = metrics?.malignant_auc ?? metrics?.auc ?? null;
  const hasHistory = data.learning_curve.length > 0 || data.model_history.length > 0;

  return (
    <div className="space-y-5">
      <PageHeader
        title={PAGE_TITLE}
        description={PAGE_DESCRIPTION}
        icon={Activity}
        actions={
          <>
            <Badge variant={data.ready ? "success" : "warning"}>
              {data.ready ? (
                <>
                  <CheckCircle2 className="h-3 w-3" /> Real validation evaluation
                </>
              ) : (
                <>
                  <Clock className="h-3 w-3" /> Awaiting artefacts
                </>
              )}
            </Badge>
            <Badge variant="neutral">
              <Crosshair className="h-3 w-3" /> {data.model_version}
            </Badge>
            {data.ready && (
              <Badge variant="neutral">
                <Target className="h-3 w-3" /> n = {formatNumber(data.n_eval)}
              </Badge>
            )}
          </>
        }
      />

      {!data.ready && (
        <EmptyState
          variant="awaiting"
          title="No evaluation to report yet"
          description={
            data.instructions ??
            "The validation embeddings or the global checkpoint are missing. Nothing is simulated on this page."
          }
          icon={Clock}
          action={
            <Button variant="outline" onClick={() => void q.refetch()}>
              <RefreshCw className="h-3.5 w-3.5" /> Re-check
            </Button>
          }
        />
      )}

      {metrics && (
        <>
          {/* -------------------------------------------- headline metrics */}
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            <StatCard
              label="Macro F1 (4-class)"
              value={formatPercent(metrics.macro_f1)}
              hint="primary multi-class metric · unweighted mean over present classes"
              icon={Gauge}
              tone="accent"
            />
            <StatCard
              label="Malignant AUC"
              value={malignantAuc === null ? "--" : formatMetric(malignantAuc)}
              hint="summed malignant probability vs benign, rank-based"
              icon={Activity}
              tone="violet"
            />
            <StatCard
              label="Malignant recall"
              value={formatPercent(metrics.recall)}
              hint={`specificity ${formatPercent(metrics.specificity)} · the screening trade-off`}
              icon={Target}
              tone="emerald"
            />
            <StatCard
              label="False negatives"
              value={formatNumber(metrics.false_negatives)}
              hint="malignant cases predicted benign"
              icon={ShieldAlert}
              tone={metrics.false_negatives > 0 ? "red" : "emerald"}
            />
            <StatCard
              label="Balanced accuracy"
              value={formatPercent(metrics.balanced_accuracy)}
              hint="mean per-class recall, insensitive to class imbalance"
              icon={TrendingUp}
              tone="slate"
            />
            <StatCard
              label="Accuracy (secondary)"
              value={formatPercent(metrics.accuracy)}
              hint="reported for context only — dominated by the benign majority"
              icon={CheckCircle2}
              tone="slate"
            />
          </div>

          <div className="flex items-start gap-2 rounded-xl border border-amber-300/70 bg-amber-50 p-3">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
            <p className="text-[11px] leading-relaxed text-amber-800">
              Read accuracy last. On this dataset benign lesions are the majority, so a model can
              look accurate while missing melanomas. The headline numbers above are the ones to
              judge: macro F1, malignant AUC and malignant recall. Worst class here:{" "}
              <strong className="capitalize">
                {metrics.worst_class_recall ? humanize(metrics.worst_class_recall) : "n/a"}
              </strong>
              .
            </p>
          </div>
        </>
      )}

      {data.ready && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Section
            title="ROC curves"
            description="One-vs-rest per class plus the summed malignant-vs-benign axis, from real predicted probabilities."
            actions={<Badge variant="neutral">threshold sweep</Badge>}
          >
            <RocChart summary={data} />
          </Section>

          <Section
            title="Confusion matrix"
            description="Where the head's errors actually land, class by class."
            actions={<Badge variant="neutral">n = {formatNumber(data.n_eval)}</Badge>}
          >
            <ConfusionMatrix summary={data} />
          </Section>
        </div>
      )}

      {data.ready && (
        <Section
          title="Per-class metrics"
          description="One-vs-rest precision, recall and specificity. Low recall on a malignant class is the failure that matters clinically."
          actions={<Badge variant="neutral">held-out validation split</Badge>}
        >
          <PerClassTable summary={data} />
        </Section>
      )}

      {hasHistory && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Section
            title="Learning curve"
            description="Global metrics recorded after each completed federated round."
            actions={
              <Badge variant="neutral">
                {data.learning_curve.filter((point) => point.evaluated).length} completed
              </Badge>
            }
          >
            <LearningCurves points={data.learning_curve} />
          </Section>

          <Section
            title="Model history"
            description="Every aggregated checkpoint this project has produced, with the metrics recorded when it was created."
            actions={<Badge variant="neutral">{data.model_history.length} versions</Badge>}
          >
            <ModelHistoryTable summary={data} />
          </Section>
        </div>
      )}

      <Section title="How to read this report" description="Scope and honesty notes for the numbers above.">
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="rounded-xl border border-ink-950/10 bg-ink-950/[0.02] p-4">
            <p className="text-xs font-semibold text-ink-900">What was evaluated</p>
            <p className="mt-1 text-[11px] leading-relaxed text-ink-600">
              The aggregated head ({data.model_version}) over{" "}
              {data.ready ? formatNumber(data.n_eval) : "the"} validation embeddings. The four
              hospital partitions come from <code className="font-mono">data/splits.json</code>;
              validation is never included in any agent&apos;s training cache.
            </p>
          </div>
          <div className="rounded-xl border border-ink-950/10 bg-ink-950/[0.02] p-4">
            <p className="text-xs font-semibold text-ink-900">What is not claimed</p>
            <p className="mt-1 text-[11px] leading-relaxed text-ink-600">
              These are research-prototype numbers on a single validation split. The head is a
              linear probe over frozen embeddings, not a clinically validated classifier. External
              data, calibration and patient-level bias were not assessed here.
            </p>
          </div>
        </div>
        <p className="mt-4 border-t border-ink-950/10 pt-3 text-[10px] leading-relaxed text-ink-500">
          {RESEARCH_DISCLAIMER}
        </p>
      </Section>
    </div>
  );
}
