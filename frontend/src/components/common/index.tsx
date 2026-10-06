import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Clock, FlaskConical, type LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn, formatMetric, formatPercent } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/* Stat card                                                           */
/* ------------------------------------------------------------------ */

export function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  tone = "accent",
  sub,
  isReal = true,
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  icon?: LucideIcon;
  tone?: "accent" | "emerald" | "amber" | "red" | "violet" | "slate";
  sub?: string;
  /** false => value came from simulation, render the demo marker */
  isReal?: boolean;
}) {
  const tones: Record<string, { bg: string; text: string; bar: string }> = {
    accent: { bg: "bg-accent-100", text: "text-accent-600", bar: "from-lilac-300 to-periwinkle-400" },
    emerald: { bg: "bg-emerald-100", text: "text-emerald-600", bar: "from-emerald-300 to-emerald-500" },
    amber: { bg: "bg-amber-100", text: "text-amber-600", bar: "from-amber-300 to-amber-500" },
    red: { bg: "bg-red-100", text: "text-red-600", bar: "from-red-300 to-red-500" },
    violet: { bg: "bg-violet-100", text: "text-violet-600", bar: "from-violet-300 to-violet-500" },
    slate: { bg: "bg-slate-100", text: "text-slate-600", bar: "from-slate-300 to-slate-500" },
  };
  const t = tones[tone];

  return (
    <div className="glass-panel group relative overflow-hidden p-4">
      <div className={cn("absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r", t.bar)} />
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-ink-600">
            {label}
            {!isReal && <DemoTag compact />}
          </p>
          <p className="mt-1.5 text-2xl font-semibold tabular-nums tracking-tight text-ink-950">
            {value}
          </p>
          {hint && <p className="mt-0.5 truncate text-[11px] text-ink-500">{hint}</p>}
          {sub && <div className="mt-2">{sub}</div>}
        </div>
        {Icon && (
          <div className={cn("shrink-0 rounded-lg p-2 transition-transform group-hover:scale-110", t.bg, t.text)}>
            <Icon className="h-4 w-4" />
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Demo-data marker                                                    */
/* ------------------------------------------------------------------ */

export function DemoTag({ compact, label }: { compact?: boolean; label?: string }) {
  const text = label ?? "ILLUSTRATIVE DEMO DATA";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded border border-amber-300 bg-amber-100 font-medium uppercase tracking-wide text-amber-700",
        compact ? "px-1 py-0 text-[8px]" : "px-2 py-0.5 text-[9px]",
      )}
      title="Value produced by the simulation harness, not by a real experiment."
    >
      <FlaskConical className={compact ? "h-2 w-2" : "h-2.5 w-2.5"} />
      {text}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Section shell                                                       */
/* ------------------------------------------------------------------ */

export function Section({
  title,
  description,
  actions,
  children,
  className,
  tone = "light",
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  tone?: "light" | "dark";
}) {
  return (
    <section
      className={cn(
        "rounded-2xl border p-5",
        tone === "light"
          ? "border-lilac-100 bg-white shadow-soft"
          : "border-white/80 bg-white/75 backdrop-blur-xl",
        className,
      )}
    >
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2
            className={cn(
              "text-sm font-semibold tracking-tight",
              tone === "light" ? "text-slate-900" : "text-ink-950",
            )}
          >
            {title}
          </h2>
          {description && (
            <p
              className={cn(
                "mt-0.5 text-xs",
                tone === "light" ? "text-slate-500" : "text-ink-600",
              )}
            >
              {description}
            </p>
          )}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Metric row                                                          */
/* ------------------------------------------------------------------ */

export function MetricRow({
  items,
  tone = "light",
}: {
  items: { label: string; value: number | null | undefined; format?: "pct" | "raw"; digits?: number; emphasise?: boolean }[];
  tone?: "light" | "dark";
}) {
  return (
    <dl
      className={cn(
        "grid gap-3",
        items.length <= 2 ? "grid-cols-2" : items.length === 3 ? "grid-cols-3" : "grid-cols-4",
      )}
    >
      {items.map((it) => (
        <div key={it.label} className="min-w-0">
          <dt
            className={cn(
              "truncate text-[10px] font-medium uppercase tracking-wider",
              tone === "light" ? "text-slate-400" : "text-ink-500",
            )}
          >
            {it.label}
          </dt>
          <dd
            className={cn(
              "mt-0.5 text-lg font-semibold tabular-nums",
              it.emphasise
                ? "text-accent-600"
                : tone === "light"
                  ? "text-slate-900"
                  : "text-ink-950",
            )}
          >
            {it.format === "pct"
              ? formatPercent(it.value, it.digits ?? 1)
              : formatMetric(it.value)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/* ------------------------------------------------------------------ */
/* Empty / awaiting state                                              */
/* ------------------------------------------------------------------ */

export function EmptyState({
  title,
  description,
  icon: Icon = Clock,
  variant = "awaiting",
  action,
}: {
  title: string;
  description?: string;
  icon?: LucideIcon;
  variant?: "awaiting" | "empty" | "error";
  action?: ReactNode;
}) {
  const cfg = {
    awaiting: { ring: "border-amber-300/70 bg-amber-100", icon: "text-amber-600", text: "text-amber-800" },
    empty: { ring: "border-ink-950/10 bg-ink-950/5", icon: "text-ink-600", text: "text-ink-800" },
    error: { ring: "border-red-300/70 bg-red-100", icon: "text-red-600", text: "text-red-800" },
  }[variant];

  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-ink-950/15 px-6 py-12 text-center">
      <div className={cn("rounded-xl border p-3", cfg.ring)}>
        <Icon className={cn("h-5 w-5", cfg.icon)} />
      </div>
      <h3 className={cn("mt-3 text-sm font-semibold", cfg.text)}>{title}</h3>
      {description && (
        <p className="mt-1 max-w-md text-xs leading-relaxed text-ink-600">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Scheduled-for-later-week placeholder                                */
/* ------------------------------------------------------------------ */

/**
 * Rendered on every roadmap route beyond the implemented week. Deliberately
 * states what is missing rather than faking a screen — the whole point of the
 * incremental build rule.
 */
export function ScheduledPage({
  week,
  title,
  planned,
}: {
  week: number;
  title: string;
  planned: string;
}) {
  return (
    <div className="mx-auto max-w-3xl py-10">
      <div className="glass-panel overflow-hidden">
        <div className="border-b border-ink-950/10 bg-gradient-to-r from-lilac-200/70 via-blush-200/50 to-transparent px-6 py-5">
          <Badge variant="warning" className="mb-2">
            <Clock className="h-3 w-3" /> Scheduled · Week {week}
          </Badge>
          <h1 className="text-xl font-semibold tracking-tight text-ink-950">{title}</h1>
          <p className="mt-1 text-sm text-ink-600">{planned}</p>
        </div>

        <div className="space-y-4 px-6 py-6">
          <div className="flex items-start gap-3 rounded-xl border border-amber-300/70 bg-amber-100 p-4">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
            <div>
              <p className="text-sm font-medium text-amber-900">
                This module is intentionally not built yet
              </p>
              <p className="mt-1 text-xs leading-relaxed text-amber-700/90">
                The project is being delivered week by week. This build implements{" "}
                <strong className="text-amber-900">weeks 1 through 6</strong>. Showing a
                placeholder rather than fabricated numbers keeps every figure in this
                application traceable to code that actually ran.
              </p>
            </div>
          </div>

          <div className="rounded-xl border border-ink-950/10 bg-white/70 p-4">
            <p className="text-xs font-medium uppercase tracking-wider text-ink-500">
              What lands here in week {week}
            </p>
            <ul className="mt-2 space-y-1.5 text-xs text-ink-700">
              <li>· Backend endpoints under `/api/{title.toLowerCase().replace(/\s+/g, "-")}`</li>
              <li>· Wired to the federated engine planned from week 7</li>
              <li>· Backed by real results — no illustrative placeholders</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Animated number                                                     */
/* ------------------------------------------------------------------ */

export function AnimatedValue({
  value,
  format = "pct",
  digits = 1,
  className,
}: {
  value: number | null | undefined;
  format?: "pct" | "raw" | "int";
  digits?: number;
  className?: string;
}) {
  const text =
    value === null || value === undefined
      ? "--"
      : format === "pct"
        ? formatPercent(value, digits)
        : format === "int"
          ? Math.round(value).toLocaleString()
          : formatMetric(value);
  return (
    <motion.span className={className} initial={{ opacity: 0.4 }} animate={{ opacity: 1 }}>
      {text}
    </motion.span>
  );
}