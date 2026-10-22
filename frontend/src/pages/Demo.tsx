import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowRight,
  Check,
  CheckCircle2,
  ClipboardCheck,
  Copy,
  Presentation,
  RefreshCw,
  XCircle,
} from "lucide-react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/AppLayout";
import { EmptyState, Section } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { api } from "@/services/api";
import type { DemoStep } from "@/types";
import { cn, formatDateTime } from "@/lib/utils";

function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };

  return (
    <button
      type="button"
      onClick={() => void copy()}
      className="group flex w-full items-start gap-2 rounded-lg border border-ink-950/10 bg-ink-950/[0.03] px-2.5 py-2 text-left font-mono text-[10px] leading-relaxed text-ink-700 transition-colors hover:border-ink-950/20 hover:bg-ink-950/[0.05]"
      title="Copy command"
    >
      <span className="min-w-0 flex-1 break-all">{command}</span>
      {copied ? (
        <Check className="mt-0.5 h-3 w-3 shrink-0 text-emerald-600" />
      ) : (
        <Copy className="mt-0.5 h-3 w-3 shrink-0 text-ink-400 group-hover:text-ink-700" />
      )}
    </button>
  );
}

function StepCard({ step }: { step: DemoStep }) {
  return (
    <li
      className={cn(
        "flex flex-col rounded-xl border bg-white/70 p-4",
        step.ready
          ? "border-emerald-200"
          : step.required
            ? "border-amber-200"
            : "border-ink-950/10",
      )}
    >
      <div className="flex items-start gap-3">
        <span
          className={cn(
            "flex h-7 w-7 shrink-0 items-center justify-center rounded-full font-mono text-[11px] font-semibold",
            step.ready
              ? "bg-emerald-100 text-emerald-700"
              : step.required
                ? "bg-amber-100 text-amber-700"
                : "bg-ink-950/5 text-ink-500",
          )}
        >
          {String(step.number).padStart(2, "0")}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <p className="text-sm font-medium text-ink-950">{step.title}</p>
            {!step.required && <Badge variant="outline">optional</Badge>}
          </div>
          <p className="mt-1 text-xs leading-relaxed text-ink-600">{step.description}</p>
        </div>
        <Badge variant={step.ready ? "success" : step.required ? "warning" : "neutral"}>
          {step.ready ? (
            <CheckCircle2 className="h-3 w-3" />
          ) : (
            <XCircle className="h-3 w-3" />
          )}
          {step.ready ? "ready" : "blocked"}
        </Badge>
      </div>

      <div className="mt-3 flex-1 rounded-lg bg-ink-950/[0.025] px-3 py-2">
        <p className="text-[10px] uppercase tracking-wider text-ink-500">Live evidence</p>
        <p className="mt-0.5 text-[11px] leading-relaxed text-ink-700">{step.evidence}</p>
      </div>

      {(step.route || (step.command && !step.ready)) && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {step.route && (
            <Button asChild variant="outline" size="sm">
              <Link to={step.route}>
                Open step
                <ArrowRight className="h-3 w-3" />
              </Link>
            </Button>
          )}
          {step.command && !step.ready && (
            <div className="min-w-[12rem] flex-1">
              <CopyCommand command={step.command} />
            </div>
          )}
        </div>
      )}
    </li>
  );
}

export default function Demo() {
  const { data, isLoading, isFetching, error, refetch } = useQuery({
    queryKey: ["demo", "checklist"],
    queryFn: api.demoChecklist,
    refetchInterval: 15_000,
    retry: 1,
  });

  const steps = data?.steps ?? [];
  const blocked = steps.filter((step) => step.required && !step.ready);
  const progress = data && data.total > 0 ? Math.round((data.completed / data.total) * 100) : 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Judge Demo"
        description="A twelve-step walkthrough derived from this machine's live state. Each step either shows its evidence or the exact command that produces it."
        icon={Presentation}
        actions={
          <Button variant="outline" size="sm" onClick={() => void refetch()} disabled={isFetching}>
            <RefreshCw className={cn("h-3.5 w-3.5", isFetching && "animate-spin")} />
            Re-check
          </Button>
        }
      />

      {error ? (
        <Section title="Checklist unavailable" tone="dark">
          <EmptyState
            title="Could not build the demo checklist"
            description={error instanceof Error ? error.message : "Unknown error."}
            icon={AlertTriangle}
            variant="error"
          />
        </Section>
      ) : isLoading || !data ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 4 }).map((_, index) => (
            <div key={index} className="h-28 animate-pulse rounded-xl bg-ink-950/5" />
          ))}
        </div>
      ) : (
        <>
          <div
            className={cn(
              "rounded-2xl border p-5",
              data.ready
                ? "border-emerald-200 bg-emerald-50/60"
                : "border-amber-200 bg-amber-50/60",
            )}
          >
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div className="flex items-start gap-3">
                <span
                  className={cn(
                    "flex h-10 w-10 items-center justify-center rounded-xl",
                    data.ready ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700",
                  )}
                >
                  <ClipboardCheck className="h-5 w-5" />
                </span>
                <div>
                  <p className="text-sm font-semibold text-ink-950">
                    {data.ready ? "Ready to demo" : "Not ready for a full demo"}
                  </p>
                  <p className="mt-0.5 max-w-2xl text-xs leading-relaxed text-ink-600">
                    {data.ready
                      ? "Every required step has live evidence. Optional steps can still be blocked without invalidating the walkthrough."
                      : `${blocked.length} required step${blocked.length === 1 ? "" : "s"} still need${blocked.length === 1 ? "s" : ""} attention. The commands below are exact, not generic advice.`}
                  </p>
                </div>
              </div>
              <div className="min-w-[10rem]">
                <div className="flex items-baseline justify-between text-xs">
                  <span className="text-ink-600">Steps ready</span>
                  <span className="font-mono text-ink-900">
                    {data.completed}/{data.total}
                  </span>
                </div>
                <div className="mt-1.5 h-2 overflow-hidden rounded-full bg-white">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all",
                      data.ready ? "bg-emerald-500" : "bg-amber-500",
                    )}
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="mt-1.5 text-right font-mono text-[10px] text-ink-500">
                  generated {formatDateTime(data.generated_at)}
                </p>
              </div>
            </div>
            <p className="mt-3 border-t border-ink-950/10 pt-3 text-[11px] leading-relaxed text-ink-600">
              {data.disclaimer}
            </p>
          </div>

          {blocked.length > 0 && (
            <div className="flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2.5">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
              <p className="text-[11px] leading-relaxed text-amber-800">
                Blocked: {blocked.map((step) => step.title).join(", ")}. Run the command
                on each card, then press Re-check. The page also refreshes itself every
                15 seconds and whenever the backend emits a system event.
              </p>
            </div>
          )}

          <ol className="grid gap-4 lg:grid-cols-2">
            {steps.map((step) => (
              <StepCard key={step.key} step={step} />
            ))}
          </ol>

          <Section title="How this page stays honest" tone="dark">
            <div className="space-y-2 text-xs leading-relaxed text-ink-600">
              <p>
                The checklist is not a static script. It calls the same integration
                diagnostics, registry cache probes, runner snapshot, privacy summary and
                audit counts as the pages it links to. If a checkpoint disappears or a
                cache is deleted, the corresponding step flips back to blocked on the
                next request.
              </p>
              <p>
                Steps marked optional (currently explainability) can be blocked without
                making the demo impossible; the rest gate the overall ready state. A
                missing artefact is never replaced with a placeholder number.
              </p>
            </div>
          </Section>
        </>
      )}
    </div>
  );
}
