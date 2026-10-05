/**
 * Week 5 (b) — model registry.
 *
 * Every row is a global model version produced by an actual federated round. There
 * is no seed data here: if the table is empty, that is because no round has
 * completed, and the page says so.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Award,
  Box,
  History,
  Layers,
  Loader2,
  Star,
  Users,
} from "lucide-react";

import { api } from "@/services/api";
import { type ModelVersion, type StrategyKind } from "@/types";
import { EmptyState, Section, StatCard } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ProgressDark } from "@/components/ui/progress";
import { cn, formatDateTime, formatNumber, formatPercent } from "@/lib/utils";

const STRATEGY_LABEL: Record<StrategyKind, string> = {
  frozen: "Frozen",
  selective: "Selective",
  full: "Full",
};

export default function ModelRegistry() {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: api.listModels });

  const recommend = useMutation({
    mutationFn: api.recommendModel,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["models"] });
      void qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });

  const list = models.data ?? [];
  const active = list.find((m) => m.status === "active") ?? null;
  const recommended = list.find((m) => m.is_recommended) ?? null;
  const bestAcc = list.reduce<number | null>((acc, m) => {
    if (m.accuracy === null) return acc;
    return acc === null ? m.accuracy : Math.max(acc, m.accuracy);
  }, null);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-white">Model Registry</h1>
          <p className="mt-1 max-w-3xl text-sm text-navy-400">
            Every global model version, its training strategy, the round it came from, its
            held-out metrics and its lifecycle status.
          </p>
        </div>
        <Badge variant={list.length > 0 ? "success" : "warning"}>
          {list.length} version{list.length === 1 ? "" : "s"}
        </Badge>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Active model"
          value={active?.version ?? "none"}
          hint={active?.strategy ? `${STRATEGY_LABEL[active.strategy]} strategy` : "no version yet"}
          icon={Star}
          tone={active ? "emerald" : "slate"}
        />
        <StatCard
          label="Recommended"
          value={recommended?.version ?? "none"}
          hint={recommended?.training_round ? `from round ${recommended.training_round}` : "unset"}
          icon={Award}
          tone={recommended ? "accent" : "slate"}
        />
        <StatCard
          label="Best accuracy"
          value={bestAcc === null ? "--" : formatPercent(bestAcc)}
          hint="across all versions"
          icon={Box}
          tone={bestAcc === null ? "slate" : "emerald"}
        />
        <StatCard
          label="Total versions"
          value={list.length}
          hint="one per completed round"
          icon={History}
          tone="violet"
        />
      </div>

      <Section
        title="Version history"
        description="Newest first. Promoting a version sets it active and demotes the incumbent."
      >
        {models.isLoading ? (
          <p className="flex items-center gap-2 text-xs text-navy-500">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading registry...
          </p>
        ) : models.isError ? (
          <EmptyState
            variant="error"
            icon={AlertTriangle}
            title="Could not load the registry"
            description={models.error.message}
          />
        ) : list.length === 0 ? (
          <EmptyState
            icon={Layers}
            variant="awaiting"
            title="No model versions yet"
            description="A version is published at the end of each completed federated round. Run a round from the federated dashboard and this table fills in."
          />
        ) : (
          <div className="space-y-3">
            {list.map((m) => (
              <VersionCard
                key={m.id}
                model={m}
                onPromote={() => recommend.mutate(m.id)}
                promoting={recommend.isPending && recommend.variables === m.id}
              />
            ))}
            {recommend.isError && (
              <p className="text-[11px] text-red-300">
                Could not promote: {recommend.error.message}
              </p>
            )}
          </div>
        )}
      </Section>
    </div>
  );
}

function VersionCard({
  model: m,
  onPromote,
  promoting,
}: {
  model: ModelVersion;
  onPromote: () => void;
  promoting: boolean;
}) {
  const metrics = [
    { label: "Accuracy", value: m.accuracy },
    { label: "Precision", value: m.precision },
    { label: "Recall", value: m.recall },
    { label: "Specificity", value: m.specificity },
    { label: "Macro F1", value: m.f1 },
    { label: "AUC", value: m.auc },
  ];

  return (
    <div
      className={cn(
        "rounded-xl border bg-navy-950/40 p-4",
        m.status === "active" ? "border-emerald-500/30" : "border-white/10",
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-semibold text-white">{m.version}</p>
            {m.strategy && (
              <Badge variant="neutral">
                <Layers className="h-3 w-3" /> {STRATEGY_LABEL[m.strategy]}
              </Badge>
            )}
            <Badge
              variant={
                m.status === "active"
                  ? "success"
                  : m.status === "archived"
                    ? "neutral"
                    : m.status === "training"
                      ? "info"
                      : "default"
              }
            >
              {m.status}
            </Badge>
            {m.is_recommended && (
              <Badge variant="warning">
                <Star className="h-3 w-3" /> recommended
              </Badge>
            )}
          </div>
          <p className="mt-1 flex flex-wrap gap-x-3 text-[10px] text-navy-500">
            {m.training_round !== null && <span>round {m.training_round}</span>}
            <span>{formatDateTime(m.created_at)}</span>
            {m.agents !== null && (
              <span className="flex items-center gap-1">
                <Users className="h-2.5 w-2.5" /> {m.agents} agents
              </span>
            )}
          </p>
        </div>
        {m.status !== "active" && (
          <Button
            size="sm"
            variant="ghost"
            onClick={onPromote}
            disabled={promoting}
            className="shrink-0"
          >
            {promoting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Star className="h-3.5 w-3.5" />}
            Promote
          </Button>
        )}
      </div>

      <div className="mt-3 grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {metrics.map((x) => (
          <div key={x.label} className="min-w-0">
            <p className="text-[9px] uppercase tracking-wider text-navy-500">{x.label}</p>
            <p className="mt-0.5 text-sm font-semibold tabular-nums text-white">
              {x.value === null ? "—" : formatPercent(x.value)}
            </p>
            {x.value !== null && (
              <div className="mt-1">
                <ProgressDark value={x.value * 100} />
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-white/10 pt-2 text-[10px] text-navy-500">
        {m.trainable_params !== null && (
          <span>{formatNumber(m.trainable_params)} trainable params</span>
        )}
        {m.total_params !== null && (
          <span>{formatNumber(m.total_params)} total params</span>
        )}
        {m.loss !== null && <span>loss {m.loss.toFixed(4)}</span>}
        {m.notes && <span className="min-w-0 flex-1 truncate">{m.notes}</span>}
      </div>
    </div>
  );
}