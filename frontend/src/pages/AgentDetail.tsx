import { Link, useParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  Building2,
  DownloadCloud,
  PauseCircle,
  PlayCircle,
  ShieldCheck,
} from "lucide-react";
import { PageHeader } from "@/components/layout/AppLayout";
import { Section, EmptyState, MetricRow } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  useAgent,
  usePauseAgentTraining,
  useStartAgentTraining,
  useSyncAgent,
} from "@/hooks/useAgents";
import { CANCER_CLASSES, CANCER_CLASS_INFO } from "@/types";
import { cn, formatDateTime, formatNumber, formatPercent } from "@/lib/utils";

/** MetricRow only renders numbers, so string fields get their own row. */
function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-[10px] font-medium uppercase tracking-wider text-ink-500">
        {label}
      </span>
      <span className="truncate text-sm font-semibold text-ink-950">{value}</span>
    </div>
  );
}

export default function AgentDetail() {
  const { id } = useParams<{ id: string }>();
  const agentId = Number(id);
  const { data: agent, isLoading, error } = useAgent(agentId);
  const train = useStartAgentTraining();
  const pause = usePauseAgentTraining();
  const sync = useSyncAgent();

  if (error) {
    return (
      <>
        <PageHeader title="Hospital Detail" icon={Building2} />
        <EmptyState
          variant="error"
          icon={AlertTriangle}
          title={String((error as { status?: number }).status) === "404" ? "Agent not found" : "Backend is not reachable"}
          description={
            String((error as { status?: number }).status) === "404"
              ? `No hospital agent with id ${id}. Valid ids are 1-4.`
              : "Start the FastAPI server to load agent detail."
          }
        />
      </>
    );
  }

  if (isLoading || !agent) {
    return (
      <>
        <PageHeader title="Hospital Detail" description="Loading agent record…" icon={Building2} />
        <div className="glass-panel h-64 animate-pulse" />
      </>
    );
  }

  const dist = agent.class_distribution ?? {};
  const total = Object.values(dist).reduce((s, n) => s + n, 0);
  const busy = train.isPending || pause.isPending || sync.isPending;
  const neverTrained = agent.rounds_participated === 0;

  return (
    <>
      <PageHeader
        title={agent.name}
        description={`${agent.agent_id} · ${agent.location ?? "location not recorded"}`}
        icon={Building2}
        actions={
          <Button asChild variant="ghost" size="sm">
            <Link to="/agents">
              <ArrowLeft /> All agents
            </Link>
          </Button>
        }
      />

      <div className="grid gap-5 lg:grid-cols-3">
        <Section
          title="Local dataset"
          description="What this hospital holds privately."
          className="lg:col-span-2"
        >
          <MetricRow
            items={[
              { label: "Images", value: agent.dataset_size },
              { label: "Classes", value: Object.keys(dist).length },
              {
                label: "Maj. class share",
                value: total ? Math.max(...Object.values(dist)) / total : null,
                format: "pct",
              },
              {
                label: "Maj ÷ min",
                value: total ? Math.max(...Object.values(dist)) / Math.min(...Object.values(dist)) : null,
                digits: 1,
              },
            ]}
          />

          <div className="space-y-3">
            {CANCER_CLASSES.map((c) => {
              const n = dist[c] ?? 0;
              const frac = total ? n / total : 0;
              return (
                <div key={c}>
                  <div className="mb-1 flex items-center justify-between text-xs">
                    <span className="flex items-center gap-1.5 text-ink-800">
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ background: CANCER_CLASS_INFO[c].color }}
                      />
                      {CANCER_CLASS_INFO[c].label}
                    </span>
                    <span className="tabular-nums text-ink-600">
                      {formatNumber(n)} · {formatPercent(frac)}
                    </span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-ink-950/10">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${frac * 100}%`, background: CANCER_CLASS_INFO[c].color }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Section>

        <Section title="Federated status" description="Participation and local controls." tone="dark">
          <div className="space-y-3">
            <InfoRow label="Status" value={agent.status} />
            <InfoRow label="Model version" value={agent.model_version} />
            <InfoRow label="Rounds participated" value={String(agent.rounds_participated)} />
            <InfoRow
              label="Last sync"
              value={agent.last_sync ? formatDateTime(agent.last_sync) : "never"}
            />
            <InfoRow label="Privacy status" value={agent.privacy_status} />

            <div className="flex flex-wrap gap-2 pt-2">
              <Button
                size="sm"
                disabled={busy || agent.status === "training"}
                onClick={() => train.mutate(agent.id)}
              >
                <PlayCircle /> Train
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={busy || agent.status !== "training"}
                onClick={() => pause.mutate(agent.id)}
              >
                <PauseCircle /> Pause
              </Button>
              <Button size="sm" variant="outline" disabled={busy} onClick={() => sync.mutate(agent.id)}>
                <DownloadCloud /> Sync
              </Button>
            </div>
          </div>
        </Section>
      </div>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Section
          title="Local validation metrics"
          description="Measured on this site's own held-out split, never shared as raw data."
        >
          {neverTrained ? (
            <EmptyState
              icon={PlayCircle}
              variant="awaiting"
              title="No local training run yet"
              description="Metrics appear after this site completes a training round. Aggregated metrics live in the federated dashboard from Week 8."
            />
          ) : (
            <MetricRow
              items={[
                { label: "Accuracy", value: agent.accuracy, format: "pct" },
                { label: "Precision", value: agent.precision, format: "pct" },
                { label: "Recall", value: agent.recall, format: "pct" },
                { label: "F1", value: agent.f1_score, format: "pct" },
              ]}
            />
          )}
        </Section>

        <Section
          title="What leaves this site"
          description="The complete transmission surface for one hospital."
          actions={<Badge variant="success">images: 0</Badge>}
        >
          <ul className="space-y-2.5 text-xs text-ink-700">
            {[
              ["Dermatite images", "Never transmitted", true],
              ["Patient identifiers", "Never transmitted", true],
              ["Model gradients / updates", "Transmitted, masked to size only", false],
              ["Aggregate loss and sample count", "Transmitted for weighting", false],
              ["Class histogram of local partition", "Transmitted to aggregator", false],
            ].map(([label, value, safe]) => (
              <li key={label as string} className="flex items-start gap-2">
                <span
                  className={cn(
                    "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[9px] font-bold",
                    safe ? "bg-emerald-100 text-emerald-700" : "bg-sky-100 text-sky-700",
                  )}
                >
                  {safe ? "✓" : "→"}
                </span>
                <span>
                  <span className="font-medium text-ink-900">{label as string}</span>
                  <span className="text-ink-500"> — {value as string}</span>
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-3 flex items-start gap-1.5 rounded-lg bg-emerald-50 p-2.5 text-[11px] text-emerald-800">
            <ShieldCheck className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            Every transmission is written to the audit ledger with direction, payload size and
            round number.
          </p>
        </Section>
      </div>
    </>
  );
}