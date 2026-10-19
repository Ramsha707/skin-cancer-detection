import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowDownToLine,
  ArrowUpFromLine,
  CheckCircle2,
  Database,
  EyeOff,
  FileLock2,
  Images,
  Lock,
  Network,
  Shield,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import { PageHeader } from "@/components/layout/AppLayout";
import { Badge } from "@/components/ui/badge";
import { EmptyState, Section, StatCard } from "@/components/common";
import { api } from "@/services/api";
import {
  CURRENT_IMPLEMENTED_WEEK,
  type PrivacyInvariant,
  type PrivacyLedgerEntry,
  type PrivacySummaryOut,
} from "@/types";
import { cn, formatDateTime, formatNumber } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const DIRECTION_LABEL: Record<string, string> = {
  hospital_to_aggregator: "Hospital → aggregator",
  aggregator_to_hospital: "Aggregator → hospital",
};

function formatKb(kb: number | null): string {
  if (kb === null || kb === undefined) return "—";
  if (kb >= 1024) return `${(kb / 1024).toFixed(2)} MB`;
  return `${kb.toFixed(1)} KB`;
}

function directionIcon(direction: string | null) {
  if (direction === "hospital_to_aggregator") return ArrowUpFromLine;
  if (direction === "aggregator_to_hospital") return ArrowDownToLine;
  return Network;
}

/* ------------------------------------------------------------------ */
/* Invariants                                                          */
/* ------------------------------------------------------------------ */

function InvariantList({ invariants }: { invariants: PrivacyInvariant[] }) {
  return (
    <ul className="space-y-2.5">
      {invariants.map((item) => (
        <li
          key={item.key}
          className={cn(
            "flex items-start gap-3 rounded-xl border p-3.5",
            item.enforced ? "border-emerald-300/70 bg-emerald-50" : "border-red-300/70 bg-red-50",
          )}
        >
          {item.enforced ? (
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
          ) : (
            <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
          )}
          <div className="min-w-0">
            <p
              className={cn(
                "text-sm font-medium",
                item.enforced ? "text-emerald-900" : "text-red-900",
              )}
            >
              {item.statement}
            </p>
            <p
              className={cn(
                "mt-0.5 text-[11px] leading-relaxed",
                item.enforced ? "text-emerald-700/90" : "text-red-700/90",
              )}
            >
              {item.evidence}
            </p>
          </div>
          <Badge variant={item.enforced ? "success" : "destructive"} className="ml-auto shrink-0">
            {item.enforced ? "enforced" : "violated"}
          </Badge>
        </li>
      ))}
    </ul>
  );
}

/* ------------------------------------------------------------------ */
/* Ledger                                                              */
/* ------------------------------------------------------------------ */

function Ledger({ entries }: { entries: PrivacyLedgerEntry[] }) {
  if (!entries.length) {
    return (
      <EmptyState
        title="No transmissions recorded yet"
        description="Run a federated round and the weight-update traffic appears here."
        icon={Network}
        variant="empty"
      />
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-ink-950/10 text-[10px] uppercase tracking-wider text-ink-500">
            <th className="py-2 pr-3 font-medium">When</th>
            <th className="py-2 pr-3 font-medium">Event</th>
            <th className="py-2 pr-3 font-medium">Direction</th>
            <th className="py-2 pr-3 font-medium">Agent</th>
            <th className="py-2 pr-3 font-medium">Payload</th>
            <th className="py-2 font-medium">Round</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => {
            const Icon = directionIcon(e.direction);
            return (
              <tr key={e.id} className="border-b border-ink-950/5 last:border-0">
                <td className="whitespace-nowrap py-2 pr-3 text-ink-600">
                  {formatDateTime(e.timestamp)}
                </td>
                <td className="py-2 pr-3">
                  <div className="flex items-center gap-1.5">
                    <Icon className="h-3.5 w-3.5 shrink-0 text-ink-500" />
                    <span className="font-medium text-ink-900">{e.label || e.event_type}</span>
                  </div>
                </td>
                <td className="py-2 pr-3 text-ink-600">
                  {e.direction ? DIRECTION_LABEL[e.direction] ?? e.direction : "—"}
                </td>
                <td className="py-2 pr-3 text-ink-600">{e.agent_id ?? "—"}</td>
                <td className="py-2 pr-3 tabular-nums text-ink-600">
                  {e.payload_type ? (
                    <span className="inline-flex items-center gap-1">
                      <FileLock2 className="h-3 w-3 text-ink-400" />
                      {e.payload_type} · {formatKb(e.payload_kb)}
                    </span>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="py-2 tabular-nums text-ink-600">{e.round_number ?? "—"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Agent traces                                                        */
/* ------------------------------------------------------------------ */

function AgentTraces({ summary }: { summary: PrivacySummaryOut }) {
  const agents = useMemo(
    () => [...summary.agents].sort((a, b) => a.agent_id.localeCompare(b.agent_id)),
    [summary.agents],
  );

  if (!agents.length) {
    return (
      <EmptyState
        title="No agents registered"
        description="Hospital agents appear here once they are seeded."
        icon={Database}
        variant="empty"
      />
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead>
          <tr className="border-b border-ink-950/10 text-[10px] uppercase tracking-wider text-ink-500">
            <th className="py-2 pr-3 font-medium">Agent</th>
            <th className="py-2 pr-3 font-medium">Privacy</th>
            <th className="py-2 pr-3 font-medium">Uploads</th>
            <th className="py-2 pr-3 font-medium">Downloads</th>
            <th className="py-2 pr-3 font-medium">Rounds</th>
            <th className="py-2 font-medium">Last activity</th>
          </tr>
        </thead>
        <tbody>
          {agents.map((a) => (
            <tr key={a.agent_id} className="border-b border-ink-950/5 last:border-0">
              <td className="py-2 pr-3">
                <span className="font-medium text-ink-900">{a.name}</span>
                <span className="ml-1.5 text-ink-400">{a.agent_id}</span>
              </td>
              <td className="py-2 pr-3">
                <Badge variant={a.privacy_status === "protected" ? "success" : "destructive"}>
                  {a.privacy_status}
                </Badge>
              </td>
              <td className="py-2 pr-3 tabular-nums text-ink-600">
                {a.uploads} · {formatKb(a.upload_kb)}
              </td>
              <td className="py-2 pr-3 tabular-nums text-ink-600">
                {a.downloads} · {formatKb(a.download_kb)}
              </td>
              <td className="py-2 pr-3 tabular-nums text-ink-600">{a.rounds_participated}</td>
              <td className="py-2 whitespace-nowrap text-ink-600">
                {a.last_activity ? formatDateTime(a.last_activity) : "—"}
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

export default function Privacy() {
  const [summary, setSummary] = useState<PrivacySummaryOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .privacySummary(50)
      .then((s) => {
        if (!cancelled) setSummary(s);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to load privacy summary.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const allEnforced = summary ? summary.invariants.every((i) => i.enforced) : true;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Privacy Centre"
        description={`What leaves each hospital, and what structurally cannot. Every number is derived from the audit ledger. Weeks 1–${CURRENT_IMPLEMENTED_WEEK} implemented.`}
        icon={Shield}
        actions={
          summary ? (
            <Badge variant={allEnforced ? "success" : "destructive"}>
              {allEnforced ? (
                <>
                  <ShieldCheck className="h-3 w-3" /> All invariants hold
                </>
              ) : (
                <>
                  <AlertTriangle className="h-3 w-3" /> Invariant violated
                </>
              )}
            </Badge>
          ) : undefined
        }
      />

      {loading ? (
        <Section title="Loading" description="Reading the ledger and agent roster.">
          <div className="h-24 animate-pulse rounded-xl bg-ink-950/5" />
        </Section>
      ) : error ? (
        <Section title="Privacy summary unavailable" tone="dark">
          <EmptyState
            title="Could not load the privacy summary"
            description={error}
            variant="error"
            icon={AlertTriangle}
          />
        </Section>
      ) : summary ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Raw images shared"
              value={formatNumber(summary.counters.raw_images_shared)}
              hint="Structurally impossible by design"
              icon={EyeOff}
              tone="emerald"
            />
            <StatCard
              label="Model updates shared"
              value={summary.counters.model_updates_shared ? "Yes" : "No"}
              hint="Weights only, never pixels"
              icon={ArrowUpFromLine}
              tone="accent"
            />
            <StatCard
              label="Total update size"
              value={formatKb(summary.counters.total_update_size_kb)}
              hint="Tensor payloads in the ledger"
              icon={FileLock2}
              tone="slate"
            />
            <StatCard
              label="Transmissions logged"
              value={formatNumber(summary.counters.transmissions_logged)}
              hint={`${summary.counters.broadcasts_to_hospitals} broadcasts to hospitals`}
              icon={Network}
              tone="violet"
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Detections performed"
              value={formatNumber(summary.counters.detections_performed)}
              hint="Inference ran locally"
              icon={Database}
              tone="accent"
            />
            <StatCard
              label="Explanations generated"
              value={formatNumber(summary.counters.explainability_generated)}
              hint="Grad-CAM on live uploads"
              icon={Images}
              tone="violet"
            />
            <StatCard
              label="Agents protected"
              value={`${summary.counters.agents_protected} / ${summary.counters.total_agents}`}
              hint="Privacy status = protected"
              icon={Lock}
              tone={summary.counters.agents_protected === summary.counters.total_agents ? "emerald" : "amber"}
            />
            <StatCard
              label="Mechanism"
              value="FedAvg"
              hint="Aggregation, not data pooling"
              icon={Network}
              tone="slate"
            />
          </div>

          <Section
            title="Privacy invariants"
            description="Checked live against the ledger on every request, not asserted in prose."
            tone="dark"
          >
            <InvariantList invariants={summary.invariants} />
          </Section>

          <Section
            title="Per-agent trace"
            description="How much traffic each hospital sent and received."
            tone="dark"
          >
            <AgentTraces summary={summary} />
          </Section>

          <Section
            title="Transmission ledger"
            description="Every weight update and broadcast, most recent first."
            tone="dark"
          >
            <Ledger entries={summary.ledger} />
          </Section>

          <div className="grid gap-5 lg:grid-cols-2">
            <Section title="Mechanism" description="How privacy is preserved." tone="dark">
              <p className="text-xs leading-relaxed text-ink-600">{summary.mechanism}</p>
            </Section>
            <Section title="Guarantee" description="What exactly is promised." tone="dark">
              <p className="text-xs leading-relaxed text-ink-600">{summary.guarantee}</p>
            </Section>
          </div>

          <div className="rounded-xl border border-ink-950/10 bg-ink-950/5 p-4 text-xs leading-relaxed text-ink-600">
            <p className="font-medium text-ink-800">Scope of this claim</p>
            <p className="mt-1">
              This panel describes what crosses the network in <em>this</em> prototype: weight
              tensors and metadata. It is not a differential-privacy guarantee, and it does not
              claim resistance to gradient-inversion attacks. What it does prove is that no raw
              image and no per-image identifier is ever serialised into a transmission, because the
              audit writer only accepts a tensor size for weight payloads.
            </p>
          </div>
        </>
      ) : null}
    </div>
  );
}
