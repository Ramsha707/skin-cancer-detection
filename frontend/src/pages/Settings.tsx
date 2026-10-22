import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Database,
  ExternalLink,
  PlugZap,
  Radio,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Terminal,
} from "lucide-react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/AppLayout";
import { EmptyState, Section, StatCard } from "@/components/common";
import { Badge, StatusDot } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useLiveEvents } from "@/hooks/useLiveEvents";
import { api, API_BASE } from "@/services/api";
import type { IntegrationModule } from "@/types";
import { cn, formatDateTime, formatNumber } from "@/lib/utils";

function ModuleRow({ module }: { module: IntegrationModule }) {
  return (
    <tr className="border-b border-ink-950/5 last:border-0">
      <td className="py-2.5 pr-3">
        <p className="font-medium text-ink-900">{module.label}</p>
        <p className="text-[11px] text-ink-500">{module.detail}</p>
        {!module.ready && module.command && (
          <p className="mt-1 flex items-start gap-1 rounded-md bg-ink-950/[0.04] px-2 py-1 font-mono text-[10px] text-ink-700">
            <Terminal className="mt-0.5 h-3 w-3 shrink-0" />
            {module.command}
          </p>
        )}
      </td>
      <td className="whitespace-nowrap py-2.5 pr-3">
        <Badge variant={module.ready ? "success" : module.required ? "destructive" : "warning"}>
          {module.ready ? "ready" : module.required ? "blocked" : "optional"}
        </Badge>
      </td>
      <td className="whitespace-nowrap py-2.5 text-right">
        {module.route ? (
          <Link
            to={module.route}
            className="inline-flex items-center gap-1 text-[11px] font-medium text-accent-700 hover:underline"
          >
            Open <ExternalLink className="h-3 w-3" />
          </Link>
        ) : (
          <span className="text-[11px] text-ink-400">—</span>
        )}
      </td>
    </tr>
  );
}

export default function Settings() {
  const integration = useQuery({
    queryKey: ["integration"],
    queryFn: api.integration,
    refetchInterval: 10_000,
    retry: 1,
  });
  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
    retry: 1,
  });
  const { status, events, lastEventAt, lastHeartbeatAt, reconnects, error } = useLiveEvents();

  const data = integration.data;
  const modules = data?.modules ?? [];
  const readyCount = modules.filter((m) => m.ready).length;
  const requiredBlocked = modules.filter((m) => m.required && !m.ready);

  const runtime: [string, string][] = [
    ["API base URL", API_BASE],
    ["Backend", health.data ? `${health.data.api} · v${health.data.version}` : health.isLoading ? "checking…" : "unreachable"],
    ["Database", data ? data.modules.find((m) => m.key === "database")?.detail ?? "unknown" : "—"],
    ["Pre-trained model", health.data?.pretrained_model ?? "—"],
    ["Dataset", health.data?.dataset ?? "—"],
    ["Demo mode", health.data ? (health.data.demo_mode ? "on (dataset not loaded)" : "off") : "—"],
    ["Server time", health.data ? formatDateTime(health.data.server_time) : "—"],
  ];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings & Integration"
        description="Every subsystem checked against live state, plus the health of the WebSocket that connects them. Nothing here is editable at runtime: configuration is environment-driven, so this page reports rather than pretends."
        icon={Settings2}
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              void integration.refetch();
              void health.refetch();
            }}
            disabled={integration.isFetching || health.isFetching}
          >
            <RefreshCw
              className={cn(
                "h-3.5 w-3.5",
                (integration.isFetching || health.isFetching) && "animate-spin",
              )}
            />
            Re-check
          </Button>
        }
      />

      {integration.error ? (
        <Section title="Diagnostics unavailable" tone="dark">
          <EmptyState
            title="Could not reach the integration endpoint"
            description={
              integration.error instanceof Error
                ? integration.error.message
                : "Unknown error."
            }
            icon={AlertTriangle}
            variant="error"
          />
        </Section>
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="System readiness"
              value={data ? (data.ready ? "Ready" : "Blocked") : "—"}
              hint={
                requiredBlocked.length
                  ? `${requiredBlocked.length} required module(s) blocked`
                  : "All required modules report ready"
              }
              icon={data?.ready ? ShieldCheck : AlertTriangle}
              tone={data?.ready ? "emerald" : "amber"}
            />
            <StatCard
              label="Modules ready"
              value={data ? `${readyCount} / ${modules.length}` : "—"}
              hint="Live checks, not configuration flags"
              icon={PlugZap}
              tone="accent"
            />
            <StatCard
              label="WebSocket subscribers"
              value={data ? formatNumber(data.websocket.subscribers) : "—"}
              hint="Server-side connections, this browser included"
              icon={Radio}
              tone="violet"
            />
            <StatCard
              label="Audit ledger rows"
              value={data ? formatNumber(data.counts.audit_events) : "—"}
              hint={`${data?.counts.rounds ?? 0} rounds, ${data?.counts.model_versions ?? 0} model versions`}
              icon={Database}
              tone="slate"
            />
          </div>

          <div className="grid gap-5 xl:grid-cols-[1.5fr_1fr]">
            <Section
              title="Integration matrix"
              description="Each row calls the same check its owning endpoint uses."
              tone="dark"
            >
              {integration.isLoading ? (
                <div className="h-40 animate-pulse rounded-xl bg-ink-950/5" />
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-ink-950/10 text-[10px] uppercase tracking-wider text-ink-500">
                        <th className="py-2 pr-3 font-medium">Subsystem</th>
                        <th className="py-2 pr-3 font-medium">State</th>
                        <th className="py-2 font-medium" />
                      </tr>
                    </thead>
                    <tbody>
                      {modules.map((module) => (
                        <ModuleRow key={module.key} module={module} />
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Section>

            <div className="space-y-5">
              <Section
                title="Live event bus"
                description="One WebSocket for the whole console."
                tone="dark"
                actions={
                  <Badge variant={status === "live" ? "success" : status === "reconnecting" ? "warning" : "neutral"}>
                    <StatusDot
                      tone={status === "live" ? "emerald" : status === "reconnecting" ? "amber" : "slate"}
                      pulse={status !== "live"}
                    />
                    {status}
                  </Badge>
                }
              >
                <dl className="divide-y divide-ink-950/5 text-xs">
                  <div className="flex items-center justify-between gap-3 py-2">
                    <dt className="text-ink-600">Endpoint</dt>
                    <dd className="font-mono text-ink-800">/ws/training</dd>
                  </div>
                  <div className="flex items-center justify-between gap-3 py-2">
                    <dt className="text-ink-600">Events buffered here</dt>
                    <dd className="tabular-nums text-ink-800">{formatNumber(events.length)}</dd>
                  </div>
                  <div className="flex items-center justify-between gap-3 py-2">
                    <dt className="text-ink-600">Last event</dt>
                    <dd className="font-mono text-ink-800">
                      {lastEventAt ? formatDateTime(lastEventAt) : "none yet"}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-3 py-2">
                    <dt className="text-ink-600">Last heartbeat</dt>
                    <dd className="font-mono text-ink-800">
                      {lastHeartbeatAt ? formatDateTime(lastHeartbeatAt) : "waiting"}
                    </dd>
                  </div>
                  <div className="flex items-center justify-between gap-3 py-2">
                    <dt className="text-ink-600">Connections opened</dt>
                    <dd className="tabular-nums text-ink-800">{reconnects}</dd>
                  </div>
                </dl>
                {error && <p className="mt-2 text-[11px] text-amber-700">{error}</p>}
                <p className="mt-3 rounded-lg border border-ink-950/10 bg-ink-950/[0.03] p-2.5 text-[11px] leading-relaxed text-ink-600">
                  The socket carries notifications only. Every page still reads its numbers
                  from REST, so a dropped frame delays a refresh instead of producing a
                  wrong value.
                </p>
              </Section>

              <Section title="Runtime" description="Environment and process facts." tone="dark">
                <dl className="divide-y divide-ink-950/5">
                  {runtime.map(([key, value]) => (
                    <div key={key} className="flex items-start justify-between gap-3 py-2">
                      <dt className="shrink-0 text-xs text-ink-600">{key}</dt>
                      <dd className="min-w-0 truncate text-right font-mono text-[11px] text-ink-800">
                        {value}
                      </dd>
                    </div>
                  ))}
                </dl>
              </Section>
            </div>
          </div>

          <Section title="Configuration" description="Why this page has no save button." tone="dark">
            <div className="space-y-2 text-xs leading-relaxed text-ink-600">
              <p>
                Hyperparameters (local epochs, learning rate, FedAvg participation) are read
                from the backend settings module and the environment at process start. The UI
                deliberately does not write them: a browser-side override would make a
                federated run irreproducible and would let the demo diverge from the audit
                ledger.
              </p>
              <p>
                To change them, set the corresponding environment variable and restart the
                API. The federated page then shows the new values in its control hints.
              </p>
            </div>
          </Section>
        </>
      )}
    </div>
  );
}
