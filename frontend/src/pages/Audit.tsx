import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Radio,
  RefreshCw,
  Search,
  ScrollText,
} from "lucide-react";

import { PageHeader } from "@/components/layout/AppLayout";
import { EmptyState, Section, StatCard } from "@/components/common";
import { Badge, StatusDot } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useLiveEvents } from "@/hooks/useLiveEvents";
import { api } from "@/services/api";
import type { AuditEventType, AuditLog } from "@/types";
import { cn, formatDateTime, formatNumber } from "@/lib/utils";

const EVENT_LABELS: Record<AuditEventType, string> = {
  local_training_started: "Local training started",
  local_training_completed: "Local training completed",
  model_update_generated: "Model update generated",
  model_update_transferred: "Model update transferred",
  fedavg_started: "FedAvg started",
  fedavg_completed: "FedAvg completed",
  global_model_generated: "Global model generated",
  model_broadcast: "Global model broadcast",
  agent_synchronized: "Agent synchronized",
  detection_performed: "Detection performed",
  explainability_generated: "Explainability generated",
  round_completed: "Round completed",
};

const EVENT_ORDER = Object.keys(EVENT_LABELS) as AuditEventType[];

function eventTone(eventType: string): "success" | "info" | "warning" | "neutral" {
  if (eventType === "round_completed" || eventType === "fedavg_completed") return "success";
  if (eventType === "model_update_transferred" || eventType === "model_broadcast") return "info";
  if (eventType === "detection_performed" || eventType === "explainability_generated") {
    return "warning";
  }
  return "neutral";
}

function statusVariant(status: AuditLog["status"]): "success" | "warning" | "destructive" {
  if (status === "success") return "success";
  if (status === "pending") return "warning";
  return "destructive";
}

function DetailPanel({ log }: { log: AuditLog }) {
  const entries = Object.entries(log.details ?? {});
  return (
    <div className="rounded-lg border border-ink-950/10 bg-ink-950/[0.025] p-3">
      <dl className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        <div>
          <dt className="text-[10px] uppercase tracking-wider text-ink-500">Event id</dt>
          <dd className="font-mono text-xs text-ink-800">{log.id}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-wider text-ink-500">Timestamp</dt>
          <dd className="font-mono text-xs text-ink-800">{log.timestamp}</dd>
        </div>
        <div>
          <dt className="text-[10px] uppercase tracking-wider text-ink-500">Event type</dt>
          <dd className="font-mono text-xs text-ink-800">{log.event_type}</dd>
        </div>
        {entries.length ? (
          entries.map(([key, value]) => (
            <div key={key}>
              <dt className="text-[10px] uppercase tracking-wider text-ink-500">{key}</dt>
              <dd className="break-all font-mono text-xs text-ink-800">
                {typeof value === "string" ? value : JSON.stringify(value)}
              </dd>
            </div>
          ))
        ) : (
          <div className="sm:col-span-2 lg:col-span-3">
            <dt className="text-[10px] uppercase tracking-wider text-ink-500">Details</dt>
            <dd className="text-xs text-ink-500">No structured details on this event.</dd>
          </div>
        )}
      </dl>
    </div>
  );
}

function LiveTicker() {
  const { status, events, lastHeartbeatAt, lastEventAt, reconnects, error } = useLiveEvents();
  const recent = useMemo(
    () => [...events].reverse().slice(0, 12),
    [events],
  );

  return (
    <Section
      title="Live stream"
      description="Every event is a notification; the table refetches the ledger."
      actions={
        <Badge variant={status === "live" ? "success" : status === "reconnecting" ? "warning" : "neutral"}>
          <StatusDot tone={status === "live" ? "emerald" : status === "reconnecting" ? "amber" : "slate"} pulse={status !== "live"} />
          {status}
        </Badge>
      }
      tone="dark"
    >
      <div className="mb-3 grid grid-cols-2 gap-2 text-[11px]">
        <div className="rounded-lg border border-ink-950/10 p-2">
          <p className="text-ink-500">Last event</p>
          <p className="truncate font-mono text-ink-800">
            {lastEventAt ? formatDateTime(lastEventAt) : "none yet"}
          </p>
        </div>
        <div className="rounded-lg border border-ink-950/10 p-2">
          <p className="text-ink-500">Last heartbeat</p>
          <p className="truncate font-mono text-ink-800">
            {lastHeartbeatAt ? formatDateTime(lastHeartbeatAt) : "waiting"}
          </p>
        </div>
      </div>
      {error && <p className="mb-2 text-[11px] text-amber-700">{error}</p>}
      <p className="mb-2 text-[11px] text-ink-500">
        {events.length} buffered · {reconnects} connection{reconnects === 1 ? "" : "s"}
      </p>
      {recent.length === 0 ? (
        <p className="rounded-lg border border-dashed border-ink-950/15 p-4 text-xs text-ink-500">
          No live events yet. Start a federated round or run a detection.
        </p>
      ) : (
        <ol className="space-y-1.5">
          {recent.map((event) => (
            <li key={`${event.type}-${event.id}`} className="flex items-start gap-2 text-[11px]">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent-400" />
              <div className="min-w-0">
                <p className="truncate text-ink-800">
                  <span className="font-medium">{String(event.payload.label ?? event.payload.event ?? event.type)}</span>
                  {typeof event.payload.message === "string" && (
                    <span className="text-ink-500"> — {event.payload.message}</span>
                  )}
                </p>
                <p className="font-mono text-[10px] text-ink-400">
                  {event.source} · {formatDateTime(event.timestamp)}
                </p>
              </div>
            </li>
          ))}
        </ol>
      )}
    </Section>
  );
}

export default function Audit() {
  const [eventType, setEventType] = useState<string>("");
  const [round, setRound] = useState<string>("");
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<number | null>(null);
  const { events } = useLiveEvents();

  const roundNumber = round.trim() === "" ? undefined : Number(round);
  const filtersValid = roundNumber === undefined || Number.isFinite(roundNumber);
  const auditFilters = {
    event_type: eventType || undefined,
    round_number: filtersValid ? roundNumber : undefined,
  };

  const { data, isLoading, isFetching, error, refetch } = useQuery<AuditLog[]>({
    queryKey: ["audit", auditFilters],
    queryFn: () => api.listAuditLogs(250, auditFilters),
    refetchInterval: 15_000,
    enabled: filtersValid,
  });

  const rows = useMemo(() => {
    const term = query.trim().toLowerCase();
    if (!term) return data ?? [];
    return (data ?? []).filter(
      (log) =>
        log.message.toLowerCase().includes(term) ||
        log.event_type.toLowerCase().includes(term) ||
        (log.agent_id ?? "").toLowerCase().includes(term),
    );
  }, [data, query]);

  const auditBuffer = useMemo(
    () => events.filter((e) => e.type === "audit").length,
    [events],
  );

  const roundNumbers = useMemo(() => {
    const numbers = new Set<number>();
    for (const log of data ?? []) if (log.round_number !== null) numbers.add(log.round_number);
    return [...numbers].sort((a, b) => b - a).slice(0, 12);
  }, [data]);

  const all = data ?? [];
  const failures = all.filter((log) => log.status !== "success").length;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Audit Logs"
        description="The append-only ledger every privacy and dashboard counter derives from. Filter it, expand a row, and watch new events arrive over the WebSocket."
        icon={ScrollText}
        actions={
          <Button variant="outline" size="sm" onClick={() => void refetch()} disabled={isFetching}>
            <RefreshCw className={cn("h-3.5 w-3.5", isFetching && "animate-spin")} />
            Refresh
          </Button>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Rows loaded" value={formatNumber(all.length)} hint="Newest first, limit 250" icon={ScrollText} tone="slate" />
        <StatCard label="Rounds represented" value={formatNumber(roundNumbers.length)} hint="Distinct round numbers in this view" icon={Activity} tone="accent" />
        <StatCard label="Non-success rows" value={formatNumber(failures)} hint="Failed or pending events" icon={AlertTriangle} tone={failures ? "amber" : "emerald"} />
        <StatCard label="Live audit events" value={formatNumber(auditBuffer)} hint="Buffered by this browser session" icon={Radio} tone="violet" />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.7fr_1fr]">
        <Section
          title="Ledger"
          description="Server-side filters; search narrows the loaded page only."
          actions={
            <div className="flex flex-wrap items-end gap-2">
              <label className="text-[10px] uppercase tracking-wider text-ink-500">
                Type
                <select
                  value={eventType}
                  onChange={(event) => setEventType(event.target.value)}
                  className="mt-1 block h-8 rounded-lg border border-ink-950/10 bg-white px-2 text-xs text-ink-800"
                >
                  <option value="">All types</option>
                  {EVENT_ORDER.map((type) => (
                    <option key={type} value={type}>
                      {EVENT_LABELS[type]}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-[10px] uppercase tracking-wider text-ink-500">
                Round
                <input
                  type="number"
                  min={1}
                  value={round}
                  onChange={(event) => setRound(event.target.value)}
                  placeholder="any"
                  className="mt-1 block h-8 w-20 rounded-lg border border-ink-950/10 bg-white px-2 text-xs text-ink-800"
                />
              </label>
            </div>
          }
        >
          <div className="relative mb-3">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-ink-400" />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter loaded rows by message, type or agent"
              className="h-9 w-full rounded-lg border border-ink-950/10 bg-white pl-9 pr-3 text-xs text-ink-800 placeholder:text-ink-400"
            />
          </div>

          {!filtersValid ? (
            <EmptyState
              title="Round filter is not a number"
              description="Clear the round field or enter an integer round number."
              icon={AlertTriangle}
              variant="error"
            />
          ) : isLoading ? (
            <div className="h-40 animate-pulse rounded-xl bg-ink-950/5" />
          ) : error ? (
            <EmptyState
              title="Could not load the ledger"
              description={error instanceof Error ? error.message : "Unknown error."}
              icon={AlertTriangle}
              variant="error"
            />
          ) : rows.length === 0 ? (
            <EmptyState
              title="No matching events"
              description="Run a federated round or a detection, or clear the filters."
              icon={ScrollText}
              variant="empty"
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-ink-950/10 text-[10px] uppercase tracking-wider text-ink-500">
                    <th className="w-6 py-2 pr-2 font-medium" />
                    <th className="py-2 pr-3 font-medium">When</th>
                    <th className="py-2 pr-3 font-medium">Event</th>
                    <th className="py-2 pr-3 font-medium">Agent</th>
                    <th className="py-2 pr-3 font-medium">Round</th>
                    <th className="py-2 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((log) => {
                    const isOpen = expanded === log.id;
                    return (
                      <FragmentRow
                        key={log.id}
                        log={log}
                        open={isOpen}
                        onToggle={() => setExpanded(isOpen ? null : log.id)}
                      />
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Section>

        <LiveTicker />
      </div>
    </div>
  );
}

function FragmentRow({
  log,
  open,
  onToggle,
}: {
  log: AuditLog;
  open: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr
        onClick={onToggle}
        className={cn(
          "cursor-pointer border-b border-ink-950/5 transition-colors hover:bg-ink-950/[0.025]",
          open && "bg-ink-950/[0.025]",
        )}
      >
        <td className="py-2 pr-2 text-ink-400">
          {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
        </td>
        <td className="whitespace-nowrap py-2 pr-3 text-ink-600">{formatDateTime(log.timestamp)}</td>
        <td className="py-2 pr-3">
          <div className="flex items-center gap-1.5">
            <Badge variant={eventTone(log.event_type)} className="shrink-0">
              {EVENT_LABELS[log.event_type] ?? log.event_type}
            </Badge>
            <span className="truncate text-ink-600">{log.message}</span>
          </div>
        </td>
        <td className="py-2 pr-3 font-mono text-ink-600">{log.agent_id ?? "—"}</td>
        <td className="py-2 pr-3 tabular-nums text-ink-600">{log.round_number ?? "—"}</td>
        <td className="py-2">
          <Badge variant={statusVariant(log.status)}>{log.status}</Badge>
        </td>
      </tr>
      {open && (
        <tr className="border-b border-ink-950/5">
          <td />
          <td colSpan={5} className="pb-3 pr-3">
            <DetailPanel log={log} />
          </td>
        </tr>
      )}
    </>
  );
}
