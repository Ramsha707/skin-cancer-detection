import { useMemo } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  Building2,
  CircleSlash,
  DownloadCloud,
  Hospital,
  PauseCircle,
  PlayCircle,
  Scale,
  ShieldCheck,
  Users,
} from "lucide-react";
import { PageHeader } from "@/components/layout/AppLayout";
import { Section, StatCard, EmptyState } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  useAgents,
  usePauseAgentTraining,
  useStartAgentTraining,
  useSyncAgent,
} from "@/hooks/useAgents";
import { CANCER_CLASSES, CANCER_CLASS_INFO, type CancerClass } from "@/types";
import { formatNumber, formatPercent } from "@/lib/utils";
import type { HospitalAgent } from "@/types";

const STATUS_VARIANT: Record<
  HospitalAgent["status"],
  "success" | "info" | "warning" | "neutral" | "destructive"
> = {
  training: "success",
  synced: "info",
  paused: "warning",
  offline: "neutral",
  idle: "neutral",
  error: "destructive",
};

/** Dominant class share, used to explain how non-IID a site really is. */
function dominantClass(a: HospitalAgent): CancerClass {
  const entries = Object.entries(a.class_distribution ?? {}).sort((x, y) => y[1] - x[1]);
  const key = entries[0]?.[0];
  return CANCER_CLASSES.includes(key as CancerClass) ? (key as CancerClass) : "benign_lesion";
}

function dominantShare(a: HospitalAgent): number {
  const total = Object.values(a.class_distribution ?? {}).reduce((s, n) => s + n, 0);
  if (!total) return 0;
  return Math.max(...Object.values(a.class_distribution ?? {})) / total;
}

function ClassMix({ agent }: { agent: HospitalAgent }) {
  const dist = agent.class_distribution ?? {};
  const total = Object.values(dist).reduce((s, n) => s + n, 0);

  if (!total) {
    return <p className="text-[11px] text-ink-500">No class distribution recorded.</p>;
  }

  return (
    <div>
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-ink-950/10">
        {CANCER_CLASSES.map((c) => {
          const n = dist[c] ?? 0;
          if (!n) return null;
          return (
            <div
              key={c}
              style={{ width: `${(n / total) * 100}%`, background: CANCER_CLASS_INFO[c].color }}
              title={`${CANCER_CLASS_INFO[c].label}: ${formatNumber(n)} (${formatPercent(n / total)})`}
            />
          );
        })}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">
        {CANCER_CLASSES.map((c) => (
          <span key={c} className="flex items-center gap-1 text-[10px] text-ink-600">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: CANCER_CLASS_INFO[c].color }}
            />
            {formatPercent((dist[c] ?? 0) / total)}
          </span>
        ))}
      </div>
    </div>
  );
}

export default function Agents() {
  const { data: agents, isLoading, error } = useAgents();
  const train = useStartAgentTraining();
  const pause = usePauseAgentTraining();
  const sync = useSyncAgent();

  // Skew is reported across sites, because a per-site number would hide the
  // thing that makes this federated: no two hospitals see the same case mix.
  const skew = useMemo(() => {
    const sized = (agents ?? []).filter((a) => a.dataset_size > 0);
    if (sized.length < 2) return null;
    const sizes = sized.map((a) => a.dataset_size);
    return Math.max(...sizes) / Math.min(...sizes);
  }, [agents]);

  const totalSamples = (agents ?? []).reduce((s, a) => s + a.dataset_size, 0);
  const busy = train.isPending || pause.isPending || sync.isPending;

  if (error) {
    return (
      <>
        <PageHeader title="Hospital Agents" icon={Hospital} />
        <EmptyState
          variant="error"
          icon={AlertTriangle}
          title="Backend is not reachable"
          description="The agent roster is served by the FastAPI API. Start it with 'python -m uvicorn app.main:app --app-dir backend --reload' from the repository root."
        />
      </>
    );
  }

  if (isLoading) {
    return (
      <>
        <PageHeader title="Hospital Agents" description="Loading agent partitions…" icon={Hospital} />
        <div className="grid gap-4 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="glass-panel h-56 animate-pulse" />
          ))}
        </div>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title="Hospital Agents"
        description="Four simulated hospital sites, each holding an isolated, non-IID partition of the HAM10000 training split. Only model updates cross site boundaries."
        icon={Hospital}
        actions={
          <Button asChild variant="outline" size="sm">
            <Link to="/privacy">
              <ShieldCheck /> Privacy guarantees
            </Link>
          </Button>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Agent sites"
          value={(agents ?? []).length}
          hint="simulated in one process"
          icon={Building2}
          tone="accent"
        />
        <StatCard
          label="Training images held"
          value={formatNumber(totalSamples)}
          hint="never transmitted"
          icon={Users}
          tone="emerald"
        />
        <StatCard
          label="Site size skew"
          value={skew ? `${skew.toFixed(2)}×` : "—"}
          hint="largest ÷ smallest partition"
          icon={Scale}
          tone={skew && skew > 1.5 ? "amber" : "violet"}
        />
        <StatCard
          label="Currently training"
          value={(agents ?? []).filter((a) => a.status === "training").length}
          hint={`${(agents ?? []).filter((a) => a.status === "paused").length} paused`}
          icon={Activity}
          tone="violet"
        />
      </div>

      {skew !== null && (
        <p className="mt-4 rounded-xl border border-ink-950/10 bg-white p-3 text-[11px] leading-relaxed text-ink-600">
          Partition sizes are capped at 1.5× the mean so no site is starved of data.
          Unconstrained Dirichlet sampling produced a 16.9× spread (205 vs 3,462 images);
          the current split keeps sizes near 1,413–2,050 while the <strong>class mix</strong>{" "}
          stays deliberately skewed, which is the part that actually exercises federated
          averaging.
        </p>
      )}

      <div className="mt-5 grid gap-4 md:grid-cols-2">
        {(agents ?? []).map((a, i) => (
          <motion.div
            key={a.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
          >
            <Section
              title={a.name}
              description={a.location ?? "Location not recorded"}
              actions={<Badge variant={STATUS_VARIANT[a.status]}>{a.status}</Badge>}
            >
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-[11px] text-ink-500">
                  <span className="font-mono">{a.agent_id}</span>
                  <span>·</span>
                  <span>{formatNumber(a.dataset_size)} images</span>
                  <span>·</span>
                  <span>{a.rounds_participated} rounds</span>
                </div>

                <div>
                  <p className="mb-1.5 text-[10px] font-medium uppercase tracking-widest text-ink-500">
                    Local class mix
                  </p>
                  <ClassMix agent={a} />
                  <p className="mt-1.5 text-[10px] text-ink-500">
                    {formatPercent(dominantShare(a))} of this site&apos;s images are{" "}
                    {CANCER_CLASS_INFO[dominantClass(a)].label.toLowerCase()}.
                  </p>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center">
                  <div className="rounded-lg border border-ink-950/10 p-2">
                    <p className="text-[9px] uppercase tracking-wider text-ink-500">Model</p>
                    <p className="truncate font-mono text-[11px] text-ink-800">{a.model_version}</p>
                  </div>
                  <div className="rounded-lg border border-ink-950/10 p-2">
                    <p className="text-[9px] uppercase tracking-wider text-ink-500">Privacy</p>
                    <p className="text-[11px] text-emerald-600">{a.privacy_status}</p>
                  </div>
                  <div className="rounded-lg border border-ink-950/10 p-2">
                    <p className="text-[9px] uppercase tracking-wider text-ink-500">Last sync</p>
                    <p className="text-[11px] text-ink-700">
                      {a.last_sync ? new Date(a.last_sync).toLocaleDateString() : "never"}
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <Button
                    size="sm"
                    disabled={busy || a.status === "training"}
                    onClick={() => train.mutate(a.id)}
                  >
                    <PlayCircle /> Train
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={busy || a.status !== "training"}
                    onClick={() => pause.mutate(a.id)}
                  >
                    <PauseCircle /> Pause
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={busy}
                    onClick={() => sync.mutate(a.id)}
                  >
                    <DownloadCloud /> Sync
                  </Button>
                  <Button asChild size="sm" variant="ghost">
                    <Link to={`/agents/${a.id}`}>Detail</Link>
                  </Button>
                </div>

                {a.status === "synced" && (
                  <p className="rounded-lg bg-sky-50 p-2 text-[10px] text-sky-700">
                    Received global weights. The audit ledger logs this as a download into the
                    hospital, so both directions of every transfer stay visible.
                  </p>
                )}
              </div>
            </Section>
          </motion.div>
        ))}

        {!agents?.length && (
          <EmptyState
            icon={CircleSlash}
            variant="empty"
            title="No agents registered"
            description="The four hospital sites are seeded on first backend start. Delete skinfl.db and restart the API to recreate them."
          />
        )}
      </div>
    </>
  );
}