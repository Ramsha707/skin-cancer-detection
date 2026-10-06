import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity,
  Building2,
  Database,
  Network,
  ShieldCheck,
  Target,
  TrendingUp,
  Layers,
  Boxes,
  FlaskConical,
  ArrowRight,
  AlertTriangle,
} from "lucide-react";
import { PageHeader } from "@/components/layout/AppLayout";
import { Section, StatCard, DemoTag, EmptyState } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ProgressDark } from "@/components/ui/progress";
import { useDashboard } from "@/hooks/useDashboard";
import { api } from "@/services/api";
import { useQuery } from "@tanstack/react-query";
import {
  CANCER_CLASSES,
  CANCER_CLASS_INFO,
  CURRENT_IMPLEMENTED_WEEK,
} from "@/types";
import { formatDateTime, formatNumber, formatPercent } from "@/lib/utils";
import { cn } from "@/lib/utils";

export default function Dashboard() {
  const { data, isLoading, error } = useDashboard();
  const { data: agents } = useQuery({ queryKey: ["agents"], queryFn: api.listAgents, retry: 1 });
  const { data: rounds } = useQuery({ queryKey: ["rounds"], queryFn: api.listRounds, retry: 1 });

  if (error) {
    return (
      <>
        <PageHeader title="Dashboard" />
        <EmptyState
          variant="error"
          icon={AlertTriangle}
          title="Backend is not reachable"
          description="Start the FastAPI server with 'python -m uvicorn app.main:app --app-dir backend --reload' from the repository root, then this page will populate from the live database."
        />
      </>
    );
  }

  if (isLoading || !data) {
    return (
      <>
        <PageHeader title="Dashboard" description="Loading live system state…" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="glass-panel h-28 animate-pulse" />
          ))}
        </div>
      </>
    );
  }

  const recentRounds = (rounds ?? []).slice(0, 5).reverse();
  const totalAssigned = data.total_samples || 1;

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Live federated state across four hospital agents. Every figure below is read from the backend database."
        actions={
          <>
            <Button asChild variant="outline" size="sm">
              <Link to="/federated-training">
                <Network /> Federated Training
              </Link>
            </Button>
            <Button asChild size="sm">
              <Link to="/detection">
                <Target /> Detect Lesion
              </Link>
            </Button>
          </>
        }
      />

      {/* ── Headline counters ───────────────────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Hospital agents"
          value={data.total_agents}
          hint={`${data.active_agents} currently active`}
          icon={Building2}
          tone="accent"
        />
        <StatCard
          label="Federated round"
          value={data.current_round}
          hint={`${data.total_rounds_completed} completed`}
          icon={Network}
          tone="violet"
        />
        <StatCard
          label="Global model"
          value={data.global_model_version ?? "—"}
          hint={data.global_model_version ? `after round ${data.current_round}` : "no rounds yet"}
          icon={Boxes}
          tone="emerald"
        />
        <StatCard
          label="Raw images shared"
          value={data.privacy.raw_images_shared}
          hint="across all rounds and agents"
          icon={ShieldCheck}
          tone="emerald"
        />
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Global accuracy"
          value={formatPercent(data.global_accuracy)}
          icon={Activity}
          tone="accent"
          isReal={!data.demo_mode}
        />
        <StatCard
          label="Global recall"
          value={formatPercent(data.global_recall)}
          hint="sensitivity — false negatives matter most"
          icon={TrendingUp}
          tone="amber"
          isReal={!data.demo_mode}
        />
        <StatCard
          label="Global F1"
          value={formatPercent(data.global_f1)}
          icon={FlaskConical}
          tone="violet"
          isReal={!data.demo_mode}
        />
        <StatCard
          label="Cancer classes"
          value={data.cancer_classes}
          hint="melanoma · BCC · actinic keratosis · benign"
          icon={Target}
          tone="slate"
        />
      </div>

      <div className="mt-6 grid gap-5 lg:grid-cols-3">
        {/* ── Privacy panel ────────────────────────────────────────── */}
        <Section
          title="Privacy status"
          description="What actually leaves each hospital boundary."
          tone="dark"
          actions={<Badge variant="success">Protected</Badge>}
        >
          <div className="space-y-4">
            <div className="rounded-xl border border-emerald-300/70 bg-emerald-100 p-4 text-center">
              <p className="text-[10px] font-medium uppercase tracking-widest text-emerald-700">
                Raw patient images shared
              </p>
              <p className="mt-1 text-4xl font-bold tabular-nums text-emerald-600">
                {data.privacy.raw_images_shared}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-ink-950/10 bg-white/70 p-3">
                <p className="text-[10px] uppercase tracking-wider text-ink-500">Updates shared</p>
                <p className="mt-0.5 text-sm font-semibold text-accent-600">
                  {data.privacy.model_updates_shared ? "YES" : "NO"}
                </p>
              </div>
              <div className="rounded-lg border border-ink-950/10 bg-white/70 p-3">
                <p className="text-[10px] uppercase tracking-wider text-ink-500">Protected sites</p>
                <p className="mt-0.5 text-sm font-semibold text-ink-950">
                  {data.privacy.agents_protected}/{data.privacy.total_agents}
                </p>
              </div>
              <div className="rounded-lg border border-ink-950/10 bg-white/70 p-3">
                <p className="text-[10px] uppercase tracking-wider text-ink-500">Transmissions</p>
                <p className="mt-0.5 text-sm font-semibold text-ink-950">
                  {formatNumber(data.privacy.transmissions_logged)}
                </p>
              </div>
              <div className="rounded-lg border border-ink-950/10 bg-white/70 p-3">
                <p className="text-[10px] uppercase tracking-wider text-ink-500">Total payload</p>
                <p className="mt-0.5 text-sm font-semibold text-ink-950">
                  {formatNumber(data.privacy.total_update_size_kb, 0)} KB
                </p>
              </div>
            </div>

            <p className="text-[11px] leading-relaxed text-ink-600">{data.privacy.guarantee}</p>

            <Button asChild variant="outline" size="sm" className="w-full">
              <Link to="/privacy">
                Privacy Center <ArrowRight />
              </Link>
            </Button>
          </div>
        </Section>

        {/* ── Agent roster ─────────────────────────────────────────── */}
        <Section
          title="Hospital agents"
          description="Each holds an isolated partition of the dermoscopic dataset."
          actions={
            <Button asChild variant="ghost" size="sm">
              <Link to="/agents">View all</Link>
            </Button>
          }
        >
          <div className="space-y-2.5">
            {(agents ?? []).map((a) => (
              <Link
                key={a.id}
                to={`/agents/${a.id}`}
                className="block rounded-xl border border-slate-200 p-3 transition-colors hover:border-accent-400/50 hover:bg-accent-500/5"
              >
                <div className="flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-slate-900">{a.name}</p>
                    <p className="font-mono text-[10px] text-slate-400">{a.agent_id}</p>
                  </div>
                  <Badge
                    variant={
                      a.status === "training"
                        ? "success"
                        : a.status === "synced"
                          ? "info"
                          : a.status === "paused"
                            ? "warning"
                            : "neutral"
                    }
                  >
                    {a.status}
                  </Badge>
                </div>

                <div className="mt-2.5">
                  <div className="mb-1 flex justify-between text-[10px] text-slate-500">
                    <span>{formatNumber(a.dataset_size)} images</span>
                    <span>{formatPercent(a.accuracy)} acc</span>
                  </div>
                  <ProgressDark value={a.dataset_size} max={Math.max(totalAssigned, 1)} />
                </div>
              </Link>
            ))}
            {!agents?.length && (
              <EmptyState
                icon={Building2}
                variant="empty"
                title="No agents registered"
                description="Agents are seeded on first backend start."
              />
            )}
          </div>
        </Section>

        {/* ── Class distribution ───────────────────────────────────── */}
        <Section
          title="Global class distribution"
          description="Partitioned across the four hospital agents."
          actions={data.demo_mode && <DemoTag />}
        >
          {data.dataset.loaded ? (
            <div className="space-y-3">
              {CANCER_CLASSES.map((c) => {
                const count = data.dataset.class_counts?.[c] ?? 0;
                const frac = data.dataset.total_images ? count / data.dataset.total_images : 0;
                return (
                  <div key={c}>
                    <div className="mb-1 flex items-center justify-between text-xs">
                      <span className="flex items-center gap-1.5 text-slate-700">
                        <span
                          className="h-2 w-2 rounded-full"
                          style={{ background: CANCER_CLASS_INFO[c].color }}
                        />
                        {CANCER_CLASS_INFO[c].label}
                      </span>
                      <span className="tabular-nums text-slate-500">
                        {formatNumber(count)} · {formatPercent(frac)}
                      </span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{ width: `${frac * 100}%`, background: CANCER_CLASS_INFO[c].color }}
                      />
                    </div>
                  </div>
                );
              })}
              {data.dataset.imbalance_ratio !== null && (
                <p className="rounded-lg bg-amber-50 p-2.5 text-[11px] text-amber-800">
                  Majority-to-minority ratio is{" "}
                  <strong>{data.dataset.imbalance_ratio.toFixed(1)}×</strong>. Class weighting is
                  applied during local training to avoid majority-class bias.
                </p>
              )}
            </div>
          ) : (
            <EmptyState
              icon={Database}
              title="No dataset loaded"
              description={data.dataset.message}
            />
          )}
        </Section>
      </div>

      {/* ── Recent rounds + roadmap ────────────────────────────────── */}
      <div className="mt-5 grid gap-5 lg:grid-cols-3">
        <Section
          title="Recent federated rounds"
          className="lg:col-span-2"
          tone="light"
          actions={
            <Button asChild variant="ghost" size="sm">
              <Link to="/federated-training">Open dashboard</Link>
            </Button>
          }
        >
          {recentRounds.length ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-[10px] uppercase tracking-wider text-slate-400">
                    <th className="pb-2 pr-4">Round</th>
                    <th className="pb-2 pr-4">Status</th>
                    <th className="pb-2 pr-4">Agents</th>
                    <th className="pb-2 pr-4">Global model</th>
                    <th className="pb-2 pr-4">Accuracy</th>
                    <th className="pb-2 pr-4">Recall</th>
                    <th className="pb-2">Completed</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {recentRounds.map((r) => (
                    <tr key={r.id} className="text-slate-700">
                      <td className="py-2.5 pr-4 font-mono text-xs font-medium">{r.round_number}</td>
                      <td className="py-2.5 pr-4">
                        <Badge
                          variant={
                            r.status === "completed"
                              ? "success"
                              : r.status === "running"
                                ? "info"
                                : r.status === "failed"
                                  ? "destructive"
                                  : "neutral"
                          }
                        >
                          {r.status}
                        </Badge>
                      </td>
                      <td className="py-2.5 pr-4 tabular-nums">{r.participating_agents}</td>
                      <td className="py-2.5 pr-4 font-mono text-xs">
                        {r.global_model_version ?? "--"}
                      </td>
                      <td className="py-2.5 pr-4 tabular-nums">
                        {formatPercent(r.global_accuracy)}
                        {!data.demo_mode && null}
                      </td>
                      <td className="py-2.5 pr-4 tabular-nums">{formatPercent(r.global_recall)}</td>
                      <td className="py-2.5 text-xs text-slate-400">
                        {formatDateTime(r.completed_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState
              icon={Network}
              title="No federated rounds yet"
              description="Start a round from the federated training dashboard to see aggregated global metrics here."
              action={
                <Button asChild size="sm">
                  <Link to="/federated-training">
                    <Network /> Open training dashboard
                  </Link>
                </Button>
              }
            />
          )}
        </Section>

        <Section title="Delivery progress" description={`Week ${CURRENT_IMPLEMENTED_WEEK} of 12 implemented`}>
          <div className="space-y-2">
            {[
              { w: 1, t: "UI/UX + foundation" },
              { w: 2, t: "Backend + database" },
              { w: 3, t: "Hospital agents" },
              { w: 4, t: "Cancer detection" },
              { w: 5, t: "Pre-trained model" },
              { w: 6, t: "Experiments" },
              { w: 7, t: "FedAvg engine" },
              { w: 8, t: "Federated dashboard" },
              { w: 9, t: "Evaluation" },
              { w: 10, t: "XAI + privacy" },
            ].map((r) => {
              const done = r.w <= CURRENT_IMPLEMENTED_WEEK;
              return (
              <motion.div
                key={r.w}
                initial={{ opacity: 0, x: -6 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: r.w * 0.03 }}
                className={cn(
                  "flex items-center gap-3 rounded-lg border px-3 py-2",
                  done ? "border-accent-200 bg-accent-50" : "border-slate-200",
                )}
              >
                <span
                  className={cn(
                    "flex h-5 w-5 items-center justify-center rounded font-mono text-[9px] font-bold",
                    done ? "bg-accent-500 text-white" : "bg-slate-200 text-slate-500",
                  )}
                >
                  {r.w}
                </span>
                <span className={cn("flex-1 text-xs", done ? "text-slate-900" : "text-slate-500")}>
                  {r.t}
                </span>
                <span className={cn("text-[9px] font-medium", done ? "text-emerald-600" : "text-slate-400")}>
                  {done ? "done" : "planned"}
                </span>
              </motion.div>
              );
            })}
          </div>

          <Button asChild variant="outline" size="sm" className="mt-4 w-full">
            <Link to="/timeline">
              <Layers /> Full roadmap
            </Link>
          </Button>
        </Section>
      </div>
    </>
  );
}