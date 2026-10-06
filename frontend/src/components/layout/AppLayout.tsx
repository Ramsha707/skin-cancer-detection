import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, Menu, PanelLeftClose, PanelLeftOpen, X, type LucideIcon } from "lucide-react";
import { Sidebar } from "@/components/layout/Sidebar";
import { Badge } from "@/components/ui/badge";
import { TitleIcon } from "@/components/common";
import { SystemStatusContext, useDashboard } from "@/hooks/useDashboard";
import { RESEARCH_DISCLAIMER } from "@/types";
import { cn } from "@/lib/utils";

/** Persistent medical-safety banner. Shown on every single page, without exception. */
export function SafetyBanner() {
  return (
    <div className="flex items-start gap-2.5 border-b border-amber-200 bg-amber-100/80 px-4 py-2.5">
      <Activity className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
      <p className="text-[11px] leading-relaxed text-amber-800">{RESEARCH_DISCLAIMER}</p>
    </div>
  );
}

function LiveStatusPill() {
  const { data } = useDashboard();
  if (!data) return null;
  const online = data.active_agents > 0;
  return (
    <div className="hidden items-center gap-3 md:flex">
      <Badge variant="neutral">
        <span
          className={cn(
            "h-1.5 w-1.5 rounded-full",
            online ? "animate-pulse bg-emerald-400" : "bg-lilac-500",
          )}
        />
        {data.active_agents}/{data.total_agents} agents
      </Badge>
      <Badge variant="neutral">Round {data.current_round}</Badge>
      <Badge variant={data.demo_mode ? "warning" : "success"}>
        {data.demo_mode ? "Demo mode" : "Live data"}
      </Badge>
    </div>
  );
}

export function AppLayout({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { data } = useDashboard();

  const sysStatus = {
    demoMode: data?.demo_mode ?? true,
    realModelLoaded: data?.real_model_loaded ?? false,
    datasetLoaded: data?.dataset.loaded ?? false,
    datasetName: data?.dataset.name ?? null,
    totalSamples: data?.total_samples ?? 0,
  };

  return (
    <SystemStatusContext.Provider value={sysStatus}>
      <div className="flex h-screen w-full overflow-hidden bg-transparent">
        {/* Desktop sidebar */}
        <div className="hidden lg:block">
          <Sidebar collapsed={collapsed} />
        </div>

        {/* Mobile drawer */}
        <AnimatePresence>
          {mobileOpen && (
            <>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setMobileOpen(false)}
                className="fixed inset-0 z-40 bg-ink-950/30 backdrop-blur-sm lg:hidden"
              />
              <motion.div
                initial={{ x: -280 }}
                animate={{ x: 0 }}
                exit={{ x: -280 }}
                transition={{ type: "spring", damping: 26, stiffness: 260 }}
                className="fixed inset-y-0 left-0 z-50 lg:hidden"
              >
                <Sidebar collapsed={false} onNavigate={() => setMobileOpen(false)} />
                <button
                  onClick={() => setMobileOpen(false)}
                  aria-label="Close navigation"
                  className="absolute -right-11 top-4 rounded-lg bg-white p-2 text-ink-700 shadow-soft"
                >
                  <X className="h-4 w-4" />
                </button>
              </motion.div>
            </>
          )}
        </AnimatePresence>

        {/* Main column */}
        <div className="flex min-w-0 flex-1 flex-col">
          <header className="flex items-center gap-3 border-b border-ink-950/10 bg-white/70 px-4 py-3 backdrop-blur-xl">
            <button
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation"
              className="rounded-lg p-2 text-ink-700 transition-colors hover:bg-ink-950/5 hover:text-ink-950 lg:hidden"
            >
              <Menu className="h-4 w-4" />
            </button>

            <button
              onClick={() => setCollapsed((c) => !c)}
              aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
              className="hidden rounded-lg p-2 text-ink-600 transition-colors hover:bg-ink-950/5 hover:text-ink-950 lg:block"
            >
              {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
            </button>

            <Link to="/" className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-ink-900">
                Privacy-Preserving Multi-Class Skin Cancer Detection
              </p>
              <p className="truncate text-[11px] text-ink-500">
                Federated learning across four hospital agents
              </p>
            </Link>

            <LiveStatusPill />
          </header>

          <SafetyBanner />

          <main className="scrollbar-thin relative flex-1 overflow-y-auto">
            <div className="grid-backdrop absolute inset-0 opacity-40" aria-hidden />
            <div className="relative mx-auto w-full max-w-[1400px] px-4 py-6 md:px-6">
              <AnimatePresence mode="wait">
                <motion.div
                  key={location.pathname}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.18, ease: "easeOut" }}
                >
                  {children}
                </motion.div>
              </AnimatePresence>
            </div>
          </main>
        </div>
      </div>
    </SystemStatusContext.Provider>
  );
}

export function PageHeader({
  title,
  description,
  actions,
  icon: Icon,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
  icon?: LucideIcon;
}) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex items-center gap-3">
        {Icon && <TitleIcon icon={Icon} />}
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink-950 md:text-2xl">{title}</h1>
          {description && <p className="mt-1 max-w-3xl text-sm text-ink-600">{description}</p>}
        </div>
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}