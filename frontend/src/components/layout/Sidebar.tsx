import { NavLink } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity,
  Building2,
  Cpu,
  FlaskConical,
  GitBranch,
  LayoutDashboard,
  Network,
  ScrollText,
  Settings,
  ShieldCheck,
  Sparkles,
  Target,
  Boxes,
  Waypoints,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { CURRENT_IMPLEMENTED_WEEK } from "@/types";

export interface NavItem {
  label: string;
  to: string;
  icon: LucideIcon;
  /** Roadmap week this route belongs to; > CURRENT means "scheduled". */
  week: number;
  description: string;
}

export const NAV_ITEMS: NavItem[] = [
  { label: "Dashboard", to: "/dashboard", icon: LayoutDashboard, week: 1, description: "System overview and live status" },
  { label: "Hospital Agents", to: "/agents", icon: Building2, week: 3, description: "Four isolated hospital agents" },
  { label: "Cancer Detection", to: "/detection", icon: Target, week: 4, description: "Upload and classify a lesion" },
  { label: "Federated Training", to: "/federated-training", icon: Network, week: 8, description: "Live FedAvg round orchestration" },
  { label: "Model Performance", to: "/performance", icon: Activity, week: 9, description: "Accuracy, recall, ROC, confusion matrix" },
  { label: "Architecture", to: "/architecture", icon: GitBranch, week: 1, description: "Six-layer system design" },
  { label: "Privacy Center", to: "/privacy", icon: ShieldCheck, week: 10, description: "Zero raw-image transmission proof" },
  { label: "Experiments", to: "/experiments", icon: FlaskConical, week: 6, description: "Frozen vs selective vs full" },
  { label: "Audit Logs", to: "/audit", icon: ScrollText, week: 10, description: "Chronological event ledger" },
    { label: "Model Registry", to: "/models", icon: Boxes, week: 12, description: "Global model versions" },
  { label: "Model Info", to: "/models/pretrained", icon: Cpu, week: 5, description: "MedSigLIP-448 introspection" },
  { label: "Timeline", to: "/timeline", icon: Waypoints, week: 12, description: "12-week delivery roadmap" },
  { label: "Settings", to: "/settings", icon: Settings, week: 11, description: "Runtime configuration" },
];

export function Sidebar({ collapsed, onNavigate }: { collapsed: boolean; onNavigate?: () => void }) {
  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r border-ink-950/10 bg-white/80 backdrop-blur-xl transition-all duration-300",
        collapsed ? "w-[72px]" : "w-[264px]",
      )}
    >
      {/* Brand */}
      <div className={cn("flex items-center gap-3 border-b border-ink-950/10 px-4 py-5", collapsed && "justify-center px-2")}>
        <div className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-lilac-400 to-periwinkle-400 shadow-glow">
          <Sparkles className="h-5 w-5 text-white" />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <p className="truncate text-sm font-semibold tracking-tight text-ink-950">SkinFL</p>
            <p className="truncate text-[10px] uppercase tracking-widest text-accent-600">
              Federated Oncology
            </p>
          </div>
        )}
      </div>

      {/* Nav */}
      <nav className="scrollbar-thin flex-1 space-y-0.5 overflow-y-auto px-3 py-4">
        {NAV_ITEMS.map((item) => {
          const scheduled = item.week > CURRENT_IMPLEMENTED_WEEK;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={onNavigate}
              title={collapsed ? item.label : undefined}
              className={({ isActive }) =>
                cn(
                  "group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all",
                  isActive
                    ? "bg-accent-100 font-medium text-accent-700 shadow-soft"
                    : "text-ink-700 hover:bg-ink-950/5 hover:text-ink-950",
                  collapsed && "justify-center px-0",
                )
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.span
                      layoutId="nav-active"
                      className="absolute left-0 top-1/2 h-6 w-0.5 -translate-y-1/2 rounded-r bg-accent-400"
                    />
                  )}
                  <item.icon
                    className={cn(
                      "h-4 w-4 shrink-0 transition-colors",
                      isActive ? "text-accent-500" : "text-ink-600 group-hover:text-accent-500",
                    )}
                  />
                  {!collapsed && (
                    <>
                      <span className="flex-1 truncate">{item.label}</span>
                      {scheduled && (
                        <span className="rounded bg-ink-950/[0.06] px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wide text-ink-500">
                          W{item.week}
                        </span>
                      )}
                    </>
                  )}
                </>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Footer */}
      {!collapsed && (
        <div className="border-t border-ink-950/10 p-4">
          <div className="rounded-xl border border-accent-200 bg-accent-100/80 p-3">
            <div className="flex items-center gap-2 text-[11px] font-medium text-accent-700">
              <ShieldCheck className="h-3.5 w-3.5" />
              Raw images shared: 0
            </div>
            <p className="mt-1 text-[10px] leading-relaxed text-ink-600">
              Only model weight updates leave each hospital.
            </p>
          </div>
          <p className="mt-3 text-center text-[9px] uppercase tracking-widest text-ink-400">
            Research prototype · not a device
          </p>
        </div>
      )}
    </aside>
  );
}