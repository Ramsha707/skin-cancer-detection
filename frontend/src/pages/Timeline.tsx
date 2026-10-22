import { CalendarClock, ExternalLink } from "lucide-react";
import { Link } from "react-router-dom";

import { PageHeader } from "@/components/layout/AppLayout";
import { CURRENT_IMPLEMENTED_WEEK, ROADMAP } from "@/types";
import { cn } from "@/lib/utils";

export default function Timeline() {
  return (
    <div>
      <PageHeader
        title="Project Timeline"
        description={`The 12-week delivery plan. Weeks 1 to ${CURRENT_IMPLEMENTED_WEEK} are implemented in this build; the remainder are sequenced but not yet built.`}
        icon={CalendarClock}
      />
      <ol className="relative space-y-3 border-l border-ink-950/10 pl-6">
        {ROADMAP.map((w) => (
          <li key={w.week} className="relative">
            <span
              className={cn(
                "absolute -left-[31px] top-3 flex h-4 w-4 items-center justify-center rounded-full border-2",
                w.status === "complete"
                  ? "border-accent-400 bg-accent-500"
                  : w.status === "in_progress"
                    ? "border-amber-400 bg-amber-500"
                    : "border-ink-200 bg-white",
              )}
            />
            <div
              className={cn(
                "rounded-xl border p-4",
                w.status === "complete"
                  ? "border-accent-300 bg-accent-100"
                  : "border-ink-950/10 bg-ink-950/5",
              )}
            >
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-accent-500">
                  W{String(w.week).padStart(2, "0")}
                </span>
                <p className="flex-1 text-sm font-medium text-ink-950">{w.title}</p>
                {w.route && (
                  <Link
                    to={w.route}
                    className="inline-flex items-center gap-1 text-[11px] text-accent-700 hover:underline"
                  >
                    Open <ExternalLink className="h-3 w-3" />
                  </Link>
                )}
                <span
                  className={cn(
                    "rounded px-2 py-0.5 text-[10px] font-medium",
                    w.status === "complete"
                      ? "bg-emerald-100 text-emerald-700"
                      : "bg-ink-950/10 text-ink-600",
                  )}
                >
                  {w.status}
                </span>
              </div>
              <p className="mt-1 text-xs text-ink-600">{w.summary}</p>
            </div>
          </li>
        ))}
      </ol>

      <div className="mt-8 rounded-xl border border-ink-950/10 bg-white/80 p-4 text-xs leading-relaxed text-ink-600">
        <p className="font-medium text-ink-800">How completion is decided</p>
        <p className="mt-1">
          A week is marked complete only when its page reads a working endpoint backed by
          executed code. Week 11 adds the live event bus and the audit/settings consoles;
          Week 12 adds the derived judge-demo checklist. No route renders placeholder
          metrics in place of either.
        </p>
      </div>
    </div>
  );
}
