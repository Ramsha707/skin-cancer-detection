import { ROADMAP } from "@/types";
import { PageHeader } from "@/components/layout/AppLayout";
import { ScheduledPage } from "@/components/common";

export default function Timeline() {
  return (
    <div>
      <PageHeader
        title="Project Timeline"
        description="The 12-week delivery plan. Weeks 1 to 6 are implemented in this build; the remainder are sequenced but not yet built."
      />
      <ol className="relative space-y-3 border-l border-white/10 pl-6">
        {ROADMAP.map((w) => (
          <li key={w.week} className="relative">
            <span
              className={`absolute -left-[31px] top-3 flex h-4 w-4 items-center justify-center rounded-full border-2 ${
                w.status === "complete"
                  ? "border-accent-400 bg-accent-500"
                  : w.status === "in_progress"
                    ? "border-amber-400 bg-amber-500"
                    : "border-navy-700 bg-navy-900"
              }`}
            />
            <div
              className={`rounded-xl border p-4 ${
                w.status === "complete"
                  ? "border-accent-400/20 bg-accent-500/5"
                  : "border-white/10 bg-navy-900/40"
              }`}
            >
              <div className="flex items-center gap-3">
                <span className="font-mono text-xs text-accent-300">W{String(w.week).padStart(2, "0")}</span>
                <p className="flex-1 text-sm font-medium text-white">{w.title}</p>
                <span
                  className={`rounded px-2 py-0.5 text-[10px] font-medium ${
                    w.status === "complete" ? "bg-emerald-500/15 text-emerald-300" : "bg-white/5 text-navy-500"
                  }`}
                >
                  {w.status}
                </span>
              </div>
              <p className="mt-1 text-xs text-navy-400">{w.summary}</p>
            </div>
          </li>
        ))}
      </ol>

      <div className="mt-10">
        <ScheduledPage
          week={12}
          title="Run Judge Demo"
          planned="A scripted twelve-step walkthrough that starts four agents and ends on the privacy confirmation screen."
        />
      </div>
    </div>
  );
}