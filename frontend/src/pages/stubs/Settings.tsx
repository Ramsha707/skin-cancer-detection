import { Settings2 } from "lucide-react";
import { ScheduledPage } from "@/components/common";
import { api, API_BASE } from "@/services/api";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";

export default function Settings() {
  const { data, isLoading, error } = useQuery({ queryKey: ["health"], queryFn: api.health, retry: false });

  const rows: [string, string][] = [
    ["API base URL", API_BASE],
    ["Backend", data ? `${data.api} · v${data.version}` : isLoading ? "checking…" : "unreachable"],
    ["Database", data?.database ?? "--"],
    ["Pre-trained model", data?.pretrained_model ?? "--"],
    ["Dataset", data?.dataset ?? "--"],
    ["Demo mode", data ? (data.demo_mode ? "on" : "off") : "--"],
    ["Server time", data?.server_time ?? "--"],
  ];

  return (
    <div className="space-y-6">
      <section className="glass-panel p-5">
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings2 className="h-4 w-4 text-accent-500" />
            <h2 className="text-sm font-semibold text-ink-950">Runtime status</h2>
          </div>
          {error && <Badge variant="destructive">backend unreachable</Badge>}
        </div>
        <dl className="divide-y divide-ink-950/5">
          {rows.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between gap-4 py-2.5">
              <dt className="text-xs text-ink-600">{k}</dt>
              <dd className="truncate font-mono text-xs text-ink-800">{v}</dd>
            </div>
          ))}
        </dl>
      </section>

      <ScheduledPage
        week={11}
        title="Configuration & Controls"
        planned="Editable federated hyperparameters (rounds, agents, local epochs, FedAvg participation fraction, learning rates) written to a backend config store."
      />
    </div>
  );
}