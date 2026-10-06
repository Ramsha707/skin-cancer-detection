import { motion } from "framer-motion";
import {
  BrainCircuit,
  Boxes,
  Database,
  Lock,
  Network,
  Server,
  Stethoscope,
  Upload,
  type LucideIcon,
} from "lucide-react";
import { PageHeader } from "@/components/layout/AppLayout";
import { Badge } from "@/components/ui/badge";

/** The six-layer architecture, exactly as implemented. */
const LAYERS: {
  n: number;
  title: string;
  icon: LucideIcon;
  tone: string;
  body: string;
  detail: string[];
}[] = [
  {
    n: 1,
    title: "Hospital Agents",
    icon: Stethoscope,
    tone: "accent",
    body: "Four independent sites, each running its own agent process abstraction.",
    detail: ["Agent A · Agent B", "Agent C · Agent D", "No shared filesystem", "Own local model copy"],
  },
  {
    n: 2,
    title: "Local Data",
    icon: Database,
    tone: "emerald",
    body: "Each site holds a patient-level partition of the dermoscopic dataset.",
    detail: ["HAM10000 metadata", "Patient ID preserved", "Non-IID class mix", "Never replicated"],
  },
  {
    n: 3,
    title: "Local Preprocessing",
    icon: Upload,
    tone: "violet",
    body: "Identical transform pipeline across sites so updates are comparable.",
    detail: ["Resize 448×448 bicubic", "Rescale 1/255", "Normalise mean=std=0.5", "Augment + class weights"],
  },
  {
    n: 4,
    title: "Cancer AI Model",
    icon: BrainCircuit,
    tone: "amber",
    body: "MedSigLIP-448 adapted into a four-class lesion classifier.",
    detail: ["Vision tower backbone", "Text-prompt-init head", "3 freeze strategies", "Grad-CAM ready"],
  },
  {
    n: 5,
    title: "Federated Learning",
    icon: Network,
    tone: "accent",
    body: "FedAvg merges weight deltas; raw images never leave the site.",
    detail: ["Local training", "Secure weight transfer", "Size-weighted FedAvg", "Broadcast + next round"],
  },
  {
    n: 6,
    title: "Explainability",
    icon: Boxes,
    tone: "slate",
    body: "Prediction trace plus gradient-weighted attribution over the vision tower.",
    detail: ["Grad-CAM heatmap", "Region overlay", "Prediction + confidence", "No clinical claim"],
  },
];

const TONES: Record<string, { bg: string; text: string; border: string }> = {
  accent: { bg: "bg-accent-200", text: "text-accent-800", border: "border-accent-300" },
  emerald: { bg: "bg-emerald-200", text: "text-emerald-800", border: "border-emerald-300" },
  violet: { bg: "bg-violet-200", text: "text-violet-800", border: "border-violet-300" },
  amber: { bg: "bg-amber-200", text: "text-amber-800", border: "border-amber-300" },
  slate: { bg: "bg-ink-950/10", text: "text-ink-800", border: "border-ink-950/10" },
};

export default function Architecture() {
  return (
    <>
      <PageHeader
        title="System Architecture"
        description="Six layers, from four hospital agents to explainable prediction. This is the design that the running code implements."
        icon={Network}
      />

      {/* Chain header */}
      <div className="glass-panel mb-6 p-5">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {["Hospital Agents", "Local Data", "Preprocessing", "Cancer Model", "FedAvg", "Global Model", "Broadcast", "Explainability"].map(
            (s, i, arr) => (
              <div key={s} className="flex items-center gap-2">
                <span className="rounded-lg border border-ink-950/10 bg-ink-950/5 px-2.5 py-1.5 font-medium text-ink-900">
                  {s}
                </span>
                {i < arr.length - 1 && <span className="text-accent-400">→</span>}
              </div>
            ),
          )}
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        {LAYERS.map((layer, i) => {
          const t = TONES[layer.tone];
          return (
            <motion.div
              key={layer.n}
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06, duration: 0.35 }}
              className={`relative rounded-2xl border bg-white p-5 shadow-soft ${t.border}`}
            >
              <div className="flex items-start gap-4">
                <div className={`rounded-xl p-2.5 ${t.bg} ${t.text}`}>
                  <layer.icon className="h-5 w-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] text-ink-500">
                      LAYER {String(layer.n).padStart(2, "0")}
                    </span>
                  </div>
                  <h2 className="mt-0.5 text-base font-semibold text-ink-950">{layer.title}</h2>
                  <p className="mt-1 text-xs leading-relaxed text-ink-600">{layer.body}</p>

                  <ul className="mt-3 flex flex-wrap gap-1.5">
                    {layer.detail.map((d) => (
                      <li
                        key={d}
                        className="rounded-md border border-ink-950/10 bg-ink-950/5 px-2 py-1 text-[10px] text-ink-700"
                      >
                        {d}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {i < LAYERS.length - 1 && (
                <div className="absolute -bottom-3 left-8 z-10 flex h-6 w-6 items-center justify-center rounded-full border border-ink-950/10 bg-white text-accent-600 shadow-soft">
                  <span className="text-[10px]">↓</span>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Privacy invariant */}
      <div className="mt-6 rounded-2xl border border-emerald-300/70 bg-emerald-100 p-5">
        <div className="flex items-start gap-3">
          <Lock className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
          <div>
            <h3 className="text-sm font-semibold text-emerald-900">The privacy invariant</h3>
            <p className="mt-1 text-xs leading-relaxed text-emerald-800/90">
              Layer 2 and layer 5 never connect. A hospital agent exposes exactly three operations to
              the aggregator — <code className="rounded bg-emerald-900/10 px-1 font-mono">fit()</code>,{" "}
              <code className="rounded bg-emerald-900/10 px-1 font-mono">update()</code> and{" "}
              <code className="rounded bg-emerald-900/10 px-1 font-mono">load()</code> — all of which accept
              and return tensors. There is no code path by which a pixel array reaches the central
              server, which is why the raw-image counter is structurally zero rather than
              merely reported as zero.
            </p>
          </div>
        </div>
      </div>

      {/* Model adaptation chain */}
      <div className="mt-5 rounded-2xl border border-ink-950/10 bg-white p-5 shadow-soft">
        <div className="mb-3 flex items-center gap-2">
          <Server className="h-4 w-4 text-accent-600" />
          <h3 className="text-sm font-semibold text-ink-950">Model adaptation chain</h3>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {[
            "Pre-trained Model\nMedSigLIP-448",
            "Cancer-specific Adaptation\n4-class head",
            "Fine-tuning\nFrozen / selective / full",
            "Prediction\n4-class softmax",
            "Explainability\nGrad-CAM",
          ].map((s, i, arr) => {
            const [title, sub] = s.split("\n");
            return (
              <div key={title} className="flex items-center gap-2">
                <div className="rounded-xl border border-ink-950/10 bg-white px-3 py-2 text-center shadow-soft">
                  <p className="text-[11px] font-semibold text-ink-950">{title}</p>
                  <p className="text-[10px] text-ink-600">{sub}</p>
                </div>
                {i < arr.length - 1 && <span className="text-accent-400">→</span>}
              </div>
            );
          })}
        </div>
        <div className="mt-4 flex gap-2">
          <Badge variant="info">Backbone: SigLIP vision tower, 27 layers</Badge>
          <Badge variant="info">Head init: text-prompt embeddings</Badge>
          <Badge variant="warning">Not clinically validated</Badge>
        </div>
      </div>
    </>
  );
}