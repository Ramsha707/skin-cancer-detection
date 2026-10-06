import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BrainCircuit,
  Cpu,
  Layers,
  Lock,
  Network,
  Play,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  Waypoints,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CANCER_CLASSES, CANCER_CLASS_INFO, CURRENT_IMPLEMENTED_WEEK } from "@/types";

const CONCEPTS = [
  {
    icon: Network,
    title: "Multi-Agent",
    body: "Four independent hospital agents each keep their own dermoscopic dataset and their own local model weights.",
  },
  {
    icon: Waypoints,
    title: "Federated Learning",
    body: "FedAvg merges weight updates — never data — into a single global model that improves for everyone.",
  },
  {
    icon: Stethoscope,
    title: "Cancer Detection",
    body: "MedSigLIP-448, pre-trained on medical imagery, adapted to four-way lesion classification.",
  },
  {
    icon: ShieldCheck,
    title: "Privacy Preservation",
    body: "Zero raw patient images cross a hospital boundary at any point in the pipeline.",
  },
];

const PIPELINE = [
  { step: "01", label: "Hospital agents", detail: "4 sites, isolated data" },
  { step: "02", label: "Local training", detail: "MedSigLIP fine-tune" },
  { step: "03", label: "Weight update", detail: "delta only, no pixels" },
  { step: "04", label: "FedAvg", detail: "size-weighted average" },
  { step: "05", label: "Global model", detail: "Global-vN broadcast" },
];

export default function Landing() {
  return (
    <div className="-mx-4 -my-6 md:-mx-6">
      {/* ── Hero ─────────────────────────────────────────────────────── */}
      <section className="relative overflow-hidden border-b border-ink-950/10">
        <div className="grid-backdrop absolute inset-0 opacity-60" aria-hidden />
        <div
          className="pointer-events-none absolute -left-32 top-0 h-[420px] w-[420px] rounded-full bg-blush-400/40 blur-[120px]"
          aria-hidden
        />
        <div
          className="pointer-events-none absolute -right-24 bottom-0 h-[380px] w-[380px] rounded-full bg-blue-300/40 blur-[120px]"
          aria-hidden
        />

        <div className="relative mx-auto max-w-[1400px] px-4 py-20 md:px-6 md:py-28">
          <div className="grid items-center gap-14 lg:grid-cols-[1.1fr_0.9fr]">
            <div>
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Badge variant="default" className="mb-6 gap-1.5 py-1 pl-1.5 pr-3">
                  <span className="rounded-full bg-accent-500 px-2 py-0.5 text-[9px] font-bold text-white">
                    WEEK 1–6
                  </span>
                  End-to-end federated prototype
                </Badge>

                <h1 className="text-balance text-4xl font-bold leading-[1.08] tracking-tight text-ink-950 md:text-6xl">
                  Privacy-Preserving AI for
                  <br />
                  <span className="text-gradient">Collaborative Skin Cancer</span> Detection
                </h1>

                <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-600">
                  Enabling hospitals to collaboratively train cancer-detection AI{" "}
                  <span className="font-medium text-accent-600">without sharing raw patient images</span>.
                </p>

                <div className="mt-9 flex flex-wrap gap-3">
                  <Button asChild size="lg">
                    <Link to="/dashboard">
                      Launch Dashboard <ArrowRight />
                    </Link>
                  </Button>
                  <Button asChild size="lg" variant="outline">
                    <Link to="/architecture">
                      <Layers /> Explore Architecture
                    </Link>
                  </Button>
                  <Button asChild size="lg" variant="ghost">
                    <Link to="/federated-training">
                      <Play /> Run Demo
                    </Link>
                  </Button>
                </div>

                <p className="mt-8 max-w-lg text-xs leading-relaxed text-ink-500">
                  Built on{" "}
                  <code className="rounded bg-ink-950/10 px-1.5 py-0.5 font-mono text-accent-600">
                    google/medsiglip-448
                  </code>{" "}
                  — a medical vision-language model — rather than training a network from
                  scratch.
                </p>
              </motion.div>
            </div>

            {/* Live pipeline preview */}
            <motion.div
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: 0.15 }}
              className="relative"
            >
              <div className="glass-panel p-5">
                <div className="mb-4 flex items-center justify-between">
                  <p className="text-xs font-medium uppercase tracking-wider text-ink-600">
                    Federated pipeline
                  </p>
                  <span className="flex items-center gap-1.5 text-[10px] text-emerald-600">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
                    raw images shared: 0
                  </span>
                </div>

                <div className="space-y-2">
                  {PIPELINE.map((p, i) => (
                    <motion.div
                      key={p.step}
                      initial={{ opacity: 0, x: 14 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.3 + i * 0.1, duration: 0.4 }}
                      className="group flex items-center gap-3 rounded-xl border border-ink-950/10 bg-ink-950/5 px-3.5 py-3 transition-colors hover:border-lilac-300 hover:bg-lilac-50"
                    >
                      <span className="font-mono text-[10px] text-accent-500">{p.step}</span>
                      <div className="min-w-0 flex-1">
                        <p className="text-xs font-medium text-ink-950">{p.label}</p>
                        <p className="truncate text-[10px] text-ink-500">{p.detail}</p>
                      </div>
                      <ArrowRight className="h-3.5 w-3.5 shrink-0 text-ink-400 transition-colors group-hover:text-accent-400" />
                    </motion.div>
                  ))}
                </div>

                <div className="mt-4 grid grid-cols-4 gap-2">
                  {CANCER_CLASSES.map((c) => (
                    <div
                      key={c}
                      className="rounded-lg border border-ink-950/10 bg-ink-950/5 px-2 py-2 text-center"
                    >
                      <span
                        className="block h-1 w-full rounded-full"
                        style={{ background: CANCER_CLASS_INFO[c].color }}
                      />
                      <p className="mt-1.5 text-[9px] font-medium text-ink-700">
                        {CANCER_CLASS_INFO[c].shortLabel}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* ── Four key concepts ───────────────────────────────────────── */}
      <section className="border-b border-ink-950/10 bg-white">
        <div className="mx-auto max-w-[1400px] px-4 py-16 md:px-6">
          <div className="mb-10 text-center">
            <h2 className="text-2xl font-semibold tracking-tight text-ink-950 md:text-3xl">
              Four ideas hold this system together
            </h2>
            <p className="mx-auto mt-2 max-w-2xl text-sm text-ink-600">
              Each maps to a concrete part of the implemented pipeline.
            </p>
          </div>

          <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
            {CONCEPTS.map((c, i) => (
              <motion.div
                key={c.title}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-80px" }}
                transition={{ delay: i * 0.08, duration: 0.4 }}
              >
                <div className="group h-full rounded-2xl border border-ink-950/10 bg-white p-5 transition-all hover:-translate-y-1 hover:border-lilac-300 hover:shadow-glow">
                  <div className="rounded-xl bg-gradient-to-br from-blush-400 via-blue-400 to-blue-500 p-2.5 transition-transform group-hover:scale-110">
                    <c.icon className="h-5 w-5 text-white" />
                  </div>
                  <h3 className="mt-4 text-sm font-semibold text-ink-950">{c.title}</h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-ink-600">{c.body}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Why transfer learning ───────────────────────────────────── */}
      <section className="mx-auto max-w-[1400px] px-4 py-16 md:px-6">
        <div className="grid gap-8 lg:grid-cols-2">
          <div className="glass-panel p-6">
            <div className="flex items-center gap-2">
              <BrainCircuit className="h-4 w-4 text-accent-600" />
              <h2 className="text-base font-semibold text-ink-950">
                Why adapt a pre-trained model instead of training from scratch?
              </h2>
            </div>

            <ul className="mt-5 space-y-4">
              {[
                {
                  t: "Domain proximity",
                  d: "MedSigLIP was pre-trained on medical image-text pairs. Its features already encode clinically relevant visual structure, which ImageNet features would not.",
                },
                {
                  t: "Label efficiency",
                  d: "A frozen backbone needs only a linear head to become competitive. With a few hundred dermoscopic images per site, full fine-tuning would overfit long before it generalised.",
                },
                {
                  t: "Federated feasibility",
                  d: "FedAvg cost scales with trainable parameters. Keeping most of the network frozen makes each hospital round cheap enough to actually run four times per round.",
                },
                {
                  t: "Preserved knowledge",
                  d: "Selective fine-tuning keeps low-level lesion texture intact while the upper layers adapt to the four-way cancer task — measurably, not by assumption.",
                },
              ].map((item) => (
                <li key={item.t} className="flex gap-3">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-accent-400" />
                  <div>
                    <p className="text-sm font-medium text-ink-950">{item.t}</p>
                    <p className="mt-0.5 text-xs leading-relaxed text-ink-600">{item.d}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          <div className="glass-panel p-6">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-accent-600" />
              <h2 className="text-base font-semibold text-ink-950">What ships in this build</h2>
            </div>

            <div className="mt-5 space-y-3">
              {[
                { w: 1, t: "Project foundation", d: "Design system, routing, landing page", done: true },
                { w: 2, t: "Backend + database", d: "FastAPI, SQLAlchemy, six tables", done: true },
                { w: 3, t: "Hospital agents", d: "Four sites with isolated partitions", done: true },
                { w: 4, t: "Cancer detection", d: "Real MedSigLIP-448 inference", done: true },
                { w: 5, t: "Model integration", d: "Backbone analysis + cancer head", done: true },
                { w: 6, t: "Experiments", d: "Frozen vs selective vs full", done: true },
                { w: 7, t: "FedAvg engine", d: "Local train → aggregate → broadcast", done: false },
                { w: 8, t: "Federated dashboard", d: "Live topology and controls", done: false },
                { w: 9, t: "Evaluation", d: "ROC, confusion matrix, full metrics", done: false },
                { w: 10, t: "Explainability + privacy", d: "Grad-CAM and privacy centre", done: false },
              ].map((row) => (
                <div
                  key={row.w}
                  className="flex items-center gap-3 rounded-lg border border-ink-950/10 bg-ink-950/5 px-3 py-2.5"
                >
                  <span
                    className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md font-mono text-[10px] font-semibold ${
                      row.done ? "bg-emerald-100 text-emerald-700" : "bg-ink-950/10 text-ink-500"
                    }`}
                  >
                    {row.w}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className={`text-xs font-medium ${row.done ? "text-ink-950" : "text-ink-600"}`}>
                      {row.t}
                    </p>
                    <p className="truncate text-[10px] text-ink-500">{row.d}</p>
                  </div>
                  {row.done ? (
                    <span className="text-[10px] font-medium text-emerald-600">done</span>
                  ) : (
                    <span className="text-[10px] text-ink-400">planned</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────────────────── */}
      <section className="border-t border-ink-950/10 bg-gradient-to-b from-blush-100 via-blue-100 to-blue-200">
        <div className="mx-auto max-w-[1400px] px-4 py-16 text-center md:px-6">
          <h2 className="text-2xl font-semibold tracking-tight text-ink-950 md:text-3xl">
            Ready to watch four hospitals train one model?
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-sm text-ink-600">
            Start a federated round and watch weight updates flow into the aggregator while raw
            images stay exactly where they started.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Button asChild size="lg">
              <Link to="/federated-training">
                <Network /> Start Federated Training
              </Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <Link to="/detection">
                <Stethoscope /> Try Cancer Detection
              </Link>
            </Button>
          </div>

          <div className="mt-10 flex items-center justify-center gap-2 text-xs text-ink-500">
            <Lock className="h-3.5 w-3.5 text-emerald-600" />
            <span>
              Implementation status:{" "}
              <strong className="text-ink-700">week {CURRENT_IMPLEMENTED_WEEK} of 12</strong>
            </span>
            <span className="ml-3 font-mono">Ramsha707/skin-cancer</span>
          </div>
        </div>
      </section>

      {/* Floating accent */}
      <motion.div
        animate={{ y: [0, -10, 0] }}
        transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        className="pointer-events-none absolute right-10 top-1/2 hidden h-2 w-2 rounded-full bg-accent-400 xl:block"
      />
      <Sparkles className="pointer-events-none absolute left-8 top-24 hidden h-4 w-4 text-accent-500/40 xl:block" />
    </div>
  );
}