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
      <section className="relative overflow-hidden border-b border-white/10">
        <div className="grid-backdrop absolute inset-0 opacity-70" aria-hidden />
        <div
          className="pointer-events-none absolute -left-32 top-0 h-[420px] w-[420px] rounded-full bg-accent-500/20 blur-[120px]"
          aria-hidden
        />
        <div
          className="pointer-events-none absolute -right-24 bottom-0 h-[380px] w-[380px] rounded-full bg-violet-500/15 blur-[120px]"
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
                  <span className="rounded-full bg-accent-500 px-2 py-0.5 text-[9px] font-bold text-navy-950">
                    WEEK 1–8
                  </span>
                  End-to-end federated prototype
                </Badge>

                <h1 className="text-balance text-4xl font-bold leading-[1.08] tracking-tight text-white md:text-6xl">
                  Privacy-Preserving AI for
                  <br />
                  <span className="text-gradient">Collaborative Skin Cancer</span> Detection
                </h1>

                <p className="mt-6 max-w-xl text-lg leading-relaxed text-navy-300">
                  Enabling hospitals to collaboratively train cancer-detection AI{" "}
                  <span className="text-white">without sharing raw patient images</span>.
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

                <p className="mt-8 max-w-lg text-xs leading-relaxed text-navy-500">
                  Built on{" "}
                  <code className="rounded bg-white/5 px-1.5 py-0.5 font-mono text-accent-300">
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
                  <p className="text-xs font-medium uppercase tracking-wider text-navy-400">
                    Federated pipeline
                  </p>
                  <span className="flex items-center gap-1.5 text-[10px] text-emerald-300">
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
                      className="group flex items-center gap-3 rounded-xl border border-white/10 bg-navy-950/50 px-3.5 py-3 transition-colors hover:border-accent-400/30"
                    >
                      <span className="font-mono text-[10px] text-accent-400/70">{p.step}</span>
                      <div className="min-w-0 flex-1">
                        <p className="text-xs font-medium text-white">{p.label}</p>
                        <p className="truncate text-[10px] text-navy-500">{p.detail}</p>
                      </div>
                      <ArrowRight className="h-3.5 w-3.5 shrink-0 text-navy-600 transition-colors group-hover:text-accent-400" />
                    </motion.div>
                  ))}
                </div>

                <div className="mt-4 grid grid-cols-4 gap-2">
                  {CANCER_CLASSES.map((c) => (
                    <div
                      key={c}
                      className="rounded-lg border border-white/10 bg-navy-950/50 px-2 py-2 text-center"
                    >
                      <span
                        className="block h-1 w-full rounded-full"
                        style={{ background: CANCER_CLASS_INFO[c].color }}
                      />
                      <p className="mt-1.5 text-[9px] font-medium text-navy-300">
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
      <section className="border-b border-white/10 bg-navy-950/60">
        <div className="mx-auto max-w-[1400px] px-4 py-16 md:px-6">
          <div className="mb-10 text-center">
            <h2 className="text-2xl font-semibold tracking-tight text-white md:text-3xl">
              Four ideas hold this system together
            </h2>
            <p className="mx-auto mt-2 max-w-2xl text-sm text-navy-400">
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
                <div className="group h-full rounded-2xl border border-white/10 bg-navy-900/50 p-5 transition-all hover:-translate-y-1 hover:border-accent-400/30 hover:shadow-glow">
                  <div className="rounded-xl bg-gradient-to-br from-accent-500/20 to-accent-600/5 p-2.5 transition-transform group-hover:scale-110">
                    <c.icon className="h-5 w-5 text-accent-300" />
                  </div>
                  <h3 className="mt-4 text-sm font-semibold text-white">{c.title}</h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-navy-400">{c.body}</p>
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
              <BrainCircuit className="h-4 w-4 text-accent-300" />
              <h2 className="text-base font-semibold text-white">
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
                    <p className="text-sm font-medium text-white">{item.t}</p>
                    <p className="mt-0.5 text-xs leading-relaxed text-navy-400">{item.d}</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          <div className="glass-panel p-6">
            <div className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-accent-300" />
              <h2 className="text-base font-semibold text-white">What ships in this build</h2>
            </div>

            <div className="mt-5 space-y-3">
              {[
                { w: 1, t: "Project foundation", d: "Design system, routing, landing page", done: true },
                { w: 2, t: "Backend + database", d: "FastAPI, SQLAlchemy, six tables", done: true },
                { w: 3, t: "Hospital agents", d: "Four sites with isolated partitions", done: true },
                { w: 4, t: "Cancer detection", d: "Real MedSigLIP-448 inference", done: true },
                { w: 5, t: "Model integration", d: "Backbone analysis + cancer head", done: true },
                { w: 6, t: "Experiments", d: "Frozen vs selective vs full", done: true },
                { w: 7, t: "FedAvg engine", d: "Local train → aggregate → broadcast", done: true },
                { w: 8, t: "Federated dashboard", d: "Live topology and controls", done: true },
                { w: 9, t: "Evaluation", d: "ROC, confusion matrix, full metrics", done: false },
                { w: 10, t: "Explainability + privacy", d: "Grad-CAM and privacy centre", done: false },
              ].map((row) => (
                <div
                  key={row.w}
                  className="flex items-center gap-3 rounded-lg border border-white/10 bg-navy-950/40 px-3 py-2.5"
                >
                  <span
                    className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md font-mono text-[10px] font-semibold ${
                      row.done ? "bg-emerald-500/15 text-emerald-300" : "bg-white/5 text-navy-500"
                    }`}
                  >
                    {row.w}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className={`text-xs font-medium ${row.done ? "text-white" : "text-navy-400"}`}>
                      {row.t}
                    </p>
                    <p className="truncate text-[10px] text-navy-500">{row.d}</p>
                  </div>
                  {row.done ? (
                    <span className="text-[10px] font-medium text-emerald-400">done</span>
                  ) : (
                    <span className="text-[10px] text-navy-600">planned</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA ─────────────────────────────────────────────────────── */}
      <section className="border-t border-white/10 bg-gradient-to-b from-navy-950 to-navy-900">
        <div className="mx-auto max-w-[1400px] px-4 py-16 text-center md:px-6">
          <h2 className="text-2xl font-semibold tracking-tight text-white md:text-3xl">
            Ready to watch four hospitals train one model?
          </h2>
          <p className="mx-auto mt-3 max-w-2xl text-sm text-navy-400">
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

          <div className="mt-10 flex items-center justify-center gap-2 text-xs text-navy-500">
            <Lock className="h-3.5 w-3.5 text-emerald-400" />
            <span>
              Implementation status:{" "}
              <strong className="text-navy-300">week {CURRENT_IMPLEMENTED_WEEK} of 12</strong>
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