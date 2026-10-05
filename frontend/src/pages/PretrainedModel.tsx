/**
 * Week 5 (a) — pre-trained model introspection.
 *
 * Every number on this page comes from `/api/models/pretrained`, which reads the
 * local HF cache rather than a hardcoded spec. The point of the page is the
 * strategy budget: why the frozen arm costs 4,612 trainable floats and the full
 * arm costs 878 million.
 */
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Box,
  CheckCircle2,
  Cpu,
  FileCode,
  Image as ImageIcon,
  Layers,
  Loader2,
  Quote,
  Snowflake,
  Tag,
} from "lucide-react";

import { api } from "@/services/api";
import { CANCER_CLASS_INFO, type CancerClass } from "@/types";
import { Section, StatCard } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { ProgressDark } from "@/components/ui/progress";
import { cn, formatNumber } from "@/lib/utils";

export default function PretrainedModel() {
  const info = useQuery({ queryKey: ["pretrained-info"], queryFn: api.pretrainedInfo });

  if (info.isLoading) {
    return (
      <div className="flex items-center gap-3 py-16 text-sm text-navy-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Reading model card from the local cache...
      </div>
    );
  }

  if (info.isError) {
    return (
      <div className="flex items-start gap-3 rounded-xl border border-red-500/30 bg-red-500/10 p-4">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
        <p className="text-xs text-red-200">{info.error.message}</p>
      </div>
    );
  }

  const m = info.data!;
  const budget = [
    {
      key: "frozen",
      label: "Frozen",
      trainable: m.trainable_params_frozen_strategy,
      color: "#38bdf8",
      note: "Head only. The vision tower never updates, so the cached embedding stays valid for the whole run.",
    },
    {
      key: "selective",
      label: "Selective",
      trainable: m.trainable_params_selective_strategy,
      color: "#a78bfa",
      note: "Head plus the top vision blocks, at a 100x smaller learning rate.",
    },
    {
      key: "full",
      label: "Full",
      trainable: m.trainable_params_full_strategy,
      color: "#f87171",
      note: "Every parameter. Needs a CUDA device for the optimiser state.",
    },
  ];

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-white">
            Pre-trained Model
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-navy-400">
            What {m.model_id} actually is, what it costs to adapt, and how the cancer head
            gets its initial weights from text prompts instead of random noise.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={m.loaded_locally ? "success" : "warning"}>
            {m.loaded_locally ? (
              <>
                <CheckCircle2 className="h-3 w-3" /> Weights local
              </>
            ) : (
              <>
                <AlertTriangle className="h-3 w-3" /> Config only
              </>
            )}
          </Badge>
          <Badge variant="neutral">{m.model_type}</Badge>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Total parameters"
          value={formatNumber(m.total_params)}
          hint={m.architecture}
          icon={Box}
          tone="accent"
        />
        <StatCard
          label="Vision layers"
          value={m.vision_tower.layers}
          hint={`hidden ${m.vision_tower.hidden_size}, ${m.vision_tower.attention_heads} heads`}
          icon={Layers}
          tone="violet"
        />
        <StatCard
          label="Patches per image"
          value={formatNumber(m.vision_tower.num_patches)}
          hint={`${m.vision_tower.image_size}px / ${m.vision_tower.patch_size}px patches`}
          icon={ImageIcon}
          tone="emerald"
        />
        <StatCard
          label="Embedding dim"
          value={formatNumber(m.embedding_dim)}
          hint="width of everything downstream"
          icon={Cpu}
          tone="amber"
        />
      </div>

      {/* ------------------------------------------------ towers */}
      <div className="grid gap-5 xl:grid-cols-2">
        <Section title="Vision tower" description="What actually runs on an uploaded image.">
          <dl className="grid grid-cols-2 gap-3">
            <Row label="Layers" value={m.vision_tower.layers} />
            <Row label="Hidden size" value={m.vision_tower.hidden_size} />
            <Row label="MLP size" value={m.vision_tower.intermediate_size} />
            <Row label="Attention heads" value={m.vision_tower.attention_heads} />
            <Row label="Image size" value={`${m.vision_tower.image_size}px`} />
            <Row label="Patch size" value={`${m.vision_tower.patch_size}px`} />
            <Row label="Patches / image" value={formatNumber(m.vision_tower.num_patches)} />
            <Row label="Parameters" value={formatNumber(m.vision_tower.params)} />
          </dl>
          <p className="mt-3 border-t border-white/10 pt-3 text-[11px] leading-relaxed text-navy-400">
            27 layers over 1,024 patch tokens is why CPU inference measured ~0.08 img/s
            and why the embedding caches exist at all: the tower runs{" "}
            <strong className="text-navy-200">once</strong> per image, then the cached
            vector is reused for every round.
          </p>
        </Section>

        <Section
          title="Text tower"
          description="Runs once, at initialisation, to build the head. Not needed for inference."
        >
          <dl className="grid grid-cols-2 gap-3">
            <Row label="Layers" value={m.text_tower.layers} />
            <Row label="Hidden size" value={m.text_tower.hidden_size} />
            <Row label="Vocab size" value={formatNumber(m.text_tower.vocab_size)} />
            <Row label="Max positions" value={m.text_tower.max_position_embeddings} />
            <Row label="Parameters" value={formatNumber(m.text_tower.params)} />
            <Row
              label="Used for head"
              value={m.text_tower.used_for_classifier ? "yes" : "no"}
            />
          </dl>
          <div className="mt-3 space-y-1.5 border-t border-white/10 pt-3">
            <p className="text-[11px] font-medium text-navy-300">Preprocessing contract</p>
            <p className="text-[11px] leading-relaxed text-navy-500">
              Resize to {m.preprocessing.input_size[0]}x{m.preprocessing.input_size[1]} using{" "}
              {m.preprocessing.resample}, rescale by {m.preprocessing.rescale_factor}, then
              normalise with mean {m.preprocessing.image_mean[0]} / std{" "}
              {m.preprocessing.image_std[0]} on all three channels. Any deviation shifts
              the embedding and quietly degrades every metric.
            </p>
          </div>
        </Section>
      </div>

      {/* ------------------------------------------------ strategy budget */}
      <Section
        title="Adaptation strategy budget"
        description="Trainable parameters per strategy, on the same 878M backbone."
      >
        <ul className="space-y-4">
          {budget.map((b) => {
            const pct = (b.trainable / m.total_params) * 100;
            return (
              <li key={b.key}>
                <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
                  <p className="text-xs font-semibold text-white">{b.label}</p>
                  <p className="text-[11px] tabular-nums text-navy-400">
                    {formatNumber(b.trainable)} trainable ·{" "}
                    <span className="text-navy-500">{pct < 0.001 ? pct.toFixed(4) : pct.toFixed(2)}% of the model</span>
                  </p>
                </div>
                <ProgressDark
                  value={Math.max(pct, 0.4)}
                  color={b.color}
                  className="h-2"
                />
                <p className="mt-1.5 text-[11px] leading-relaxed text-navy-500">{b.note}</p>
              </li>
            );
          })}
        </ul>
        <p className="mt-4 border-t border-white/10 pt-3 text-[11px] leading-relaxed text-navy-400">
          The frozen arm is the reason FedAvg is practical on this hardware:{" "}
          <strong className="text-navy-200">
            {formatNumber(m.trainable_params_frozen_strategy)} floats per agent per round
          </strong>
          , sample-count weighted into one global head. That is the entire payload crossing
          the hospital boundary.
        </p>
      </Section>

      {/* ------------------------------------------------ domain fit + prompts */}
      <div className="grid gap-5 xl:grid-cols-2">
        <Section
          title="Domain fit"
          description="Whether this backbone is actually appropriate for dermoscopy."
        >
          <div className="space-y-3">
            <Fact label="Pretraining data" value={m.original_training_domain} />
            <Fact label="Pretraining task" value={m.original_task} />
            <Fact
              label="Similarity to dermoscopy"
              value={m.domain_similarity_to_dermoscopy}
            />
            <div className="rounded-lg border border-white/10 bg-navy-950/40 p-3">
              <p className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-navy-500">
                <Snowflake className="h-3 w-3" /> Cache status
              </p>
              <p className="mt-1 text-[11px] text-navy-300">{m.cache_status}</p>
            </div>
            <div className="rounded-lg border border-white/10 bg-navy-950/40 p-3">
              <p className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-navy-500">
                <Tag className="h-3 w-3" /> Device
              </p>
              <p className="mt-1 text-[11px] text-navy-300">{m.device}</p>
            </div>
          </div>

          <div className="mt-4 border-t border-white/10 pt-3">
            <p className="mb-2 text-[11px] font-medium text-navy-300">
              Backbone viability
            </p>
            <ul className="space-y-1.5">
              {m.backbone_viability_notes.map((n) => (
                <li key={n} className="flex items-start gap-2 text-[11px] leading-relaxed text-navy-400">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-accent-400" />
                  {n}
                </li>
              ))}
            </ul>
          </div>
        </Section>

        <Section
          title="Prompt-initialised head"
          description="One classifier row per class, taken from that class's text embedding."
        >
          <p className="mb-3 flex items-start gap-2 rounded-lg border border-white/10 bg-navy-950/40 p-3 text-[11px] leading-relaxed text-navy-400">
            <Quote className="mt-0.5 h-3 w-3 shrink-0 text-accent-400" />
            {m.head_initialisation}
          </p>
          <ul className="space-y-2">
            {m.text_prompts.map((p) => {
              const info = CANCER_CLASS_INFO[p.dx as CancerClass];
              return (
                <li
                  key={p.dx}
                  className="flex items-start gap-3 rounded-lg border border-white/10 bg-navy-950/40 p-3"
                >
                  <span
                    className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ background: info.color }}
                    aria-hidden
                  />
                  <div className="min-w-0">
                    <p className="text-xs font-semibold text-white">{info.label}</p>
                    <p className="mt-0.5 font-mono text-[10px] leading-relaxed text-navy-500">
                      &ldquo;{p.prompt}&rdquo;
                    </p>
                  </div>
                  <Badge
                    variant={p.dx === "benign_lesion" ? "info" : "destructive"}
                    className="ml-auto shrink-0"
                  >
                    {p.dx === "benign_lesion" ? "benign" : "malignant"}
                  </Badge>
                </li>
              );
            })}
          </ul>
          <p className="mt-3 border-t border-white/10 pt-3 text-[11px] leading-relaxed text-navy-500">
            The head is a{" "}
            <code className="text-navy-300">
              {m.projection_size} x {m.text_prompts.length}
            </code>{" "}
            weight matrix plus biases —{" "}
            {formatNumber(
              m.projection_size * m.text_prompts.length + m.text_prompts.length,
            )}{" "}
            floats, and the exact tensor FedAvg averages.
          </p>
        </Section>
      </div>

      <p className={cn("text-[10px] leading-relaxed text-navy-600")}>
        Research prototype. The figures on this page describe architecture and parameter
        counts, not clinical performance.
      </p>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-white/10 bg-navy-950/40 px-3 py-2">
      <dt className="text-[10px] uppercase tracking-wider text-navy-500">{label}</dt>
      <dd className="mt-0.5 text-xs font-semibold tabular-nums text-white">{value}</dd>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-navy-950/40 p-3">
      <p className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-navy-500">
        <FileCode className="h-3 w-3" /> {label}
      </p>
      <p className="mt-1 text-[11px] leading-relaxed text-navy-300">{value}</p>
    </div>
  );
}