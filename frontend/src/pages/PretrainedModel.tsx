/**
 * Week 5 — pre-trained model introspection.
 *
 * Every figure on this page is served by `GET /api/models/pretrained`, which
 * sums parameter counts from the header of the local safetensors file rather
 * than quoting a paper. `cache_status` drives the header badge: when the
 * weights are not on disk the page says so instead of implying these numbers
 * were just verified against real bytes.
 */
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  Cpu,
  Database,
  Layers,
  Loader2,
  RefreshCw,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";

import { api, ApiError } from "@/services/api";
import {
  CANCER_CLASS_INFO,
  RESEARCH_DISCLAIMER,
  type PretrainedModelInfo,
} from "@/types";
import { EmptyState, Section, StatCard, TitleIcon } from "@/components/common";
import { PageHeader } from "@/components/layout/AppLayout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn, formatNumber } from "@/lib/utils";

/** 878300338 -> "878.3M"; exact counts still shown in the detail rows. */
function formatParams(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return formatNumber(value);
}

function SpecGrid({
  items,
}: {
  items: { label: string; value: string | number; mono?: boolean }[];
}) {
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
      {items.map((it) => (
        <div key={it.label} className="min-w-0">
          <dt className="truncate text-[10px] font-medium uppercase tracking-wider text-ink-500">
            {it.label}
          </dt>
          <dd
            className={cn(
              "mt-0.5 truncate text-sm font-semibold text-ink-950",
              it.mono && "font-mono text-xs",
            )}
            title={String(it.value)}
          >
            {it.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function strategiesOf(d: PretrainedModelInfo) {
  return [
    {
      key: "frozen",
      label: "Frozen backbone",
      note: "Text-prompt-initialised head only — the tower never moves.",
      trainable: d.trainable_params_frozen_strategy,
      tone: "bg-emerald-400",
    },
    {
      key: "selective",
      label: "Selective fine-tuning",
      note: "Head plus the deepest 20% of vision-encoder layers.",
      trainable: d.trainable_params_selective_strategy,
      tone: "bg-accent-400",
    },
    {
      key: "full",
      label: "Full fine-tuning",
      note: "Every parameter is trainable — highest capacity, highest overfit risk.",
      trainable: d.trainable_params_full_strategy,
      tone: "bg-violet-400",
    },
  ];
}

export default function PretrainedModel() {
  const q = useQuery({
    queryKey: ["pretrained-model"],
    queryFn: api.pretrainedInfo,
  });

  if (q.isLoading) {
    return (
      <div className="space-y-5">
<PageHeader
        title="Pre-trained Model"
        description="MedSigLIP-448 architecture introspection."
        icon={Cpu}
      />
        <EmptyState
          title="Reading the local model header…"
          description="Summing tensor shapes from the safetensors file in the HuggingFace cache."
          icon={Loader2}
        />
      </div>
    );
  }

  if (q.isError || !q.data) {
    const offline = q.error instanceof ApiError && q.error.status >= 500;
    return (
      <div className="space-y-5">
<PageHeader
        title="Pre-trained Model"
        description="MedSigLIP-448 architecture introspection."
        icon={Cpu}
      />
        <EmptyState
          variant="error"
          title={offline ? "Backend did not respond" : "Model introspection failed"}
          description={
            q.error instanceof Error
              ? q.error.message
              : "The endpoint returned an unexpected payload."
          }
          icon={AlertTriangle}
          action={
            <Button variant="outline" onClick={() => void q.refetch()}>
              <RefreshCw className="h-3.5 w-3.5" /> Retry
            </Button>
          }
        />
      </div>
    );
  }

  const d = q.data;
  const cached = d.cache_status === "ok";
  const strategies = strategiesOf(d);

  return (
    <div className="space-y-5">
      {/* ------------------------------------------------ page header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <TitleIcon icon={Cpu} />
          <div>
            <h1 className="text-xl font-semibold tracking-tight text-ink-950">
              Pre-trained Model
            </h1>
            <p className="mt-1 max-w-3xl text-sm text-ink-600">
              {d.model_id} — the vision-language backbone every other week builds on.
              Parameter counts below are summed from the tensor header of the weight
              file on this machine, not from a datasheet.
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={cached ? "success" : d.cache_status === "partial" ? "info" : "warning"}>
            {cached ? (
              <>
                <CheckCircle2 className="h-3 w-3" /> Weights cached locally
              </>
            ) : d.cache_status === "partial" ? (
              <>
                <AlertTriangle className="h-3 w-3" /> Config only
              </>
            ) : (
              <>
                <AlertTriangle className="h-3 w-3" /> Weights not on disk
              </>
            )}
          </Badge>
          <Badge variant="neutral">
            <Cpu className="h-3 w-3" /> {d.device}
          </Badge>
        </div>
      </div>

      {/* ------------------------------------------------ headline stats */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Total parameters"
          value={formatParams(d.total_params)}
          hint={formatNumber(d.total_params)}
          icon={Layers}
          tone="accent"
          isReal={cached}
        />
        <StatCard
          label="Vision tower"
          value={formatParams(d.vision_tower.params)}
          hint={`${d.vision_tower.layers} layers · patch ${d.vision_tower.patch_size}`}
          icon={Database}
          tone="violet"
          isReal={cached}
        />
        <StatCard
          label="Text tower"
          value={formatParams(d.text_tower.params)}
          hint={`${d.text_tower.vocab_size} vocab · ${d.text_tower.max_position_embeddings} positions`}
          icon={Brain}
          tone="emerald"
          isReal={cached}
        />
        <StatCard
          label="Embedding width"
          value={formatNumber(d.embedding_dim)}
          hint={`${formatNumber(d.vision_tower.num_patches)} patches per image`}
          icon={Cpu}
          tone="amber"
        />
      </div>

      {/* ------------------------------------------------ tower geometry */}
      <div className="grid gap-5 xl:grid-cols-2">
        <Section
          title="Vision tower"
          description="The encoder that turns a 448×448 dermoscopy image into a 1152-d embedding."
        >
          <SpecGrid
            items={[
              { label: "Layers", value: d.vision_tower.layers },
              { label: "Hidden size", value: d.vision_tower.hidden_size },
              { label: "MLP intermediate", value: d.vision_tower.intermediate_size },
              { label: "Attention heads", value: d.vision_tower.attention_heads },
              { label: "Patch size", value: `${d.vision_tower.patch_size}px` },
              { label: "Image size", value: `${d.vision_tower.image_size}px` },
              { label: "Tokens per image", value: formatNumber(d.vision_tower.num_patches) },
              { label: "Parameters", value: formatParams(d.vision_tower.params), mono: true },
              { label: "Trainable when frozen", value: formatNumber(d.vision_tower.trainable_when_frozen) },
            ]}
          />
        </Section>

        <Section
          title="Text tower"
          description="Supplies the class prompts that initialise the head — never used for classification at inference."
        >
          <SpecGrid
            items={[
              { label: "Layers", value: d.text_tower.layers },
              { label: "Hidden size", value: d.text_tower.hidden_size },
              { label: "MLP intermediate", value: d.text_tower.intermediate_size },
              { label: "Attention heads", value: d.text_tower.attention_heads },
              { label: "Vocabulary", value: formatNumber(d.text_tower.vocab_size) },
              { label: "Max positions", value: d.text_tower.max_position_embeddings },
              { label: "Projection", value: d.projection_size },
              { label: "Parameters", value: formatParams(d.text_tower.params), mono: true },
              { label: "Used for classification", value: d.text_tower.used_for_classifier ? "Prompt init only" : "No" },
            ]}
          />
        </Section>
      </div>

      {/* ------------------------------------------------ adaptation budget */}
      <Section
        title="Parameter budget by adaptation strategy"
        description="The three strategies the experiments week compares. Head parameters are exact: embedding width × classes + biases."
        actions={
          <Badge variant="neutral">
            Head = {formatNumber(d.trainable_params_frozen_strategy)} params
          </Badge>
        }
      >
        <div className="space-y-4">
          {strategies.map((s) => {
            const pct = d.total_params > 0 ? s.trainable / d.total_params : 0;
            return (
              <div key={s.key}>
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-ink-950">{s.label}</p>
                    <p className="text-xs text-ink-600">{s.note}</p>
                  </div>
                  <p className="font-mono text-xs text-ink-700">
                    {formatNumber(s.trainable)} trainable ·{" "}
                    {formatNumber(d.total_params - s.trainable)} frozen
                  </p>
                </div>
                <div className="mt-2 flex items-center gap-3">
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-950/15">
                    <div
                      className={cn("h-full rounded-full", s.tone)}
                      style={{ width: `${Math.max(pct * 100, 0.5)}%` }}
                    />
                  </div>
                  <span className="w-16 shrink-0 text-right text-xs font-semibold tabular-nums text-ink-800">
                    {pct < 0.001 ? "<0.1%" : `${(pct * 100).toFixed(pct < 0.01 ? 2 : 1)}%`}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
        <p className="mt-4 border-t border-ink-950/10 pt-3 text-[11px] leading-relaxed text-ink-600">
          Frozen-strategy percentages are tiny on purpose: a 4,612-parameter head over a
          frozen 878M backbone is what makes federated averaging tractable.
        </p>
      </Section>

      {/* ------------------------------------------------ preprocessing */}
      <div className="grid gap-5 xl:grid-cols-2">
        <Section
          title="Preprocessing contract"
          description="Identical at every hospital site — a transform mismatch would silently break FedAvg."
        >
          <SpecGrid
            items={[
              { label: "Input size", value: `${d.preprocessing.input_size[0]} × ${d.preprocessing.input_size[1]}` },
              { label: "Resample", value: d.preprocessing.resample },
              { label: "Rescale factor", value: d.preprocessing.rescale_factor },
              { label: "Channel mean", value: d.preprocessing.image_mean.join(", ") },
              { label: "Channel std", value: d.preprocessing.image_std.join(", ") },
              { label: "Architecture", value: d.architecture },
            ]}
          />
        </Section>

        <Section
          title="Local resolution"
          description="Where the cache lives and how it was verified."
        >
          <dl className="space-y-3">
            <div className="flex items-start justify-between gap-4">
              <dt className="text-xs uppercase tracking-wider text-ink-500">Cache status</dt>
              <dd className="text-right text-sm font-semibold text-ink-950">
                {d.cache_status}
              </dd>
            </div>
            <div className="flex items-start justify-between gap-4">
              <dt className="text-xs uppercase tracking-wider text-ink-500">Weights present</dt>
              <dd className="text-sm font-semibold text-ink-950">
                {d.loaded_locally ? "yes" : "no"}
              </dd>
            </div>
            <div className="flex items-start justify-between gap-4">
              <dt className="text-xs uppercase tracking-wider text-ink-500">Device</dt>
              <dd className="text-sm font-semibold text-ink-950">{d.device}</dd>
            </div>
            <div className="flex items-start gap-4">
              <dt className="shrink-0 text-xs uppercase tracking-wider text-ink-500">
                Snapshot
              </dt>
              <dd
                className="min-w-0 break-all font-mono text-xs text-ink-700"
                title={d.resolved_path ?? "not cached"}
              >
                {d.resolved_path ?? "not cached"}
              </dd>
            </div>
          </dl>
        </Section>
      </div>

      {/* ------------------------------------------------ domain assessment */}
      <Section
        title="Domain assessment"
        description="Why a general medical vision-language model is a defensible backbone for dermoscopy."
        actions={
          <Badge variant={d.suitable_as_backbone ? "success" : "destructive"}>
            <ShieldCheck className="h-3 w-3" />
            {d.suitable_as_backbone ? "Suitable backbone" : "Not suitable"}
          </Badge>
        }
      >
        <div className="space-y-4">
          <div>
            <p className="text-[10px] font-medium uppercase tracking-wider text-ink-500">
              Original training domain
            </p>
            <p className="mt-1 text-sm leading-relaxed text-ink-800">
              {d.original_training_domain}
            </p>
          </div>
          <div>
            <p className="text-[10px] font-medium uppercase tracking-wider text-ink-500">
              Original task
            </p>
            <p className="mt-1 text-sm leading-relaxed text-ink-800">{d.original_task}</p>
          </div>
          <div>
            <p className="text-[10px] font-medium uppercase tracking-wider text-ink-500">
              Original classes
            </p>
            <p className="mt-1 text-sm leading-relaxed text-ink-800">
              {d.original_classes}
            </p>
          </div>
          <div>
            <p className="text-[10px] font-medium uppercase tracking-wider text-ink-500">
              Output format
            </p>
            <p className="mt-1 text-sm leading-relaxed text-ink-800">{d.output_format}</p>
          </div>
          <div>
            <p className="text-[10px] font-medium uppercase tracking-wider text-ink-500">
              Similarity to dermoscopy
            </p>
            <p className="mt-1 text-sm leading-relaxed text-ink-800">
              {d.domain_similarity_to_dermoscopy}
            </p>
          </div>
          <ul className="space-y-1.5 rounded-xl border border-ink-950/10 bg-lilac-50 p-4">
            {d.backbone_viability_notes.map((note) => (
              <li key={note} className="flex gap-2 text-xs leading-relaxed text-ink-700">
                <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" />
                {note}
              </li>
            ))}
          </ul>
        </div>
      </Section>

      {/* ------------------------------------------------ head initialisation */}
      <Section
        title="Head initialisation"
        description="Each class weight row starts at the text embedding of its prompt — epoch 0 is not a coin flip."
        actions={<Badge variant="neutral">4 classes</Badge>}
      >
        <p className="text-sm leading-relaxed text-ink-800">{d.head_initialisation}</p>

        <div className="mt-4 overflow-hidden rounded-xl border border-ink-950/10">
          <table className="w-full text-left text-sm">
            <thead className="bg-lilac-50 text-[10px] uppercase tracking-wider text-ink-500">
              <tr>
                <th className="px-4 py-2 font-medium">Class</th>
                <th className="px-4 py-2 font-medium">Prompt</th>
                <th className="px-4 py-2 text-right font-medium">Polarity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-950/10">
              {d.text_prompts.map((tp) => {
                const info = CANCER_CLASS_INFO[tp.dx];
                return (
                  <tr key={tp.dx} className="bg-white">
                    <td className="px-4 py-2.5">
                      <span className="flex items-center gap-2 font-medium text-ink-950">
                        <span
                          className="h-2.5 w-2.5 shrink-0 rounded-full"
                          style={{ backgroundColor: info.color }}
                        />
                        {info.label}
                      </span>
                    </td>
                    <td className="px-4 py-2.5 font-mono text-xs text-ink-700">
                      “{tp.prompt}”
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <Badge variant={info.malignant ? "destructive" : "success"}>
                        {info.malignant ? "malignant" : "benign"}
                      </Badge>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <p className="mt-4 border-t border-ink-950/10 pt-3 text-[10px] leading-relaxed text-ink-500">
          {RESEARCH_DISCLAIMER}
        </p>
      </Section>

      {/* ------------------------------------------------ provenance note */}
      <p className="px-1 text-[11px] leading-relaxed text-ink-500">
        <SlidersHorizontal className="mr-1.5 inline h-3 w-3" />
        {cached
          ? "Counts read from the local safetensors header; architecture read from config.json."
          : "Local weights are absent, so totals come from the published model card and the tower split is an estimate — `cache_status` above says so."}
      </p>
    </div>
  );
}
