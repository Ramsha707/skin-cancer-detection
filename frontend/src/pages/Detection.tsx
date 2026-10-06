/**
 * Week 4 — cancer detection.
 *
 * Everything here reads live from `/api/detection/*`. The one design decision
 * worth stating: the upload control is only enabled when `status.ready` is true,
 * and otherwise the page shows the exact command to run. A page that accepts an
 * image and then returns a 503 is worse than one that explains why it cannot.
 */
import { useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Cpu,
  FileImage,
  Gauge,
  HardDrive,
  Loader2,
  Upload,
  X,
} from "lucide-react";

import { api, ApiError } from "@/services/api";
import {
  CANCER_CLASS_INFO,
  RESEARCH_DISCLAIMER,
  type CancerClass,
  type DetectionResult,
} from "@/types";
import {
  EmptyState,
  Section,
  StatCard,
} from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ProgressDark } from "@/components/ui/progress";
import {
  cn,
  formatDuration,
  formatNumber,
  formatPercent,
  formatDateTime,
} from "@/lib/utils";

export default function Detection() {
  const qc = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const status = useQuery({ queryKey: ["detection-status"], queryFn: api.detectionStatus });
  const classes = useQuery({ queryKey: ["detection-classes"], queryFn: api.detectionClasses });
  const history = useQuery({ queryKey: ["detections"], queryFn: api.listDetections });

  const predict = useMutation({
    mutationFn: api.detect,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["detections"] });
      void qc.invalidateQueries({ queryKey: ["detection-status"] });
      void qc.invalidateQueries({ queryKey: ["dashboard"] });
    },
  });

  function choose(file: File) {
    setSelected(file);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(file);
    });
    predict.reset();
  }

  function clear() {
    setSelected(null);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return null;
    });
    predict.reset();
    if (inputRef.current) inputRef.current.value = "";
  }

  const st = status.data;
  const ready = st?.ready ?? false;
  const cacheRows = Object.entries(st?.caches_ready ?? {});

  return (
    <div className="space-y-5">
      {/* ------------------------------------------------ page header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-ink-950">
            Cancer Detection
          </h1>
          <p className="mt-1 max-w-3xl text-sm text-ink-600">
            Upload a dermoscopy image. MedSigLIP-448 embeds it, the 4,612-parameter
            linear head scores the four classes, and the full probability vector is
            returned so the result is auditable rather than a bare label.
          </p>
        </div>
        <Badge variant={ready ? "success" : "warning"}>
          {ready ? (
            <>
              <CheckCircle2 className="h-3 w-3" /> Model ready
            </>
          ) : (
            <>
              <AlertTriangle className="h-3 w-3" /> Awaiting extraction
            </>
          )}
        </Badge>
      </div>

      {/* ------------------------------------------------ status strip */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Model version"
          value={st?.model_version ?? "--"}
          hint={st?.is_real_inference ? "Fine-tuned checkpoint" : "Zero-shot prompt head"}
          icon={Brain}
          tone={st?.is_real_inference ? "emerald" : "amber"}
          isReal={st?.is_real_inference ?? false}
        />
        <StatCard
          label="Trainable head"
          value={classes.data ? formatNumber(classes.data.head.trainable_params) : "--"}
          hint="floats, frozen backbone"
          icon={Cpu}
          tone="violet"
        />
        <StatCard
          label="Embedding width"
          value={classes.data ? formatNumber(classes.data.head.embedding_dim) : "--"}
          hint="MedSigLIP output dim"
          icon={Activity}
          tone="accent"
        />
        <StatCard
          label="Caches ready"
          value={`${cacheRows.filter(([, v]) => v).length}/${cacheRows.length || 6}`}
          hint="embedding extracts on disk"
          icon={HardDrive}
          tone={ready ? "emerald" : "amber"}
        />
      </div>

      <div className="grid gap-5 xl:grid-cols-5">
        {/* ------------------------------------------------ upload */}
        <Section
          title="Upload an image"
          description="JPG, JPEG or PNG. Kept on the local host; never transmitted to a hospital."
          className="xl:col-span-2"
        >
          {!ready && st?.instructions && (
            <div className="mb-4 rounded-xl border border-amber-400/40 bg-amber-100 p-3">
              <p className="text-xs font-medium text-amber-900">
                Detection is not available on this checkout yet
              </p>
              <p className="mt-1 break-words text-[11px] leading-relaxed text-amber-800/80">
                {st.instructions}
              </p>
            </div>
          )}

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const f = e.dataTransfer.files?.[0];
              if (f) choose(f);
            }}
            className={cn(
              "relative flex min-h-[240px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition-colors",
              dragging
                ? "border-accent-400 bg-accent-100"
                : "border-ink-950/15 bg-ink-950/5 hover:border-accent-400/50 hover:bg-accent-50",
              !ready && "pointer-events-none opacity-50",
            )}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png"
              className="hidden"
              disabled={!ready}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) choose(f);
              }}
            />
            {preview ? (
              <img
                src={preview}
                alt="Selected dermoscopy image"
                className="max-h-48 rounded-lg object-contain"
              />
            ) : (
              <>
                <Upload className="h-8 w-8 text-ink-500" />
                <p className="mt-3 text-sm font-medium text-ink-800">
                  Drop a dermoscopy image here
                </p>
                <p className="mt-1 text-xs text-ink-500">or click to browse</p>
              </>
            )}
          </div>

          {selected && (
            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between gap-3 text-xs text-ink-600">
                <span className="flex min-w-0 items-center gap-2">
                  <FileImage className="h-3.5 w-3.5 shrink-0" />
                  <span className="truncate">{selected.name}</span>
                  <span className="shrink-0 text-ink-400">
                    {(selected.size / 1024).toFixed(0)} KB
                  </span>
                </span>
                <button
                  onClick={clear}
                  className="shrink-0 text-ink-500 transition-colors hover:text-red-400"
                  aria-label="Clear selection"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <Button
                onClick={() => predict.mutate(selected)}
                disabled={predict.isPending}
                className="w-full"
              >
                {predict.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" /> Running inference
                  </>
                ) : (
                  <>
                    <Activity className="h-4 w-4" /> Run detection
                  </>
                )}
              </Button>
            </div>
          )}

          {predict.isError && (
            <div className="mt-3 flex items-start gap-2 rounded-lg border border-red-400/40 bg-red-100 p-3">
              <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-red-600" />
              <p className="text-[11px] leading-relaxed text-red-800">
                {predict.error instanceof ApiError
                  ? predict.error.message
                  : "Inference failed unexpectedly."}
              </p>
            </div>
          )}

          <p className="mt-4 border-t border-ink-950/10 pt-3 text-[10px] leading-relaxed text-ink-500">
            {RESEARCH_DISCLAIMER}
          </p>
        </Section>

        {/* ------------------------------------------------ result */}
        <Section
          title="Result"
          description={predict.data ? "From the live endpoint." : "Awaiting an image."}
          className="xl:col-span-3"
        >
          {predict.data ? (
            <ResultPanel result={predict.data} />
          ) : (
            <EmptyState
              icon={FileImage}
              variant="empty"
              title="No prediction yet"
              description="Upload a dermoscopy image to see the four-class probability distribution, the predicted label and the model version that produced it."
            />
          )}
        </Section>
      </div>

      {/* ------------------------------------------------ taxonomy + history */}
      <div className="grid gap-5 xl:grid-cols-2">
        <Section
          title="Class taxonomy"
          description="Four classes, three of them malignant. The head is initialised from these prompts."
        >
          {classes.isLoading ? (
            <p className="text-xs text-ink-500">Loading taxonomy...</p>
          ) : (
            <ul className="space-y-2.5">
              {(classes.data?.classes ?? []).map((c) => {
                const info = CANCER_CLASS_INFO[c.name];
                return (
                  <li
                    key={c.name}
                    className="flex items-start gap-3 rounded-lg border border-ink-950/10 bg-ink-950/5 p-3"
                  >
                    <span
                      className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full"
                      style={{ background: info.color }}
                      aria-hidden
                    />
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="text-xs font-semibold text-ink-950">{info.label}</p>
<Badge variant={c.malignant ? "destructive" : "info"}>
                        {c.malignant ? "malignant" : "benign"}
                      </Badge>
                        <span className="text-[10px] text-ink-400">HAM10000: {info.hamDx}</span>
                      </div>
                      <p className="mt-0.5 truncate font-mono text-[10px] text-ink-500">
                        &ldquo;{c.prompt}&rdquo;
                      </p>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
          <p className="mt-4 border-t border-ink-950/10 pt-3 text-[11px] leading-relaxed text-ink-600">
            HAM10000 contains <strong className="text-ink-800">no</strong> squamous cell
            carcinoma images. Its seven <code className="text-ink-700">dx</code> values
            include <code className="text-ink-700">akiec</code> — actinic keratosis — which
            is kept as its own class rather than relabelled{" "}
            <code className="text-ink-700">scc</code>. Relabelling would fabricate ground
            truth on all 327 of those images.
          </p>
        </Section>

        <Section
          title="Recent predictions"
          description="Persisted locally. Newest first."
          actions={
            history.data && history.data.length > 0 ? (
              <span className="text-[11px] text-ink-500">{history.data.length} stored</span>
            ) : null
          }
        >
          {history.isLoading ? (
            <p className="text-xs text-ink-500">Loading history...</p>
          ) : history.isError ? (
            <EmptyState
              variant="error"
              icon={AlertTriangle}
              title="Could not load history"
              description={history.error.message}
            />
          ) : (history.data?.length ?? 0) === 0 ? (
            <EmptyState
              icon={Activity}
              variant="empty"
              title="No predictions recorded"
              description="Every detection is written to the local database so the demo has a real, growing audit trail."
            />
          ) : (
            <ul className="max-h-[320px] space-y-2 overflow-y-auto pr-1">
              {history.data!.map((d) => {
                const info = CANCER_CLASS_INFO[d.predicted_class as CancerClass];
                return (
                  <li
                    key={d.id}
                    className="flex items-center gap-3 rounded-lg border border-ink-950/10 bg-ink-950/5 px-3 py-2"
                  >
                    <span
                      className="h-2 w-2 shrink-0 rounded-full"
                      style={{ background: info?.color ?? "#64748b" }}
                      aria-hidden
                    />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs text-ink-800">
                        {info?.label ?? d.predicted_class}
                      </p>
                      <p className="truncate text-[10px] text-ink-400">
                        {d.image_name} · {formatDateTime(d.created_at)}
                      </p>
                    </div>
                    <div className="shrink-0 text-right">
                      <p className="text-xs font-semibold tabular-nums text-ink-900">
                        {formatPercent(d.confidence)}
                      </p>
                      <p className="text-[10px] text-ink-400">
                        {d.inference_time ? formatDuration(d.inference_time) : "--"}
                      </p>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}
        </Section>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Result panel                                                        */
/* ------------------------------------------------------------------ */

function ResultPanel({ result }: { result: DetectionResult }) {
  const info = CANCER_CLASS_INFO[result.predicted_class as CancerClass];

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      <div
        className="rounded-xl border p-4"
style={{
            borderColor: `${info.color}44`,
            background: `${info.color}12`,
          }}
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-ink-600">
              Predicted diagnosis
            </p>
            <p className="mt-1 text-xl font-semibold tracking-tight text-ink-950">
              {info?.label ?? result.predicted_class}
            </p>
          </div>
          <div className="text-right">
            <p className="text-2xl font-semibold tabular-nums text-ink-950">
              {formatPercent(result.confidence)}
            </p>
            <p className="text-[10px] uppercase tracking-wider text-ink-500">
              confidence
            </p>
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
<Badge variant={result.is_malignant ? "destructive" : "info"}>
                    {result.is_malignant ? "Malignant" : "Benign"}
                  </Badge>
          <Badge variant="neutral">
            {result.is_real_inference ? "Fine-tuned model" : "Zero-shot prompt head"}
          </Badge>
          <Badge variant="neutral">{result.model_version}</Badge>
          {result.inference_time && (
            <Badge variant="neutral">
              <Gauge className="h-3 w-3" /> {formatDuration(result.inference_time)}
            </Badge>
          )}
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-wider text-ink-600">
          Class probabilities
        </p>
        <ul className="space-y-2">
          {result.probabilities.map((p) => {
            const ci = CANCER_CLASS_INFO[p.dx as CancerClass];
            return (
              <li key={p.dx}>
                <div className="mb-1 flex items-center justify-between text-[11px]">
                  <span className="text-ink-700">{ci?.label ?? p.dx}</span>
                  <span className="tabular-nums text-ink-600">
                    {formatPercent(p.probability)}
                  </span>
                </div>
                <ProgressDark value={p.probability * 100} color={ci?.color} />
              </li>
            );
          })}
        </ul>
      </div>

      {result.note && (
        <div className="flex items-start gap-2 rounded-lg border border-ink-950/10 bg-ink-950/5 p-3">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
          <p className="text-[11px] leading-relaxed text-ink-600">{result.note}</p>
        </div>
      )}

      <p className="text-[11px] leading-relaxed text-ink-500">
        Recorded {formatDateTime(result.created_at)} as a {result.modality} image. The
        probabilities are a softmax over the four prompt-initialised head rows, so they
        are directly comparable across classes and sum to 1.
      </p>
    </motion.div>
  );
}