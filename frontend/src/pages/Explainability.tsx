import type { ChangeEvent, DragEvent } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Crosshair,
  ImageUp,
  Layers,
  ScanSearch,
  Sparkles,
  Trash2,
} from "lucide-react";
import { PageHeader } from "@/components/layout/AppLayout";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EmptyState, Section, StatCard, TitleIcon } from "@/components/common";
import { api } from "@/services/api";
import {
  CANCER_CLASSES,
  CANCER_CLASS_INFO,
  CURRENT_IMPLEMENTED_WEEK,
  type CancerClass,
  type ExplainabilityCapability,
  type ExplainabilityResult,
} from "@/types";
import { cn, formatPercent } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/* Heat-map rendering                                                  */
/* ------------------------------------------------------------------ */

const RAMP = ["#0b1020", "#3b1d6e", "#8b2f9e", "#d946a0", "#f97316", "#fee08b"];

function heatColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const hex = (h: string) => [
    parseInt(h.slice(1, 3), 16),
    parseInt(h.slice(3, 5), 16),
    parseInt(h.slice(5, 7), 16),
  ];
  const [r1, g1, b1] = hex(RAMP[i]);
  const [r2, g2, b2] = hex(RAMP[i + 1]);
  const mix = (a: number, b: number) => Math.round(a + (b - a) * f);
  return `rgb(${mix(r1, r2)}, ${mix(g1, g2)}, ${mix(b1, b2)})`;
}

function HeatMapGrid({ grid }: { grid: number[][] }) {
  const cells = useMemo(() => {
    const rows = grid.length;
    const cols = rows ? grid[0].length : 0;
    const out: { key: string; color: string }[] = [];
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        out.push({ key: `${r}-${c}`, color: heatColor(grid[r][c]) });
      }
    }
    return { out, rows, cols };
  }, [grid]);

  if (!cells.rows) {
    return (
      <div className="flex h-48 items-center justify-center rounded-xl border border-dashed border-ink-950/15 text-xs text-ink-500">
        No attribution map
      </div>
    );
  }

  return (
    <div
      className="grid overflow-hidden rounded-xl border border-ink-950/10"
      style={{ gridTemplateColumns: `repeat(${cells.cols}, minmax(0, 1fr))` }}
    >
      {cells.out.map((cell) => (
        <div key={cell.key} className="aspect-square" style={{ backgroundColor: cell.color }} />
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Capability gate                                                     */
/* ------------------------------------------------------------------ */

function CapabilityGate({ capability }: { capability: ExplainabilityCapability }) {
  return (
    <Section
      title="Grad-CAM is not available in this environment"
      description="The pipeline refuses to draw a heatmap it cannot actually compute."
      tone="dark"
    >
      <div className="flex items-start gap-3 rounded-xl border border-amber-300/70 bg-amber-100 p-4">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        <div className="text-sm text-amber-900">
          <p className="font-medium">Explanation requires torch plus the pretrained vision tower.</p>
          <p className="mt-1 text-xs leading-relaxed text-amber-700/90">{capability.reason}</p>
          <p className="mt-2 text-xs leading-relaxed text-amber-700/90">
            Install the runtime and weights, then reload this page. Until then every field below
            stays empty rather than showing a synthetic overlay — a fabricated heatmap would imply
            a trained model looked at your image when nothing did.
          </p>
        </div>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
        {[
          { label: "Method", value: capability.method },
          { label: "Vision model", value: capability.model_id },
          { label: "Device", value: capability.device ?? "—" },
          { label: "Classes", value: String(capability.classes.length) },
        ].map((item) => (
          <div key={item.label} className="rounded-lg border border-ink-950/10 bg-white/60 p-3">
            <dt className="text-[10px] font-medium uppercase tracking-wider text-ink-500">
              {item.label}
            </dt>
            <dd className="mt-0.5 truncate font-medium text-ink-900">{item.value}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

/* ------------------------------------------------------------------ */
/* Result panel                                                        */
/* ------------------------------------------------------------------ */

function ResultPanel({ result }: { result: ExplainabilityResult }) {
  const info = result.predicted_class ? CANCER_CLASS_INFO[result.predicted_class] : null;

  return (
    <div className="space-y-5">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Predicted class"
          value={info?.shortLabel ?? "—"}
          hint={info?.label}
          icon={ScanSearch}
          tone={info?.malignant ? "red" : "emerald"}
          isReal={result.is_real_inference}
        />
        <StatCard
          label="Confidence"
          value={formatPercent(result.confidence, 1)}
          hint={result.head_kind === "fine-tuned" ? "trained head" : "zero-shot prompt head"}
          icon={Activity}
          tone="accent"
          isReal={result.is_real_inference}
        />
        <StatCard
          label="Explained target"
          value={result.target_class ? CANCER_CLASS_INFO[result.target_class].shortLabel : "—"}
          hint={result.target_class ? CANCER_CLASS_INFO[result.target_class].label : undefined}
          icon={Crosshair}
          tone="violet"
        />
        <StatCard
          label="Grid"
          value={`${result.grid_size} × ${result.grid_size}`}
          hint={`${result.method} attribution cells`}
          icon={Layers}
          tone="slate"
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Section
          title="Attribution overlay"
          description="Heated cells mark the image regions that pushed the score for the selected class."
          tone="dark"
        >
          {result.overlay_png ? (
            <img
              src={result.overlay_png}
              alt="Grad-CAM overlay"
              className="w-full rounded-xl border border-ink-950/10"
            />
          ) : (
            <EmptyState
              title="No overlay produced"
              description="The map was computed but a colour blend could not be rendered."
              variant="empty"
            />
          )}
        </Section>

        <Section
          title="Raw attribution map"
          description="The numeric grid before it is drawn back over the pixels."
          tone="dark"
        >
          <HeatMapGrid grid={result.grid} />
        </Section>
      </div>

      <Section
        title="Where the model looked"
        description="The strongest cells, named by their position on the image."
        tone="dark"
      >
        {result.top_regions.length ? (
          <ul className="space-y-2">
            {result.top_regions.map((region, i) => (
              <li
                key={`${region.row}-${region.column}`}
                className="flex items-center gap-3 rounded-xl border border-ink-950/10 bg-white/60 p-3"
              >
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-blush-400 to-blue-500 text-xs font-semibold text-white">
                  {i + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium capitalize text-ink-950">{region.region}</p>
                  <p className="text-[11px] text-ink-500">
                    cell ({region.row}, {region.column})
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-24 overflow-hidden rounded-full bg-ink-950/10">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-blush-400 to-blue-500"
                      style={{ width: `${Math.round(region.intensity * 100)}%` }}
                    />
                  </div>
                  <span className="w-10 text-right text-xs tabular-nums text-ink-700">
                    {Math.round(region.intensity * 100)}%
                  </span>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <EmptyState
            title="No concentrated region"
            description="Attribution is spread evenly across the image, so no single area stands out."
            variant="empty"
          />
        )}
      </Section>

      <div className="rounded-xl border border-ink-950/10 bg-ink-950/5 p-4 text-xs leading-relaxed text-ink-600">
        <p className="font-medium text-ink-800">How to read this</p>
        <p className="mt-1">
          Grad-CAM back-propagates the {result.head_kind ?? "model"} score for{" "}
          <strong>{result.target_class ? CANCER_CLASS_INFO[result.target_class].label : "the class"}</strong>{" "}
          through the final vision-tower block and pools the activation gradients. Brighter cells
          are regions the score is most sensitive to. It shows <em>where</em> the model attends, not{" "}
          <em>why</em>, and it is not a lesion segmentation — treat it as a debugging aid, never as
          evidence for a diagnosis.
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function Explainability() {
  const [capability, setCapability] = useState<ExplainabilityCapability | null>(null);
  const [loadingCap, setLoadingCap] = useState(true);
  const [targetClass, setTargetClass] = useState<CancerClass | "auto">("auto");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<ExplainabilityResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .explainabilityStatus()
      .then((c) => {
        if (!cancelled) setCapability(c);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Failed to read capability.");
      })
      .finally(() => {
        if (!cancelled) setLoadingCap(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectFile = useCallback((next: File | null) => {
    setResult(null);
    setError(null);
    setFile(next);
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return next ? URL.createObjectURL(next) : null;
    });
  }, []);

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      const dropped = e.dataTransfer.files?.[0];
      if (dropped) selectFile(dropped);
    },
    [selectFile],
  );

  const onPick = (e: ChangeEvent<HTMLInputElement>) => {
    const picked = e.target.files?.[0];
    if (picked) selectFile(picked);
  };

  const run = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.explainImage(file, targetClass === "auto" ? undefined : targetClass);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Explanation request failed.");
    } finally {
      setBusy(false);
    }
  };

  const available = capability?.available ?? false;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Explainability"
        description={`Grad-CAM heatmaps for a live upload. Real attention attribution over the vision tower — no illustrative overlays. Weeks 1–${CURRENT_IMPLEMENTED_WEEK} implemented.`}
        icon={Sparkles}
        actions={
          <Badge variant={available ? "success" : "warning"}>
            {available ? (
              <>
                <CheckCircle2 className="h-3 w-3" /> Ready
              </>
            ) : (
              <>
                <AlertTriangle className="h-3 w-3" /> Unavailable
              </>
            )}
          </Badge>
        }
      />

      {loadingCap ? (
        <Section title="Checking capability" description="Looking for torch and the vision weights.">
          <div className="h-24 animate-pulse rounded-xl bg-ink-950/5" />
        </Section>
      ) : capability && !capability.available ? (
        <CapabilityGate capability={capability} />
      ) : (
        <div className="grid gap-5 lg:grid-cols-[minmax(0,360px)_1fr]">
          <div className="space-y-5">
            <Section
              title="Upload a lesion image"
              description="Heatmaps are computed only for the image in this request. Nothing is stored."
              tone="dark"
            >
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                className={cn(
                  "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors",
                  dragging
                    ? "border-accent-500 bg-accent-500/5"
                    : "border-ink-950/15 hover:border-ink-950/25",
                )}
              >
                {preview ? (
                  <img
                    src={preview}
                    alt="Selected lesion"
                    className="max-h-48 rounded-lg border border-ink-950/10"
                  />
                ) : (
                  <>
                    <div className="rounded-xl border border-ink-950/10 bg-white/60 p-3">
                      <ImageUp className="h-5 w-5 text-ink-600" />
                    </div>
                    <p className="mt-3 text-sm font-medium text-ink-800">
                      Drop an image or click to browse
                    </p>
                    <p className="mt-0.5 text-[11px] text-ink-500">JPEG or PNG, up to the server limit</p>
                  </>
                )}
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png"
                  className="hidden"
                  onChange={onPick}
                />
              </div>

              {file && (
                <div className="mt-3 flex items-center justify-between gap-2 rounded-lg border border-ink-950/10 bg-white/60 p-2 text-xs">
                  <span className="truncate text-ink-700">{file.name}</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      selectFile(null);
                      if (inputRef.current) inputRef.current.value = "";
                    }}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
                </div>
              )}

              <div className="mt-4">
                <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-ink-500">
                  Explain this class
                </p>
                <div className="flex flex-wrap gap-1.5">
                  <button
                    type="button"
                    onClick={() => setTargetClass("auto")}
                    className={cn(
                      "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                      targetClass === "auto"
                        ? "border-accent-500 bg-accent-500/10 text-accent-700"
                        : "border-ink-950/15 text-ink-600 hover:border-ink-950/30",
                    )}
                  >
                    Auto (predicted)
                  </button>
                  {CANCER_CLASSES.map((c) => (
                    <button
                      key={c}
                      type="button"
                      onClick={() => setTargetClass(c)}
                      className={cn(
                        "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                        targetClass === c
                          ? "border-accent-500 bg-accent-500/10 text-accent-700"
                          : "border-ink-950/15 text-ink-600 hover:border-ink-950/30",
                      )}
                    >
                      {CANCER_CLASS_INFO[c].shortLabel}
                    </button>
                  ))}
                </div>
              </div>

              <Button
                className="mt-4 w-full"
                disabled={!file || busy}
                onClick={run}
              >
                {busy ? "Computing attribution…" : "Generate heatmap"}
              </Button>

              {error && (
                <p className="mt-3 rounded-lg border border-red-300/70 bg-red-100 p-2 text-xs text-red-800">
                  {error}
                </p>
              )}
            </Section>

            <Section title="Why only live uploads" description="A deliberate privacy choice." tone="dark">
              <p className="text-xs leading-relaxed text-ink-600">
                Stored detection results deliberately carry no image reference, so there is nothing
                to re-explain later. A heatmap can only be produced while the uploaded file is in
                hand, inside the request that received it. Nothing on disk links a patient image to
                a result row.
              </p>
            </Section>
          </div>

          <div>
            {result ? (
              result.available ? (
                <ResultPanel result={result} />
              ) : (
                <Section title="Explanation unavailable" tone="dark">
                  <EmptyState
                    title="Could not compute a Grad-CAM map"
                    description={result.reason ?? "The model runtime refused to produce an explanation."}
                    variant="error"
                    icon={AlertTriangle}
                  />
                </Section>
              )
            ) : (
              <Section title="Heatmap" description="Upload an image to see where the model attends." tone="dark">
                <EmptyState
                  title="No explanation yet"
                  description="Pick an image on the left and press Generate heatmap. The attribution map appears here."
                  icon={Sparkles}
                />
              </Section>
            )}
          </div>
        </div>
      )}

      {capability && capability.available && (
        <Section
          title="Method"
          description="What runs under the hood, in one paragraph."
          actions={<TitleIcon icon={Layers} />}
          tone="dark"
        >
          <p className="text-xs leading-relaxed text-ink-600">
            The image is embedded by the {capability.model_id} vision tower. The final transformer
            block's patch tokens are retained with gradients, the selected class logit is
            back-propagated, and the gradients are channel-pooled, ReLU-rectified and normalised
            onto a {result?.grid_size ?? 32}×{result?.grid_size ?? 32} grid. The grid is then
            upsampled over the pixels for the overlay. Device: {capability.device ?? "—"}.
          </p>
        </Section>
      )}
    </div>
  );
}
