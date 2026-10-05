"""Week 6: transfer-learning strategy comparison.

Three strategies are registered up front. On this machine only two of them can
actually run, and the third records *why* it did not rather than reporting a
number:

* `frozen` -- linear probe over cached embeddings. Runs in seconds at full scale.
  This is the real, runnable experiment.
* `selective` -- unfreeze the last N transformer blocks of the vision tower.
  Runs, but each epoch re-runs the forward pass, so it is capped to a subset.
* `full` -- every parameter trainable. Gated behind an explicit budget check that
  refuses on CPU-only hosts, because 878M parameters in Adam needs roughly 17.5 GB
  of optimiser state alone and minutes per step on this hardware.

A refused experiment is still recorded, with `is_real_result = False` and a note
saying what was missing. The Experiments page renders those as "not run: requires
GPU" instead of showing an empty metric block that reads like a failure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services import registry
from app.services.dataset import CANCER_CLASSES, MALIGNANT_CLASSES
from app.services.inference import evaluate_head_on
from app.services.metrics import evaluate
from ml.models import CancerHead
from ml.training.head import inverse_frequency_weights, train_head

log = logging.getLogger(__name__)

# Memory floor for full fine-tuning: 878M params in fp32 with Adam moments.
# Kept for the explanation text, not as the gate -- see
# `ComputeBudget.supports_full_finetune` for why RAM alone does not qualify.
FULL_FINETUNE_MIN_BYTES = 17_500_000_000


@dataclass
class ComputeBudget:
    """What this host can actually afford. Decides which arms are runnable."""

    total_ram_bytes: int
    cuda_available: bool
    model_params: int = 878_000_000

    @property
    def supports_full_finetune(self) -> bool:
        """Full fine-tuning needs a GPU, not just RAM.

        RAM alone is misleading: this host has 34 GB, which clears the ~17.5 GB
        optimiser-state floor, but at the measured 0.08 img/s for Torch CPU a
        single 6,925-image epoch is ~24 h and a 3-epoch run is over three days.
        Enough memory to not crash is not the same as enough compute to finish,
        so the gate is a CUDA device.
        """
        return self.cuda_available

    @property
    def reason_if_not(self) -> str | None:
        if self.supports_full_finetune:
            return None
        return (
            f"Not run: full fine-tuning of all {self.model_params / 1e6:.0f}M parameters "
            f"needs a CUDA device. Its optimiser state alone is ~17.5 GB (fp32 weights, "
            f"gradients and two Adam moments), and while this host reports "
            f"{self.total_ram_bytes / 1e9:.1f} GB of RAM -- enough not to crash -- the "
            "measured CPU throughput of 0.08 img/s puts one epoch at ~24 h. "
            "Selective unfreezing of the last blocks answers the same 'adapt more of "
            "the backbone' question at a fraction of the cost."
        )


def detect_budget() -> ComputeBudget:
    """Read what this host can afford, without requiring torch.

    The backend venv has no torch, so probing `torch.cuda` would make every
    `/api/experiments` response a 500. A missing torch is itself the answer:
    there is no CUDA device, which is what the budget check needs to know.
    """
    try:
        total = _physical_memory_bytes()
    except (OSError, AttributeError):
        # Non-Windows host, or ctypes unavailable. Report 0 rather than guessing:
        # an honest "unknown" must not read as "affordable".
        total = 0

    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except ModuleNotFoundError:
        cuda = False

    return ComputeBudget(total_ram_bytes=total, cuda_available=cuda)


def _physical_memory_bytes() -> int:
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MemoryStatusEx()
    stat.dwLength = ctypes.sizeof(MemoryStatusEx)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return int(stat.ullTotalPhys)


@dataclass
class ExperimentResult:
    name: str
    strategy: str
    status: str
    is_real_result: bool
    notes: str
    metrics: dict = field(default_factory=dict)
    trainable_layers: str | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    train_seconds: float | None = None
    epochs: int | None = None
    learning_rate: float | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "status": self.status,
            "is_real_result": self.is_real_result,
            "notes": self.notes,
            "metrics": self.metrics,
            "trainable_layers": self.trainable_layers,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "train_seconds": self.train_seconds,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
        }


def frozen_probe(
    train_split,
    test_split,
    *,
    init: CancerHead | None = None,
    epochs: int = 60,
    learning_rate: float = 0.5,
) -> ExperimentResult:
    """Full-scale, fully runnable. Linear probe over frozen embeddings.

    This is the headline experiment: at full dataset scale it trains in seconds
    and is evaluated on a held-out patient-level test split.
    """
    t0 = time.perf_counter()
    weights = inverse_frequency_weights(train_split.labels, len(CANCER_CLASSES))
    result = train_head(
        train_split.embeddings,
        train_split.labels,
        init=init or registry.get_zero_shot_head(),
        epochs=epochs,
        learning_rate=learning_rate,
        class_weight=weights,
    )
    elapsed = time.perf_counter() - t0

    metrics = evaluate_head_on(result.head, test_split)
    return ExperimentResult(
        name=f"Frozen linear probe ({train_split.size} train / {test_split.size} test)",
        strategy="frozen",
        status="completed",
        is_real_result=bool(train_split.real and test_split.real),
        notes=(
            "Vision tower frozen throughout; only the 4,612-parameter linear head was "
            "trained. Evaluated on the patient-level held-out test split."
        ),
        metrics=metrics,
        trainable_layers="classifier head only (linear, 1152 -> 4)",
        trainable_params=result.head.param_count,
        total_params=result.head.param_count,
        train_seconds=elapsed,
        epochs=epochs,
        learning_rate=learning_rate,
    )


def selective_unfreeze(
    train_split,
    test_split,
    *,
    last_blocks: int = 2,
    max_images: int = 1400,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    budget: ComputeBudget | None = None,
) -> ExperimentResult:
    """Unfreeze the last N vision blocks and train on a capped subset of *images*.

    This runs a real backward pass through the vision tower: the cached-embedding
    shortcut is not usable here, because the whole point of selective unfreezing
    is that the backbone weights change, and cached embeddings are frozen outputs
    of the old backbone. So this reads image paths from the split manifest and
    re-runs the forward pass per image.

    Each epoch costs a full forward+backward at the measured ~0.57 img/s, so a
    6,925-image epoch is ~3.4 h. `max_images` bounds a single run to a defensible
    duration and the reported note states the subset size rather than implying
    full-dataset coverage.
    """
    budget = budget or detect_budget()

    paths, labels = _stride_images(train_split, max_images)
    n = len(paths)

    t0 = time.perf_counter()
    log.info("selective: %d images, %d epoch(s)", n, epochs)

    import torch
    from PIL import Image

    from medsiglip import MedSiglipClassifier

    classifier = MedSiglipClassifier()
    model = classifier.model
    vision = model.model.vision_model if hasattr(model, "model") else model.vision_model
    encoder = vision.vision_model.encoder
    layers = list(encoder.layers)
    trainable_layers = layers[-last_blocks:]
    for block in trainable_layers:
        for param in block.parameters():
            param.requires_grad_(True)

    trainable = sum(p.numel() for p in trainable_layers.parameters())
    total_params = registry.load_model_card()["total_params"]

    prompt_head = registry.get_zero_shot_head()
    head_params = torch.nn.Parameter(torch.from_numpy(prompt_head.weights.T.copy()))
    bias_params = torch.nn.Parameter(torch.from_numpy(prompt_head.bias.copy()))
    optimiser = torch.optim.AdamW(
        [*trainable_layers.parameters(), head_params, bias_params],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    criterion = torch.nn.CrossEntropyLoss()
    label_tensor = torch.from_numpy(labels.astype(np.int64))

    history: list[float] = []
    try:
        for epoch in range(epochs):
            running, seen = 0.0, 0
            for i, path in enumerate(paths):
                pixels = classifier.preprocess(Image.open(path).convert("RGB")).unsqueeze(0)

                # hidden_states[-1] is the tower output; the leading token is the
                # pooled representation the linear head consumes.
                feats = vision(
                    pixel_values=pixels, output_hidden_states=True
                ).hidden_states[-1][:, 0]
                logits = torch.nn.functional.linear(feats, head_params, bias_params)

                loss = criterion(logits.squeeze(0), label_tensor[i])
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

                running += float(loss)
                seen += 1
                if i % 25 == 0:
                    log.info(
                        "  epoch %d %d/%d loss %.4f", epoch + 1, i, n, running / max(seen, 1)
                    )
            history.append(running / max(seen, 1))

        elapsed = time.perf_counter() - t0
        metrics = _evaluate_trained_tower(model, head_params, bias_params, test_split)
        return ExperimentResult(
            name=f"Selective unfreeze, last {last_blocks} blocks ({n} images)",
            strategy="selective",
            status="completed",
            is_real_result=True,
            notes=(
                f"Selective: last {last_blocks} of {len(layers)} vision blocks unfrozen "
                f"({trainable / 1e6:.1f}M trainable of {total_params / 1e6:.0f}M). "
                f"Real forward+backward pass on a {n}-image strided subset "
                f"(1 epoch): a full epoch at the measured 0.57 img/s would take ~3.4 h. "
                f"Evaluated on the held-out test split through the updated tower."
            ),
            metrics=metrics,
            trainable_layers=f"vision_model.encoder.layers[-{last_blocks}:]",
            trainable_params=trainable,
            total_params=total_params,
            train_seconds=elapsed,
            epochs=epochs,
            learning_rate=learning_rate,
        )
    finally:
        for block in trainable_layers:
            for param in block.parameters():
                param.requires_grad_(False)
        del trainable_layers, optimiser
        log.info("selective finished, peak params restored to requires_grad=False")


def _stride_images(split, max_images: int) -> tuple[np.ndarray, list[Path], np.ndarray]:
    """Strided image subset, preserving the non-IID class mix.

    Head-slicing would take a contiguous block, which the partitioner built to be
    class-skewed. Striding samples across the whole partition instead.
    """
    n = min(max_images, split.size)
    keep = np.linspace(0, split.size - 1, n).astype(int)
    return [Path(split.paths[i]) for i in keep], split.labels[keep]


def _evaluate_trained_tower(model, head_params, bias_params, test_split) -> dict:
    """Score the updated tower on the held-out split by re-embedding its images.

    The cached test embeddings are stale for this arm: they were produced by the
    pre-training backbone. Re-embedding is the only way to report a number that
    reflects the weights that were actually trained.
    """
    import torch
    from PIL import Image

    from medsiglip import MedSiglipClassifier

    classifier = MedSiglipClassifier()
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for i, path in enumerate(test_split.paths):
            pixels = classifier.preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
            feats = model.model.vision_model(
                pixel_values=pixels, output_hidden_states=True
            ).hidden_states[-1][:, 0]
            logits = torch.nn.functional.linear(feats, head_params, bias_params)
            preds.append(int(logits.argmax(dim=-1)))
            labels.append(int(test_split.labels[i]))
            if i % 100 == 0:
                log.info("  eval %d/%d", i, test_split.size)
    report = evaluate(
        np.array(labels, dtype=np.int64),
        np.array(preds, dtype=np.int64),
        n_classes=len(CANCER_CLASSES),
        class_names=CANCER_CLASSES,
        malignant_indices=[
            CANCER_CLASSES.index(c) for c in MALIGNANT_CLASSES
        ],
        y_score=np.eye(len(CANCER_CLASSES))[preds],
    )
    payload = report.as_dict()
    payload["is_real_result"] = True
    return payload


def full_finetune(budget: ComputeBudget | None = None) -> ExperimentResult:
    """Registered but refused on CPU-only hosts. Records why."""
    budget = budget or detect_budget()
    return ExperimentResult(
        name="Full fine-tune (all 878M parameters)",
        strategy="full",
        status="skipped",
        is_real_result=False,
        notes=budget.reason_if_not
        or "Not run: full fine-tuning was not attempted on this host.",
        metrics={},
        trainable_layers="all",
        trainable_params=registry.load_model_card()["total_params"],
        total_params=registry.load_model_card()["total_params"],
    )


def zero_shot_baseline(test_split) -> ExperimentResult:
    """The prompt head, evaluated untrained. The control every other arm beats."""
    head = registry.get_zero_shot_head()
    metrics = evaluate_head_on(head, test_split)
    return ExperimentResult(
        name="Zero-shot prompt head (no training)",
        strategy="frozen",
        status="completed",
        is_real_result=bool(test_split.real),
        notes=(
            "Control condition. The four-class head initialised purely from MedSigLIP "
            "text-prompt embeddings, with zero gradient steps."
        ),
        metrics=metrics,
        trainable_layers="none (frozen at prompt initialisation)",
        trainable_params=0,
        total_params=registry.load_model_card()["total_params"],
        train_seconds=0.0,
    )


def run_all(
    train_split, test_split, *, budget: ComputeBudget | None = None
) -> list[ExperimentResult]:
    """The full comparison, including the arm that cannot run."""
    budget = budget or detect_budget()
    results = [zero_shot_baseline(test_split), frozen_probe(train_split, test_split)]
    results.append(full_finetune(budget))
    return results


def persist(results: list[ExperimentResult]) -> Path:
    """Write the comparison to disk so the UI can render it without a rerun."""
    import json

    out = registry.CHECKPOINT_DIR / "experiments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump([r.as_dict() for r in results], fh, indent=2)
    return out


def persistence_path() -> Path:
    return registry.CHECKPOINT_DIR / "experiments.json"


__all__ = [
    "ComputeBudget",
    "ExperimentResult",
    "detect_budget",
    "frozen_probe",
    "full_finetune",
    "persist",
    "persistence_path",
    "run_all",
    "selective_unfreeze",
    "zero_shot_baseline",
]