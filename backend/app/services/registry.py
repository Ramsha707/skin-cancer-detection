"""Registry and cache loading for trained heads.

Bridges the on-disk embedding caches to the running FastAPI process. Loading is
lazy and cached, because the OpenVINO model takes ~30 s to construct and the
backend should not pay that on import.

Every loader returns `is_real=True` only when the underlying artefacts actually
exist on disk. When they do not, callers fall back to the zero-shot text
classifier and every response carries `is_real_inference = False`, so the
distinction survives into the API and the UI rather than being papered over.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.config import REPO_ROOT, settings
from ml.models import CANCER_CLASSES, CLASS_PROMPTS, CancerHead, EmbeddingCache

log = logging.getLogger(__name__)

EMBEDDING_DIR = REPO_ROOT / "data" / "embeddings"
SPLIT_PATH = REPO_ROOT / "data" / "splits.json"
CHECKPOINT_DIR = REPO_ROOT / "data" / "checkpoints"

AGENT_SLUGS = ["agent_a", "agent_b", "agent_c", "agent_d"]

# Output width of the vision tower. Everything downstream sizes itself from this:
# the head weight matrix, the embedding caches and the FedAvg payload.
EMBEDDING_DIM = 1152


@dataclass
class CachedSplit:
    embeddings: np.ndarray
    labels: np.ndarray
    image_ids: list[str]
    real: bool
    paths: list[Path] = field(default_factory=list)

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])


def cache_dir() -> Path:
    return EMBEDDING_DIR


def load_split(name: str, agent: str | None = None) -> CachedSplit:
    """Load one embedding cache. Falls back to a synthetic placeholder.

    The placeholder is clearly flagged (`real=False`) and is a fixed deterministic
    array, so an un-extracted checkout renders a warning rather than either
    crashing or silently reporting fabricated accuracy.
    """
    filename = f"{name}.npy" if agent is None else f"{name}_{agent}.npy"
    cache = EmbeddingCache(EMBEDDING_DIR / filename)
    if cache.exists():
        try:
            embeddings, labels, ids = cache.load()
            return CachedSplit(
                embeddings,
                labels,
                ids,
                real=True,
                paths=load_paths(name, agent),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            log.warning("could not read %s: %s", filename, exc)

    n = 8 if name == "train" else 16
    log.warning("embedding cache %s missing; using placeholder", filename)
    return CachedSplit(
        embeddings=np.zeros((n, 1152), dtype=np.float32),
        labels=np.zeros(n, dtype=np.int64),
        image_ids=[f"placeholder_{i}" for i in range(n)],
        real=False,
    )


def load_paths(name: str, agent: str | None = None) -> list[Path]:
    """Image paths for a split, in the same order as its embedding cache.

    Only the selective-unfreeze arm needs this, because it re-runs the vision
    tower instead of reading cached embeddings. `EmbeddingCache` deliberately does
    not store paths next to the vectors, so they come from `splits.json`.
    """
    if not SPLIT_PATH.is_file():
        return []
    with SPLIT_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    block = data["agents"][agent] if agent else data.get(name)
    if not block:
        return []
    return [Path(p) for p in block["paths"]]


def head_param_count() -> int:
    """Trainable floats in the cancer head: one weight row and bias per class.

    Single source of truth. Week 4 renders it on the detection page, Week 7 uses
    it as the FedAvg payload size, and `app.api.models` builds the strategy
    budget from it. Derived rather than hardcoded so a taxonomy change cannot
    silently leave one page reporting a stale count.
    """
    return (EMBEDDING_DIM + 1) * len(class_names())


def caches_ready() -> dict[str, bool]:
    """Which artefacts exist. Drives the 'not built yet' states in the UI."""
    def have(name: str, agent: str | None = None) -> bool:
        filename = f"{name}.npy" if agent is None else f"{name}_{agent}.npy"
        return (EMBEDDING_DIR / filename).is_file() and (
            EMBEDDING_DIR / filename.replace(".npy", ".json")
        ).is_file()

    return {
        "val": have("val"),
        "test": have("test"),
        **{f"train_{a}": have("train", a) for a in AGENT_SLUGS},
    }


@lru_cache(maxsize=1)
def load_model_card() -> dict:
    """Static architecture facts, read from the local HF cache where possible."""
    card: dict = {
        "model_id": settings.pretrained_model_id,
        "architecture": "SiglipModel (dual-tower)",
        "vision_tower": {
            "layers": 27,
            "hidden_size": 1152,
            "mlp_size": 4304,
            "attention_heads": 16,
            "patch_size": 14,
        },
        "text_tower": {
            "layers": 27,
            "hidden_size": 1152,
            "vocab_size": 32_000,
            "max_positions": 64,
        },
        "input": {"size": 448, "resample": "bicubic", "mean": 0.5, "std": 0.5},
        "embedding_dim": 1152,
        "total_params": 878_000_000,
        "source": "local HuggingFace cache",
    }
    return card


@lru_cache(maxsize=1)
def _zero_shot_head() -> CancerHead:
    """Prompt-initialised head used before any training has happened.

    Constructing this needs the text tower, which loads the full 878M model. That
    is too slow to do at import time, so it is deferred until a request actually
    needs a prediction.
    """
    from medsiglip import MedSiglipClassifier

    clf = MedSiglipClassifier(model_id=settings.pretrained_model_id)
    embeddings = clf.embed_texts(CLASS_PROMPTS).numpy()
    return CancerHead.from_text_prompts(embeddings)


def get_zero_shot_head() -> CancerHead:
    return CancerHead.from_state(_zero_shot_head().state_dict())


def model_available() -> bool:
    """Whether the pretrained weights are present locally.

    Cheap: it checks the resolved path, not a model load. `pretrained_info` uses
    this to report whether the build is wired to real weights or is config-only.
    """
    try:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            settings.pretrained_model_id, local_files_only=True
        ) is not None
    except (OSError, ValueError, ImportError, TypeError, KeyError):
        # Any failure here means "not confidently available". Reporting false is
        # the safe direction: it never claims weights exist when they do not.
        log.debug("pretrained weights not resolvable locally", exc_info=True)
        return False


def latest_checkpoint() -> Path | None:
    """Newest trained checkpoint on disk, without loading any model.

    Deliberately cheap: `detection/status` is polled by the UI and must not pay
    the 878M-parameter model load just to answer "is anything trained yet".
    """
    if not CHECKPOINT_DIR.is_dir():
        return None
    checkpoints = sorted(
        CHECKPOINT_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return checkpoints[0] if checkpoints else None


def load_trained_head() -> tuple[CancerHead, str, bool]:
    """Best available head, newest first.

    Returns `(head, version, trained)`. When nothing is trained this is the
    zero-shot prompt head, with `trained=False` so the caller can label it.
    """
    path = latest_checkpoint()
    if path is not None:
        with np.load(path) as data:
            head = CancerHead.from_state(
                {"weights": data["weights"], "bias": data["bias"]}
            )
        return head, path.stem, True
    return get_zero_shot_head(), "zeroshot-prompt", False


def save_head(head: CancerHead, version: str) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"{version}.npz"
    np.savez(path, **head.state_dict())
    return path


def class_prompts() -> list[str]:
    return list(CLASS_PROMPTS)


def class_names() -> list[str]:
    return list(CANCER_CLASSES)


__all__ = [
    "AGENT_SLUGS",
    "EMBEDDING_DIR",
    "CachedSplit",
    "caches_ready",
    "class_names",
    "class_prompts",
    "get_zero_shot_head",
    "load_model_card",
    "load_split",
    "load_trained_head",
    "save_head",
]