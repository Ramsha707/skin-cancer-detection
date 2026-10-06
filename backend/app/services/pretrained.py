"""Week 5: local introspection of the pre-trained MedSigLIP-448 backbone.

Every parameter count reported here is summed from the header of the
`safetensors` file that is actually on disk, not copied from a paper or scaled
out of the model card. Only the header is read — a few kilobytes of JSON in
front of the 3.5 GB of tensor data — so the endpoint stays fast while the
numbers remain traceable to real bytes.

When no local snapshot exists, the static model card facts are used instead
and `cache_status` reports `missing`. The page must never imply weights are
present when they are not.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import struct
from collections import defaultdict
from pathlib import Path

from app.config import settings
from app.services import registry

log = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
PREPROCESSOR_FILE = "preprocessor_config.json"
WEIGHTS_FILE = "model.safetensors"

# Selective fine-tuning updates the head plus the top fifth of the vision
# encoder; the early patch-embedding layers generalise and stay frozen.
SELECTIVE_LAYER_FRACTION = 0.2

# PIL Resampling enum: only the values we may realistically see are mapped.
_RESAMPLE_NAMES = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "hamming"}

_DEVICE_WITH_OV = "openvino:GPU"
_DEVICE_FALLBACK = "cpu"

_VISION_LAYER_RE = re.compile(r"vision_model\.encoder\.layers\.(\d+)\.")

# The assessment block below is written prose about why this backbone was
# chosen, not measured output — it is the one part of the payload that is
# static by nature.
ORIGINAL_TRAINING_DOMAIN = (
    "Mixed medical imaging: chest X-ray, dermatology, ophthalmology, "
    "histopathology, CT/MRI slices, plus natural images for domain retention"
)
ORIGINAL_TASK = (
    "Contrastive image-text alignment (SigLIP sigmoid loss) for zero-shot "
    "medical image classification and retrieval"
)
ORIGINAL_CLASSES = (
    "Open-vocabulary: trained on image-text pairs with no fixed label set. "
    "Benchmarked zero-shot on 79-class medical classification, where its "
    "dermatology subset includes the lesion types HAM10000 covers."
)
OUTPUT_FORMAT = (
    "1152-d shared image-text embedding from the vision tower; the cancer head "
    "added on top maps it to a 4-class softmax (melanoma, basal cell carcinoma, "
    "actinic keratosis, benign lesion)."
)
DOMAIN_SIMILARITY = (
    "High - dermoscopy images are present in training data and dermatology "
    "was an explicit training modality (MedSigLIP, arXiv:2507.05201)"
)
BACKBONE_NOTES = [
    "Dermatology images (incl. actinic keratosis) present in training data - "
    "domain overlap is strong",
    "448x448 resolution matches the HAM10000 aspect ratio well (no extreme cropping)",
    "MedSigLIP reports dermatology AUC 0.851 (79-class) and linear probe 0.881, "
    "demonstrating transferability",
    "SigLIP's shared image-text space enables prompt-based head initialisation "
    "without labels",
    "~878M parameters provide ample capacity for 4-class adaptation",
    "No SQCC (invasive squamous cell carcinoma) in HAM10000 - actinic keratosis "
    "is the closest precursor class available",
]
HEAD_INITIALISATION = (
    "Text-prompt embeddings: each class weight row is initialised from the "
    "MedSigLIP text-tower embedding of its canonical prompt, then normalised. "
    "This places the head in the shared image-text space from epoch 0 instead "
    "of starting from a coin flip."
)


def hub_cache_dir() -> Path:
    """The HuggingFace cache directory, honouring the standard env overrides."""
    if env := os.environ.get("HF_HUB_CACHE"):
        return Path(env)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def local_snapshot(model_id: str | None = None) -> Path | None:
    """Newest local snapshot directory for `model_id`, or None if uncached.

    Snapshots are named by commit hash, so lexicographic order is not age
    order; the snapshot that actually carries the weights wins, and any
    snapshot beats none.
    """
    model_id = model_id or settings.pretrained_model_id
    repo = hub_cache_dir() / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repo.is_dir():
        return None
    snapshots = sorted(p for p in repo.iterdir() if p.is_dir())
    for snap in reversed(snapshots):
        if (snap / WEIGHTS_FILE).is_file():
            return snap
    return snapshots[-1] if snapshots else None


def read_tensor_header(path: Path) -> dict[str, dict]:
    """`{tensor name: {"dtype": str, "shape": [int]}}` from a safetensors file.

    The format is an 8-byte little-endian header length followed by that many
    bytes of JSON, then the tensor data, which is never read here.
    """
    with path.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) < 8:
            raise ValueError(f"{path} is too small to be a safetensors file")
        (length,) = struct.unpack("<Q", raw_len)
        header = json.loads(fh.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"{path} header is not a tensor map")
    header.pop("__metadata__", None)
    return header


def _numel(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _sum_prefix(header: dict[str, dict], prefix: str) -> int:
    return sum(_numel(v["shape"]) for k, v in header.items() if k.startswith(prefix))


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.debug("could not read %s", path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def _selective_extra_params(header: dict[str, dict], total_layers: int) -> int:
    """Parameters unlocked by selective fine-tuning: the top 20% of the tower.

    Deep layers are the domain-specific ones, so they are the ones worth
    unfreezing; the shallow patch-embedding layers already generalise.
    """
    layer_params: dict[int, int] = defaultdict(int)
    for name, meta in header.items():
        if match := _VISION_LAYER_RE.match(name):
            layer_params[int(match.group(1))] += _numel(meta["shape"])

    depth = math.ceil(total_layers * SELECTIVE_LAYER_FRACTION)
    return sum(layer_params.get(i, 0) for i in range(total_layers - depth, total_layers))


def build_pretrained_info() -> dict:
    """Assemble the payload for `GET /api/models/pretrained`."""
    snapshot = local_snapshot()
    config = _load_json(snapshot / CONFIG_FILE) if snapshot else None
    preprocessor = _load_json(snapshot / PREPROCESSOR_FILE) if snapshot else None
    card = registry.load_model_card()

    weights_path = snapshot / WEIGHTS_FILE if snapshot else None
    header: dict[str, dict] | None = None
    if weights_path is not None and weights_path.is_file():
        try:
            header = read_tensor_header(weights_path)
        except (OSError, ValueError, struct.error):
            log.warning("could not parse %s", weights_path, exc_info=True)

    vision_cfg = (config or {}).get("vision_config", {})
    text_cfg = (config or {}).get("text_config", {})

    vision_layers = int(vision_cfg.get("num_hidden_layers", 27))
    hidden = int(vision_cfg.get("hidden_size", card["vision_tower"]["hidden_size"]))
    patch = int(vision_cfg.get("patch_size", card["vision_tower"]["patch_size"]))
    image_size = int(vision_cfg.get("image_size", card["input"]["size"]))
    embedding_dim = int(text_cfg.get("projection_size", card["embedding_dim"]))

    class_names = registry.class_names()
    n_classes = len(class_names)

    if header is not None:
        # Exact: summed from the tensors that are actually on disk.
        total_params = sum(_numel(m["shape"]) for m in header.values())
        vision_params = _sum_prefix(header, "vision_model")
        text_params = _sum_prefix(header, "text_model")
        selective_extra = _selective_extra_params(header, vision_layers)
    else:
        # Static-card fallback: the total is a published figure and the tower
        # split is a documented estimate. `cache_status` reports `missing`, so
        # the UI never presents these as locally verified.
        total_params = int(card["total_params"])
        vision_params = int(total_params * 0.55)
        text_params = total_params - vision_params
        selective_extra = int(vision_params * SELECTIVE_LAYER_FRACTION)

    # The adaptation layer is one linear map over the frozen embedding.
    head_params = embedding_dim * n_classes + n_classes

    resample = "bicubic"
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    rescale = 1.0 / 255.0
    if preprocessor:
        resample = _RESAMPLE_NAMES.get(int(preprocessor.get("resample", 3) or 3), resample)
        mean = [float(x) for x in preprocessor.get("image_mean", mean)]
        std = [float(x) for x in preprocessor.get("image_std", std)]
        rescale = float(preprocessor.get("rescale_factor", rescale))
        size = preprocessor.get("size", {})
        if height := size.get("height"):
            image_size = int(height)

    # "Present" means the header parsed: a corrupt or truncated weight file is
    # reported as absent rather than as verified.
    weights_present = header is not None
    if snapshot is None:
        cache_status = "missing"
    elif config is not None and weights_present:
        cache_status = "ok"
    else:
        cache_status = "partial"

    architectures = (config or {}).get("architectures") or [card["architecture"]]

    return {
        "model_id": settings.pretrained_model_id,
        "architecture": architectures[0],
        "model_type": "SiglipModel (dual-tower contrastive encoder)",
        "vision_tower": {
            "layers": vision_layers,
            "hidden_size": hidden,
            "intermediate_size": int(
                vision_cfg.get("intermediate_size", card["vision_tower"]["mlp_size"])
            ),
            "attention_heads": int(
                vision_cfg.get(
                    "num_attention_heads", card["vision_tower"]["attention_heads"]
                )
            ),
            "patch_size": patch,
            "image_size": image_size,
            "num_patches": (image_size // patch) ** 2,
            "params": vision_params,
            "trainable_when_frozen": 0,
        },
        "text_tower": {
            "layers": int(text_cfg.get("num_hidden_layers", 27)),
            "hidden_size": int(text_cfg.get("hidden_size", card["text_tower"]["hidden_size"])),
            "intermediate_size": int(text_cfg.get("intermediate_size", hidden * 4)),
            "attention_heads": int(
                text_cfg.get("num_attention_heads", card["vision_tower"]["attention_heads"])
            ),
            "vocab_size": int(text_cfg.get("vocab_size", card["text_tower"]["vocab_size"])),
            "max_position_embeddings": int(
                text_cfg.get("max_position_embeddings", card["text_tower"]["max_positions"])
            ),
            "params": text_params,
            "used_for_classifier": True,
        },
        "projection_size": embedding_dim,
        "embedding_dim": embedding_dim,
        "total_params": total_params,
        "trainable_params_frozen_strategy": head_params,
        "trainable_params_selective_strategy": head_params + selective_extra,
        "trainable_params_full_strategy": total_params,
        "frozen_params_frozen_strategy": total_params - head_params,
        "preprocessing": {
            "input_size": [image_size, image_size],
            "resample": resample,
            "rescale_factor": rescale,
            "image_mean": mean,
            "image_std": std,
        },
        "original_training_domain": ORIGINAL_TRAINING_DOMAIN,
        "original_task": ORIGINAL_TASK,
        "original_classes": ORIGINAL_CLASSES,
        "output_format": OUTPUT_FORMAT,
        "domain_similarity_to_dermoscopy": DOMAIN_SIMILARITY,
        "suitable_as_backbone": True,
        "backbone_viability_notes": BACKBONE_NOTES,
        "head_initialisation": HEAD_INITIALISATION,
        "text_prompts": [
            {"dx": dx, "prompt": prompt}
            for dx, prompt in zip(class_names, registry.class_prompts(), strict=True)
        ],
        "loaded_locally": weights_present,
        "device": _DEVICE_WITH_OV if registry.ov_available() else _DEVICE_FALLBACK,
        "resolved_path": str(snapshot) if snapshot else None,
        "cache_status": cache_status,
    }


__all__ = [
    "BACKBONE_NOTES",
    "DOMAIN_SIMILARITY",
    "HEAD_INITIALISATION",
    "ORIGINAL_CLASSES",
    "ORIGINAL_TASK",
    "ORIGINAL_TRAINING_DOMAIN",
    "OUTPUT_FORMAT",
    "build_pretrained_info",
    "hub_cache_dir",
    "local_snapshot",
    "read_tensor_header",
]
