"""Week 10: gradient-weighted class activation mapping for the SigLIP vision tower.

This module is the *only* place that produces a heatmap. It is deliberately
separated from the numpy head so that the explanation can be honest about a
capability the checkout may not have:

- A Grad-CAM explanation needs the actual vision transformer's spatial token
  grid and its gradients. That requires a torch build plus the local
  `google/medsiglip-448` weights.
- When either is missing, `availability()` reports why and the API returns an
  explicit not-available payload. Nothing here ever returns a fabricated
  attention map: a plausible-looking random heatmap is exactly the honesty
  failure this project is built to avoid.

The approach is a Grad-CAM over the last vision encoder layer's token grid. The
linear head here is `logits = W @ normalize(pooled)`, so the class score is a
scalar function of every spatial token that contributed to the pooled embedding.
Grad-CAM weights each spatial location by the pooled gradient of the target class
logit, then rectifies and normalises the result into a `grid x grid` map.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PATCH_SIZE = 14
IMAGE_SIZE = 448
GRID = IMAGE_SIZE // PATCH_SIZE  # 32x32 spatial token grid


@dataclass
class Availability:
    """Why a real Grad-CAM can or cannot be produced on this host."""

    available: bool
    reason: str
    device: str | None = None
    model_id: str | None = None


def _torch_present() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "torch is not importable in this Python environment."
    return True, ""


def _weights_present(model_id: str) -> tuple[bool, str]:
    """Cheap check: can the pretrained config resolve from the local cache?

    Mirrors `app.services.registry.model_available` so the two capability gates
    never disagree about whether the weights are on disk.
    """
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(model_id, local_files_only=True)
    except (OSError, ValueError, ImportError, TypeError, KeyError):
        return (
            False,
            f"the pretrained weights for '{model_id}' are not in the local "
            "HuggingFace cache.",
        )
    return True, ""


def availability(model_id: str = "google/medsiglip-448") -> Availability:
    """Whether a real Grad-CAM can run here, and the honest reason if not.

    Never loads the model, so this is safe to call on a hot endpoint that the UI
    polls. The order matters: the cheapest checks run first.
    """
    ok, reason = _torch_present()
    if not ok:
        return Availability(available=False, reason=reason, model_id=model_id)

    ok, reason = _weights_present(model_id)
    if not ok:
        return Availability(available=False, reason=reason, model_id=model_id)

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover -- guarded above
        device = "cpu"
    return Availability(available=True, reason="", device=device, model_id=model_id)


@lru_cache(maxsize=1)
def _load(model_id: str):
    """Construct the vision model once. ~30 s, so cached for the process life.

    Only called on a request that already passed `availability()`. Returns a tuple
    of `(model, processor, device)`.
    """
    import torch
    from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotImageClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, processor, device


def _vision_tower(model):
    """The SiglipVisionModel inside the dual-tower classification model.

    Attribute names differ slightly across transformers versions, so this probes
    the known locations rather than assuming one.
    """
    for path in ("vision_model", "vision_tower", "model.vision_model"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("could not locate the SigLIP vision tower on the model")


def gradcam_grid(
    image_path: Path | str,
    logits_fn,
    *,
    target_index: int,
    model_id: str = "google/medsiglip-448",
) -> np.ndarray:
    """Compute a normalised Grad-CAM grid for one image and one target class.

    `logits_fn(embedding)` maps a `(1, 1152)` pooled embedding to a `(1, 4)`
    logit array. It is the *real* trained head, so the map explains the same
    classifier the detection endpoint serves -- not a proxy.

    Returns a float32 `(GRID, GRID)` array in [0, 1], or raises RuntimeError with
    an operator-readable message. The caller must have checked `availability()`.
    """
    import torch
    from PIL import Image

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")

    model, processor, device = _load(model_id)
    tower = _vision_tower(model)

    image = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    activations: dict[str, "torch.Tensor"] = {}

    def hook(_module, _args, output):
        # transformers may hand back a bare tensor or a ModelOutput here.
        tensor = output if torch.is_tensor(output) else getattr(output, "last_hidden_state", None)
        if tensor is None:
            raise RuntimeError("vision tower output has no hidden state to hook")
        tensor.retain_grad()
        activations["a"] = tensor

    handle = tower.encoder.layers[-1].register_forward_hook(hook)
    try:
        tower.zero_grad(set_to_none=True)
        out = tower(pixel_values=pixel_values)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out

        # The pooled embedding is the vision-tower output; run the real head on it
        # and backprop the target class logit into the spatial tokens.
        pooled = activated_pool(hidden)
        logits = logits_fn(pooled.detach().cpu().numpy())
        score = torch.tensor(
            float(logits[0, target_index]), dtype=hidden.dtype, device=device
        )
        tower.zero_grad(set_to_none=True)
        score.backward()

        maps = activations.get("a")
        if maps is None or maps.grad is None:
            raise RuntimeError("Grad-CAM hook never captured an activation gradient")

        grads = maps.grad          # (1, tokens, dim)
        acts = maps.detach()       # (1, tokens, dim)
    finally:
        handle.remove()

    return _gradcam_from_tokens(acts, grads)


def activated_pool(hidden) -> "torch.Tensor":
    """Pool the token grid down to the single embedding the head consumes.

    SigLIP pools with a learned probe and then projects; both live on the vision
    tower output, so if a pooler_output is present it is authoritative. Otherwise
    mean-pool the patch tokens, which is the correct reduction for the spatial
    grid and is what makes the Grad-CAM attribution meaningful.
    """
    import torch

    if hasattr(hidden, "pooler_output") and hidden.pooler_output is not None:
        return hidden.pooler_output
    if isinstance(hidden, torch.Tensor) and hidden.dim() == 3:
        return hidden[:, 1:].mean(dim=1)
    if hasattr(hidden, "last_hidden_state"):
        state = hidden.last_hidden_state
        if state.dim() == 3:
            return state[:, 1:].mean(dim=1)
        return state
    return hidden


def _gradcam_from_tokens(activations: "torch.Tensor", grads: "torch.Tensor") -> np.ndarray:
    """Turn `(1, tokens, dim)` activations + grads into a normalised grid.

    `tokens` should be the patch grid; a leading CLS token is dropped when the
    count is one more than a perfect square.
    """
    acts = activations[0].detach().cpu().numpy()
    grads_np = grads[0].detach().cpu().numpy()

    n = acts.shape[0]
    side = int(round(np.sqrt(n)))
    if side * side != n:
        # Drop the CLS token if present and see if the remainder squares up.
        if n - 1 > 0 and int(round(np.sqrt(n - 1))) ** 2 == n - 1:
            acts = acts[1:]
            grads_np = grads_np[1:]
            side = int(round(np.sqrt(acts.shape[0])))
        else:
            raise RuntimeError(
                f"cannot reshape {n} vision tokens into a square spatial grid"
            )

    weights = grads_np.mean(axis=-1)                 # (tokens,) channel-pooled grads
    cam = (acts * weights[:, None]).sum(axis=-1)     # (tokens,)
    cam = np.maximum(cam, 0.0)                       # rectify: ReLU of Grad-CAM
    cam = cam.reshape(side, side)

    span = float(cam.max() - cam.min())
    if span <= 1e-12:
        # A flat map is an honest "no spatial preference", not an error. Return
        # zeros rather than normalising noise into a vivid-looking overlay.
        return np.zeros((GRID, GRID), dtype=np.float32)

    cam = (cam - cam.min()) / span
    return _resize_nearest(cam, GRID).astype(np.float32)


def _resize_nearest(grid: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour upsample so the map lands on the 448px image cleanly."""
    src = grid.shape[0]
    if src == size:
        return grid
    idx = (np.arange(size) * src // size).clip(0, src - 1)
    return grid[np.ix_(idx, idx)]


__all__ = [
    "GRID",
    "IMAGE_SIZE",
    "PATCH_SIZE",
    "Availability",
    "availability",
    "gradcam_grid",
]
