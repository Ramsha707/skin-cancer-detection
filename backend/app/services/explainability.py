"""Week 10: explainability service.

Ties the Grad-CAM grid (`ml.explainability`) to the real trained head and renders
a heatmap overlay as a self-contained data URI, so the frontend needs no extra
image endpoint and the response is a single auditable payload.

The honesty contract:

- The map is computed from the *same* head `is` the serving head
  (`registry.load_trained_head`). If that head is the zero-shot prompt head the
  payload says so via `head_kind`, exactly like the detection endpoint.
- When torch or the pretrained weights are missing, `build_explanation` returns
  an unavailable payload with the reason. It never draws a placeholder overlay.
- The class attributed is the *predicted* class by default, but any class can be
  requested so a reviewer can ask "what would make this benign?".
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np

from app.services import registry
from ml import explainability as xai
from ml.models import CANCER_CLASSES

log = logging.getLogger(__name__)

# The overlay colour ramp: low attribution is transparent, high attribution is
# a saturated red. Kept as an explicit breakpoint list so the look is stable.
_RAMP = [
    (0.0, (0, 0, 0, 0)),
    (0.25, (59, 130, 246, 60)),
    (0.5, (250, 204, 21, 120)),
    (0.75, (249, 115, 22, 170)),
    (1.0, (220, 38, 38, 210)),
]


def _parse_model_id() -> str:
    from app.config import settings

    return getattr(settings, "pretrained_model_id", "google/medsiglip-448")


def availability() -> xai.Availability:
    return xai.availability(_parse_model_id())


def _apply_ramp(grid: np.ndarray) -> np.ndarray:
    """Map [0,1] attribution to an RGBA uint8 array via the ramp."""
    stops = np.array([s[0] for s in _RAMP], dtype=np.float32)
    colors = np.array([s[1] for s in _RAMP], dtype=np.float32)
    channels = [
        np.interp(grid, stops, colors[:, c]).astype(np.uint8) for c in range(4)
    ]
    return np.stack(channels, axis=-1)


def _overlay_data_uri(image_path: Path, grid: np.ndarray) -> str | None:
    """Blend the heatmap over the resized image and return a PNG data URI.

    Uses Pillow only (already a project dependency). Returns `None` if Pillow is
    somehow unavailable, so a missing renderer degrades to "no image, but the
    numeric grid is still returned" rather than failing the request.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover -- Pillow is a listed dependency
        log.warning("Pillow missing; returning grid without an overlay image")
        return None

    base = (
        Image.open(image_path)
        .convert("RGB")
        .resize((xai.IMAGE_SIZE, xai.IMAGE_SIZE), Image.BILINEAR)
    )
    rgba = _apply_ramp(grid)
    # The grid is coarser than the image; upsample the colour map to the image
    # size so the two layers composite. Nearest keeps each attribution cell crisp
    # rather than smearing a hot spot across its neighbours.
    heat = Image.fromarray(rgba, mode="RGBA").resize(
        (xai.IMAGE_SIZE, xai.IMAGE_SIZE), Image.NEAREST
    )
    blended = Image.alpha_composite(base.convert("RGBA"), heat).convert("RGB")

    buf = io.BytesIO()
    blended.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _top_regions(grid: np.ndarray, *, top_k: int = 3) -> list[dict]:
    """The most-attributed coarse regions, for a text summary next to the map.

    Bins the grid into a 4x4 layout and reports the strongest cells with their
    normalised coordinates, so a reader who cannot see the image still gets a
    location in words ("upper-left", "centre").
    """
    side = grid.shape[0]
    bins = 4
    step = side // bins
    cells: list[tuple[float, int, int]] = []
    for by in range(bins):
        for bx in range(bins):
            block = grid[by * step : (by + 1) * step, bx * step : (bx + 1) * step]
            cells.append((float(block.mean()), bx, by))
    cells.sort(reverse=True)

    vertical = ["upper", "upper-middle", "lower-middle", "lower"]
    horizontal = ["left", "centre-left", "centre-right", "right"]

    out = []
    for mean, bx, by in cells[:top_k]:
        out.append(
            {
                "intensity": round(mean, 4),
                "column": bx,
                "row": by,
                "region": f"{vertical[by]} {horizontal[bx]}",
            }
        )
    return out


def build_explanation(
    image_path: Path | str,
    *,
    target_class: str | None = None,
    include_overlay: bool = True,
) -> dict:
    """Full explainability payload for one live-uploaded image.

    Never fabricates a map. If the capability is absent, or the target class is
    unknown, the returned dict carries `available=False` and a reason.
    """
    path = Path(image_path)
    model_id = _parse_model_id()

    if target_class is not None and target_class not in CANCER_CLASSES:
        return {
            "available": False,
            "reason": (
                f"unknown class '{target_class}'. Expected one of "
                f"{', '.join(CANCER_CLASSES)}."
            ),
            "model_id": model_id,
            "method": "grad-cam",
        }

    cap = xai.availability(model_id)
    if not cap.available:
        return {
            "available": False,
            "reason": cap.reason,
            "model_id": model_id,
            "method": "grad-cam",
        }

    head, version, trained = registry.load_trained_head()

    def logits_fn(embedding: np.ndarray) -> np.ndarray:
        return head.logits(np.asarray(embedding, dtype=np.float32))

    # Score the image first: the predicted class decides the default attribution
    # target, so there is exactly one Grad-CAM pass in the common case.
    probs = None
    try:
        from app.services.inference import embed_image

        probs = head.probabilities(embed_image(path))[0]
    except Exception:
        log.warning("could not score the image for the explainability payload", exc_info=True)

    predicted_index = int(np.argmax(probs)) if probs is not None else 0
    predicted_class = CANCER_CLASSES[predicted_index]
    target = target_class or predicted_class
    target_index = CANCER_CLASSES.index(target)

    try:
        grid = xai.gradcam_grid(path, logits_fn, target_index=target_index, model_id=model_id)
    except (FileNotFoundError, RuntimeError) as exc:
        log.exception("grad-cam failed")
        return {
            "available": False,
            "reason": f"the heatmap could not be computed: {exc}",
            "model_id": model_id,
            "method": "grad-cam",
        }

    overlay = _overlay_data_uri(path, grid) if include_overlay else None

    return {
        "available": True,
        "reason": None,
        "method": "grad-cam",
        "model_id": model_id,
        "head_kind": "fine-tuned" if trained else "zero-shot-prompt",
        "model_version": version,
        "is_real_inference": trained,
        "predicted_class": predicted_class,
        "target_class": target,
        "target_index": target_index,
        "confidence": round(float(probs[predicted_index]), 6) if probs is not None else None,
        "grid_size": int(grid.shape[0]),
        "grid": [[round(float(v), 4) for v in row] for row in grid.tolist()],
        "overlay_png": overlay,
        "top_regions": _top_regions(grid),
        "device": cap.device,
    }


__all__ = ["availability", "build_explanation"]
