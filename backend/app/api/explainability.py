"""Week 10: explainability endpoints.

Two calls:

- `GET /api/explainability/status` — a cheap capability probe the UI polls to
  decide between the upload panel and an instruction panel.
- `POST /api/explainability/generate` — explain one freshly-uploaded image.

There is no endpoint that explains a *stored* detection: the upload directory is
not linked back to the `detection_results` rows by design, so a heatmap is only
produced for the image in the current request. That keeps the explanation bound to
the bytes the reviewer actually submitted.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database.base import get_db
from app.schemas import ExplainabilityCapability, ExplainabilityResult
from app.services import audit as audit_service
from app.services import explainability as xai_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/explainability", tags=["explainability"])

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
UPLOAD_DIR = Path(__file__).resolve().parents[3] / "data" / "uploads"


@router.get("/status", response_model=ExplainabilityCapability)
def explainability_status() -> dict:
    """Whether a real Grad-CAM can run here, with the honest reason if not.

    Never loads the model, so it is safe to call on every page visit.
    """
    cap = xai_service.availability()
    return {
        "available": cap.available,
        "reason": cap.reason,
        "method": "grad-cam",
        "model_id": cap.model_id or settings.pretrained_model_id,
        "device": cap.device,
    }


@router.post("/generate", response_model=ExplainabilityResult)
async def generate_explanation(
    file: UploadFile = File(...),
    target_class: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> dict:
    """Compute a Grad-CAM overlay for one uploaded image.

    Returns `available=False` with a reason when the capability is absent, rather
    than an error status: an unavailable explanation is a normal state the UI
    renders as an instruction, not a failure.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"unsupported file type '{suffix or 'unknown'}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    payload = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(payload) > limit:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"image is {len(payload) / 1024 / 1024:.1f} MB, "
                f"limit is {settings.max_upload_mb} MB"
            ),
        )
    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="uploaded file is empty")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}{suffix}"
    target = UPLOAD_DIR / name
    target.write_bytes(payload)

    result = xai_service.build_explanation(target, target_class=target_class)

    # Only a map that actually rendered is a real explainability event. A refused
    # capability is not logged as if it produced something.
    if result.get("available"):
        audit_service.log(
            db,
            event_type=audit_service.EVENT_EXPLAINABILITY_GENERATED,
            message=(
                f"Grad-CAM generated for {file.filename or name} "
                f"(target: {result.get('target_class')})."
            ),
            details={
                "method": result.get("method"),
                "target_class": result.get("target_class"),
                "model_version": result.get("model_version"),
                "head_kind": result.get("head_kind"),
                "grid_size": result.get("grid_size"),
            },
        )
        db.commit()

    return result


__all__ = ["router"]
