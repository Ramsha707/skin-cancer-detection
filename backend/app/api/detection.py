"""Week 4: cancer detection endpoints.

Upload -> preprocess -> predict, with the real-vs-simulated distinction carried
in the response body rather than inferred by the client.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.database.base import get_db
from app.models import DetectionResult
from app.schemas import (
    CancerClassOut,
    CancerHeadOut,
    DetectionClassesOut,
    DetectionResultOut,
    DetectionStatusOut,
)
from app.services import audit as audit_service
from app.services import inference, registry

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/detection", tags=["detection"])

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}
UPLOAD_DIR = Path(__file__).resolve().parents[3] / "data" / "uploads"


@router.get("", response_model=list[DetectionResultOut])
def list_detections(db: Session = Depends(get_db)) -> list[DetectionResult]:
    return list(
        db.scalars(select(DetectionResult).order_by(DetectionResult.created_at.desc()).limit(50))
    )


@router.post("/predict", response_model=DetectionResultOut)
async def predict_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DetectionResult:
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

    try:
        result = inference.predict(target)
    except FileNotFoundError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail="image could not be read"
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    row = DetectionResult(
        image_name=file.filename or name,
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        model_version=result.model_version,
        inference_time=result.inference_seconds,
        strategy="frozen-linear-probe",
        is_real_inference=result.is_real_inference,
        modality="dermoscopy",
        note=result.note,
        probabilities=result.probabilities,
    )
    db.add(row)

    audit_service.log(
        db,
        event_type=audit_service.EVENT_DETECTION_PERFORMED,
        message=f"Classified {row.image_name} as {result.predicted_class} "
        f"({result.confidence:.1%}).",
        details={
            "model_version": result.model_version,
            "is_real_inference": result.is_real_inference,
        },
    )
    db.commit()
    db.refresh(row)
    return row


@router.get("/classes", response_model=DetectionClassesOut)
def list_classes() -> dict:
    """Class taxonomy plus the prompts the head is built from."""
    from app.services.dataset import MALIGNANT_CLASSES

    prompts = registry.class_prompts()
    return {
        "classes": [
            CancerClassOut(
                name=name, prompt=prompt, malignant=name in MALIGNANT_CLASSES
            ).model_dump()
            for name, prompt in zip(registry.class_names(), prompts, strict=True)
        ],
        "head": CancerHeadOut(
            embedding_dim=registry.load_model_card()["embedding_dim"],
            trainable_params=registry.head_param_count(),
        ).model_dump(),
        "caches_ready": registry.caches_ready(),
    }


@router.get("/status", response_model=DetectionStatusOut)
def detection_status() -> dict:
    """What can this endpoint actually do right now.

    The UI reads this to decide between a working upload panel and an explicit
    'run the extraction script' instruction, rather than rendering an upload box
    that would fail.
    """
    ready = registry.caches_ready()
    checkpoint = registry.latest_checkpoint()
    trained = checkpoint is not None
    version = checkpoint.stem if trained else "zeroshot-prompt"

    if not ready["val"]:
        instructions = (
            "Run .\\backend-venv\\Scripts\\python.exe scripts/build_splits.py, then "
            ".\\.venv-ov\\Scripts\\python.exe scripts/extract_embeddings.py --split val "
            "(~35 min on the iGPU)."
        )
    elif not trained:
        instructions = (
            "No fine-tuned checkpoint in data/checkpoints. "
            "Predictions use the zero-shot prompt head."
        )
    else:
        instructions = None

    return {
        "ready": bool(trained and ready["val"]),
        "model_version": version,
        "is_real_inference": trained,
        "caches_ready": ready,
        "instructions": instructions,
    }