"""Week 5: pre-trained model information endpoints; week 7: model registry rows.

Serves the MedSigLIP-448 introspection that the `/models/pretrained` page
renders, plus the list of global model versions produced by federated rounds.
The registry *page* arrives in week 12; the rows themselves are real from week 7
onward, so the list is served now rather than stubbed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.models import ModelVersion
from app.schemas import ModelVersionOut, PretrainedModelInfo
from app.services.pretrained import build_pretrained_info

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/pretrained", response_model=PretrainedModelInfo)
def pretrained_model_info() -> dict:
    """Architecture, parameter budget, preprocessing and domain assessment.

    Parameter counts are summed from the local safetensors header when the
    weights are cached; `cache_status` tells the client which case it got.
    """
    return build_pretrained_info()


@router.get("", response_model=list[ModelVersionOut])
def list_models(db: Session = Depends(get_db)) -> list[ModelVersion]:
    """Global model versions, newest first."""
    return list(
        db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()))
    )


@router.get("/{model_id}", response_model=ModelVersionOut)
def get_model(model_id: int, db: Session = Depends(get_db)) -> ModelVersion:
    row = db.get(ModelVersion, model_id)
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, detail=f"model version {model_id} not found"
        )
    return row


__all__ = ["router"]
