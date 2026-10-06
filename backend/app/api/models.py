"""Week 5: pre-trained model information endpoints.

Serves the MedSigLIP-448 introspection that the `/models/pretrained` page
renders. All payload assembly lives in `app.services.pretrained` so the same
numbers can be asserted directly in tests without booting FastAPI.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas import PretrainedModelInfo
from app.services.pretrained import build_pretrained_info

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/pretrained", response_model=PretrainedModelInfo)
def pretrained_model_info() -> dict:
    """Architecture, parameter budget, preprocessing and domain assessment.

    Parameter counts are summed from the local safetensors header when the
    weights are cached; `cache_status` tells the client which case it got.
    """
    return build_pretrained_info()


@router.get("", response_model=list)
def list_models() -> list:
    """Placeholder — trained model versions arrive with the registry week."""
    return []


__all__ = ["router"]
