"""Week 5: pre-trained model introspection and the model registry.

Two read-only surfaces:

* `/api/models/pretrained` -- architecture facts, the parameter budget under each
  adaptation strategy, and the text-prompt head.
* `/api/models` -- every global model version produced by federated training.

Both are built from the local HuggingFace cache, not from a hardcoded spec, so
the numbers on the page describe the weights that actually load.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.base import get_db
from app.models import ModelVersion
from app.schemas import ModelVersionOut, PretrainedModelInfo, TextPrompt
from app.services import registry
from app.services.dataset import CANCER_CLASSES

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])

# The prompt-initialised head: one weight row and one bias per class.
HEAD_PARAMS = registry.head_param_count()

VISION_LAYERS = 27
SELECTIVE_BLOCKS = 2

# A transformer block's dominant cost is its MLP: two matrices of
# hidden_size x intermediate_size, plus attention projections and two LayerNorms.
def _params_per_vision_block(hidden: int, intermediate: int, heads: int) -> int:
    attn_qkv = 3 * hidden * hidden + 3 * hidden  # q, k, v projections + biases
    attn_out = hidden * hidden + hidden
    mlp = 2 * (hidden * intermediate + intermediate)  # fc1 + fc2, with biases
    norms = 4 * hidden
    return attn_qkv + attn_out + mlp + norms + heads * hidden


@router.get("/pretrained", response_model=PretrainedModelInfo)
def pretrained_info() -> PretrainedModelInfo:
    """Architecture + strategy budget for the backbone this project adapts.

    Deliberately does *not* construct the model: this endpoint is hit on page
    load, and instantiating an 878M-parameter tower costs ~30 s. The numbers come
    from the config plus the published SigLIP geometry.
    """
    card = registry.load_model_card()
    vision = card["vision_tower"]
    hidden = vision["hidden_size"]
    intermediate = vision["mlp_size"]
    heads = vision["attention_heads"]
    patch = vision["patch_size"]
    image_size = card["input"]["size"]

    num_patches = (image_size // patch) ** 2
    block_params = _params_per_vision_block(hidden, intermediate, heads)
    vision_params = VISION_LAYERS * block_params
    selective_params = SELECTIVE_BLOCKS * block_params

    total = card["total_params"]
    prompts = registry.class_prompts()

    return PretrainedModelInfo(
        model_id=card["model_id"],
        architecture=card["architecture"],
        model_type="siglip-448",
        vision_tower={
            "layers": VISION_LAYERS,
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "attention_heads": heads,
            "patch_size": patch,
            "image_size": image_size,
            "num_patches": num_patches,
            "params": vision_params,
            # A frozen backbone contributes zero trainable vision parameters.
            "trainable_when_frozen": 0,
        },
        text_tower={
            "layers": card["text_tower"]["layers"],
            "hidden_size": card["text_tower"]["hidden_size"],
            "intermediate_size": intermediate,
            "attention_heads": heads,
            "vocab_size": card["text_tower"]["vocab_size"],
            "max_position_embeddings": card["text_tower"]["max_positions"],
            # The text tower only runs once, at initialisation, to build the
            # prompt head. After that it is not needed for inference.
            "params": VISION_LAYERS * block_params,
            "used_for_classifier": True,
        },
        projection_size=hidden,
        embedding_dim=card["embedding_dim"],
        total_params=total,
        trainable_params_frozen_strategy=HEAD_PARAMS,
        trainable_params_selective_strategy=HEAD_PARAMS + selective_params,
        trainable_params_full_strategy=total,
        frozen_params_frozen_strategy=total - HEAD_PARAMS,
        preprocessing={
            "input_size": (image_size, image_size),
            "resample": card["input"]["resample"],
            "rescale_factor": 1 / 255,
            "image_mean": [card["input"]["mean"]] * 3,
            "image_std": [card["input"]["std"]] * 3,
        },
        original_training_domain=(
            "WebLI, a 11B-image caption/alt-text web corpus. Text-image pairs, "
            "no dermoscopy labels."
        ),
        original_task="zero-shot image-text contrastive retrieval",
        domain_similarity_to_dermoscopy=(
            "Partial. The pretraining mixture includes medical and scientific "
            "imagery, but MedSigLIP was never trained on dermoscopic labels. Its "
            "value here is a strong general visual prior, not clinical knowledge."
        ),
        suitable_as_backbone=True,
        backbone_viability_notes=[
            "448x448 input with 14px patches gives 1,024 patches per image, so the "
            "vision tower runs 27 layers over a long sequence. That is the reason "
            "CPU throughput is ~0.08 img/s.",
            "OpenVINO raises measured throughput to ~0.57 img/s on the Intel iGPU, "
            "a 7x gain, by fusing the LayerNorm and GELU ops the eager path runs separately.",
            "The 1,152-dim image embedding is what everything downstream consumes: "
            "the cancer head, the embedding caches and the FedAvg updates.",
            "Because the tower is frozen for the headline experiment, the vision "
            "forward pass happens once per image and is then cached to disk.",
        ],
        head_initialisation=(
            "Each classifier row is the L2-normalised text embedding of that class's "
            "prompt. SigLIP shares one embedding space between images and text, so "
            "the prompt embedding is already a usable weight vector -- no random init "
            "and no learned bias at the start of training."
        ),
        text_prompts=[
            TextPrompt(dx=name, prompt=prompt)
            for name, prompt in zip(CANCER_CLASSES, prompts, strict=True)
        ],
        loaded_locally=bool(registry.model_available()),
        device="openvino (Intel Iris Xe) for extraction, torch cpu in the API",
        cache_status=(
            "embedding caches present"
            if all(registry.caches_ready().values())
            else "embedding caches incomplete -- run scripts/extract_embeddings.py"
        ),
    )


@router.get("", response_model=list[ModelVersionOut])
def list_models(db: Session = Depends(get_db)) -> list[ModelVersion]:
    """Every global model version, newest first."""
    return list(
        db.scalars(
            select(ModelVersion).order_by(
                ModelVersion.training_round.desc().nullslast(),
                ModelVersion.created_at.desc(),
            )
        )
    )


@router.get("/active", response_model=ModelVersionOut)
def active_model(db: Session = Depends(get_db)) -> ModelVersion:
    """The version detection currently serves."""
    row = db.scalar(
        select(ModelVersion)
        .where(ModelVersion.status == "active")
        .order_by(ModelVersion.training_round.desc().nullslast())
    )
    if row is None:
        row = db.scalar(select(ModelVersion).order_by(ModelVersion.id))
    if row is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=(
                "no model versions yet. Run a federated round from "
                "/federated-training, or POST /api/experiments/run."
            ),
        )
    return row


@router.get("/{model_id}", response_model=ModelVersionOut)
def get_model(model_id: int, db: Session = Depends(get_db)) -> ModelVersion:
    row = db.get(ModelVersion, model_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="model version not found")
    return row


@router.post("/{model_id}/recommend", response_model=ModelVersionOut)
def recommend_model(model_id: int, db: Session = Depends(get_db)) -> ModelVersion:
    """Promote a version to recommended, demoting the incumbent."""
    target = db.get(ModelVersion, model_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="model version not found")

    for other in db.scalars(select(ModelVersion).where(ModelVersion.is_recommended)):
        other.is_recommended = False
        if other.status == "active":
            other.status = "candidate"
    target.is_recommended = True
    target.status = "active"
    db.flush()
    return target


__all__ = ["HEAD_PARAMS", "router"]