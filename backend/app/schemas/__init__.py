"""Pydantic v2 schemas.

Every `model_config` sets `from_attributes = True` so these serialise straight
from SQLAlchemy rows. Shapes are kept in lockstep with
`frontend/src/types/index.ts`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------- literals --

AgentStatus = Literal["idle", "training", "paused", "synced", "offline", "error"]
PrivacyStatus = Literal["protected", "compromised"]
RoundStatus = Literal["pending", "running", "aggregating", "completed", "failed"]
TrainingStatus = Literal["idle", "running", "paused", "completed", "error"]
StrategyKind = Literal["frozen", "selective", "full"]
ModelStatus = Literal["candidate", "active", "archived", "training"]
# `skipped` is a real outcome, not an error: an arm can be refused by the host's
# compute budget (see `ComputeBudgetOut`) and must be reported as skipped rather
# than silently dropped or reported as completed.
ExperimentStatus = Literal["awaiting", "running", "completed", "failed", "skipped"]
# Mirrors `app.services.dataset.CANCER_CLASSES`. HAM10000 has no squamous cell
# carcinoma images, so actinic keratosis is reported under its own name rather
# than being relabelled as SCC.
CancerClass = Literal["melanoma", "basal_cell_carcinoma", "actinic_keratosis", "benign_lesion"]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------- agent --


class HospitalAgentOut(ORMModel):
    id: int
    name: str
    agent_id: str
    location: str | None = None
    status: AgentStatus = "idle"
    dataset_size: int = 0
    model_version: str = "unassigned"
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None
    loss: float | None = None
    privacy_status: PrivacyStatus = "protected"
    last_sync: datetime | None = None
    rounds_participated: int = 0
    class_distribution: dict[str, int] = Field(default_factory=dict)
    local_epochs: int = 0
    device: str | None = None


class AgentRoundState(BaseModel):
    """Live per-agent state during a federated round."""

    agent_id: str
    agent_name: str
    status: AgentStatus = "idle"
    samples: int = 0
    local_loss: float | None = None
    local_accuracy: float | None = None
    local_recall: float | None = None
    local_f1: float | None = None
    weight_delta_norm: float | None = None
    update_size_kb: float | None = None
    phase: str | None = None


# ------------------------------------------------------------------ rounds --


class TrainingRoundOut(ORMModel):
    id: int
    round_number: int
    status: RoundStatus = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    participating_agents: int = 0
    global_model_version: str | None = None
    global_accuracy: float | None = None
    global_precision: float | None = None
    global_recall: float | None = None
    global_f1: float | None = None
    global_auc: float | None = None
    global_loss: float | None = None
    duration_seconds: float | None = None
    samples_processed: int | None = None


class FederatedState(BaseModel):
    status: TrainingStatus = "idle"
    current_round: int = 0
    global_model_version: str | None = None
    aggregation_status: str = "idle"
    training_status: TrainingStatus = "idle"
    agents: list[AgentRoundState] = Field(default_factory=list)


class RoundDetail(BaseModel):
    round: TrainingRoundOut
    agents: list[AgentRoundState] = Field(default_factory=list)


class StartResponse(BaseModel):
    ok: bool = True
    round: int


class RunRoundsRequest(BaseModel):
    rounds: int = Field(default=5, ge=1, le=50)


class RunRoundsResponse(BaseModel):
    ok: bool = True
    rounds: int


class OkResponse(BaseModel):
    ok: bool = True


# ------------------------------------------------------------------ models --


class ModelVersionOut(ORMModel):
    id: int
    version: str
    strategy: StrategyKind | None = None
    training_round: int | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    specificity: float | None = None
    f1: float | None = None
    auc: float | None = None
    loss: float | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    agents: int | None = None
    status: ModelStatus = "candidate"
    is_recommended: bool = False
    created_at: datetime
    notes: str | None = None


class VisionTowerInfo(BaseModel):
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    patch_size: int
    image_size: int
    num_patches: int
    params: int
    trainable_when_frozen: int


class TextTowerInfo(BaseModel):
    layers: int
    hidden_size: int
    intermediate_size: int
    attention_heads: int
    vocab_size: int
    max_position_embeddings: int
    params: int
    used_for_classifier: bool


class PreprocessingInfo(BaseModel):
    input_size: tuple[int, int]
    resample: str
    rescale_factor: float
    image_mean: list[float]
    image_std: list[float]


class TextPrompt(BaseModel):
    dx: CancerClass
    prompt: str


class PretrainedModelInfo(BaseModel):
    """Everything the Week 5 'Model Information' page renders."""

    model_id: str
    architecture: str
    model_type: str
    vision_tower: VisionTowerInfo
    text_tower: TextTowerInfo
    projection_size: int
    embedding_dim: int
    total_params: int
    trainable_params_frozen_strategy: int
    trainable_params_selective_strategy: int
    trainable_params_full_strategy: int
    frozen_params_frozen_strategy: int
    preprocessing: PreprocessingInfo
    original_training_domain: str
    original_task: str
    domain_similarity_to_dermoscopy: str
    suitable_as_backbone: bool
    backbone_viability_notes: list[str]
    head_initialisation: str
    text_prompts: list[TextPrompt]
    loaded_locally: bool
    device: str
    resolved_path: str | None = None
    cache_status: str


# ------------------------------------------------------------- experiments --


class ExperimentOut(ORMModel):
    id: int
    name: str
    model: str
    strategy: StrategyKind | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    specificity: float | None = None
    f1: float | None = None
    auc: float | None = None
    trainable_layers: str | None = None
    trainable_params: int | None = None
    total_params: int | None = None
    train_seconds: float | None = None
    status: ExperimentStatus = "awaiting"
    is_real_result: bool = False
    notes: str | None = None
    created_at: datetime


class RunExperimentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=160)


# ---------------------------------------------------------------- detection --


class ClassProbability(BaseModel):
    dx: str
    label: str
    probability: float
    malignant: bool


class DetectionResultOut(ORMModel):
    id: int
    image_name: str
    predicted_class: str
    predicted_label: str = ""
    confidence: float
    is_malignant: bool = False
    model_version: str
    inference_time: float | None = None
    strategy: str | None = None
    is_real_inference: bool = False
    modality: str | None = None
    created_at: datetime
    probabilities: list[ClassProbability] = Field(default_factory=list)
    note: str | None = None


# ---------------------------------------------------------------- detection --


class DetectionStatusOut(BaseModel):
    """What the detection endpoint can actually do right now.

    The frontend reads `ready` to choose between a working upload panel and an
    explicit run-the-script instruction, so an un-extracted checkout never
    presents an upload box that would fail.
    """

    ready: bool
    model_version: str
    is_real_inference: bool
    caches_ready: dict[str, bool] = Field(default_factory=dict)
    instructions: str | None = None


class CancerClassOut(BaseModel):
    name: CancerClass
    prompt: str
    malignant: bool


class CancerHeadOut(BaseModel):
    """Shape of the trainable head. Shared by Week 4 and Week 7 pages."""

    type: str = "linear-probe"
    embedding_dim: int
    trainable_params: int


class DetectionClassesOut(BaseModel):
    """The taxonomy, the prompts behind it, and the head that consumes both."""

    classes: list[CancerClassOut]
    head: CancerHeadOut
    caches_ready: dict[str, bool] = Field(default_factory=dict)


# ---------------------------------------------------------------- federated --


class FederatedStatusOut(BaseModel):
    """Whether a round can run, and what crosses the boundary when it does.

    `param_count` / `param_kb` are the whole privacy story in two numbers: the
    aggregator receives this many floats per agent, and never an image.
    """

    ready: bool
    caches_ready: dict[str, bool] = Field(default_factory=dict)
    param_count: int
    param_kb: float
    aggregation: str
    instructions: str | None = None


# --------------------------------------------------------------- experiments --


class ComputeBudgetOut(BaseModel):
    """What this host can actually afford. Explains a skipped `full` arm."""

    cuda_available: bool
    total_ram_gb: float
    supports_full_finetune: bool


class PlannedExperimentOut(BaseModel):
    """An arm that has not run yet, with the reason it has not run."""

    name: str
    strategy: StrategyKind
    status: str
    note: str


class ExperimentComparisonOut(BaseModel):
    """`GET /api/experiments`.

    `status` is `not_run` until a run is persisted, at which point `experiments`
    is populated. `planned` is only present while `not_run`, so the UI can never
    render a planned arm as if it were a measurement.
    """

    status: Literal["not_run", "complete"]
    budget: ComputeBudgetOut
    caches_ready: dict[str, bool] = Field(default_factory=dict)
    experiments: list[ExperimentOut] = Field(default_factory=list)
    planned: list[PlannedExperimentOut] = Field(default_factory=list)
    source: Literal["database", "disk"] | None = None


class ExperimentRunOut(BaseModel):
    """`POST /api/experiments/run`. `status` is `partial` when an arm was skipped."""

    status: Literal["complete", "partial"]
    experiments: list[ExperimentOut]


# -------------------------------------------------------------------- audit --


class AuditLogOut(ORMModel):
    id: int
    timestamp: datetime
    agent_id: str | None = None
    event_type: str
    message: str
    status: str
    round_number: int | None = None
    details: dict[str, Any] | None = None


# --------------------------------------------------------------- dashboard --


class PrivacySummary(BaseModel):
    raw_images_shared: int = 0
    model_updates_shared: bool = True
    total_update_size_kb: float = 0.0
    transmissions_logged: int = 0
    agents_protected: int = 0
    total_agents: int = 0
    mechanism: str = (
        "Only parameter tensors cross the hospital boundary. No image, pixel array "
        "or patient identifier is ever serialised for transmission."
    )
    guarantee: str = (
        "Raw patient images shared: 0. The aggregator calls only fit(), update() and "
        "load() on each agent, all of which exchange tensors, so there is no code path "
        "by which an image reaches the central server."
    )


class DatasetInfo(BaseModel):
    name: str = "HAM10000"
    loaded: bool = False
    total_images: int = 0
    total_patients: int = 0
    class_counts: dict[str, int] = Field(default_factory=dict)
    imbalance_ratio: float | None = None
    patient_level_split: bool = False
    splits: dict[str, int] = Field(default_factory=dict)
    source_path: str | None = None
    message: str = "No dataset configured."


class DashboardSummary(BaseModel):
    total_agents: int = 0
    active_agents: int = 0
    current_round: int = 0
    global_model_version: str | None = None
    global_accuracy: float | None = None
    global_recall: float | None = None
    global_f1: float | None = None
    global_auc: float | None = None
    cancer_classes: int = 4
    total_samples: int = 0
    total_model_versions: int = 0
    total_rounds_completed: int = 0
    privacy: PrivacySummary = Field(default_factory=PrivacySummary)
    dataset: DatasetInfo = Field(default_factory=DatasetInfo)
    demo_mode: bool = True
    real_model_loaded: bool = False


# ------------------------------------------------------------------ system --


class SystemStatus(BaseModel):
    api: str = "online"
    database: str = "connected"
    pretrained_model: str = "not loaded"
    dataset: str = "not configured"
    demo_mode: bool = True
    version: str
    server_time: datetime


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    detail: str
    components: dict[str, str] = Field(default_factory=dict)
