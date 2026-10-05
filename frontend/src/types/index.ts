/**
 * Canonical domain types.
 *
 * These mirror the Pydantic schemas in `backend/app/schemas/` one-to-one so the
 * frontend never has to guess the wire format. Keep both sides in sync.
 */

/* ------------------------------------------------------------------ */
/* Cancer classes                                                      */
/* ------------------------------------------------------------------ */

/**
 * Four-class taxonomy over HAM10000.
 *
 * HAM10000 ships **no** squamous cell carcinoma images. Its seven `dx` values are
 * nv, mel, bkl, bcc, akiec, vasc, df - it contains actinic keratosis (`akiec`,
 * 327 images), a UV-damage precursor that is clinically distinct from SCC. It is
 * labelled as its own class rather than being relabelled `scc`, because doing so
 * would fabricate ground truth on those images.
 */
export const CANCER_CLASSES = [
  "melanoma",
  "basal_cell_carcinoma",
  "actinic_keratosis",
  "benign_lesion",
] as const;

export type CancerClass = (typeof CANCER_CLASSES)[number];

export interface CancerClassInfo {
  key: CancerClass;
  label: string;
  shortLabel: string;
  /** HAM10000 dx string this class is derived from. */
  hamDx: string;
  malignant: boolean;
  /** Tailwind-ish hex used for charts and legends. */
  color: string;
  description: string;
}

export const CANCER_CLASS_INFO: Record<CancerClass, CancerClassInfo> = {
  melanoma: {
    key: "melanoma",
    label: "Melanoma",
    shortLabel: "MEL",
    hamDx: "melanoma",
    malignant: true,
    color: "#ef4444",
    description:
      "Malignant melanocytic lesion. Highest clinical urgency of the four classes and the main driver of false-negative cost.",
  },
  basal_cell_carcinoma: {
    key: "basal_cell_carcinoma",
    label: "Basal Cell Carcinoma",
    shortLabel: "BCC",
    hamDx: "bcc",
    malignant: true,
    color: "#f59e0b",
    description:
      "Most common malignant epidermal tumour. Slow-growing but locally invasive; must not be missed.",
  },
  actinic_keratosis: {
    key: "actinic_keratosis",
    label: "Actinic Keratosis",
    shortLabel: "AKIEC",
    hamDx: "akiec",
    malignant: true,
    color: "#8b5cf6",
    description:
      "UV-induced keratinocyte intraepithelial neoplasia and the precursor to SCC. The rarest class here (327 images), so it is the hardest to learn and the main driver of low macro-F1.",
  },
  benign_lesion: {
    key: "benign_lesion",
    label: "Benign Lesion",
    shortLabel: "BENIGN",
    hamDx: "nv, bkl, df",
    malignant: false,
    color: "#22d3ee",
    description:
      "Melanocytic nevi, benign keratosis-like lesions and dermatofibromas. About 80% of the four-class set, which makes this the majority / hard-negative class.",
  },
};

export const MALIGNANT_CLASSES: CancerClass[] = CANCER_CLASSES.filter(
  (c) => CANCER_CLASS_INFO[c].malignant,
);

/* ------------------------------------------------------------------ */
/* Agents                                                              */
/* ------------------------------------------------------------------ */

export type AgentStatus = "idle" | "training" | "paused" | "synced" | "offline" | "error";
export type PrivacyStatus = "protected" | "compromised";

export interface HospitalAgent {
  id: number;
  name: string;
  agent_id: string;
  location: string | null;
  status: AgentStatus;
  dataset_size: number;
  model_version: string;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  loss: number | null;
  privacy_status: PrivacyStatus;
  last_sync: string | null;
  rounds_participated: number;
  class_distribution: Record<string, number>;
  local_epochs: number;
  device: string | null;
}

export interface ClassDistributionItem {
  dx: CancerClass;
  label: string;
  count: number;
  fraction: number;
}

/* ------------------------------------------------------------------ */
/* Federated training                                                  */
/* ------------------------------------------------------------------ */

export type RoundStatus = "pending" | "running" | "aggregating" | "completed" | "failed";
export type TrainingStatus = "idle" | "running" | "paused" | "completed" | "error";

export interface TrainingRound {
  id: number;
  round_number: number;
  status: RoundStatus;
  started_at: string | null;
  completed_at: string | null;
  participating_agents: number;
  global_model_version: string | null;
  global_accuracy: number | null;
  global_loss: number | null;
  global_f1: number | null;
  global_recall: number | null;
  duration_seconds: number | null;
  samples_processed: number | null;
}

export interface AgentRoundState {
  agent_id: string;
  agent_name: string;
  status: AgentStatus;
  samples: number;
  local_loss: number | null;
  local_accuracy: number | null;
  local_recall: number | null;
  local_f1: number | null;
  weight_delta_norm: number | null;
  update_size_kb: number | null;
  phase: string | null;
}

/* ------------------------------------------------------------------ */
/* Models                                                              */
/* ------------------------------------------------------------------ */

export type StrategyKind = "frozen" | "selective" | "full";
export type ModelStatus = "candidate" | "active" | "archived" | "training";

export interface ModelVersion {
  id: number;
  version: string;
  strategy: StrategyKind | null;
  training_round: number | null;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  specificity: number | null;
  f1: number | null;
  auc: number | null;
  trainable_params: number | null;
  total_params: number | null;
  agents: number | null;
  status: ModelStatus;
  is_recommended: boolean;
  created_at: string;
  notes: string | null;
}

/* ------------------------------------------------------------------ */
/* Experiments                                                         */
/* ------------------------------------------------------------------ */

export type ExperimentStatus = "awaiting" | "running" | "completed" | "failed";

export interface Experiment {
  id: number;
  name: string;
  strategy: StrategyKind | null;
  learning_rate: number | null;
  epochs: number | null;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  specificity: number | null;
  f1: number | null;
  auc: number | null;
  trainable_layers: string | null;
  trainable_params: number | null;
  total_params: number | null;
  train_seconds: number | null;
  status: ExperimentStatus;
  is_real_result: boolean;
  notes: string | null;
  created_at: string;
}

/* ------------------------------------------------------------------ */
/* Detection                                                           */
/* ------------------------------------------------------------------ */

export interface ClassProbability {
  dx: CancerClass;
  label: string;
  probability: number;
  malignant: boolean;
}

export interface DetectionResult {
  id: number;
  image_name: string;
  predicted_class: CancerClass;
  predicted_label: string;
  confidence: number;
  is_malignant: boolean;
  model_version: string;
  inference_time: number | null;
  strategy: string | null;
  is_real_inference: boolean;
  modality: string | null;
  created_at: string;
  probabilities: ClassProbability[];
  note: string | null;
}

export interface DetectionStatus {
  ready: boolean;
  model_version: string;
  is_real_inference: boolean;
  caches_ready: Record<string, boolean>;
  instructions: string | null;
}

export interface CancerHeadInfo {
  type: string;
  embedding_dim: number;
  trainable_params: number;
}

export interface DetectionClasses {
  classes: CancerClassInfo[];
  head: CancerHeadInfo;
  caches_ready: Record<string, boolean>;
}

/* ------------------------------------------------------------------ */
/* Audit                                                               */
/* ------------------------------------------------------------------ */

export type AuditEventType =
  | "local_training_started"
  | "local_training_completed"
  | "model_update_generated"
  | "model_update_transferred"
  | "fedavg_started"
  | "fedavg_completed"
  | "global_model_generated"
  | "model_broadcast"
  | "agent_synchronized"
  | "detection_performed"
  | "explainability_generated"
  | "round_completed";

export interface AuditLog {
  id: number;
  timestamp: string;
  agent_id: string | null;
  event_type: AuditEventType;
  message: string;
  status: "success" | "pending" | "failed";
  round_number: number | null;
  details: Record<string, unknown> | null;
}

/* ------------------------------------------------------------------ */
/* Pretrained model introspection (Week 5)                             */
/* ------------------------------------------------------------------ */

export interface PretrainedModelInfo {
  model_id: string;
  architecture: string;
  model_type: string;
  vision_tower: {
    layers: number;
    hidden_size: number;
    intermediate_size: number;
    attention_heads: number;
    patch_size: number;
    image_size: number;
    num_patches: number;
    params: number;
    trainable_when_frozen: number;
  };
  text_tower: {
    layers: number;
    hidden_size: number;
    intermediate_size: number;
    attention_heads: number;
    vocab_size: number;
    max_position_embeddings: number;
    params: number;
    used_for_classifier: boolean;
  };
  projection_size: number;
  embedding_dim: number;
  total_params: number;
  trainable_params_frozen_strategy: number;
  trainable_params_selective_strategy: number;
  trainable_params_full_strategy: number;
  frozen_params_frozen_strategy: number;
  preprocessing: {
    input_size: [number, number];
    resample: string;
    rescale_factor: number;
    image_mean: number[];
    image_std: number[];
  };
  original_training_domain: string;
  original_task: string;
  domain_similarity_to_dermoscopy: string;
  suitable_as_backbone: boolean;
  backbone_viability_notes: string[];
  head_initialisation: string;
  text_prompts: { dx: CancerClass; prompt: string }[];
  loaded_locally: boolean;
  device: string;
  resolved_path: string | null;
  cache_status: string;
}

/* ------------------------------------------------------------------ */
/* Dashboard aggregate                                                 */
/* ------------------------------------------------------------------ */

export interface PrivacySummary {
  raw_images_shared: number;
  model_updates_shared: boolean;
  total_update_size_kb: number;
  transmissions_logged: number;
  agents_protected: number;
  total_agents: number;
  mechanism: string;
  guarantee: string;
}

export interface DashboardSummary {
  total_agents: number;
  active_agents: number;
  current_round: number;
  global_model_version: string | null;
  global_accuracy: number | null;
  global_recall: number | null;
  global_f1: number | null;
  global_auc: number | null;
  cancer_classes: number;
  total_samples: number;
  total_model_versions: number;
  total_rounds_completed: number;
  privacy: PrivacySummary;
  dataset: DatasetInfo;
  demo_mode: boolean;
  real_model_loaded: boolean;
}

export interface DatasetInfo {
  name: string;
  loaded: boolean;
  total_images: number;
  total_patients: number;
  class_counts: Record<string, number>;
  imbalance_ratio: number | null;
  patient_level_split: boolean;
  splits: Record<string, number>;
  source_path: string | null;
  message: string;
}

/* ------------------------------------------------------------------ */
/* System health                                                       */
/* ------------------------------------------------------------------ */

export interface SystemStatus {
  api: string;
  database: string;
  pretrained_model: string;
  dataset: string;
  demo_mode: boolean;
  version: string;
  server_time: string;
}

/* ------------------------------------------------------------------ */
/* Week-by-week roadmap                                                */
/* ------------------------------------------------------------------ */

export interface WeekMilestone {
  week: number;
  title: string;
  summary: string;
  status: "complete" | "in_progress" | "planned";
  route?: string;
}

export const ROADMAP: WeekMilestone[] = [
  { week: 1, title: "UI/UX + Project Foundation", summary: "Design system, routing, landing page", status: "complete", route: "/" },
  { week: 2, title: "Backend + Database", summary: "FastAPI, SQLAlchemy, SQLite schema", status: "complete" },
  { week: 3, title: "Hospital Agents", summary: "Four isolated hospital agents with local data", status: "complete", route: "/agents" },
  { week: 4, title: "Cancer Detection Interface", summary: "Upload, preprocess, predict, explain", status: "planned", route: "/detection" },
  { week: 5, title: "Pre-trained Model Integration", summary: "MedSigLIP-448 analysis + cancer adaptation", status: "planned", route: "/models/pretrained" },
  { week: 6, title: "Fine-Tuning + Experiments", summary: "Frozen vs selective vs full comparison", status: "planned", route: "/experiments" },
  { week: 7, title: "Federated Learning Engine", summary: "FedAvg local train / aggregate / broadcast", status: "planned", route: "/federated-training" },
  { week: 8, title: "Federated Dashboard", summary: "Interactive topology, controls, live rounds", status: "planned", route: "/federated-training" },
  { week: 9, title: "Evaluation", summary: "Full metric suite + ROC + confusion matrix", status: "planned", route: "/performance" },
  { week: 10, title: "Explainability + Privacy", summary: "Grad-CAM heatmaps and privacy centre", status: "planned", route: "/explainability" },
  { week: 11, title: "Full-System Integration", summary: "WebSocket event bus across all modules", status: "planned" },
  { week: 12, title: "Testing + Final Demo", summary: "Run Judge Demo and presentation polish", status: "planned" },
];

/**
 * Highest roadmap week actually implemented in this build.
 *
 * This must be bumped as weeks land. It is 3 right now: week 3's agent roster,
 * class-mix view and per-site controls read live from the backend, while the
 * routes for weeks 4-8 exist as navigation shells and must stay `planned`.
 */
export const CURRENT_IMPLEMENTED_WEEK = 3;

export const RESEARCH_DISCLAIMER =
  "RESEARCH PROTOTYPE — NOT A MEDICAL DIAGNOSIS. This system is intended for academic and research demonstration only. It has not been clinically validated and must not be used to make medical decisions.";

export const DEMO_DATA_LABEL = "ILLUSTRATIVE DEMO DATA";