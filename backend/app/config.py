"""Runtime configuration.

Every setting can be overridden with an environment variable of the same name
(prefixed `SKINFL_`) or via a `.env` file in the repository root.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SKINFL_",
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SkinFL Federated Oncology API"
    version: str = "0.8.0"
    debug: bool = True

    # ---- database ----
    database_url: str = f"sqlite:///{REPO_ROOT / 'skinfl.db'}"

    # ---- base model ----
    # Confirmed present in the local HuggingFace cache.
    pretrained_model_id: str = "google/medsiglip-448"
    model_device: str = "auto"  # auto | cpu | cuda

    # ---- dataset ----
    # HAM10000 root. Expected layout:
    #   <root>/images/ISIC_xxxxxxx.jpg
    #   <root>/metadata.csv   (columns: image, dx, dx_type, lesion_id, ...)
    dataset_root: Path | None = None
    dataset_metadata: str = "metadata.csv"

    # ---- federated simulation ----
    num_agents: int = 4
    num_classes: int = 4
    default_rounds: int = 5
    local_epochs: int = 1
    learning_rate: float = 1e-3
    fedavg_learning_rate: float = 1.0

    # ---- detection ----
    max_upload_mb: int = 12
    confidence_threshold: float = 0.0

    # ---- CORS ----
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    @property
    def dataset_images_dir(self) -> Path | None:
        if self.dataset_root is None:
            return None
        return self.dataset_root / "images"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
