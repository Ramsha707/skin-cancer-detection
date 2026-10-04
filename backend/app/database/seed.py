"""First-run seeding.

Creates the four hospital agents and registers the three transfer-learning
experiments. Experiment rows are created with `is_real_result=False` and
`status='awaiting'`: the Experiments page renders those as *Awaiting Experiment*
rather than displaying invented numbers.

Partitioning is derived from the real dataset when HAM10000 is present, so the
per-agent `dataset_size` and `class_distribution` reflect actual data.
"""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Experiment, HospitalAgent
from app.services import dataset as dataset_service

log = logging.getLogger(__name__)

AGENT_DEFINITIONS = [
    {
        "name": "Hospital Agent A",
        "agent_id": "agent-a",
        "location": "Northside Teaching Hospital",
        "device": "cpu",
    },
    {
        "name": "Hospital Agent B",
        "agent_id": "agent-b",
        "location": "Riverside Medical Centre",
        "device": "cpu",
    },
    {
        "name": "Hospital Agent C",
        "agent_id": "agent-c",
        "location": "Eastgate Dermatology Unit",
        "device": "cpu",
    },
    {
        "name": "Hospital Agent D",
        "agent_id": "agent-d",
        "location": "South Valley Clinic",
        "device": "cpu",
    },
]

EXPERIMENT_DEFINITIONS = [
    {
        "name": "A — Frozen backbone feature extractor",
        "strategy": "frozen",
        "learning_rate": 1e-3,
        "epochs": 10,
        "trainable_layers": "classification head only (4-class linear)",
        "notes": (
            "Entire SigLIP vision tower frozen. Only the 4-class head is trained. "
            "Cheapest per round, hardest to overfit, and the strongest baseline "
            "when each hospital holds only a few hundred images."
        ),
    },
    {
        "name": "B — Selective fine-tuning (upper tower)",
        "strategy": "selective",
        "learning_rate": 1e-5,
        "epochs": 10,
        "trainable_layers": "vision tower layers 22-27 + post-layernorm + head",
        "notes": (
            "Top 6 of 27 encoder blocks plus the head, at a 100x smaller learning "
            "rate. Chosen because the early blocks encode generic edge and texture "
            "operators that dermoscopy still depends on."
        ),
    },
    {
        "name": "C — Full fine-tuning",
        "strategy": "full",
        "learning_rate": 1e-5,
        "epochs": 10,
        "trainable_layers": "all vision tower layers + head",
        "notes": (
            "Only justified if the local partition is large. At HAM10000 per-site "
            "sizes this is expected to overfit and is the most expensive option "
            "for FedAvg, so it is gated behind a feasibility check."
        ),
    },
]


def seed_agents(db: Session) -> None:
    existing = db.scalar(select(func.count()).select_from(HospitalAgent)) or 0
    if existing:
        return

    manifest = dataset_service.load_manifest()

    if manifest.loaded:
        train_idx, _, _ = dataset_service.patient_level_split(manifest)
        partitions = dataset_service.partition_for_agents(
            train_idx, manifest, settings.num_agents, non_iid=True
        )
        sizes = [len(p) for p in partitions]
        histograms = [dataset_service.class_histogram(manifest, p) for p in partitions]
        log.info(
            "seeding %d agents from %s (%d train images, sizes %s)",
            settings.num_agents,
            manifest.name,
            len(train_idx),
            sizes,
        )
    else:
        # No dataset: sizes stay 0 and are filled in by the federated engine's
        # simulation mode. Nothing here is presented as real.
        sizes = [0] * settings.num_agents
        histograms = [{} for _ in range(settings.num_agents)]
        log.warning("no dataset found - agents seeded with empty partitions, demo mode active")

    for i, definition in enumerate(AGENT_DEFINITIONS[: settings.num_agents]):
        db.add(
            HospitalAgent(
                **definition,
                dataset_size=sizes[i],
                class_distribution=histograms[i],
                status="idle",
                privacy_status="protected",
                model_version="unassigned",
            )
        )
    db.flush()


def seed_experiments(db: Session) -> None:
    existing = db.scalar(select(func.count()).select_from(Experiment)) or 0
    if existing:
        return

    for definition in EXPERIMENT_DEFINITIONS:
        db.add(
            Experiment(
                model=settings.pretrained_model_id,
                status="awaiting",
                is_real_result=False,
                total_params=None,
                **definition,
            )
        )
    db.flush()


def seed_all(db: Session) -> None:
    seed_agents(db)
    seed_experiments(db)
    db.commit()
