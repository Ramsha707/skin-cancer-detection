"""Write the canonical split to disk so other interpreters can consume it.

Split logic lives in `app.services.dataset` and must not be duplicated: the
embedding caches are extracted per-partition, so if this script and the backend
ever disagreed about which images belong to which site, the federated results
would silently describe a partition that the rest of the system never uses.

This script runs under the *backend* venv. `scripts/extract_embeddings.py` runs
under the OpenVINO venv, which has no pydantic-settings and therefore cannot
import the dataset service. Rather than fork the splitter, that script reads this
JSON.

Usage:
    .\\backend-venv\\Scripts\\python.exe scripts/build_splits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.dataset import (  # noqa: E402
    CANCER_CLASSES,
    load_manifest,
    partition_for_agents,
    patient_level_split,
)

AGENT_SLUGS = ["agent_a", "agent_b", "agent_c", "agent_d"]
SPLIT_PATH = REPO_ROOT / "data" / "splits.json"


def main() -> int:
    manifest = load_manifest()
    if not manifest.loaded:
        print(manifest.message)
        return 2

    train, val, test = patient_level_split(manifest)
    partitions = partition_for_agents(train, manifest, len(AGENT_SLUGS))

    def describe(indices: list[int]) -> dict:
        recs = manifest.subset(indices)
        counts: dict[str, int] = {c: 0 for c in CANCER_CLASSES}
        for rec in recs:
            counts[CANCER_CLASSES[rec.class_index]] += 1
        return {
            "count": len(recs),
            "class_counts": counts,
            "image_ids": [r.image_id for r in recs],
            "paths": [str(r.path) for r in recs],
            "labels": [r.class_index for r in recs],
        }

    payload = {
        "dataset": manifest.name,
        "patients": manifest.patients,
        "val": describe(val),
        "test": describe(test),
        "agents": {
            slug: describe(part) for slug, part in zip(AGENT_SLUGS, partitions, strict=True)
        },
        "train_size": len(train),
    }

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SPLIT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    print(f"wrote {SPLIT_PATH}")
    print(f"  train {len(train)} split across {len(AGENT_SLUGS)} agents")
    print(f"  val {payload['val']['count']}  test {payload['test']['count']}")
    for slug, part in payload["agents"].items():
        mix = ", ".join(f"{k}={v}" for k, v in part["class_counts"].items() if v)
        print(f"  {slug}: {part['count']}  [{mix}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())