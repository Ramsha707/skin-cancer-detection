"""Extract MedSigLIP vision-tower embeddings for every split and agent partition.

This is the expensive step: 8,659 images at ~1.4 img/s on the OpenVINO iGPU is
roughly 1.7 hours, and roughly 30 hours on CPU-only torch. Everything downstream
trains on the output, so it runs once.

Runs under the OpenVINO venv, which does not have pydantic-settings and so cannot
import `app.services.dataset`. The split itself comes from `data/splits.json`,
written by `scripts/build_splits.py` under the backend venv, so there is exactly
one implementation of the patient-level split and the caches cannot drift from it.

Design choices that matter:

* **Per-partition train caches.** Each hospital agent gets its own file, so the
  aggregator never holds one table it could slice into any site's data. That keeps
  "no raw images at the server" structural rather than a promise.
* **Shared val/test.** Every agent is scored against the same held-out set, which
  is what makes FedAvg's global metric meaningful.
* **Resume-aware.** Checkpoints every `CHECKPOINT_EVERY` images, so an
  interrupted 1.7-hour run continues instead of restarting.

Usage:
    .\\.venv-ov\\Scripts\\python.exe scripts/extract_embeddings.py --split train
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.models import EmbeddingCache  # noqa: E402

CACHE_ROOT = REPO_ROOT / "data" / "embeddings"
SPLIT_PATH = REPO_ROOT / "data" / "splits.json"
CHECKPOINT_EVERY = 100


def cache_path(split: str, agent: str | None) -> Path:
    if agent is None:
        return CACHE_ROOT / f"{split}.npy"
    return CACHE_ROOT / f"{split}_{agent}.npy"


def load_split(split: str) -> dict:
    if not SPLIT_PATH.is_file():
        raise SystemExit(
            f"{SPLIT_PATH} not found. Run scripts/build_splits.py with the backend "
            "venv first so the caches match the backend's patient-level split."
        )
    with SPLIT_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _partial_paths(out: Path) -> tuple[Path, Path]:
    return (
        out.with_name(out.stem + "_partial.npy"),
        out.with_name(out.stem + "_partial.json"),
    )


def extract(
    paths: list[str],
    labels: list[int],
    image_ids: list[str],
    out: EmbeddingCache,
    *,
    batch: int,
) -> None:
    if out.exists() and out.size == len(paths):
        print(f"[skip] {out.path.name} already complete ({out.size} images)")
        return

    from medsiglip_ov import MedSiglipOpenVINO

    clf = MedSiglipOpenVINO()
    print(f"[device] OpenVINO on {clf.device}")

    out.path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[np.ndarray] = []
    done_ids: list[str] = []
    done_labels: list[int] = []
    start = 0

    partial_emb, partial_meta = _partial_paths(out.path)
    if partial_emb.is_file() and partial_meta.is_file():
        chunks.append(np.load(partial_emb))
        with partial_meta.open(encoding="utf-8") as fh:
            meta = json.load(fh)
        done_ids = meta["image_ids"]
        done_labels = meta["labels"]
        start = len(done_ids)
        print(f"[resume] {start} embeddings already extracted")

    t0 = time.perf_counter()
    for i in range(start, len(paths), batch):
        chunk = paths[i : i + batch]
        emb = np.asarray(clf.embed_images(chunk), dtype=np.float32)
        chunks.append(emb)
        done_labels.extend(labels[i : i + len(chunk)])
        done_ids.extend(image_ids[i : i + len(chunk)])

        processed = i + len(chunk) - start
        elapsed = time.perf_counter() - t0
        rate = processed / elapsed if elapsed else 0.0
        remaining = (len(paths) - i - len(chunk)) / rate if rate else 0.0
        print(
            f"  {i + len(chunk)}/{len(paths)}  {rate:.2f} img/s  eta {remaining / 60:.1f} min",
            flush=True,
        )

        if (i // batch) % max(1, CHECKPOINT_EVERY // batch) == 0:
            np.save(partial_emb, np.vstack(chunks))
            with partial_meta.open("w", encoding="utf-8") as fh:
                json.dump({"labels": done_labels, "image_ids": done_ids}, fh)

    embeddings = np.vstack(chunks)
    out.save(
        embeddings,
        np.asarray(done_labels),
        done_ids,
        extracted_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    for leftover in (partial_emb, partial_meta):
        if leftover.is_file():
            leftover.unlink()

    print(f"[done] {out.path.name}: {embeddings.shape}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="debug: cap images per split")
    args = p.parse_args()

    data = load_split(args.split)

    def run(name: str, block: dict) -> None:
        paths = block["paths"]
        labels = block["labels"]
        ids = block["image_ids"]
        if args.limit:
            # Stride sampling rather than a head slice, so the class mix survives.
            stride = max(1, len(paths) // args.limit)
            keep = list(range(0, len(paths), stride))[: args.limit]
            paths = [paths[i] for i in keep]
            labels = [labels[i] for i in keep]
            ids = [ids[i] for i in keep]
        mix = ", ".join(
            f"{c}={labels.count(i)}" for i, c in enumerate(data.get("class_names", []))
        )
        print(f"\n=== {name}: {len(paths)} images {mix} ===")
        extract(paths, labels, ids, EmbeddingCache(cache_path(args.split, name)), batch=args.batch)

    if args.split == "train":
        for slug in sorted(data["agents"]):
            run(slug, data["agents"][slug])
    else:
        run(None, data[args.split])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())