"""HAM10000 dataset discovery, patient-level splitting and non-IID partitioning.

Two leakage guarantees are enforced here, because both are easy to get wrong
and impossible to detect after the fact:

1. **Patient-level separation.** HAM10000 contains multiple lesions per patient
   (`lesion_id`). A random image split puts two lesions from the same patient in
   both train and test, which inflates every metric. `patient_level_split()`
   groups by patient before splitting.

2. **No raw-image egress.** `partition_for_agents()` returns *index lists*. The
   aggregator receives counts and class histograms, never image tensors.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.config import settings

log = logging.getLogger(__name__)

CANCER_CLASSES = [
    "melanoma",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "benign_lesion",
]

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# HAM10000 `dx` value -> our four-class taxonomy.
#
# HAM10000 contains **no** squamous cell carcinoma images. Its seven `dx` values
# are nv, mel, bkl, bcc, akiec, vasc, df; it ships actinic keratosis (`akiec`, 327
# images) where a naive pipeline would expect `scc`. Actinic keratosis is a UV
# keratinocyte intraepithelial neoplasia and a precursor to SCC, but it is
# clinically distinct, so it is labelled as its own class here rather than being
# silently relabelled as SCC. That keeps every ground-truth label truthful.
#
# `vasc` (142 vascular lesions) is excluded as neither cancer nor a lesion the
# four-class brief describes.
DX_TO_CLASS: dict[str, str] = {
    "mel": "melanoma",
    "bcc": "basal_cell_carcinoma",
    "akiec": "actinic_keratosis",
    "nv": "benign_lesion",
    "bkl": "benign_lesion",
    "df": "benign_lesion",
}

MALIGNANT_CLASSES = ["melanoma", "basal_cell_carcinoma", "actinic_keratosis"]


@dataclass
class DatasetRecord:
    """One image row. `patient` is the leakage-prevention key."""

    image_id: str
    path: Path
    dx: str
    patient: str
    lesion_id: str
    class_index: int


@dataclass
class DatasetManifest:
    """Everything downstream code needs, computed once at startup."""

    name: str = "HAM10000"
    loaded: bool = False
    root: Path | None = None
    metadata_path: Path | None = None
    records: list[DatasetRecord] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    patients: int = 0
    message: str = "No dataset configured."

    @property
    def total_images(self) -> int:
        return len(self.records)

    @property
    def imbalance_ratio(self) -> float | None:
        counts = list(self.class_counts.values())
        if not counts or min(counts) == 0:
            return None
        return float(max(counts) / min(counts))

    def subset(self, indices: list[int]) -> list[DatasetRecord]:
        return [self.records[i] for i in indices]


def _candidate_roots() -> list[Path]:
    """Places a HAM10000 checkout plausibly lives, in priority order."""
    roots: list[Path] = []
    if settings.dataset_root:
        roots.append(Path(settings.dataset_root))

    repo = Path(__file__).resolve().parents[3]
    # The layout shipped with this project:
    #   data/HAM10000_metadata.csv
    #   data/HAM10000_images_part_1/*.jpg
    #   data/HAM10000_images_part_2/*.jpg
    roots += [
        repo / "data",
        repo / "datasets" / "HAM10000",
        repo / "data" / "ham10000",
        repo / "data" / "HAM10000",
        repo / "HAM10000",
        Path.cwd() / "data",
    ]
    return roots


def _find_metadata(root: Path) -> Path | None:
    for name in (
        settings.dataset_metadata,
        "HAM10000_metadata.csv",
        "metadata.csv",
        "HAM10000_metadata",
    ):
        candidate = root / name
        if candidate.is_file():
            return candidate
    matches = sorted(root.rglob("*metadata*.csv"))
    return matches[0] if matches else None


def _image_dirs(root: Path) -> list[Path]:
    """All directories that may hold image files, longest-name-first."""
    candidates = [root / "images", root / "image", root / "HAM10000_images"]
    candidates += sorted(p for p in root.glob("HAM10000_images*") if p.is_dir())
    return [p for p in candidates if p.is_dir()]


def load_manifest(force: bool = False) -> DatasetManifest:
    """Read `metadata.csv` and resolve each row to an existing image file."""
    if force:
        load_manifest.cache_clear()

    for root in _candidate_roots():
        meta = _find_metadata(root)
        if meta is None:
            continue

        image_dirs = _image_dirs(root)

        records: list[DatasetRecord] = []
        counts = {c: 0 for c in CANCER_CLASSES}
        patients: set[str] = set()

        try:
            with meta.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    dx = (row.get("dx") or "").strip().lower()
                    cls = DX_TO_CLASS.get(dx)
                    if cls is None:
                        continue

                    # This checkout uses `image_id`; the canonical HAM10000
                    # release uses `image`. Accept both.
                    image_id = (row.get("image_id") or row.get("image") or "").strip()
                    if not image_id:
                        continue

                    path = _resolve_image(image_dirs, image_id)
                    if path is None:
                        continue

                    # HAM10000 has no explicit patient column; lesion_id is the
                    # documented grouping key and is what the authors use for
                    # patient-aware evaluation.
                    lesion = (row.get("lesion_id") or image_id).strip()
                    patient = (row.get("patient") or lesion).strip()

                    records.append(
                        DatasetRecord(
                            image_id=image_id,
                            path=path,
                            dx=dx,
                            patient=patient,
                            lesion_id=lesion,
                            class_index=CANCER_CLASSES.index(cls),
                        )
                    )
                    counts[cls] += 1
                    patients.add(patient)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            log.warning("could not parse %s: %s", meta, exc)
            continue

        if records:
            log.info(
                "loaded %d images (%d patients) from %s",
                len(records),
                len(patients),
                root,
            )
            return DatasetManifest(
                loaded=True,
                root=root,
                metadata_path=meta,
                records=records,
                class_counts=counts,
                patients=len(patients),
                message=(
                    f"{len(records)} images across {len(patients)} patients loaded from "
                    f"{meta.name}. Splits are patient-level."
                ),
            )

    return DatasetManifest(
        loaded=False,
        message=(
            "HAM10000 not found. Set SKINFL_DATASET_ROOT to the folder containing "
            "HAM10000_metadata.csv and the HAM10000_images_part_* directories, "
            "or continue in demo mode."
        ),
    )


load_manifest = lru_cache(maxsize=1)(load_manifest)  # type: ignore[assignment]


def _resolve_image(image_dirs: list[Path], image_id: str) -> Path | None:
    """Resolve a metadata `image_id` to a file.

    HAM10000 metadata stores ids without an extension (``ISIC_0024006``) while the
    image directories hold ``ISIC_0024006.jpg``, so match on stem.
    """
    for directory in image_dirs:
        candidate = directory / image_id
        if candidate.is_file():
            return candidate
        stem = Path(image_id).stem
        for suffix in _IMAGE_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate

    # lru_cache needs a hashable key, hence the tuple.
    index = _image_index(tuple(image_dirs))
    return index.get(Path(image_id).stem)


@lru_cache(maxsize=8)
def _image_index(image_dirs: tuple[Path, ...]) -> dict[str, Path]:
    """Map every image stem to its path, so 10k rows resolve in one pass."""
    index: dict[str, Path] = {}
    for directory in image_dirs:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                index.setdefault(path.stem, path)
    return index


# --------------------------------------------------------------------------
# Patient-level splitting
# --------------------------------------------------------------------------


def patient_level_split(
    manifest: DatasetManifest,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split by patient, never by image, while stratifying on class.

    HAM10000 groups images into lesions rather than reporting patient ids, so
    `lesion_id` is the grouping key here; a lesion can contribute several images
    and must never appear in two buckets.

    Stratification happens at the *patient* level: each patient is labelled with
    its dominant class, patients are shuffled within each class, and each class's
    patients are then apportioned across train/val/test. This keeps the ~2/3
    benign majority from swamping the rarer classes without ever splitting a
    patient.
    """
    import random

    if not manifest.records:
        return [], [], []

    by_patient: dict[str, list[int]] = {}
    for idx, rec in enumerate(manifest.records):
        by_patient.setdefault(rec.patient, []).append(idx)

    # Label each patient by its most frequent class, then by class index for
    # deterministic tie-breaking.
    dominant: dict[str, int] = {}
    for patient, idxs in by_patient.items():
        tally: dict[int, int] = {}
        for i in idxs:
            ci = manifest.records[i].class_index
            tally[ci] = tally.get(ci, 0) + 1
        dominant[patient] = min(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    rng = random.Random(seed)
    by_class: dict[int, list[str]] = {}
    for patient in sorted(by_patient):
        by_class.setdefault(dominant[patient], []).append(patient)
    for patients in by_class.values():
        rng.shuffle(patients)

    val_patients: list[str] = []
    test_patients: list[str] = []
    train_patients: list[str] = []

    for class_index in sorted(by_class):
        patients = by_class[class_index]
        k = len(patients)
        if k == 0:
            continue
        n_val = max(1, round(k * val_fraction)) if k > 2 else 0
        n_test = max(1, round(k * test_fraction)) if k > 2 else 0
        if n_val + n_test >= k:
            n_val = n_test = 0
        val_patients += patients[:n_val]
        test_patients += patients[n_val : n_val + n_test]
        train_patients += patients[n_val + n_test :]

    def flatten(group: list[str]) -> list[int]:
        return sorted(i for p in group for i in by_patient[p])

    return flatten(train_patients), flatten(val_patients), flatten(test_patients)


def stratified_indices(manifest: DatasetManifest, indices: list[int], seed: int = 42) -> list[int]:
    """Shuffle deterministically while keeping the class mix."""
    import random

    out = list(indices)
    random.Random(seed).shuffle(out)
    return out


def partition_for_agents(
    indices: list[int],
    manifest: DatasetManifest,
    num_agents: int,
    *,
    non_iid: bool = True,
    seed: int = 7,
    size_ratio: float = 1.5,
) -> list[list[int]]:
    """Split a set of indices across `num_agents` hospital sites.

    With `non_iid=True` the class histogram is deliberately skewed per site
    (Dirichlet-style), which is what real hospital referral patterns look like
    and what makes federated averaging a non-trivial problem rather than a
    formality.

    `size_ratio` caps how far the partition sizes may drift from the mean, so the
    skew shows up in class mix rather than leaving one site with too little data
    to train on at all.
    """
    import random

    rng = random.Random(seed)
    remaining = stratified_indices(manifest, indices, seed)
    if not remaining:
        return [[] for _ in range(num_agents)]

    # Bucket indices by class.
    by_class: dict[int, list[int]] = {}
    for i in remaining:
        by_class.setdefault(manifest.records[i].class_index, []).append(i)

    for bucket in by_class.values():
        rng.shuffle(bucket)

    agent_indices: list[list[int]] = [[] for _ in range(num_agents)]

    if not non_iid:
        cursor = 0
        total = len(remaining)
        for a in range(num_agents):
            size = total // num_agents + (1 if a < total % num_agents else 0)
            agent_indices[a] = remaining[cursor : cursor + size]
            cursor += size
        return agent_indices

    # Non-IID: give each agent a random concentration of each class.
    for pool in by_class.values():
        cursor = 0
        weights = [rng.betavariate(0.5, 0.5) + 1e-3 for _ in range(num_agents)]
        total_w = sum(weights)
        for a in range(num_agents):
            take = round(len(pool) * weights[a] / total_w)
            take = max(0, min(take, len(pool) - cursor))
            agent_indices[a].extend(pool[cursor : cursor + take])
            cursor += take
        # Distribute any remainder to whichever agents came up short.
        leftover = pool[cursor:]
        for offset, idx in enumerate(leftover):
            agent_indices[(offset + cursor) % num_agents].append(idx)

    rebalance_agent_sizes(agent_indices, manifest, size_ratio=size_ratio)

    for a in agent_indices:
        rng.shuffle(a)
    return agent_indices


def rebalance_agent_sizes(
    agent_indices: list[list[int]],
    manifest: DatasetManifest,
    *,
    size_ratio: float = 1.5,
) -> None:
    """Bound agent partition sizes while leaving the class mix skewed.

    Unconstrained Dirichlet sampling produced one site with 3,462 images next to
    one with 205. That is realistic but too thin to train on, so surplus indices
    are moved from the largest agent to the smallest until every site sits within
    `size_ratio` of the mean. The bound is symmetric about the mean, so the
    largest site ends up at most `size_ratio` times the smallest.

    When choosing what to move, the donor's most over-represented class is given
    up first, so rebalancing does not quietly flatten the non-IID skew that makes
    federated averaging interesting.
    """
    import math

    num_agents = len(agent_indices)
    total = sum(len(a) for a in agent_indices)
    if num_agents == 0 or total == 0:
        return

    mean = total / num_agents
    spread = math.sqrt(size_ratio)
    max_allowed = max(1, math.ceil(mean * spread))
    min_allowed = max(1, math.floor(mean / spread))

    def counts_for(agent: int) -> dict[int, int]:
        tally: dict[int, int] = {}
        for i in agent_indices[agent]:
            ci = manifest.records[i].class_index
            tally[ci] = tally.get(ci, 0) + 1
        return tally

    # Move one index at a time from the largest agent to the smallest until every
    # site sits inside the band. Total size is conserved, so if one agent is
    # under the floor some other agent must be above the mean; taking from the
    # current largest and giving to the current smallest therefore converges.
    while True:
        sizes = [len(a) for a in agent_indices]
        if all(min_allowed <= s <= max_allowed for s in sizes):
            break

        donor = max(range(num_agents), key=lambda a: sizes[a])
        taker = min(range(num_agents), key=lambda a: sizes[a])
        if sizes[donor] - sizes[taker] <= 1:
            # Cannot improve without moving onto a band edge; stop rather than
            # oscillate.
            break

        tally = counts_for(donor)
        if not tally:
            break
        # Give up the class the donor holds most of, keeping the rare classes
        # where they already are.
        top_class = max(tally.items(), key=lambda kv: (kv[1], kv[0]))[0]
        position = next(
            i
            for i, idx in enumerate(agent_indices[donor])
            if manifest.records[idx].class_index == top_class
        )
        moved = agent_indices[donor].pop(position)
        agent_indices[taker].append(moved)


def class_histogram(manifest: DatasetManifest, indices: list[int]) -> dict[str, int]:
    counts = {c: 0 for c in CANCER_CLASSES}
    for i in indices:
        counts[CANCER_CLASSES[manifest.records[i].class_index]] += 1
    return counts
