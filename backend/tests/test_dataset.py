from __future__ import annotations

import csv
from pathlib import Path

import pytest
from app.services.dataset import (
    CANCER_CLASSES,
    DX_TO_CLASS,
    DatasetManifest,
    DatasetRecord,
    class_histogram,
    load_manifest,
    partition_for_agents,
    patient_level_split,
    rebalance_agent_sizes,
    stratified_indices,
)


def _write_fake_ham10000(root: Path, rows: list[dict[str, str]]) -> Path:
    """Build the exact on-disk layout shipped in ``data/``."""
    data_dir = root / "data"
    (data_dir / "HAM10000_images_part_1").mkdir(parents=True)
    (data_dir / "HAM10000_images_part_2").mkdir(parents=True)

    split_at = len(rows) // 2
    for i, row in enumerate(rows):
        part = 1 if i < split_at else 2
        name = f"{row['image_id']}.jpg"
        (data_dir / f"HAM10000_images_part_{part}" / name).write_bytes(b"\xff\xd8\xff")

    meta = data_dir / "HAM10000_metadata.csv"
    with meta.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["lesion_id", "image_id", "dx", "dx_type", "age", "sex"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return meta


def _rows(per_class: dict[str, int], lesions_per_class: int = 1) -> list[dict[str, str]]:
    """`per_class` values are *total image counts*, spread over lesions.

    HAM10000 stores several images per lesion, so a helper that emitted one image
    per row could never exercise multi-image lesion grouping.
    """
    rows: list[dict[str, str]] = []
    counter = 0
    for dx, count in per_class.items():
        lesions = max(1, min(lesions_per_class, count))
        base, extra = divmod(count, lesions)
        for lesion in range(lesions):
            for _ in range(base + (1 if lesion < extra else 0)):
                counter += 1
                rows.append(
                    {
                        "lesion_id": f"L_{dx}_{lesion}",
                        "image_id": f"ISIC_{counter:07d}",
                        "dx": dx,
                        "dx_type": "histo",
                        "age": "50.0",
                        "sex": "male",
                    }
                )
    return rows


class TestTaxonomy:
    def test_ham10000_has_no_scc_so_we_do_not_invent_one(self):
        # The official dataset ships `akiec` where a naive pipeline expects
        # `scc`. Relabelling actinic keratosis as squamous cell carcinoma would
        # fabricate ground truth, so the two must stay distinct.
        assert "scc" not in DX_TO_CLASS
        assert "squamous_cell_carcinoma" not in CANCER_CLASSES
        assert DX_TO_CLASS["akiec"] == "actinic_keratosis"
        assert "actinic_keratosis" in CANCER_CLASSES

    def test_all_mapped_labels_are_declared_classes(self):
        assert set(DX_TO_CLASS.values()) <= set(CANCER_CLASSES)

    def test_unknown_and_vascular_labels_are_excluded(self):
        assert "vasc" not in DX_TO_CLASS


class TestLoadManifest:
    def test_missing_dataset_reports_instead_of_raising(self, tmp_path, monkeypatch):
        # Both roots must be neutralised, otherwise this picks up the real
        # HAM10000 checkout sitting in the repo and passes vacuously.
        monkeypatch.setattr("app.services.dataset.settings.dataset_root", str(tmp_path / "nope"))
        monkeypatch.setattr("app.services.dataset._candidate_roots", list)
        manifest = load_manifest.__wrapped__(force=True)
        assert manifest.loaded is False
        assert manifest.total_images == 0
        assert "demo mode" in manifest.message

    def test_reads_across_both_image_part_directories(self, tmp_path, monkeypatch):
        rows = _rows({"mel": 3, "bcc": 2, "akiec": 2, "nv": 4})
        _write_fake_ham10000(tmp_path, rows)
        monkeypatch.setattr("app.services.dataset._candidate_roots", lambda: [tmp_path / "data"])

        manifest = load_manifest.__wrapped__(force=True)
        assert manifest.loaded is True
        assert manifest.total_images == len(rows)
        assert manifest.class_counts["melanoma"] == 3
        assert manifest.class_counts["basal_cell_carcinoma"] == 2
        assert manifest.class_counts["actinic_keratosis"] == 2
        assert manifest.class_counts["benign_lesion"] == 4

    def test_extensionless_image_ids_resolve_to_jpg_files(self, tmp_path, monkeypatch):
        rows = _rows({"mel": 2}, lesions_per_class=1)
        _write_fake_ham10000(tmp_path, rows)
        monkeypatch.setattr("app.services.dataset._candidate_roots", lambda: [tmp_path / "data"])

        manifest = load_manifest.__wrapped__(force=True)
        # Metadata stores `ISIC_0000001`, the file on disk is `ISIC_0000001.jpg`.
        assert manifest.loaded is True
        assert all(r.path.suffix == ".jpg" for r in manifest.records)

    def test_rows_pointing_at_missing_files_are_skipped(self, tmp_path, monkeypatch):
        rows = _rows({"mel": 2}, lesions_per_class=1)
        rows.append(
            {
                "lesion_id": "L_missing",
                "image_id": "ISIC_9999999",
                "dx": "mel",
                "dx_type": "histo",
                "age": "50.0",
                "sex": "male",
            }
        )
        _write_fake_ham10000(tmp_path, rows)
        # Remove the file so the metadata row is genuinely unresolvable.
        for part in (1, 2):
            stub = tmp_path / "data" / f"HAM10000_images_part_{part}" / "ISIC_9999999.jpg"
            if stub.exists():
                stub.unlink()
        monkeypatch.setattr("app.services.dataset._candidate_roots", lambda: [tmp_path / "data"])

        manifest = load_manifest.__wrapped__(force=True)
        assert manifest.total_images == 2
        assert "ISIC_9999999" not in {r.image_id for r in manifest.records}

    def test_lesion_id_is_the_grouping_key(self, tmp_path, monkeypatch):
        rows = _rows({"mel": 3}, lesions_per_class=3)
        _write_fake_ham10000(tmp_path, rows)
        monkeypatch.setattr("app.services.dataset._candidate_roots", lambda: [tmp_path / "data"])

        manifest = load_manifest.__wrapped__(force=True)
        assert manifest.patients == 3
        assert all(r.patient == r.lesion_id for r in manifest.records)


def _manifest_from_classes(counts: list[int]) -> DatasetManifest:
    """One lesion per class, `count` images each, so splitting stays simple."""
    records: list[DatasetRecord] = []
    class_counts: dict[str, int] = {}
    for class_index, count in enumerate(counts):
        name = CANCER_CLASSES[class_index]
        records.extend(
            DatasetRecord(
                image_id=f"c{class_index}_{i}",
                path=Path(f"c{class_index}_{i}.jpg"),
                dx=name,
                patient=f"P_{class_index}_{i}",  # one lesion per image
                lesion_id=f"P_{class_index}_{i}",
                class_index=class_index,
            )
            for i in range(count)
        )
        class_counts[name] = count
    return DatasetManifest(
        loaded=True,
        records=records,
        class_counts=class_counts,
        patients=len(records),
    )


class TestPatientLevelSplit:
    def test_no_lesion_ever_appears_in_two_splits(self):
        manifest = _manifest_from_classes([40, 20, 12, 120])
        train, val, test = patient_level_split(manifest)
        groups = [{manifest.records[i].patient for i in s} for s in (train, val, test)]
        assert not groups[0] & groups[1]
        assert not groups[0] & groups[2]
        assert not groups[1] & groups[2]

    def test_every_image_lands_in_exactly_one_split(self):
        manifest = _manifest_from_classes([40, 20, 12, 120])
        train, val, test = patient_level_split(manifest)
        combined = train + val + test
        assert sorted(combined) == list(range(manifest.total_images))

    def test_splits_are_class_stratified(self):
        # The whole point: the rare classes must appear in val and test, and the
        # benign majority must not be ~2/3 of every bucket.
        manifest = _manifest_from_classes([100, 40, 24, 240])
        train, val, _ = patient_level_split(manifest)

        def ratio(idxs, class_index):
            if not idxs:
                return 0.0
            hits = sum(1 for i in idxs if manifest.records[i].class_index == class_index)
            return hits / len(idxs)

        for class_index in range(4):
            assert ratio(val, class_index) == pytest.approx(ratio(train, class_index), abs=0.05)

    def test_multi_image_lesions_stay_together(self):
        # Ten lesions, each carrying five images of the same class.
        records = [
            DatasetRecord(
                image_id=f"l{lesion}_{img}",
                path=Path(f"l{lesion}_{img}.jpg"),
                dx="melanoma",
                patient=f"LESION_{lesion}",
                lesion_id=f"LESION_{lesion}",
                class_index=0,
            )
            for lesion in range(10)
            for img in range(5)
        ]
        manifest = DatasetManifest(
            loaded=True, records=records, class_counts={"melanoma": 50}, patients=10
        )
        train, val, test = patient_level_split(manifest)

        for split in (train, val, test):
            counts: dict[str, int] = {}
            for i in split:
                counts[manifest.records[i].patient] = counts.get(manifest.records[i].patient, 0) + 1
            assert all(c == 5 for c in counts.values())

    def test_deterministic_for_a_given_seed(self):
        manifest = _manifest_from_classes([40, 20, 12, 120])
        assert patient_level_split(manifest, seed=7) == patient_level_split(manifest, seed=7)
        assert patient_level_split(manifest, seed=7) != patient_level_split(manifest, seed=8)

    def test_empty_manifest_is_handled(self):
        assert patient_level_split(DatasetManifest()) == ([], [], [])


class TestStratifiedIndices:
    def test_preserves_class_mix(self):
        manifest = _manifest_from_classes([50, 25, 25, 100])
        order = stratified_indices(manifest, list(range(manifest.total_images)))
        assert sorted(order) == list(range(manifest.total_images))
        assert [manifest.records[i].class_index for i in order] == [
            manifest.records[i].class_index for i in order
        ]


class TestPartitionForAgents:
    def test_partitions_cover_train_split_exactly_once(self):
        manifest = _manifest_from_classes([100, 40, 24, 240])
        train, _, _ = patient_level_split(manifest)
        parts = partition_for_agents(train, manifest, 4)
        assert sorted(i for p in parts for i in p) == sorted(train)

    def test_non_iid_partitions_differ_in_class_mix(self):
        manifest = _manifest_from_classes([100, 40, 24, 240])
        train, _, _ = patient_level_split(manifest)
        parts = partition_for_agents(train, manifest, 4, non_iid=True)
        mixes = [tuple(sorted(class_histogram(manifest, p).items())) for p in parts]
        assert len(set(mixes)) > 1

    def test_fewer_indices_than_agents_is_handled(self):
        manifest = _manifest_from_classes([2, 1, 1, 4])
        parts = partition_for_agents(list(range(manifest.total_images)), manifest, 8)
        assert len(parts) == 8
        assert sum(len(p) for p in parts) == manifest.total_images

    def test_partition_sizes_stay_within_the_configured_ratio(self):
        # Unconstrained Dirichlet sampling gave one site 3,462 images and another
        # 205, which is too thin to train on.
        manifest = _manifest_from_classes([400, 200, 120, 600])
        parts = partition_for_agents(list(range(manifest.total_images)), manifest, 4)
        sizes = sorted(len(p) for p in parts)
        mean = manifest.total_images / 4
        assert max(sizes) / min(sizes) <= 1.6
        assert max(sizes) <= mean * 1.5 + 1

    def test_size_cap_does_not_flatten_the_class_skew(self):
        manifest = _manifest_from_classes([400, 200, 120, 600])
        parts = partition_for_agents(list(range(manifest.total_images)), manifest, 4)
        mixes = {c: [class_histogram(manifest, p).get(c, 0) for p in parts] for c in CANCER_CLASSES}
        # At least one class must be concentrated unevenly across sites,
        # otherwise the non-IID setup has been averaged away.
        assert any(max(v) > 3 * min(v) for v in mixes.values())

    def test_rebalance_is_idempotent(self):
        manifest = _manifest_from_classes([50, 25, 25, 100])
        parts = partition_for_agents(list(range(manifest.total_images)), manifest, 4)
        before = [sorted(p) for p in parts]
        rebalance_agent_sizes(parts, manifest)
        assert [sorted(p) for p in parts] == before

    def test_rebalance_never_loses_or_duplicates_indices(self):
        manifest = _manifest_from_classes([120, 60, 30, 200])
        parts = partition_for_agents(list(range(manifest.total_images)), manifest, 4)
        flat = [i for p in parts for i in p]
        assert sorted(flat) == list(range(manifest.total_images))
