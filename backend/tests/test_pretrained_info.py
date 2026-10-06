"""Week 5: pretrained-model introspection.

The fixture builds a miniature HuggingFace-style snapshot (the real
`config.json` schema plus a hand-written safetensors header) so the assertions
are about arithmetic the parser must get right, not about whatever happens to
be in the developer's cache directory.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from app.services import pretrained
from fastapi.testclient import TestClient

MODEL_DIR = "models--google--medsiglip-448"

CONFIG = {
    "architectures": ["SiglipModel"],
    "vision_config": {
        "num_hidden_layers": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_attention_heads": 16,
        "patch_size": 14,
        "image_size": 448,
    },
    "text_config": {
        "num_hidden_layers": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_attention_heads": 16,
        "vocab_size": 32000,
        "max_position_embeddings": 64,
        "projection_size": 1152,
    },
}

PREPROCESSOR = {
    "resample": 3,
    "rescale_factor": 1.0 / 255.0,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "size": {"height": 448, "width": 448},
}

# Patch embedding (1152 x 14*14*3) + 27 encoder layers of 4x4, a text tower
# of 100x8 and the two SigLIP logits. Small enough to sum by hand:
VISION = 1152 * 588 + 27 * 16  # 677,808
TEXT = 800
TOTAL = VISION + TEXT + 2  # 678,610
HEAD = 1152 * 4 + 4  # 4,612
SELECTIVE = 6 * 16  # top 6 of 27 layers x 16 params each


def write_weights(path: Path, tensors: dict[str, list[int]]) -> None:
    header = {name: {"dtype": "F32", "shape": shape} for name, shape in tensors.items()}
    payload = json.dumps(header).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(payload)))
        fh.write(payload)


def make_snapshot(root: Path, *, weights: bool) -> Path:
    snap = root / MODEL_DIR / "snapshots" / "commit-abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    (snap / "preprocessor_config.json").write_text(
        json.dumps(PREPROCESSOR), encoding="utf-8"
    )
    if weights:
        tensors = {
            "vision_model.embeddings.patch_embedding.weight": [1152, 588],
            **{
                f"vision_model.encoder.layers.{i}.weight": [4, 4] for i in range(27)
            },
            "text_model.weight": [100, 8],
            "logit_scale": [1],
            "logit_bias": [1],
        }
        write_weights(snap / "model.safetensors", tensors)
    return snap


@pytest.fixture()
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(pretrained, "hub_cache_dir", lambda: tmp_path)
    return tmp_path


class TestTensorHeader:
    def test_reads_names_and_shapes_without_touching_tensor_data(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "w.safetensors"
        write_weights(path, {"a.weight": [3, 4], "b.bias": [5]})

        header = pretrained.read_tensor_header(path)

        assert header == {
            "a.weight": {"dtype": "F32", "shape": [3, 4]},
            "b.bias": {"dtype": "F32", "shape": [5]},
        }

    def test_rejects_a_file_that_is_not_safetensors(self, tmp_path: Path) -> None:
        path = tmp_path / "not-weights"
        path.write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError):
            pretrained.read_tensor_header(path)


class TestBuildPretrainedInfo:
    def test_parameter_counts_come_from_the_weight_file(self, cache_root: Path) -> None:
        make_snapshot(cache_root, weights=True)

        info = pretrained.build_pretrained_info()

        assert info["total_params"] == TOTAL
        assert info["vision_tower"]["params"] == VISION
        assert info["text_tower"]["params"] == TEXT
        assert info["cache_status"] == "ok"
        assert info["loaded_locally"] is True
        assert info["resolved_path"] is not None
        assert info["device"] in {"openvino:GPU", "cpu"}

    def test_strategy_budgets_are_derived_from_the_counts(self, cache_root: Path) -> None:
        make_snapshot(cache_root, weights=True)

        info = pretrained.build_pretrained_info()

        assert info["trainable_params_frozen_strategy"] == HEAD
        assert info["trainable_params_selective_strategy"] == HEAD + SELECTIVE
        assert info["trainable_params_full_strategy"] == TOTAL
        assert info["frozen_params_frozen_strategy"] == TOTAL - HEAD
        # Frozen strategy must leave the vision tower untouched.
        assert info["vision_tower"]["trainable_when_frozen"] == 0

    def test_geometry_and_preprocessing_are_read_from_config(
        self, cache_root: Path
    ) -> None:
        make_snapshot(cache_root, weights=True)

        info = pretrained.build_pretrained_info()

        assert info["architecture"] == "SiglipModel"
        assert info["vision_tower"]["num_patches"] == (448 // 14) ** 2
        assert info["vision_tower"]["layers"] == 27
        assert info["text_tower"]["vocab_size"] == 32000
        assert info["preprocessing"]["input_size"] == [448, 448]
        assert info["preprocessing"]["resample"] == "bicubic"
        assert info["preprocessing"]["image_mean"] == [0.5, 0.5, 0.5]

    def test_text_prompts_track_the_classifier_class_order(self, cache_root: Path) -> None:
        make_snapshot(cache_root, weights=True)

        info = pretrained.build_pretrained_info()

        assert [p["dx"] for p in info["text_prompts"]] == [
            "melanoma",
            "basal_cell_carcinoma",
            "actinic_keratosis",
            "benign_lesion",
        ]
        assert all(p["prompt"] for p in info["text_prompts"])

    def test_config_without_weights_reports_partial_not_ok(
        self, cache_root: Path
    ) -> None:
        make_snapshot(cache_root, weights=False)

        info = pretrained.build_pretrained_info()

        assert info["cache_status"] == "partial"
        assert info["loaded_locally"] is False

    def test_no_snapshot_falls_back_to_the_model_card(self, cache_root: Path) -> None:
        info = pretrained.build_pretrained_info()

        assert info["cache_status"] == "missing"
        assert info["loaded_locally"] is False
        assert info["resolved_path"] is None
        # The static card total, not a count from disk.
        assert info["total_params"] == 878_000_000


class TestEndpoint:
    def test_pretrained_endpoint_serves_the_payload(self, cache_root: Path) -> None:
        from app.main import app

        make_snapshot(cache_root, weights=True)

        with TestClient(app) as client:
            response = client.get("/api/models/pretrained")

        assert response.status_code == 200
        body = response.json()
        assert body["model_id"] == "google/medsiglip-448"
        assert body["total_params"] == TOTAL
        assert body["cache_status"] == "ok"
        assert len(body["text_prompts"]) == 4
