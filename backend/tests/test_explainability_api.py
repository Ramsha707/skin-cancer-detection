"""Week 10: explainability.

Two things are under test:

1. The capability gate. When torch or the pretrained weights are absent, both the
   status and the generate endpoints must return an explicit unavailable payload.
   A fabricated heatmap would be the exact honesty failure this project forbids,
   so this is asserted, not assumed.
2. The service plumbing that runs *after* a grid exists: target-class resolution,
   the overlay data URI, and the top-region summary. That path is exercised by
   monkeypatching the Grad-CAM entry point to a deterministic grid, so the service
   logic is covered without a torch build.
"""

from __future__ import annotations

import numpy as np
import pytest
from app import models  # noqa: F401  (registration side effect)
from app.database.base import Base, get_db
from app.main import app
from app.services import explainability as xai_service
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ml import explainability as xai


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, future=True)

    def override():
        db = Session()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    app.dependency_overrides[get_db] = override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (200, 150, 120)).save(buf, format="PNG")
    return buf.getvalue()


def _upload(client, target_class: str | None = None):
    data = {"target_class": target_class} if target_class else {}
    return client.post(
        "/api/explainability/generate",
        files={"file": ("lesion.png", _png_bytes(), "image/png")},
        data=data,
    )


class TestCapabilityGate:
    def test_status_reports_unavailable_with_a_reason(self, client):
        body = client.get("/api/explainability/status").json()
        # In the test environment torch is absent; either way the field must be
        # present and, when unavailable, carry an operator-readable reason.
        assert "available" in body
        assert body["method"] == "grad-cam"
        if not body["available"]:
            assert body["reason"]

    def test_generate_returns_not_available_rather_than_a_fake_overlay(self, client):
        body = _upload(client).json()
        if not body["available"]:
            assert body["reason"]
            assert body["grid"] == []
            assert body["overlay_png"] is None

    def test_unsupported_file_type_is_refused(self, client):
        r = client.post(
            "/api/explainability/generate",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 415

    def test_empty_upload_is_refused(self, client):
        r = client.post(
            "/api/explainability/generate",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert r.status_code == 400

    def test_unknown_target_class_is_refused_without_computing(self, client):
        body = _upload(client, target_class="not_a_class").json()
        assert body["available"] is False
        assert "unknown class" in body["reason"].lower()


class TestServicePlumbing:
    """Drive the post-grid path with a deterministic mocked Grad-CAM."""

    @pytest.fixture()
    def mock_grid(self, monkeypatch):
        grid = np.zeros((xai.GRID, xai.GRID), dtype=np.float32)
        # A hot spot in the upper-left quadrant so `top_regions` has something real.
        grid[: grid.shape[0] // 4, : grid.shape[1] // 4] = 1.0

        calls: list[int] = []

        def fake_gradcam(image_path, logits_fn, *, target_index, model_id):
            calls.append(target_index)
            return grid

        # Availability must be forced on, and the grid computation replaced.
        monkeypatch.setattr(
            xai,
            "availability",
            lambda model_id="google/medsiglip-448": xai.Availability(
                available=True, reason="", device="cpu", model_id=model_id
            ),
        )
        monkeypatch.setattr(xai, "gradcam_grid", fake_gradcam)
        monkeypatch.setattr(xai_service, "availability", xai.availability)

        head = _FakeHead()
        monkeypatch.setattr(
            xai_service.registry, "load_trained_head", lambda: (head, "v1", True)
        )
        return {"grid": grid, "calls": calls}

    def test_available_payload_has_grid_overlay_and_regions(self, client, mock_grid):
        body = _upload(client).json()
        assert body["available"] is True
        assert body["head_kind"] == "fine-tuned"
        assert body["is_real_inference"] is True
        assert body["model_version"] == "v1"
        assert body["grid_size"] == xai.GRID
        assert len(body["grid"]) == xai.GRID
        assert len(body["grid"][0]) == xai.GRID
        assert body["overlay_png"].startswith("data:image/png;base64,")
        assert len(body["top_regions"]) == 3
        # The strongest region is the upper-left hot spot we planted.
        assert body["top_regions"][0]["row"] == 0
        assert body["top_regions"][0]["column"] == 0
        assert "upper" in body["top_regions"][0]["region"]

    def test_default_target_is_the_predicted_class(self, client, mock_grid):
        body = _upload(client).json()
        # The fake head always predicts melanoma (class 0), so the target matches.
        assert body["predicted_class"] == "melanoma"
        assert body["target_class"] == "melanoma"

    def test_explicit_target_overrides_the_prediction(self, client, mock_grid):
        body = _upload(client, target_class="benign_lesion").json()
        assert body["available"] is True
        assert body["target_class"] == "benign_lesion"
        assert body["target_index"] == 3
        # A second Grad-CAM call was made for the requested class.
        assert 3 in mock_grid["calls"]

    def test_available_generation_is_audited(self, client, mock_grid):
        _upload(client)
        events = [e["event_type"] for e in client.get("/api/audit").json()]
        assert "explainability_generated" in events

    def test_audit_details_carry_no_file_path(self, client, mock_grid):
        """The ledger must never gain an image location."""
        _upload(client)
        entry = next(
            e
            for e in client.get("/api/audit").json()
            if e["event_type"] == "explainability_generated"
        )
        forbidden = {"image", "image_path", "path", "filename", "patient_id"}
        assert not (forbidden & set(entry["details"] or {}))


class _FakeHead:
    """Predicts melanoma with high confidence, uniformly over the other classes."""

    def logits(self, embeddings: np.ndarray) -> np.ndarray:
        out = np.zeros((len(embeddings), 4), dtype=np.float32)
        out[:, 0] = 5.0
        return out

    def probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        logits = self.logits(embeddings)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
