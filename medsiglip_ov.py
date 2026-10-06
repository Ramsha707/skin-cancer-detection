"""OpenVINO-accelerated MedSigLIP for Intel GPUs (iGPU included).

Run with the .venv-ov interpreter:
    .\.venv-ov\Scripts\python.exe classify_ov.py --folder data --labels a b

The vision tower is compiled to OpenVINO and cached to disk as an .xml/.bin
pair; the text tower stays on torch CPU since it runs once per label set and
is negligible next to per-image vision encoding.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from PIL import Image
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

from medsiglip import IMAGE_SIZE, MAX_TEXT_TOKENS, MODEL_ID, _as_tensor, iter_image_paths

CACHE_DIR = Path(__file__).parent / ".cache_openvino"


def _available_devices() -> list[str]:
    return ov.Core().available_devices


class MedSiglipOpenVINO:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "GPU",
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        available = _available_devices()
        if device not in available:
            fallback = "GPU" if "GPU" in available else available[0]
            print(f"[openvino] device '{device}' unavailable, using '{fallback}'")
            device = fallback
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.torch_model = AutoModelForZeroShotImageClassification.from_pretrained(model_id)
        self.torch_model.eval()

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        xml_path = cache_dir / "vision_model.xml"
        bin_path = cache_dir / "vision_model.bin"

        self.core = ov.Core()
        if xml_path.is_file() and bin_path.is_file():
            ov_model = self.core.read_model(str(xml_path), str(bin_path))
        else:
            ov_model = self._export(model_id, xml_path, bin_path)

        # Compile the in-memory model. Reloading a saved model loses static
        # shape info, which makes compile_model reject batched input.
        compiled = self.core.compile_model(ov_model, self.device)
        self._infer = compiled.create_infer_request()

    @staticmethod
    def _export(model_id: str, xml_path: Path, bin_path: Path):
        model = AutoModelForZeroShotImageClassification.from_pretrained(model_id)
        vision = model.model.vision_model if hasattr(model, "model") else model.vision_model

        class VisionWrapper(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                self.v = v

            def forward(self, pixel_values):
                # pooler_output, not last_hidden_state[:, 0]: SigLIP's image
                # embedding is the projected pooled output and the two are
                # near-orthogonal, so using the wrong one yields ~uniform scores
                return self.v(pixel_values).pooler_output

        example = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        ov_model = ov.convert_model(VisionWrapper(vision.eval()), example_input=example)
        # 2026.x moved save_model off the Model object onto the module
        ov.save_model(ov_model, str(xml_path), False)
        print(f"[openvino] compiled model cached to {xml_path.name}")
        return ov_model

    @staticmethod
    def load_image(source) -> Image.Image:
        if isinstance(source, Image.Image):
            image = source
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"image not found: {path}")
            image = Image.open(path)
        return image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    def embed_images(self, images: Sequence) -> np.ndarray:
        pixels = [self.load_image(img) for img in images]
        inputs = self.processor(images=pixels, return_tensors="np")
        self._infer.infer(inputs["pixel_values"])
        return np.asarray(self._infer.get_output_tensor().data)

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(texts),
            padding="max_length",
            max_length=MAX_TEXT_TOKENS,
            truncation=True,
            return_tensors="pt",
        )
        kwargs = {"input_ids": inputs["input_ids"]}
        mask = inputs.get("attention_mask")
        if mask is not None:
            kwargs["attention_mask"] = mask
        embeds = _as_tensor(self.torch_model.get_text_features(**kwargs))
        return embeds.numpy()

    def classify(self, images: Sequence, candidate_labels: Sequence[str], top_k: int | None = None):
        if not candidate_labels:
            raise ValueError("candidate_labels must not be empty")

        image_embeds = self.embed_images(images)
        text_embeds = self.embed_texts(candidate_labels)

        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)

        logit_scale = float(self.torch_model.logit_scale.exp())
        logits = logit_scale * image_embeds @ text_embeds.T
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)

        k = top_k or len(candidate_labels)
        results = []
        for row in probs:
            order = np.argsort(row)[::-1][:k]
            results.append({candidate_labels[i]: round(float(row[i]), 6) for i in order})
        return results

    def similarity_matrix(self, images: Sequence, texts: Sequence[str]) -> np.ndarray:
        image_embeds = self.embed_images(images)
        text_embeds = self.embed_texts(texts)
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)
        return image_embeds @ text_embeds.T


__all__ = ["MedSiglipOpenVINO", "iter_image_paths"]