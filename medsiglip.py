"""Wrapper around google/medsiglip-448 for zero-shot medical image classification."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

MODEL_ID = "google/medsiglip-448"
IMAGE_SIZE = 448
MAX_TEXT_TOKENS = 64


def _as_tensor(output) -> torch.Tensor:
    """transformers 5.x returns a ModelOutput from get_*_features; unwrap it."""
    if isinstance(output, torch.Tensor):
        return output
    for attr in ("pooler_output", "last_hidden_state", "image_embeds", "text_embeds"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value[:, 0] if value.dim() == 3 else value
    raise TypeError(f"cannot extract tensor from {type(output).__name__}")


class MedSiglipClassifier:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(
            model_id, dtype=dtype
        )
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def load_image(source: str | Path | Image.Image) -> Image.Image:
        if isinstance(source, Image.Image):
            image = source
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"image not found: {path}")
            image = Image.open(path)
        return image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    @torch.no_grad()
    def embed_images(self, images: Sequence[str | Path | Image.Image]):
        pixels = [self.load_image(img) for img in images]
        inputs = self.processor(images=pixels, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        return _as_tensor(self.model.get_image_features(pixel_values=pixel_values))

    @torch.no_grad()
    def embed_texts(self, texts: Sequence[str]):
        inputs = self.processor(
            text=list(texts),
            padding="max_length",
            max_length=MAX_TEXT_TOKENS,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(self.device)
        return _as_tensor(self.model.get_text_features(**kwargs))

    @torch.no_grad()
    def classify(
        self,
        images: Sequence[str | Path | Image.Image],
        candidate_labels: Sequence[str],
        top_k: int | None = None,
    ) -> list[dict]:
        if not candidate_labels:
            raise ValueError("candidate_labels must not be empty")

        image_embeds = self.embed_images(images)
        text_embeds = self.embed_texts(candidate_labels)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logits = self.model.logit_scale.exp() * image_embeds @ text_embeds.T
        probs = logits.softmax(dim=-1)

        k = top_k or len(candidate_labels)
        results = []
        for row in probs:
            scores, idx = row.topk(min(k, len(candidate_labels)))
            results.append(
                {
                    label: round(float(p), 6)
                    for p, label in zip(
                        scores, (candidate_labels[i] for i in idx), strict=True
                    )
                }
            )
        return results

    @torch.no_grad()
    def similarity_matrix(
        self, images: Sequence[str | Path | Image.Image], texts: Sequence[str]
    ) -> torch.Tensor:
        """Raw cosine similarity between every image and every text."""
        image_embeds = self.embed_images(images)
        text_embeds = self.embed_texts(texts)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return image_embeds @ text_embeds.T


def iter_image_paths(
    folder: str | Path, patterns: Iterable[str] = ("*.png", "*.jpg", "*.jpeg")
) -> list[Path]:
    root = Path(folder)
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    return sorted(p for p in paths if p.is_file())