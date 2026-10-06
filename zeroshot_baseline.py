"""Stratified zero-shot baseline for MedSigLIP on HAM10000.

Sampled across all 7 dx classes so the rare classes are represented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

DX_LABELS = {
    "mel": "melanoma",
    "nv": "melanocytic nev",
    "bcc": "basal cell carcinoma",
    "akiec": "actinic keratosis",
    "bkl": "benign keratosis",
    "vasc": "vascular lesion",
    "df": "dermatofibroma",
}

MALIGNANT = {"mel", "bcc", "akiec"}


TEMPLATES = {
    "mole": "a photo of a mole that is {label}",
    "photo": "a photo of {label}",
    "bare": "{label}",
    "lesion": "this is a {label} lesion",
    "derm": "a dermoscopy photo of {label}",
}


def find_image(image_id: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = root / f"{image_id}.jpg"
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", default="data")
    p.add_argument("--per-class", type=int, default=40)
    p.add_argument("--backend", default="ov", choices=["ov", "torch"])
    p.add_argument("--template", default="mole", choices=sorted(TEMPLATES))
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--out", default="results/zeroshot_baseline.json")
    args = p.parse_args()

    data = Path(args.data)
    roots = sorted(data.glob("HAM10000_images_part_*"))
    if not roots:
        print(f"no image folders under {data}", flush=True)
        return 2

    df = pd.read_csv(data / "HAM10000_metadata.csv")
    sample = df.groupby("dx", group_keys=False).apply(
        lambda g: g.sample(n=min(args.per_class, len(g)), random_state=0)
    )
    print(f"sampled {len(sample)} images across {sample.dx.nunique()} classes", flush=True)

    paths, classes = [], []
    missing = 0
    for _, row in sample.iterrows():
        path = find_image(row.image_id, roots)
        if path is None:
            missing += 1
            continue
        paths.append(str(path))
        classes.append(row.dx)
    if missing:
        print(f"warning: {missing} images not found on disk", flush=True)
    if not paths:
        print("no images resolved", flush=True)
        return 2

    if args.backend == "ov":
        from medsiglip_ov import MedSiglipOpenVINO
        clf = MedSiglipOpenVINO()
    else:
        from medsiglip import MedSiglipClassifier
        clf = MedSiglipClassifier()

    codes = sorted({c for c in classes})
    template = TEMPLATES[args.template]
    prompts = [template.format(label=DX_LABELS[c]) for c in codes]

    logit_scale = clf.logit_scale

    # text tower once, then vision tower in batches
    text_embeds = clf.embed_texts(prompts)
    text_embeds = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)

    prob_matrix = np.zeros((len(paths), len(codes)))
    total = len(paths)
    for start in range(0, total, args.batch):
        chunk = paths[start : start + args.batch]
        image_embeds = clf.embed_images(chunk)
        image_embeds = image_embeds / np.linalg.norm(image_embeds, axis=-1, keepdims=True)
        logits = image_embeds @ text_embeds.T
        scaled = logits * logit_scale
        exp = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        prob_matrix[start : start + len(chunk)] = exp / exp.sum(axis=-1, keepdims=True)
        print(f"  {min(start + args.batch, total)}/{total}", flush=True)

    predictions = prob_matrix.argmax(axis=1)
    y_true = [codes.index(c) for c in classes]
    names = [DX_LABELS[c] for c in codes]

    print("\n" + classification_report(y_true, predictions, target_names=names, zero_division=0))
    print("confusion matrix (rows=true):")
    cm = confusion_matrix(y_true, predictions)
    print(pd.DataFrame(cm, index=names, columns=names).to_string())

    y_true_bin = [int(c in MALIGNANT) for c in classes]
    try:
        malignant_idx = [codes.index(c) for c in ("mel", "bcc", "akiec") if c in codes]
        malignant_score = prob_matrix[:, malignant_idx].sum(axis=1)
        auc = roc_auc_score(y_true_bin, malignant_score)
        print(f"\nbinary malignant-vs-benign AUC: {auc:.4f}")
    except ValueError as exc:
        auc = None
        print(f"\nAUC not computed: {exc}")

    acc = float(np.mean([a == b for a, b in zip(predictions, y_true, strict=True)]))
    print(f"top-1 accuracy: {acc:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "backend": args.backend,
                "template": args.template,
                "n": len(paths),
                "per_class": args.per_class,
                "accuracy": acc,
                "malignant_auc": auc,
                "prompts": prompts,
                "true": [DX_LABELS[c] for c in classes],
                "pred": [DX_LABELS[codes[i]] for i in predictions],
                "paths": paths,
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())