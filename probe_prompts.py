"""Find why HAM10000 prompts collapse. Tests padding side and prompt templates."""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from medsiglip import MedSiglipClassifier

DX = {
    "mel": "melanoma",
    "nv": "melanocytic nev",
    "bcc": "basal cell carcinoma",
    "akiec": "actinic keratosis",
    "bkl": "benign keratosis",
    "vasc": "vascular lesion",
    "df": "dermatofibroma",
}

df = pd.read_csv("data/HAM10000_metadata.csv")
roots = list(Path("data").glob("HAM10000_images_part_*"))
sel = df.groupby("dx", group_keys=False).head(6)
paths, labels = [], []
for _, row in sel.iterrows():
    for r in roots:
        p = r / f"{row.image_id}.jpg"
        if p.is_file():
            paths.append(str(p))
            labels.append(row.dx)
            break

codes = sorted(DX)
clf = MedSiglipClassifier()

img = clf.embed_images(paths)
img = img / img.norm(dim=-1, keepdim=True)

def separation(text_embeds):
    t = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    m = (t @ t.T).numpy()
    off = m[~np.eye(len(m), dtype=bool)]
    return off.max(), off.mean()

for side in ("right", "left"):
    clf.processor.tokenizer.padding_side = side
    print(f"\n{'='*60}\npadding_side = {side}")
    tmpl = {
        "A card-style 'a photo of X'": [f"a photo of {DX[c]}" for c in codes],
        "B 'a dermoscopy photo of X'": [f"a dermoscopy photo of {DX[c]}" for c in codes],
        "C bare label 'X'": [DX[c] for c in codes],
        "D 'a photo of a mole that is X'": [f"a photo of a mole that is {DX[c]}" for c in codes],
        "E 'this is a X lesion'": [f"this is a {DX[c]} lesion" for c in codes],
    }
    for name, prompts in tmpl.items():
        te = clf.embed_texts(prompts)
        mx, mean = separation(te)
        sim = (img @ (te / te.norm(dim=-1, keepdim=True)).T).numpy()
        pred = sim.argmax(axis=1)
        acc = np.mean([codes[p] == l for p, l in zip(pred, labels)])
        top1 = np.mean(sim.max(axis=1))
        print(f"  {name:34s} acc {acc:5.1%} | mean sim {top1:.3f} | prompt maxcos {mx:.3f} meancos {mean:.3f}")