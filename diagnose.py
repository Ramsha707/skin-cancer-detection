"""Diagnose zero-shot behaviour on HAM10000: compare torch vs OpenVINO embeddings."""

import numpy as np
import pandas as pd

from medsiglip_ov import MedSiglipOpenVINO

PROMPTS = [
    "a dermoscopy photo of melanoma",
    "a dermoscopy photo of melanocytic nev",
    "a dermoscopy photo of basal cell carcinoma",
    "a dermoscopy photo of actinic keratosis",
    "a dermoscopy photo of benign keratosis",
    "a dermoscopy photo of vascular lesion",
    "a dermoscopy photo of dermatofibroma",
]

df = pd.read_csv("data/HAM10000_metadata.csv")
roots = [p for p in __import__("pathlib").Path("data").glob("HAM10000_images_part_*")]

sel = df.groupby("dx", group_keys=False).head(2)
paths, labels = [], []
for _, row in sel.iterrows():
    for r in roots:
        p = r / f"{row.image_id}.jpg"
        if p.is_file():
            paths.append(str(p))
            labels.append(row.dx)
            break

print(f"{len(paths)} probe images")

ov = MedSiglipOpenVINO()

# 1. does the OpenVINO image embedding match torch?
from medsiglip import MedSiglipClassifier
import torch

tc = MedSiglipClassifier()
with torch.no_grad():
    t_img = tc.embed_images(paths)
t_img = t_img / t_img.norm(dim=-1, keepdim=True)
o_img = ov.embed_images(paths)
o_img = o_img / np.linalg.norm(o_img, axis=-1, keepdims=True)
cos = np.sum(t_img.numpy() * o_img, axis=-1)
print(f"\ncos(torch, openvino) image embeds: min {cos.min():.4f} mean {cos.mean():.4f}")

t_txt = tc.embed_texts(PROMPTS)
t_txt = t_txt / t_txt.norm(dim=-1, keepdim=True)
o_txt = ov.embed_texts(PROMPTS)
o_txt = o_txt / np.linalg.norm(o_txt, axis=-1, keepdims=True)
cos_t = np.sum(t_txt.numpy() * o_txt, axis=-1)
print(f"cos(torch, openvino) text  embeds: min {cos_t.min():.4f} mean {cos_t.mean():.4f}")

# 2. what does each path predict?
for name, sim in (("torch", (t_img @ t_txt.T).numpy()), ("openvino", o_img @ o_txt.T)):
    pred = sim.argmax(axis=1)
    print(f"\n--- {name} predictions ---")
    for i, (p, lab) in enumerate(zip(paths, labels)):
        top = sorted(zip(PROMPTS, sim[i]), key=lambda x: -x[1])[:2]
        print(f"true dx={lab:6s} pred={PROMPTS[pred[i]].split('of ')[1]:28s} | {top[0][1]:.3f} {top[1][1]:.3f}")

# 3. are the prompt embeddings themselves sane?
print("\n--- prompt embedding similarity matrix (torch) ---")
print(np.round((t_txt @ t_txt.T).numpy(), 2))