"""CLI: zero-shot classify with OpenVINO (Intel GPU).

Run with the OpenVINO venv interpreter:
    .\.venv-ov\Scripts\python.exe classify_ov.py --folder data --labels "a" "b"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from medsiglip_ov import MedSiglipOpenVINO, iter_image_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--folder")

    lbl = p.add_mutually_exclusive_group(required=True)
    lbl.add_argument("--labels", nargs="+")
    lbl.add_argument("--labels-file")

    p.add_argument("--model", default="google/medsiglip-448")
    p.add_argument("--device", default="GPU", help="GPU or CPU")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.labels:
        labels = args.labels
    else:
        with open(args.labels_file, encoding="utf-8") as fh:
            labels = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

    images = [args.image] if args.image else [str(p) for p in iter_image_paths(args.folder)]
    if not labels or not images:
        print("need at least one label and one image", file=sys.stderr)
        return 2

    clf = MedSiglipOpenVINO(model_id=args.model, device=args.device)

    start = time.perf_counter()
    results = clf.classify(images, labels, top_k=args.top_k)
    elapsed = time.perf_counter() - start

    if args.json:
        print(json.dumps(dict(zip(images, results)), indent=2))
        return 0

    for path, scores in zip(images, results):
        print(f"\n{path}")
        for label, prob in scores.items():
            print(f"  {prob:7.2%}  {label}")
    print(f"\n[{clf.device}] {len(images)} image(s) in {elapsed:.2f}s ({elapsed/len(images):.2f}s each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())