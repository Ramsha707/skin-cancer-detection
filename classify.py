"""CLI: zero-shot classify images with google/medsiglip-448.

Examples:
  python classify.py --image scan.jpg --labels "no pneumonia" "pneumonia"
  python classify.py --folder data/ --labels-file labels.txt --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys

from medsiglip import MedSiglipClassifier, iter_image_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="single image path or URL-free local file")
    src.add_argument("--folder", help="directory of images (searched recursively)")

    lbl = p.add_mutually_exclusive_group(required=True)
    lbl.add_argument("--labels", nargs="+", help="candidate labels")
    lbl.add_argument("--labels-file", help="file with one label per line")

    p.add_argument("--model", default="google/medsiglip-448")
    p.add_argument("--device", default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.labels:
        labels = args.labels
    else:
        with open(args.labels_file, encoding="utf-8") as fh:
            labels = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if not labels:
        print("no candidate labels given", file=sys.stderr)
        return 2

    images = [args.image] if args.image else [str(p) for p in iter_image_paths(args.folder)]
    if not images:
        print("no images found", file=sys.stderr)
        return 2

    clf = MedSiglipClassifier(model_id=args.model, device=args.device)
    results = clf.classify(images, labels, top_k=args.top_k)

    if args.json:
        print(json.dumps(dict(zip(images, results)), indent=2))
        return 0

    for path, scores in zip(images, results):
        print(f"\n{path}")
        for label, prob in scores.items():
            print(f"  {prob:7.2%}  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())