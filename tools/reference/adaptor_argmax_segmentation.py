#!/usr/bin/env python3
"""
Reference adaptor for segmentation: per-pixel scores -> one mask per image.

The Processing TEE runs a model provider's adaptor as

    python3 adaptor.py --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/

  raw/raw_outputs.npz   `ids` (file names) + one array per model output
  predictions.csv       file,mask   (mask = path relative to predictions/)
  masks/<n>.png         one class-index mask per image (0 = background)

This one accepts a model with a single output of shape
  [N, C, H, W]  C = number of class_names: label = argmax over C
  [N, 1, H, W]  2 classes only: one logit per pixel, label = sigmoid >= 0.5
  [N, H, W]     2 classes only: same, without the channel axis
Masks are written at the model's output size; the evaluator resizes them to
the ground-truth size (nearest neighbour). Use it as-is, or as a template.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def single_output(npz):
    names = [k for k in npz.files if k != "ids"]
    if len(names) != 1:
        raise SystemExit(f"expected exactly one model output, found {names}")
    return names[0], np.asarray(npz[names[0]])


def to_masks(out, num_classes, n):
    if out.shape[0] != n:
        raise SystemExit(f"output has {out.shape[0]} rows for {n} images")
    if out.ndim == 4 and out.shape[1] == num_classes and num_classes > 1:
        return out.argmax(axis=1)
    if num_classes == 2 and (out.ndim == 3 or (out.ndim == 4 and out.shape[1] == 1)):
        logits = out.reshape(n, out.shape[-2], out.shape[-1]).astype(np.float64)
        return (logits >= 0.0).astype(np.int64)  # sigmoid(x) >= 0.5  <=>  x >= 0
    raise SystemExit(f"output shape {out.shape} does not fit {num_classes} classes; "
                     f"expected [N, {num_classes}, H, W]" + (" or [N, 1, H, W]" if num_classes == 2 else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    num_classes = len(json.loads(Path(args.spec).read_text())["class_names"])
    npz = np.load(Path(args.raw_dir) / "raw_outputs.npz", allow_pickle=False)
    ids = [str(x) for x in npz["ids"]]
    name, out = single_output(npz)
    masks = to_masks(out, num_classes, len(ids))

    out_dir = Path(args.output_dir)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "mask"])
        for i, (file_id, mask) in enumerate(zip(ids, masks)):
            rel = f"masks/{i:06d}.png"  # numbered: ids may contain folders or clash by name
            Image.fromarray(mask.astype(np.uint8), mode="L").save(out_dir / rel)
            w.writerow([file_id, rel])
    print(f"[adaptor] wrote {len(ids)} masks from output {name!r} {out.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
