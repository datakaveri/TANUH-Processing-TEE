#!/usr/bin/env python3
"""
Reference adaptor for multiclass_classification: C logits per sample -> predictions.csv.

The Processing TEE runs a model provider's adaptor as

    python3 adaptor.py --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/

  raw/raw_outputs.npz   `ids` (file names) + one array per model output
  predictions.csv       file,label,prob_0,...,prob_{C-1}   (softmax; label = argmax)

This one accepts a model with a single [N, C] output of logits, where C must
equal the number of class_names in the dataset spec. Use it as-is for models
like the BCD ResNet34, or as a template for your own adaptor.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def single_output(npz):
    names = [k for k in npz.files if k != "ids"]
    if len(names) != 1:
        raise SystemExit(f"expected exactly one model output, found {names}")
    return names[0], np.asarray(npz[names[0]], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    num_classes = len(json.loads(Path(args.spec).read_text())["class_names"])
    npz = np.load(Path(args.raw_dir) / "raw_outputs.npz", allow_pickle=False)
    ids = [str(x) for x in npz["ids"]]
    name, logits = single_output(npz)
    if logits.ndim != 2 or logits.shape != (len(ids), num_classes):
        raise SystemExit(f"output {name!r} has shape {logits.shape}, expected ({len(ids)}, {num_classes})")

    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = probs.argmax(axis=1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label"] + [f"prob_{c}" for c in range(num_classes)])
        for file_id, lab, p in zip(ids, labels, probs):
            w.writerow([file_id, int(lab)] + [repr(float(v)) for v in p])
    print(f"[adaptor] wrote {len(ids)} predictions from output {name!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
