#!/usr/bin/env python3
"""
Reference adaptor for binary_classification: one logit per sample -> predictions.csv.

The Processing TEE runs a model provider's adaptor as

    python3 adaptor.py --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/

  raw/raw_outputs.npz   `ids` (file names) + one array per model output
  predictions.csv       file,label,score   (score = P(class 1); label = score > 0.5)

This one accepts a model with a single output shaped [N], [N,1] (a logit →
sigmoid) or [N,2] (two logits → softmax, class 1). Use it as-is for models like
the OCS MobileViTV2, or as a template for your own adaptor.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

THRESHOLD = 0.5


def single_output(npz):
    names = [k for k in npz.files if k != "ids"]
    if len(names) != 1:
        raise SystemExit(f"expected exactly one model output, found {names}")
    return names[0], np.asarray(npz[names[0]], dtype=np.float64)


def positive_probability(raw):
    if raw.ndim == 2 and raw.shape[1] == 2:
        z = raw - raw.max(axis=1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=1, keepdims=True))[:, 1]
    if raw.ndim == 1 or (raw.ndim == 2 and raw.shape[1] == 1):
        return 1.0 / (1.0 + np.exp(-raw.reshape(-1)))
    raise SystemExit(f"unsupported output shape {raw.shape}: expected [N], [N,1] or [N,2]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    npz = np.load(Path(args.raw_dir) / "raw_outputs.npz", allow_pickle=False)
    ids = [str(x) for x in npz["ids"]]
    name, raw = single_output(npz)
    if raw.shape[0] != len(ids):
        raise SystemExit(f"output {name!r} has {raw.shape[0]} rows for {len(ids)} ids")
    score = positive_probability(raw)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "score"])
        for file_id, s in zip(ids, score):
            w.writerow([file_id, int(s > THRESHOLD), repr(float(s))])
    print(f"[adaptor] wrote {len(ids)} predictions from output {name!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
