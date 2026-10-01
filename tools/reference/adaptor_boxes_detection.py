#!/usr/bin/env python3
"""
Reference adaptor for object_detection: per-image detector outputs -> box rows.

The Processing TEE runs a detection job's inference with --per-sample, then the
model provider's adaptor as

    python3 adaptor.py --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/

  raw/raw_outputs.npz   ids, orig_hw [N,2], input_hw [2], and per model output
                        <name> / <name>__shape / <name>__offset (see per_image())
  predictions.csv       file,x_min,y_min,x_max,y_max,score,class   (original-image pixels)

This one accepts a model that already applies NMS and returns, per image, either
  * one output of shape [K, 6] or [1, K, 6]: x_min, y_min, x_max, y_max, score, class
  * three outputs named like boxes [K, 4] (xyxy), scores [K] and labels [K]
    (torchvision-style)
with boxes in the model's input pixels (or normalised to [0, 1]) and class as a
0-based index into class_names. Raw YOLO-style outputs without NMS need their own
decoding: use this file as a template.

LABEL_OFFSET: set to 1 for models whose labels count a background class as 0
(torchvision detection models do), so label 1 becomes class_names[0].
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

LABEL_OFFSET = 0
MAX_PER_IMAGE = 100


def per_image(npz, name, i):
    """Input i's output, exactly as the model returned it for a batch of one."""
    off = npz[name + "__offset"]
    return npz[name][off[i]:off[i + 1]].reshape(npz[name + "__shape"][i])


def detections(npz, outputs, i):
    """(boxes [K,4], scores [K], classes [K]) for input i."""
    if len(outputs) == 1:
        arr = per_image(npz, outputs[0], i)
        arr = arr.reshape(-1, arr.shape[-1]) if arr.size else np.zeros((0, 6))
        if arr.shape[-1] != 6:
            raise SystemExit(f"output {outputs[0]!r} has {arr.shape[-1]} values per box; expected 6 "
                             f"(x_min, y_min, x_max, y_max, score, class)")
        return arr[:, :4], arr[:, 4], arr[:, 5]
    pick = lambda *keys: next((o for o in outputs if any(k in o.lower() for k in keys)), None)
    b, s, c = pick("box"), pick("score", "conf"), pick("label", "class")
    if None in (b, s, c):
        raise SystemExit(f"cannot tell boxes / scores / labels apart in outputs {outputs}")
    boxes = per_image(npz, b, i).reshape(-1, 4)
    return boxes, per_image(npz, s, i).reshape(-1), per_image(npz, c, i).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    num_classes = len(json.loads(Path(args.spec).read_text())["class_names"])
    npz = np.load(Path(args.raw_dir) / "raw_outputs.npz", allow_pickle=False)
    ids = [str(x) for x in npz["ids"]]
    orig_hw = npz["orig_hw"]
    input_hw = npz["input_hw"] if "input_hw" in npz.files else None
    reserved = {"ids", "orig_hw", "input_hw"}
    outputs = [k for k in npz.files if k not in reserved and not k.endswith(("__shape", "__offset"))]
    if not outputs:
        raise SystemExit("no model outputs in raw_outputs.npz (was inference run with --per-sample?)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "x_min", "y_min", "x_max", "y_max", "score", "class"])
        for i, file_id in enumerate(ids):
            boxes, scores, classes = detections(npz, outputs, i)
            boxes = boxes.astype(np.float64)
            classes = classes.astype(np.int64) - LABEL_OFFSET
            keep = (classes >= 0) & (classes < num_classes)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
            order = np.argsort(-scores, kind="mergesort")[:MAX_PER_IMAGE]
            boxes, scores, classes = boxes[order], scores[order], classes[order]
            h0, w0 = (int(v) for v in orig_hw[i])
            if input_hw is not None:
                hin, win = (int(v) for v in input_hw)
                if boxes.size and boxes.max() <= 1.0:  # normalised to the model input
                    boxes = boxes * [win, hin, win, hin]
                boxes = boxes * [w0 / win, h0 / hin, w0 / win, h0 / hin]  # model input -> original pixels
            boxes = np.clip(boxes, 0, [w0, h0, w0, h0])
            if len(boxes) == 0:
                w.writerow([file_id, "", "", "", "", "", ""])
                continue
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
                w.writerow([file_id, f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", f"{float(s):.6f}", int(c)])
                n_boxes += 1
    print(f"[adaptor] wrote {n_boxes} boxes for {len(ids)} images from outputs {outputs}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
