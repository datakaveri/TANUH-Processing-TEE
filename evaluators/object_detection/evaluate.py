#!/usr/bin/env python3
"""
Bucket evaluator: object_detection (2D axis-aligned bounding boxes).

Scores a model's predicted boxes against a dataset's ground-truth boxes. Owned
by the platform; one script serves every detection dataset. It never touches
the model or the raw images — only two tables.

    python3 evaluate.py --predictions P --ground-truth G --spec S --results R

  predictions.csv   the adaptor's table: one row per predicted box (flexible, see below)
  ground_truth.csv  file,x_min,y_min,x_max,y_max,class   (written by the platform)
                    one row per box, in the original image's pixels; class is the
                    0-based index into class_names; an image with no objects has
                    one row with the box fields empty
  dataset_spec.json {"class_names": [...], ...}   (at least 1 class; no background)

The predictions table is read leniently:
  delimiter   , tab ; or |
  file column file / filename / image / id / ... (else the first column); an id
              may be the dataset path, the file name, or the name without extension
  box         x_min,y_min,x_max,y_max (also xmin / x1 / left, ...) or x,y,w,h
              (also x_min,y_min,width,height); original-image pixels
  score       score / confidence / conf / prob / probability; left out, every box
              scores 1 (then AP cannot rank the boxes)
  class       class / label / category / class_id / category_id: a 0-based index or
              a class name; left out, allowed only with a single class
A row with empty box fields means "no detections" for that file. Images the
adaptor does not mention have no detections — that is valid, it just scores 0
for their objects. Rows for files the ground truth does not list are ignored.

Metrics (all in [0, 1]), COCO-style: 101-point interpolated precision, at most
100 detections per image and class, greedy matching by score; averaged over the
classes that have at least one ground-truth box:
  map          AP averaged over IoU thresholds 0.50:0.05:0.95  (mAP@[.5:.95])
  ap50, ap75   AP at IoU 0.50 / 0.75
  map_giou     as map, but a detection matches a ground-truth box by Generalized
               IoU >= threshold instead of IoU
  recall_50    recall at IoU 0.50 with all (up to 100) detections
  mean_iou     for every ground-truth box, the best IoU with a same-class
               prediction in its image (0 if none), averaged over all boxes
  per_class    ap / ap50 / ap75 / recall_50 / ground-truth and predicted box counts

Exit codes:
  0   results.json written
  12  predictions invalid (unreadable number, x_max < x_min, unknown class, or no
      row matches any ground-truth file) — the model provider's fault
  other non-zero — evaluator/platform error
"""

import argparse
import csv
import io
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXIT_PREDICTIONS_INVALID = 12
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.951, 0.05), 2)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
MAX_DETS = 100


class PredictionsInvalid(Exception):
    """The adaptor's output does not satisfy the bucket contract."""


def log(msg):
    print(f"[evaluator:detection] {msg}", flush=True)


# ── reading tables leniently (same helpers as the other bucket evaluators) ──────

FILE_COLUMNS = ("file", "filename", "file_name", "filepath", "file_path", "path", "image", "image_id",
                "image_name", "image_path", "img", "id", "ids", "sample", "sample_id", "case_id", "name")
X1 = ("x_min", "xmin", "x1", "left", "x0", "bbox_x_min", "box_x_min")
Y1 = ("y_min", "ymin", "y1", "top", "y0", "bbox_y_min", "box_y_min")
X2 = ("x_max", "xmax", "x2", "right", "bbox_x_max", "box_x_max")
Y2 = ("y_max", "ymax", "y2", "bottom", "bbox_y_max", "box_y_max")
XS = ("x", "bbox_x", "box_x")
YS = ("y", "bbox_y", "box_y")
WS = ("w", "width", "bbox_w", "bbox_width", "box_w", "box_width")
HS = ("h", "height", "bbox_h", "bbox_height", "box_h", "box_height")
SCORE = ("score", "confidence", "conf", "prob", "probability", "score_1", "det_score", "objectness")
CLASS = ("class", "label", "category", "class_id", "category_id", "class_index", "label_id", "cls",
         "pred_class", "predicted_class", "class_name", "category_name")


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def read_rows(path, invalid):
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise invalid(f"{Path(path).name} is empty")
    delimiter = "\t" if str(path).endswith(".tsv") else max([",", "\t", ";", "|"], key=lines[0].count)
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text), delimiter=delimiter)
            if any(c.strip() for c in r)]
    if not rows:
        raise invalid(f"{Path(path).name} has no header")
    return rows[0], rows[1:]


def find_column(header, names, taken=()):
    normed = [norm(h) for h in header]
    for n in names:
        if n in normed and normed.index(n) not in taken:
            return normed.index(n)
    return None


def _stem(name):
    return name.rsplit(".", 1)[0] if "." in name else name


class IdMatcher:
    """Maps the ids an adaptor wrote to ground-truth files: the dataset path,
    a trailing part of it, the file name, or the name without extension."""

    def __init__(self, gt_files):
        self.exact = set(gt_files)
        self.tables = (defaultdict(list), defaultdict(list), defaultdict(list))  # suffix, base, stem
        for f in gt_files:
            parts = f.split("/")
            for i in range(1, len(parts) - 1):
                self.tables[0]["/".join(parts[i:])].append(f)
            self.tables[1][parts[-1]].append(f)
            self.tables[2][_stem(parts[-1])].append(f)

    def match(self, raw):
        pid = raw.strip().replace("\\", "/")
        while pid.startswith("./"):
            pid = pid[2:]
        pid = pid.strip("/")
        if pid in self.exact:
            return pid
        base = pid.rsplit("/", 1)[-1]
        for table, key in ((self.tables[0], pid), (self.tables[1], base), (self.tables[2], base),
                           (self.tables[2], _stem(base))):
            hits = table.get(key, [])
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise PredictionsInvalid(f"prediction id {raw!r} fits {len(hits)} files, e.g. {hits[:2]}; "
                                         f"write the id the platform gave the adaptor")
        return None


def parse_class(raw, class_names, line, invalid):
    s = raw.strip()
    if s == "" and len(class_names) == 1:
        return 0
    try:
        v = float(s)
        if v == int(v) and 0 <= int(v) < len(class_names):
            return int(v)
    except ValueError:
        names = {norm(n).replace("_", ""): i for i, n in enumerate(class_names)}
        key = norm(s).replace("_", "")
        if key in names:
            return names[key]
    raise invalid(f"line {line}: class {raw!r} is neither an index 0..{len(class_names) - 1} "
                  f"nor one of {class_names}")


def number(raw, what, line, invalid):
    try:
        v = float(raw)
    except ValueError:
        raise invalid(f"line {line}: {what} is not a number: {raw!r}")
    if not math.isfinite(v):
        raise invalid(f"line {line}: {what} is {raw!r}")
    return v


def box_columns(header, taken, invalid):
    """Column indexes for (x1, y1, x2, y2) or (x, y, w, h); mode 'xyxy' or 'xywh'."""
    cols = [find_column(header, n, taken) for n in (X1, Y1, X2, Y2)]
    if None not in cols:
        return "xyxy", cols
    x = cols[0] if cols[0] is not None else find_column(header, XS, taken)
    y = cols[1] if cols[1] is not None else find_column(header, YS, taken)
    w, h = find_column(header, WS, taken), find_column(header, HS, taken)
    if None not in (x, y, w, h):
        return "xywh", [x, y, w, h]
    raise invalid(f"need box columns x_min,y_min,x_max,y_max (or x,y,w,h); header={header}")


def read_boxes(path, class_names, invalid, need_score):
    """{file id as written: [(x1, y1, x2, y2, score, class), ...]} and format info."""
    header, rows = read_rows(path, invalid)
    fcol = find_column(header, FILE_COLUMNS)
    fcol = 0 if fcol is None else fcol
    mode, bcols = box_columns(header, (fcol,), invalid)
    taken = (fcol, *bcols)
    scol = find_column(header, SCORE, taken) if need_score else None
    ccol = find_column(header, CLASS, (*taken, scol) if scol is not None else taken)
    if ccol is None and len(class_names) != 1:
        raise invalid(f"need a class column with {len(class_names)} classes; header={header}")
    out = defaultdict(list)
    for line, row in enumerate(rows, start=2):
        cell = lambda i: row[i] if i is not None and i < len(row) else ""
        f = cell(fcol)
        if not f:
            raise invalid(f"line {line}: empty file id")
        values = [cell(i) for i in bcols]
        if all(v == "" for v in values):
            out[f]  # an image with no boxes
            continue
        a, b, c, d = (number(v, n, line, invalid) for v, n in zip(values, ("x", "y", "x2/w", "y2/h")))
        x1, y1, x2, y2 = (a, b, c, d) if mode == "xyxy" else (a, b, a + c, b + d)
        if x2 < x1 or y2 < y1:
            raise invalid(f"line {line}: box has x_max < x_min or y_max < y_min: {values}")
        score = number(cell(scol), "score", line, invalid) if scol is not None else 1.0
        cls = parse_class(cell(ccol), class_names, line, invalid) if ccol is not None else 0
        out[f].append((x1, y1, x2, y2, score, cls))
    info = {"file_column": header[fcol], "box_format": mode, "box_columns": [header[i] for i in bcols],
            "score_column": header[scol] if scol is not None else None,
            "class_column": header[ccol] if ccol is not None else None}
    return out, info


# ── matching and AP ──────────────────────────────────────────────────────────────

def pairwise_iou(a, b, generalized=False):
    """[len(a), len(b)] IoU (or GIoU) of xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, dtype=np.float64)[:, None, :], np.asarray(b, dtype=np.float64)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
    if not generalized:
        return iou
    cw = np.maximum(a[..., 2], b[..., 2]) - np.minimum(a[..., 0], b[..., 0])
    ch = np.maximum(a[..., 3], b[..., 3]) - np.minimum(a[..., 1], b[..., 1])
    hull = cw * ch
    return iou - np.where(hull > 0, (hull - union) / np.where(hull > 0, hull, 1), 0.0)


def interpolated_ap(tp, scores, num_gt):
    """COCO 101-point AP from per-detection TP flags."""
    if num_gt == 0:
        return None
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    tp = np.asarray(tp, dtype=np.float64)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / num_gt
    precision = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # envelope
    idx = np.searchsorted(recall, RECALL_POINTS, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def class_scores(gt, preds, c, generalized):
    """AP per threshold, recall@0.5, and (num_gt, num_pred) for class c."""
    num_gt = sum(sum(1 for b in boxes if b[5] == c) for boxes in gt.values())
    tps = {t: [] for t in IOU_THRESHOLDS}
    scores = []
    num_pred = 0
    for f in sorted(gt):
        g = [b[:4] for b in gt[f] if b[5] == c]
        d = sorted((b for b in preds.get(f, []) if b[5] == c), key=lambda b: -b[4])[:MAX_DETS]
        num_pred += len(d)
        if not d:
            continue
        sim = pairwise_iou([b[:4] for b in d], g, generalized)
        scores.extend(b[4] for b in d)
        for t in IOU_THRESHOLDS:
            taken = np.zeros(len(g), dtype=bool)
            for i in range(len(d)):
                best, best_j = t - 1e-12, -1  # COCO: the best unmatched box at or above t
                for j in range(len(g)):
                    if not taken[j] and sim[i, j] >= best:
                        best, best_j = sim[i, j], j
                if best_j >= 0:
                    taken[best_j] = True
                tps[t].append(1 if best_j >= 0 else 0)
    aps = {t: interpolated_ap(tps[t], scores, num_gt) for t in IOU_THRESHOLDS}
    recall50 = (sum(tps[0.5]) / num_gt) if num_gt else None
    return aps, recall50, num_gt, num_pred


def mean_best_iou(gt, preds):
    best = []
    for f, boxes in gt.items():
        for b in boxes:
            same = [p[:4] for p in preds.get(f, []) if p[5] == b[5]]
            best.append(float(pairwise_iou([b[:4]], same).max()) if same else 0.0)
    return float(np.mean(best)) if best else 0.0


def main():
    ap = argparse.ArgumentParser(description="object_detection bucket evaluator")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    class_names = spec.get("class_names") or []
    if not class_names:
        raise ValueError("object_detection needs at least 1 class name")

    gt, _ = read_boxes(args.ground_truth, class_names, ValueError, need_score=False)
    if not gt:
        raise ValueError("ground_truth.csv has no rows")
    try:
        raw, pred_format = read_boxes(args.predictions, class_names, PredictionsInvalid, need_score=True)
        matcher = IdMatcher(list(gt))
        preds, ignored = defaultdict(list), 0
        for pid, boxes in raw.items():
            f = matcher.match(pid)
            if f is None:
                ignored += 1
                continue
            preds[f].extend(boxes)
        if raw and ignored == len(raw):
            raise PredictionsInvalid(f"no prediction file id matches a ground-truth file "
                                     f"(e.g. {list(raw)[:3]}); write the ids the platform gave the adaptor")
    except PredictionsInvalid as e:
        log(f"PREDICTIONS INVALID: {e}")
        return EXIT_PREDICTIONS_INVALID
    pred_format["ignored_files"] = ignored
    pred_format["images_with_predictions"] = sum(1 for f in gt if preds.get(f))
    gt_max = max((max(b[2], b[3]) for boxes in gt.values() for b in boxes), default=0)
    pred_max = max((max(b[2], b[3]) for boxes in preds.values() for b in boxes), default=0)
    pred_format["looks_normalised"] = bool(gt_max > 2 and 0 < pred_max <= 1.0)
    if pred_format["looks_normalised"]:
        log("WARNING: predicted boxes are all within [0, 1] but ground-truth boxes are in pixels; "
            "the adaptor must write original-image pixel coordinates")

    per_class, aps, aps_giou, recalls = {}, [], [], []
    for c, name in enumerate(class_names):
        a, r50, n_gt, n_pred = class_scores(gt, preds, c, generalized=False)
        ag, _, _, _ = class_scores(gt, preds, c, generalized=True)
        per_class[name] = {"num_gt": n_gt, "num_pred": n_pred}
        if n_gt == 0:  # COCO: a class with no ground truth is left out of the means
            per_class[name].update({"ap": None, "ap50": None, "ap75": None, "recall_50": None})
            continue
        m = float(np.mean([a[t] for t in IOU_THRESHOLDS]))
        per_class[name].update({"ap": m, "ap50": a[0.5], "ap75": a[0.75], "recall_50": r50})
        aps.append((m, a[0.5], a[0.75]))
        aps_giou.append(float(np.mean([ag[t] for t in IOU_THRESHOLDS])))
        recalls.append(r50)
    if not aps:
        raise ValueError("the ground truth has no boxes of any class")
    metrics = {
        "map": float(np.mean([x[0] for x in aps])),
        "ap50": float(np.mean([x[1] for x in aps])),
        "ap75": float(np.mean([x[2] for x in aps])),
        "map_giou": float(np.mean(aps_giou)),
        "recall_50": float(np.mean(recalls)),
        "mean_iou": mean_best_iou(gt, preds),
    }
    result = {
        "status": "success",
        "task_type": "object_detection",
        "num_samples": len(gt),
        "num_gt_boxes": sum(len(b) for b in gt.values()),
        "num_pred_boxes": sum(len(b) for b in preds.values()),
        "class_names": class_names,
        "metrics": metrics,
        "per_class": per_class,
        "predictions_format": pred_format,
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"scored {len(gt)} images: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
