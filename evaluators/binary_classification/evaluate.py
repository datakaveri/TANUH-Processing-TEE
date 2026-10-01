#!/usr/bin/env python3
"""
Bucket evaluator: binary_classification.

Scores a model's predictions against a dataset's ground truth. Owned by the
platform; one script serves every binary-classification dataset. It never
touches the model or the raw data — only two CSV tables.

    python3 evaluate.py --predictions P --ground-truth G --spec S --results R

  predictions.csv   the adaptor's table; flexible (see below)
  ground_truth.csv  file,label         (written by the platform; label ∈ {0,1})
  dataset_spec.json {"class_names": [...], ...}   (exactly 2 class names)

The predictions table is read leniently:
  delimiter   , tab ; or |
  file column file / filename / image / id / path / ... (else the first column);
              an id may be the dataset path, the file name, or the name without
              extension
  score       score / prob / probability / prob_1 / p1 / confidence /
              prob_<class 1 name> / <class 1 name> / logit ... = P(class 1);
              values outside [0,1] are read as logits (sigmoid)
  label       label / pred / prediction / predicted_class ...: 0/1 or a class
              name; left out, it is score >= 0.5
At least a label or a score is needed; without a score, auc is not reported.
Rows for files the ground truth does not list are ignored and counted.

Every ground-truth row must have exactly one prediction, so every model is
scored on the same samples. Exit codes:
  0   results.json written
  12  predictions invalid (missing or conflicting prediction, unreadable
      label or score) — the model provider's fault
  other non-zero — evaluator/platform error

Metric definitions follow the previous platform evaluator for this vertical
(evaluate_model_OCS.py compute_metrics), plus fnr/fpr. Positive class = 1.
A ratio whose denominator is 0 is reported as 0.
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    roc_auc_score,
)

EXIT_PREDICTIONS_INVALID = 12


class PredictionsInvalid(Exception):
    """The adaptor's output does not satisfy the bucket contract."""


def log(msg):
    print(f"[evaluator:binary] {msg}", flush=True)


def read_table(path, required):
    """Read a CSV into a list of dicts, checking the header has `required`."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise ValueError(f"{Path(path).name}: missing column(s) {missing}; header={header}")
        rows = []
        for raw in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in raw.items()})
        return rows


def load_ground_truth(path):
    gt = {}
    for i, row in enumerate(read_table(path, ["file", "label"]), start=2):
        f = row["file"]
        if not f:
            raise ValueError(f"ground_truth.csv line {i}: empty file")
        if f in gt:
            raise ValueError(f"ground_truth.csv line {i}: duplicate file {f!r}")
        if row["label"] not in ("0", "1"):
            raise ValueError(f"ground_truth.csv line {i}: label must be 0 or 1, got {row['label']!r}")
        gt[f] = int(row["label"])
    if not gt:
        raise ValueError("ground_truth.csv has no rows")
    return gt


# ── reading a predictions table leniently (same block in every bucket evaluator) ──

FILE_COLUMNS = ("file", "filename", "file_name", "filepath", "file_path", "path", "image", "image_id",
                "image_name", "image_path", "img", "id", "ids", "sample", "sample_id", "case_id", "name")
LABEL_COLUMNS = ("label", "pred", "prediction", "predicted", "predicted_label", "pred_label",
                 "predicted_class", "pred_class", "class", "y_pred", "label_pred")


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def read_rows(path):
    """(header, rows) of a delimited table: , tab ; or | — whichever the header uses."""
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise PredictionsInvalid(f"{Path(path).name} is empty")
    delimiter = "\t" if str(path).endswith(".tsv") else max([",", "\t", ";", "|"], key=lines[0].count)
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text), delimiter=delimiter)
            if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise PredictionsInvalid(f"{Path(path).name} has no rows")
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


def parse_label(raw, class_names):
    """A class index (0, 1, 1.0) or a class name; None if neither."""
    s = raw.strip()
    try:
        v = float(s)
        if v == int(v) and 0 <= int(v) < len(class_names):
            return int(v)
        return None
    except ValueError:
        pass
    names = {norm(n).replace("_", ""): i for i, n in enumerate(class_names)}
    key = norm(s).replace("_", "")
    if key in names:
        return names[key]
    if len(class_names) == 2 and key in ("true", "yes", "positive", "false", "no", "negative"):
        return int(key in ("true", "yes", "positive"))
    return None


def parse_number(raw, what, line):
    try:
        v = float(raw)
    except ValueError:
        raise PredictionsInvalid(f"predictions line {line}: {what} is not a number: {raw!r}")
    if not math.isfinite(v):
        raise PredictionsInvalid(f"predictions line {line}: {what} is {raw!r}")
    return v


def collect(rows, fcol, gt, parse_row):
    """{gt file: parsed row}, the number of rows for files outside the ground
    truth, and an error for a missing or conflicting prediction."""
    matcher = IdMatcher(list(gt))
    preds, ignored = {}, 0
    for line, row in enumerate(rows, start=2):
        if fcol >= len(row):
            raise PredictionsInvalid(f"predictions line {line}: missing the file column")
        f = matcher.match(row[fcol])
        if f is None:
            ignored += 1
            continue
        value = parse_row(row, line)
        if f in preds and preds[f] != value:
            raise PredictionsInvalid(f"predictions line {line}: a second, different prediction for {row[fcol]!r}")
        preds[f] = value
    missing = [f for f in gt if f not in preds]
    if missing:
        raise PredictionsInvalid(f"{len(missing)} of {len(gt)} ground-truth file(s) have no prediction "
                                 f"(ids read from column {fcol + 1}), e.g. {missing[:3]}")
    return preds, ignored


# ── binary predictions ─────────────────────────────────────────────────────────

def score_columns(class_names):
    positive = norm(class_names[1])
    return ("score", "prob", "probability", "prob_1", "p_1", "p1", "probability_1", "score_1",
            "positive_prob", "prob_positive", "positive_score", "confidence", "pred_score", "y_score",
            "y_prob", f"prob_{positive}", f"p_{positive}", f"score_{positive}", f"probability_{positive}",
            positive, "logit", "logits", "logit_1", "output")


def load_predictions(path, gt, class_names):
    """Return ({file: (label, score or None)}, format info); raise
    PredictionsInvalid when the table cannot give every ground-truth file a
    prediction."""
    header, rows = read_rows(path)
    fcol = find_column(header, FILE_COLUMNS)
    fcol = 0 if fcol is None else fcol
    lcol = find_column(header, LABEL_COLUMNS, taken=(fcol,))
    scol = find_column(header, score_columns(class_names), taken=(fcol, lcol))
    if lcol is None and scol is None:
        raise PredictionsInvalid(f"predictions need a label or a score column (e.g. file,label,score); header={header}")

    def parse_row(row, line):
        label = score = None
        if scol is not None:
            score = parse_number(row[scol] if scol < len(row) else "", "score", line)
        if lcol is not None:
            label = parse_label(row[lcol] if lcol < len(row) else "", class_names)
            if label is None:
                raise PredictionsInvalid(f"predictions line {line}: label must be 0/1 or a class name "
                                         f"{class_names}, got {row[lcol]!r}")
        return label, score

    preds, ignored = collect(rows, fcol, gt, parse_row)
    transform = "none"
    if scol is not None and any(not 0.0 <= s <= 1.0 for _, s in preds.values()):
        transform = "sigmoid"  # logits
        preds = {f: (l, 1.0 / (1.0 + math.exp(-s))) for f, (l, s) in preds.items()}
    if lcol is None:
        preds = {f: (int(s >= 0.5), s) for f, (_, s) in preds.items()}
    info = {"file_column": header[fcol], "label_column": header[lcol] if lcol is not None else None,
            "score_column": header[scol] if scol is not None else None, "score_transform": transform,
            "label_from": "column" if lcol is not None else "score >= 0.5", "ignored_rows": ignored}
    return preds, info


def ratio(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def compute_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    auc = None
    if y_score is not None:  # no score column: labels only, no ranking to compute AUC from
        try:
            auc = finite_or_none(roc_auc_score(y_true, y_score))
        except ValueError:  # only one class present in the ground truth
            auc = None
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "fnr": ratio(fn, fn + tp),
        "fpr": ratio(fp, fp + tn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="binary_classification bucket evaluator")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    class_names = spec.get("class_names") or []
    if len(class_names) != 2:
        raise ValueError(f"binary_classification needs exactly 2 class_names, got {class_names}")

    gt = load_ground_truth(args.ground_truth)
    try:
        preds, pred_format = load_predictions(args.predictions, gt, class_names)
    except PredictionsInvalid as e:
        log(f"PREDICTIONS INVALID: {e}")
        return EXIT_PREDICTIONS_INVALID
    log(f"read predictions as {pred_format}")

    files = sorted(gt)  # fixed order: scores never depend on the adaptor's row order
    y_true = np.array([gt[f] for f in files], dtype=np.int64)
    y_pred = np.array([preds[f][0] for f in files], dtype=np.int64)
    y_score = None if pred_format["score_column"] is None else \
        np.array([preds[f][1] for f in files], dtype=np.float64)

    metrics = compute_metrics(y_true, y_pred, y_score)
    result = {
        "status": "success",
        "task_type": "binary_classification",
        "num_samples": int(len(files)),
        "class_names": class_names,
        "metrics": metrics,
        "confusion_matrix": metrics["confusion_matrix"],
        "label_distribution": {class_names[c]: int((y_true == c).sum()) for c in (0, 1)},
        "prediction_distribution": {class_names[c]: int((y_pred == c).sum()) for c in (0, 1)},
        "predictions_format": pred_format,
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"scored {len(files)} samples: " + ", ".join(
        f"{k}={metrics[k]:.4f}" for k in ("accuracy", "sensitivity", "specificity", "ppv", "npv", "f2")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
