#!/usr/bin/env python3
"""
Bucket evaluator: multiclass_classification.

Scores a model's predictions against a dataset's ground truth. Owned by the
platform; one script serves every multi-class dataset (ordinal ones included —
QWK is reported for all, and is meaningful when class_names are ordered).

    python3 evaluate.py --predictions P --ground-truth G --spec S --results R

  predictions.csv   the adaptor's table; flexible (see below)
  ground_truth.csv  file,label   (written by the platform; label ∈ 0..C-1)
  dataset_spec.json {"class_names": [...], ...}   (C >= 2 names, in label order)

The predictions table is read leniently:
  delimiter      , tab ; or |
  file column    file / filename / image / id / path / ... (else the first column);
                 an id may be the dataset path, the file name, or the name without
                 extension
  probabilities  one column per class, all named the same way: prob_0.. / p0.. /
                 score_0.. / logit_0.. / prob_<class name>.. / <class name>..;
                 or one column (probs / probabilities / scores / logits) holding
                 a list like [0.1, 0.9, ...]. Values outside [0,1] are read as
                 logits (softmax); rows not summing to 1 are re-normalised.
  label          label / pred / prediction / predicted_class ...: an index or a
                 class name; left out, it is the most likely class
At least a label or the probabilities are needed; without probabilities, auc
is not reported. Rows for files the ground truth does not list are ignored.

Every ground-truth row must have exactly one prediction. Exit codes:
  0   results.json written
  12  predictions invalid (missing or conflicting prediction, unreadable label
      or probability) — the model provider's fault
  other non-zero — evaluator/platform error

Metric definitions follow the previous platform evaluator for this vertical
(evaluate_model_breastcancer.py) and the dataset provider's metrics.py:
sensitivity = macro recall, ppv = macro precision, specificity / npv = mean of
the one-vs-rest per-class values. A ratio whose denominator is 0 counts as 0.
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
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

EXIT_PREDICTIONS_INVALID = 12
PROB_SUM_TOLERANCE = 1e-3


class PredictionsInvalid(Exception):
    """The adaptor's output does not satisfy the bucket contract."""


def log(msg):
    print(f"[evaluator:multiclass] {msg}", flush=True)


def read_table(path, required):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [h.strip() for h in (reader.fieldnames or [])]
        missing = [c for c in required if c not in header]
        if missing:
            raise ValueError(f"{Path(path).name}: missing column(s) {missing}; header={header}")
        return [{(k or "").strip(): (v or "").strip() for k, v in raw.items()} for raw in reader]


def parse_index(raw, num_classes):
    """A label in the platform's canonical ground truth: 0..C-1."""
    if not raw.isdigit() or int(raw) >= num_classes:
        return None
    return int(raw)


def load_ground_truth(path, num_classes):
    gt = {}
    for i, row in enumerate(read_table(path, ["file", "label"]), start=2):
        f = row["file"]
        if not f:
            raise ValueError(f"ground_truth.csv line {i}: empty file")
        if f in gt:
            raise ValueError(f"ground_truth.csv line {i}: duplicate file {f!r}")
        label = parse_index(row["label"], num_classes)
        if label is None:
            raise ValueError(f"ground_truth.csv line {i}: label must be 0..{num_classes - 1}, got {row['label']!r}")
        gt[f] = label
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


# ── multi-class predictions ────────────────────────────────────────────────────

LIST_COLUMNS = ("probs", "probabilities", "prob", "probability", "scores", "score", "logits", "outputs", "output")


def probability_columns(header, class_names, taken):
    """One column per class, all named by the same pattern, or None."""
    names = [norm(n) for n in class_names]
    patterns = [lambda i, n: f"prob_{i}", lambda i, n: f"p_{i}", lambda i, n: f"p{i}", lambda i, n: f"prob{i}",
                lambda i, n: f"probability_{i}", lambda i, n: f"score_{i}", lambda i, n: f"logit_{i}",
                lambda i, n: f"logits_{i}", lambda i, n: f"class_{i}", lambda i, n: f"prob_{n}",
                lambda i, n: f"p_{n}", lambda i, n: f"probability_{n}", lambda i, n: f"score_{n}",
                lambda i, n: f"logit_{n}", lambda i, n: n]
    partial = None
    for pattern in patterns:
        cols = [find_column(header, (pattern(i, n),), taken) for i, n in enumerate(names)]
        found = [c for c in cols if c is not None]
        if len(found) == len(cols) and len(set(cols)) == len(cols):
            return cols
        if len(found) >= 2 and partial is None:
            partial = [header[c] for c in found]
    if partial:  # e.g. prob_0..prob_2 for 4 classes: an adaptor bug, not a labels-only table
        raise PredictionsInvalid(f"probability columns {partial} cover {len(partial)} of {len(class_names)} "
                                 f"classes {class_names}")
    return None


def parse_list(raw, num_classes, line):
    text = raw.strip().strip("[]()")
    parts = [p for p in re.split(r"[,;\s]+", text) if p]
    if len(parts) != num_classes:
        raise PredictionsInvalid(f"predictions line {line}: expected {num_classes} probabilities, got {raw!r}")
    return [parse_number(p, "a probability", line) for p in parts]


def load_predictions(path, gt, class_names):
    """Return ({file: (label, probs or None)}, format info); raise
    PredictionsInvalid when the table cannot give every ground-truth file a
    prediction."""
    num_classes = len(class_names)
    header, rows = read_rows(path)
    fcol = find_column(header, FILE_COLUMNS)
    fcol = 0 if fcol is None else fcol
    lcol = find_column(header, LABEL_COLUMNS, taken=(fcol,))
    pcols = probability_columns(header, class_names, taken=(fcol, lcol))
    listcol = None if pcols else find_column(header, LIST_COLUMNS, taken=(fcol, lcol))
    if lcol is None and pcols is None and listcol is None:
        raise PredictionsInvalid(f"predictions need a label or per-class probability columns "
                                 f"(e.g. file,label,prob_0..prob_{num_classes - 1}); header={header}")

    def parse_row(row, line):
        label = probs = None
        if pcols is not None:
            probs = tuple(parse_number(row[c] if c < len(row) else "", "a probability", line) for c in pcols)
        elif listcol is not None:
            probs = tuple(parse_list(row[listcol] if listcol < len(row) else "", num_classes, line))
        if lcol is not None:
            label = parse_label(row[lcol] if lcol < len(row) else "", class_names)
            if label is None:
                raise PredictionsInvalid(f"predictions line {line}: label must be 0..{num_classes - 1} or a class "
                                         f"name {class_names}, got {row[lcol]!r}")
        return label, probs

    preds, ignored = collect(rows, fcol, gt, parse_row)
    transform = "none"
    if pcols is not None or listcol is not None:
        files = list(preds)
        p = np.array([preds[f][1] for f in files], dtype=np.float64)
        if (p < 0).any() or (p > 1).any():
            transform = "softmax"  # logits
            e = np.exp(p - p.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
        elif (np.abs(p.sum(axis=1) - 1.0) > PROB_SUM_TOLERANCE).any():
            if (p.sum(axis=1) <= 0).any():
                raise PredictionsInvalid("a row's probabilities are all 0")
            transform = "renormalised"
            p = p / p.sum(axis=1, keepdims=True)
        preds = {f: (preds[f][0], list(p[i])) for i, f in enumerate(files)}
    if lcol is None:
        preds = {f: (int(np.argmax(probs)), probs) for f, (_, probs) in preds.items()}
    prob_from = ([header[c] for c in pcols] if pcols is not None
                 else header[listcol] if listcol is not None else None)
    info = {"file_column": header[fcol], "label_column": header[lcol] if lcol is not None else None,
            "probability_columns": prob_from, "probability_transform": transform,
            "label_from": "column" if lcol is not None else "most likely class", "ignored_rows": ignored}
    return preds, info


def ratio(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def finite_or_none(x):
    x = float(x)
    return x if math.isfinite(x) else None


def compute_metrics(y_true, y_pred, y_prob, class_names):
    c = len(class_names)
    labels = np.arange(c)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_f2": float(fbeta_score(y_true, y_pred, beta=2, labels=labels, average="macro", zero_division=0)),
        "weighted_f2": float(fbeta_score(y_true, y_pred, beta=2, labels=labels, average="weighted", zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "ppv": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    try:
        m["qwk"] = finite_or_none(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        m["qwk"] = None
    m["auc"] = None
    if y_prob is not None:  # labels only: no probabilities to compute AUC from
        try:
            m["auc"] = finite_or_none(roc_auc_score(y_true, y_prob, labels=labels, multi_class="ovr", average="macro"))
        except ValueError:  # e.g. a class absent from the ground truth
            m["auc"] = None

    per_class, specs, npvs = {}, [], []
    total = int(cm.sum())
    for i in range(c):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum()) - tp
        fp = int(cm[:, i].sum()) - tp
        tn = total - tp - fn - fp
        spec, npv = ratio(tn, tn + fp), ratio(tn, tn + fn)
        specs.append(spec)
        npvs.append(npv)
        per_class[class_names[i]] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": ratio(tp, tp + fp),
            "recall": ratio(tp, tp + fn),
            "specificity": spec,
            "npv": npv,
        }
    m["specificity"] = float(np.mean(specs))
    m["npv"] = float(np.mean(npvs))
    m["confusion_matrix"] = cm.tolist()
    m["per_class"] = per_class
    return m


def main():
    ap = argparse.ArgumentParser(description="multiclass_classification bucket evaluator")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    class_names = spec.get("class_names") or []
    if len(class_names) < 2:
        raise ValueError(f"multiclass_classification needs >= 2 class_names, got {class_names}")
    num_classes = len(class_names)

    gt = load_ground_truth(args.ground_truth, num_classes)
    try:
        preds, pred_format = load_predictions(args.predictions, gt, class_names)
    except PredictionsInvalid as e:
        log(f"PREDICTIONS INVALID: {e}")
        return EXIT_PREDICTIONS_INVALID
    log(f"read predictions as {pred_format}")

    files = sorted(gt)
    y_true = np.array([gt[f] for f in files], dtype=np.int64)
    y_pred = np.array([preds[f][0] for f in files], dtype=np.int64)
    y_prob = None if pred_format["probability_columns"] is None else \
        np.array([preds[f][1] for f in files], dtype=np.float64)

    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    result = {
        "status": "success",
        "task_type": "multiclass_classification",
        "num_samples": int(len(files)),
        "class_names": class_names,
        "metrics": metrics,
        "confusion_matrix": metrics["confusion_matrix"],
        "label_distribution": {class_names[c]: int((y_true == c).sum()) for c in range(num_classes)},
        "prediction_distribution": {class_names[c]: int((y_pred == c).sum()) for c in range(num_classes)},
        "predictions_format": pred_format,
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"scored {len(files)} samples: accuracy={metrics['accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f} qwk={metrics['qwk']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
