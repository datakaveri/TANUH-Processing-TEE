#!/usr/bin/env python3
"""
Bucket evaluator: segmentation (semantic, 2D, one mask per input).

Scores a model's predicted masks against a dataset's ground-truth masks. Owned
by the platform; one script serves every segmentation dataset. It never
touches the model or the raw images — only two tables and the mask files they
point to.

    python3 evaluate.py --predictions P --ground-truth G --spec S --results R

  predictions.csv   the adaptor's table: file,mask        (flexible, see below)
  ground_truth.csv  file,mask   (written by the platform; mask = path of the
                    ground-truth mask, absolute or relative to this file)
  dataset_spec.json {"class_names": [...], ...}
                    class_names[0] is the background; at least 2 names.
                    Optional: "ignore_index" (a mask value excluded from
                    scoring, e.g. 255) and "empty_score" (see below).

A mask holds one class index per pixel (0 = background). Accepted files:
  .png / .tif / .tiff / .bmp   single-channel, palette ("P") or bilevel images;
                               an RGB image is accepted only if R = G = B
  .npy / .npz                  a 2D integer array; or, for predictions only,
                               a [C, H, W] score array (argmax is taken) or,
                               with 2 classes, a 2D probability array (>= 0.5)
With 2 classes, a 0/255 mask is read as 0/1 (the usual binary-mask export).

The predictions table is read leniently, like the classification evaluators:
  delimiter   , tab ; or |
  file column file / filename / image / id / ... (else the first column); an id
              may be the dataset path, the file name, or the name without
              extension
  mask column mask / mask_path / pred_mask / prediction / segmentation / ...
              (else the second column); paths are relative to predictions.csv
A predicted mask whose size differs from the ground truth (models usually run
at a fixed input size) is resized to the ground-truth size with
nearest-neighbour sampling, and counted in results.json.

Metrics (all in [0, 1]; averages are over the foreground classes 1..C-1):
  dice, iou            mean per-image Dice / IoU (the usual medical-imaging
                       reading: every image counts equally)
  global_dice,         Dice / IoU from pixel counts summed over the dataset
  global_iou           (large structures weigh more)
  sensitivity, ppv,    pixel recall / precision / specificity, from summed
  specificity          counts
  pixel_accuracy       correct pixels / scored pixels, all classes
  per_class            the same per foreground class, by class name
An image where a class is absent from both masks has no Dice for that class;
"empty_score" decides it: 1 (default: a correct "nothing here") or "skip"
(left out of that class's per-image mean).

Every ground-truth row must have exactly one prediction, so every model is
scored on the same samples. Exit codes:
  0   results.json written
  12  predictions invalid (missing, conflicting or unreadable mask, a class
      index outside class_names) — the model provider's fault
  other non-zero — evaluator/platform error (incl. a bad ground-truth mask)
"""

import argparse
import csv
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

EXIT_PREDICTIONS_INVALID = 12
IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".bmp", ".gif"}
ARRAY_SUFFIXES = {".npy", ".npz"}


class PredictionsInvalid(Exception):
    """The adaptor's output does not satisfy the bucket contract."""


class MaskError(Exception):
    """A mask file cannot be read as a class-index mask."""


def drop_channel_axes(arr):
    """Remove singleton channel axes ([1,H,W], [H,W,1], [1,1,H,W]) but never a
    height or width of 1, so a 1-pixel-high mask stays 2D."""
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def log(msg):
    print(f"[evaluator:segmentation] {msg}", flush=True)


# ── reading tables leniently (same block as the classification evaluators) ──────

FILE_COLUMNS = ("file", "filename", "file_name", "filepath", "file_path", "image", "image_id",
                "image_name", "image_path", "img", "id", "ids", "sample", "sample_id", "case_id", "name")
MASK_COLUMNS = ("mask", "mask_path", "mask_file", "pred_mask", "predicted_mask", "prediction",
                "pred", "segmentation", "seg", "output", "label", "annotation", "gt", "path")


def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def read_rows(path, invalid=PredictionsInvalid):
    """(header, rows) of a delimited table: , tab ; or | — whichever the header uses."""
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    if not lines:
        raise invalid(f"{Path(path).name} is empty")
    delimiter = "\t" if str(path).endswith(".tsv") else max([",", "\t", ";", "|"], key=lines[0].count)
    rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text), delimiter=delimiter)
            if any(c.strip() for c in r)]
    if len(rows) < 2:
        raise invalid(f"{Path(path).name} has no rows")
    return rows[0], rows[1:]


def find_column(header, names, taken=()):
    normed = [norm(h) for h in header]
    for n in names:
        if n in normed and normed.index(n) not in taken:
            return normed.index(n)
    return None


def file_and_mask_columns(header, invalid):
    fcol = find_column(header, FILE_COLUMNS)
    fcol = 0 if fcol is None else fcol
    mcol = find_column(header, MASK_COLUMNS, taken=(fcol,))
    if mcol is None:
        others = [i for i in range(len(header)) if i != fcol]
        if not others:
            raise invalid(f"need a file column and a mask column (e.g. file,mask); header={header}")
        mcol = others[0]
    return fcol, mcol


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


def resolve(path_text, base_dir):
    p = Path(path_text.strip())
    return p if p.is_absolute() else (base_dir / p)


# ── masks ────────────────────────────────────────────────────────────────────────

def _load_array(path, allow_scores, num_classes):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            if len(z.files) != 1:
                raise MaskError(f"{path.name} holds {len(z.files)} arrays; expected one")
            arr = z[z.files[0]]
    else:
        arr = np.load(path, allow_pickle=False)
    arr = drop_channel_axes(np.asarray(arr))
    if arr.ndim == 3 and allow_scores and arr.shape[0] == num_classes:
        return arr.argmax(axis=0)  # [C, H, W] scores
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        if allow_scores and num_classes == 2 and finite.size and finite.min() >= 0 and finite.max() <= 1 \
                and not np.all(np.equal(np.mod(finite, 1), 0)):
            return (arr >= 0.5).astype(np.int64)  # binary probability map
        if not np.all(np.isfinite(arr)) or not np.all(np.equal(np.mod(arr, 1), 0)):
            raise MaskError(f"{path.name}: float values that are not class indices")
        arr = arr.astype(np.int64)
    if arr.ndim != 2:
        raise MaskError(f"{path.name}: shape {arr.shape}; expected a 2D mask"
                        + (f" or [{num_classes}, H, W] scores" if allow_scores else ""))
    return arr


def _load_image(path):
    try:
        with Image.open(path) as im:
            im.seek(0)
            if im.mode == "P":
                arr = np.array(im)  # palette images hold class indices directly
            elif im.mode in ("RGB", "RGBA"):
                rgb = np.array(im.convert("RGB"))
                if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2])):
                    raise MaskError(f"{path.name}: a colour image; save masks as single-channel class indices")
                arr = rgb[..., 0]
            elif im.mode == "1":
                arr = np.array(im, dtype=np.uint8)
            else:
                arr = np.array(im)
    except MaskError:
        raise
    except Exception as e:  # unreadable or not an image
        raise MaskError(f"{path.name}: cannot read as an image ({e})")
    arr = drop_channel_axes(arr)
    if arr.ndim != 2:
        raise MaskError(f"{path.name}: shape {arr.shape}; expected a single-channel mask")
    return arr.astype(np.int64)


def read_mask(path, num_classes, ignore_index, allow_scores):
    """A 2D int64 class-index mask; raises MaskError."""
    if not path.is_file():
        raise MaskError(f"{path} not found")
    suffix = path.suffix.lower()
    if suffix in ARRAY_SUFFIXES:
        arr = _load_array(path, allow_scores, num_classes)
    elif suffix in IMAGE_SUFFIXES:
        arr = _load_image(path)
    else:
        raise MaskError(f"{path.name}: unsupported mask type {suffix or '(none)'}; "
                        f"use PNG/TIFF or .npy")
    arr = arr.astype(np.int64)
    if num_classes == 2 and ignore_index != 255:
        values = np.unique(arr)
        if values.size and set(values.tolist()) <= {0, 255} and 255 in values:
            arr = (arr == 255).astype(np.int64)  # a 0/255 binary export
    valid = arr if ignore_index is None else arr[arr != ignore_index]
    if valid.size and (valid.min() < 0 or valid.max() >= num_classes):
        bad = sorted(set(np.unique(valid).tolist()) - set(range(num_classes)))[:5]
        raise MaskError(f"{path.name}: class index {bad} outside 0..{num_classes - 1} "
                        f"(class_names has {num_classes})")
    return arr


def resize_nearest(mask, shape):
    h, w = shape
    if mask.max(initial=0) < 256 and mask.min(initial=0) >= 0:
        im = Image.fromarray(mask.astype(np.uint8), mode="L")
    else:
        im = Image.fromarray(mask.astype(np.int32), mode="I")
    return np.array(im.resize((w, h), resample=Image.NEAREST)).astype(np.int64)


# ── loading both sides ───────────────────────────────────────────────────────────

def load_ground_truth(path):
    header, rows = read_rows(path, invalid=ValueError)
    fcol, mcol = file_and_mask_columns(header, ValueError)
    base = Path(path).resolve().parent
    gt = {}
    for line, row in enumerate(rows, start=2):
        f = row[fcol] if fcol < len(row) else ""
        m = row[mcol] if mcol < len(row) else ""
        if not f or not m:
            raise ValueError(f"ground_truth.csv line {line}: empty file or mask")
        if f in gt:
            raise ValueError(f"ground_truth.csv line {line}: duplicate file {f!r}")
        gt[f] = resolve(m, base)
    return gt


def load_predictions(path, gt):
    """({gt file: predicted mask path}, format info)."""
    header, rows = read_rows(path)
    fcol, mcol = file_and_mask_columns(header, PredictionsInvalid)
    base = Path(path).resolve().parent
    matcher = IdMatcher(list(gt))
    preds, ignored = {}, 0
    for line, row in enumerate(rows, start=2):
        if fcol >= len(row) or mcol >= len(row) or not row[mcol]:
            raise PredictionsInvalid(f"predictions line {line}: needs a file and a mask")
        f = matcher.match(row[fcol])
        if f is None:
            ignored += 1
            continue
        mask_path = resolve(row[mcol], base)
        if f in preds and preds[f] != mask_path:
            raise PredictionsInvalid(f"predictions line {line}: a second, different mask for {row[fcol]!r}")
        preds[f] = mask_path
    missing = [f for f in gt if f not in preds]
    if missing:
        raise PredictionsInvalid(f"{len(missing)} of {len(gt)} ground-truth file(s) have no predicted mask "
                                 f"(ids read from column {fcol + 1}), e.g. {missing[:3]}")
    info = {"file_column": header[fcol], "mask_column": header[mcol], "ignored_rows": ignored}
    return preds, info


# ── scoring ──────────────────────────────────────────────────────────────────────

def ratio(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def score(gt, preds, class_names, ignore_index, empty_score):
    C = len(class_names)
    fg = range(1, C)
    tp = np.zeros(C, dtype=np.int64)
    fp = np.zeros(C, dtype=np.int64)
    fn = np.zeros(C, dtype=np.int64)
    scored_pixels = correct_pixels = 0
    per_image = {c: {"dice": [], "iou": []} for c in fg}
    empty = {c: 0 for c in fg}
    present = {c: 0 for c in fg}
    resized = 0

    for f in sorted(gt):  # fixed order: scores never depend on the adaptor's row order
        try:
            truth = read_mask(gt[f], C, ignore_index, allow_scores=False)
        except MaskError as e:
            raise ValueError(f"ground-truth mask for {f!r}: {e}")
        try:
            pred = read_mask(preds[f], C, None, allow_scores=True)
        except MaskError as e:
            raise PredictionsInvalid(f"predicted mask for {f!r}: {e}")
        if pred.shape != truth.shape:
            pred = resize_nearest(pred, truth.shape)
            resized += 1
        valid = np.ones(truth.shape, dtype=bool) if ignore_index is None else truth != ignore_index
        t, p = truth[valid], pred[valid]
        scored_pixels += int(t.size)
        correct_pixels += int((t == p).sum())
        for c in fg:
            tc, pc = t == c, p == c
            a, b, d = int((tc & pc).sum()), int((~tc & pc).sum()), int((tc & ~pc).sum())
            tp[c] += a
            fp[c] += b
            fn[c] += d
            if tc.any():
                present[c] += 1
            if a + b + d == 0:  # class absent from both masks
                empty[c] += 1
                if empty_score == "skip":
                    continue
                per_image[c]["dice"].append(1.0)
                per_image[c]["iou"].append(1.0)
                continue
            per_image[c]["dice"].append(2.0 * a / (2 * a + b + d))
            per_image[c]["iou"].append(a / (a + b + d))

    per_class = {}
    for c in fg:
        tn = scored_pixels - tp[c] - fp[c] - fn[c]
        per_class[class_names[c]] = {
            "dice": float(np.mean(per_image[c]["dice"])) if per_image[c]["dice"] else 0.0,
            "iou": float(np.mean(per_image[c]["iou"])) if per_image[c]["iou"] else 0.0,
            "global_dice": ratio(2 * tp[c], 2 * tp[c] + fp[c] + fn[c]),
            "global_iou": ratio(tp[c], tp[c] + fp[c] + fn[c]),
            "sensitivity": ratio(tp[c], tp[c] + fn[c]),
            "ppv": ratio(tp[c], tp[c] + fp[c]),
            "specificity": ratio(tn, tn + fp[c]),
            "images_with_class": present[c],
            "images_empty_in_both": empty[c],
        }

    def mean_of(key):
        return float(np.mean([per_class[class_names[c]][key] for c in fg]))

    metrics = {k: mean_of(k) for k in ("dice", "iou", "global_dice", "global_iou",
                                        "sensitivity", "ppv", "specificity")}
    metrics["pixel_accuracy"] = ratio(correct_pixels, scored_pixels)
    return metrics, per_class, resized


def main():
    ap = argparse.ArgumentParser(description="segmentation bucket evaluator")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    class_names = spec.get("class_names") or []
    if len(class_names) < 2:
        raise ValueError(f"segmentation needs at least 2 class_names (background first), got {class_names}")
    ignore_index = spec.get("ignore_index")
    if ignore_index is not None:
        ignore_index = int(ignore_index)
    empty_score = spec.get("empty_score", 1)
    if empty_score not in (1, 1.0, "skip"):
        raise ValueError(f"empty_score must be 1 or \"skip\", got {empty_score!r}")

    gt = load_ground_truth(args.ground_truth)
    try:
        preds, pred_format = load_predictions(args.predictions, gt)
        metrics, per_class, resized = score(gt, preds, class_names, ignore_index, empty_score)
    except PredictionsInvalid as e:
        log(f"PREDICTIONS INVALID: {e}")
        return EXIT_PREDICTIONS_INVALID
    pred_format["resized_to_ground_truth"] = resized
    log(f"read predictions as {pred_format}")

    result = {
        "status": "success",
        "task_type": "segmentation",
        "num_samples": int(len(gt)),
        "class_names": class_names,
        "metrics": metrics,
        "per_class": per_class,
        "ignore_index": ignore_index,
        "empty_score": empty_score,
        "predictions_format": pred_format,
    }
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"scored {len(gt)} samples: " + ", ".join(
        f"{k}={metrics[k]:.4f}" for k in ("dice", "iou", "global_dice", "pixel_accuracy")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
