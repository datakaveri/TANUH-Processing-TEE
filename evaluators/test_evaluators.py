"""
Contract tests for the bucket evaluators (stdlib unittest; needs numpy + scikit-learn).

    python3 -m unittest evaluators/test_evaluators.py

Each evaluator is run as a subprocess exactly as the Processing TEE runs it,
so the tests pin the CLI, the exit codes and the results.json shape.
"""

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARY = HERE / "binary_classification" / "evaluate.py"
MULTI = HERE / "multiclass_classification" / "evaluate.py"


def run_eval(script, gt_rows, pred_header, pred_rows, class_names):
    d = Path(tempfile.mkdtemp())
    with open(d / "ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label"])
        w.writerows(gt_rows)
    with open(d / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(pred_header)
        w.writerows(pred_rows)
    (d / "spec.json").write_text(json.dumps({"class_names": class_names}))
    proc = subprocess.run(
        [sys.executable, str(script),
         "--predictions", str(d / "predictions.csv"),
         "--ground-truth", str(d / "ground_truth.csv"),
         "--spec", str(d / "spec.json"),
         "--results", str(d / "results.json")],
        capture_output=True, text=True)
    results = json.loads((d / "results.json").read_text()) if proc.returncode == 0 else None
    return proc.returncode, results, proc.stdout + proc.stderr


BIN_GT = [["a.jpg", 0], ["b.jpg", 1], ["c.jpg", 1], ["d.jpg", 0]]
BIN_HDR = ["file", "label", "score"]
BIN_OK = [["a.jpg", 0, 0.1], ["b.jpg", 1, 0.9], ["c.jpg", 0, 0.4], ["d.jpg", 1, 0.6]]
BIN_CLASSES = ["Non-Suspicious", "Suspicious"]


class BinaryEvaluator(unittest.TestCase):
    def test_happy_path(self):
        code, res, out = run_eval(BINARY, BIN_GT, BIN_HDR, BIN_OK, BIN_CLASSES)
        self.assertEqual(code, 0, out)
        m = res["metrics"]
        self.assertEqual((m["tp"], m["tn"], m["fp"], m["fn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(m["sensitivity"], 0.5)
        self.assertAlmostEqual(m["fnr"], 0.5)
        self.assertAlmostEqual(m["fpr"], 0.5)
        self.assertEqual(res["num_samples"], 4)
        self.assertEqual(m["confusion_matrix"], [[1, 1], [1, 1]])

    def test_row_order_does_not_matter(self):
        _, a, _ = run_eval(BINARY, BIN_GT, BIN_HDR, BIN_OK, BIN_CLASSES)
        _, b, _ = run_eval(BINARY, BIN_GT, BIN_HDR, list(reversed(BIN_OK)), BIN_CLASSES)
        self.assertEqual(a["metrics"], b["metrics"])

    def test_single_class_ground_truth_gives_null_auc(self):
        gt = [["a.jpg", 1], ["b.jpg", 1]]
        preds = [["a.jpg", 1, 0.9], ["b.jpg", 0, 0.2]]
        code, res, out = run_eval(BINARY, gt, BIN_HDR, preds, BIN_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertIsNone(res["metrics"]["auc"])
        self.assertEqual(res["metrics"]["specificity"], 0.0)  # 0/0 counts as 0

    def test_invalid_predictions_exit_12(self):
        cases = {
            "missing row": BIN_OK[:3],
            "conflicting duplicate": BIN_OK + [["a.jpg", 1, 0.9]],
            "label 2": [["a.jpg", 2, 0.1]] + BIN_OK[1:],
            "score nan": [["a.jpg", 0, "nan"]] + BIN_OK[1:],
            "score text": [["a.jpg", 0, "high"]] + BIN_OK[1:],
            "ids nobody recognises": [["x" + r[0], *r[1:]] for r in BIN_OK],
        }
        for name, rows in cases.items():
            with self.subTest(name):
                code, _, out = run_eval(BINARY, BIN_GT, BIN_HDR, rows, BIN_CLASSES)
                self.assertEqual(code, 12, out)
        code, _, out = run_eval(BINARY, BIN_GT, ["file", "note"], [[r[0], "x"] for r in BIN_OK], BIN_CLASSES)
        self.assertEqual(code, 12, out)  # neither a label nor a score

    def test_flexible_columns_and_ids(self):
        want = run_eval(BINARY, BIN_GT, BIN_HDR, BIN_OK, BIN_CLASSES)[1]["metrics"]
        variants = {
            "other names": (["image_id", "prediction", "probability"], BIN_OK),
            "class-name score": (["filename", "pred", "prob_Suspicious"], BIN_OK),
            "columns reordered": (["score", "label", "file"], [[r[2], r[1], r[0]] for r in BIN_OK]),
            "ids without extension": (BIN_HDR, [[r[0][:-4], *r[1:]] for r in BIN_OK]),
            "labels as names": (BIN_HDR, [[r[0], BIN_CLASSES[r[1]], r[2]] for r in BIN_OK]),
            "extra columns + unknown row": (["file", "label", "score", "note"],
                                            [r + ["x"] for r in BIN_OK] + [["zzz.jpg", 0, 0.1, "x"]]),
            "identical duplicate": (BIN_HDR, BIN_OK + [BIN_OK[0]]),
        }
        for name, (hdr, rows) in variants.items():
            with self.subTest(name):
                code, res, out = run_eval(BINARY, BIN_GT, hdr, rows, BIN_CLASSES)
                self.assertEqual(code, 0, out)
                self.assertEqual(res["metrics"], want)
        _, res, _ = run_eval(BINARY, BIN_GT, BIN_HDR, BIN_OK + [["zzz.jpg", 0, 0.1]], BIN_CLASSES)
        self.assertEqual(res["predictions_format"]["ignored_rows"], 1)

    def test_label_derived_from_score(self):
        rows = [[r[0], r[2]] for r in BIN_OK]  # file,score: labels = score >= 0.5
        code, res, out = run_eval(BINARY, BIN_GT, ["file", "score"], rows, BIN_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertEqual(res["prediction_distribution"], {"Non-Suspicious": 2, "Suspicious": 2})
        self.assertEqual(res["predictions_format"]["label_from"], "score >= 0.5")

    def test_logits_become_probabilities(self):
        rows = [[r[0], [-2.2, 2.2, -0.4, 0.4][i]] for i, r in enumerate(BIN_OK)]
        code, res, out = run_eval(BINARY, BIN_GT, ["file", "logit"], rows, BIN_CLASSES)
        self.assertEqual(code, 0, out)
        want = run_eval(BINARY, BIN_GT, BIN_HDR, BIN_OK, BIN_CLASSES)[1]["metrics"]
        self.assertEqual(res["metrics"], want)  # same ranking and the same 0.5 threshold
        self.assertEqual(res["predictions_format"]["score_transform"], "sigmoid")

    def test_labels_only_omits_auc(self):
        code, res, out = run_eval(BINARY, BIN_GT, ["file", "label"], [r[:2] for r in BIN_OK], BIN_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertIsNone(res["metrics"]["auc"])
        self.assertAlmostEqual(res["metrics"]["accuracy"], 0.5)

    def test_semicolon_delimiter(self):
        d = Path(tempfile.mkdtemp())
        (d / "gt.csv").write_text("file,label\n" + "".join(f"{f},{l}\n" for f, l in BIN_GT))
        (d / "p.csv").write_text("file;label;score\n" + "".join(f"{f};{l};{s}\n" for f, l, s in BIN_OK))
        (d / "spec.json").write_text(json.dumps({"class_names": BIN_CLASSES}))
        proc = subprocess.run([sys.executable, str(BINARY), "--predictions", str(d / "p.csv"),
                               "--ground-truth", str(d / "gt.csv"), "--spec", str(d / "spec.json"),
                               "--results", str(d / "r.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

    def test_bad_spec_is_not_blamed_on_the_model(self):
        code, _, _ = run_eval(BINARY, BIN_GT, BIN_HDR, BIN_OK, ["only-one"])
        self.assertNotIn(code, (0, 12))


MC_CLASSES = ["A", "B", "C", "D"]
MC_HDR = ["file", "label", "prob_0", "prob_1", "prob_2", "prob_3"]
MC_GT = [["s1.dcm", 0], ["s2.dcm", 1], ["s3.dcm", 2], ["s4.dcm", 3], ["s5.dcm", 3]]
MC_OK = [
    ["s1.dcm", 0, 0.7, 0.1, 0.1, 0.1],
    ["s2.dcm", 1, 0.1, 0.7, 0.1, 0.1],
    ["s3.dcm", 3, 0.1, 0.1, 0.3, 0.5],
    ["s4.dcm", 3, 0.1, 0.1, 0.1, 0.7],
    ["s5.dcm", 2, 0.1, 0.1, 0.5, 0.3],
]


class MulticlassEvaluator(unittest.TestCase):
    def test_happy_path(self):
        code, res, out = run_eval(MULTI, MC_GT, MC_HDR, MC_OK, MC_CLASSES)
        self.assertEqual(code, 0, out)
        m = res["metrics"]
        self.assertAlmostEqual(m["accuracy"], 3 / 5)
        for key in ("macro_f1", "weighted_f1", "macro_f2", "weighted_f2", "sensitivity",
                    "specificity", "ppv", "npv", "qwk", "auc"):
            self.assertIn(key, m)
        self.assertEqual(len(m["confusion_matrix"]), 4)
        self.assertEqual(set(m["per_class"]), set(MC_CLASSES))

    def test_qwk_can_be_negative(self):
        gt = [["a", 0], ["b", 1], ["c", 2], ["d", 3]]
        rev = [["a", 3, 0, 0, 0, 1], ["b", 2, 0, 0, 1, 0], ["c", 1, 0, 1, 0, 0], ["d", 0, 1, 0, 0, 0]]
        code, res, out = run_eval(MULTI, gt, MC_HDR, rev, MC_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertLess(res["metrics"]["qwk"], 0)

    def test_absent_class_gives_null_auc(self):
        gt = [["a", 0], ["b", 1], ["c", 2]]  # class D never appears
        preds = [["a", 0, 0.7, 0.1, 0.1, 0.1], ["b", 1, 0.1, 0.7, 0.1, 0.1], ["c", 2, 0.1, 0.1, 0.7, 0.1]]
        code, res, out = run_eval(MULTI, gt, MC_HDR, preds, MC_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertIsNone(res["metrics"]["auc"])

    def test_invalid_predictions_exit_12(self):
        cases = {
            "missing row": MC_OK[:4],
            "conflicting duplicate": MC_OK + [["s1.dcm", 1, 0.1, 0.7, 0.1, 0.1]],
            "label out of range": [["s1.dcm", 4, 0.7, 0.1, 0.1, 0.1]] + MC_OK[1:],
            "probability text": [["s1.dcm", 0, "high", 0.1, 0.1, 0.1]] + MC_OK[1:],
            "all-zero row": [["s1.dcm", 0, 0, 0, 0, 0]] + MC_OK[1:],
        }
        for name, rows in cases.items():
            with self.subTest(name):
                code, _, out = run_eval(MULTI, MC_GT, MC_HDR, rows, MC_CLASSES)
                self.assertEqual(code, 12, out)

    def test_missing_prob_column_exit_12(self):
        hdr = MC_HDR[:-1]
        code, _, out = run_eval(MULTI, MC_GT, hdr, [r[:-1] for r in MC_OK], MC_CLASSES)
        self.assertEqual(code, 12, out)

    def test_flexible_columns_and_ids(self):
        want = run_eval(MULTI, MC_GT, MC_HDR, MC_OK, MC_CLASSES)[1]["metrics"]
        variants = {
            "class-name columns": (["image", "prediction", "A", "B", "C", "D"], MC_OK),
            "prob_<name> columns": (["file", "label", "prob_A", "prob_B", "prob_C", "prob_D"], MC_OK),
            "p0.. columns, no label": (["file", "p0", "p1", "p2", "p3"], [[r[0], *r[2:]] for r in MC_OK]),
            "one list column": (["file", "label", "probs"], [[r[0], r[1], json.dumps(r[2:])] for r in MC_OK]),
            "labels as names": (MC_HDR, [[r[0], MC_CLASSES[r[1]], *r[2:]] for r in MC_OK]),
            "ids without extension": (MC_HDR, [[r[0][:-4], *r[1:]] for r in MC_OK]),
            "identical duplicate + unknown row": (MC_HDR, MC_OK + [MC_OK[0], ["x.dcm", 0, 1, 0, 0, 0]]),
        }
        for name, (hdr, rows) in variants.items():
            with self.subTest(name):
                code, res, out = run_eval(MULTI, MC_GT, hdr, rows, MC_CLASSES)
                self.assertEqual(code, 0, out)
                self.assertEqual(res["metrics"], want)

    def test_logits_and_unnormalised_rows(self):
        logits = [[r[0], r[1], *[math.log(p) for p in r[2:]]] for r in MC_OK]  # softmax(log p) = p
        code, res, out = run_eval(MULTI, MC_GT, MC_HDR, logits, MC_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertEqual(res["predictions_format"]["probability_transform"], "softmax")
        want = run_eval(MULTI, MC_GT, MC_HDR, MC_OK, MC_CLASSES)[1]["metrics"]
        self.assertAlmostEqual(res["metrics"]["auc"], want["auc"])
        halved = [[r[0], r[1], *[p / 2 for p in r[2:]]] for r in MC_OK]  # sums to 0.5: re-normalised
        code, res, out = run_eval(MULTI, MC_GT, MC_HDR, halved, MC_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertEqual(res["predictions_format"]["probability_transform"], "renormalised")
        self.assertEqual(res["metrics"], want)

    def test_labels_only_omits_auc(self):
        code, res, out = run_eval(MULTI, MC_GT, ["file", "label"], [r[:2] for r in MC_OK], MC_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertIsNone(res["metrics"]["auc"])
        self.assertAlmostEqual(res["metrics"]["accuracy"], 3 / 5)


# ── segmentation (needs numpy + Pillow) ──────────────────────────────────────────

SEG = HERE / "segmentation" / "evaluate.py"
SEG_ADAPTOR = HERE.parent / "tools" / "reference" / "adaptor_argmax_segmentation.py"


def _save_mask(path, arr):
    import numpy as np
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npy":
        np.save(path, arr)
    else:
        Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="L").save(path)


def run_seg(gt_masks, pred_masks, class_names, spec_extra=None, pred_rows=None):
    """gt_masks / pred_masks: {file: (filename, array)}. Writes the GT masks next
    to ground_truth.csv (as the platform would) and the predicted masks under
    predictions/masks/, then runs the evaluator."""
    d = Path(tempfile.mkdtemp())
    with open(d / "ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "mask"])
        for file_id, (name, arr) in gt_masks.items():
            _save_mask(d / "gt_masks" / name, arr)
            w.writerow([file_id, f"gt_masks/{name}"])
    pdir = d / "predictions"
    pdir.mkdir()
    with open(pdir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "mask"])
        if pred_rows is not None:
            w.writerows(pred_rows)
        for file_id, (name, arr) in pred_masks.items():
            _save_mask(pdir / "masks" / name, arr)
            w.writerow([file_id, f"masks/{name}"])
    (d / "spec.json").write_text(json.dumps({"class_names": class_names, **(spec_extra or {})}))
    proc = subprocess.run(
        [sys.executable, str(SEG), "--predictions", str(pdir / "predictions.csv"),
         "--ground-truth", str(d / "ground_truth.csv"), "--spec", str(d / "spec.json"),
         "--results", str(d / "results.json")],
        capture_output=True, text=True)
    results = json.loads((d / "results.json").read_text()) if proc.returncode == 0 else None
    return proc.returncode, results, proc.stdout + proc.stderr


def _square(size, top, left, side, value=1):
    import numpy as np
    m = np.zeros((size, size), dtype=np.uint8)
    m[top:top + side, left:left + side] = value
    return m


SEG_CLASSES = ["background", "lesion"]


class SegmentationEvaluator(unittest.TestCase):
    def test_perfect_prediction_and_0_255_ground_truth(self):
        gt = {"a.jpg": ("a.png", _square(8, 2, 2, 4, value=255))}   # a 0/255 export
        pred = {"a.jpg": ("a.png", _square(8, 2, 2, 4))}
        code, res, out = run_seg(gt, pred, SEG_CLASSES)
        self.assertEqual(code, 0, out)
        m = res["metrics"]
        for k in ("dice", "iou", "global_dice", "global_iou", "sensitivity", "ppv", "specificity",
                  "pixel_accuracy"):
            self.assertAlmostEqual(m[k], 1.0, msg=k)

    def test_known_overlap(self):
        # truth: 2x2 at (0,0) = 4 px; prediction: 2x2 at (0,1) -> tp 2, fp 2, fn 2
        gt = {"a.jpg": ("a.png", _square(4, 0, 0, 2))}
        pred = {"a.jpg": ("a.png", _square(4, 0, 1, 2))}
        code, res, out = run_seg(gt, pred, SEG_CLASSES)
        self.assertEqual(code, 0, out)
        m = res["metrics"]
        self.assertAlmostEqual(m["dice"], 0.5)
        self.assertAlmostEqual(m["iou"], 1 / 3)
        self.assertAlmostEqual(m["sensitivity"], 0.5)
        self.assertAlmostEqual(m["ppv"], 0.5)
        self.assertAlmostEqual(m["pixel_accuracy"], 12 / 16)

    def test_per_image_mean_vs_global(self):
        # image a: perfect on a big object; image b: misses a 1-px object entirely
        gt = {"a.jpg": ("a.png", _square(8, 0, 0, 6)), "b.jpg": ("b.png", _square(8, 3, 3, 1))}
        pred = {"a.jpg": ("a.png", _square(8, 0, 0, 6)), "b.jpg": ("b.png", _square(8, 0, 0, 0))}
        code, res, out = run_seg(gt, pred, SEG_CLASSES)
        self.assertEqual(code, 0, out)
        m = res["metrics"]
        self.assertAlmostEqual(m["dice"], 0.5)                      # (1 + 0) / 2
        self.assertAlmostEqual(m["global_dice"], 72 / 73)           # 2*36 / (2*36 + 0 + 1)

    def test_prediction_at_model_size_is_resized(self):
        gt = {"a.jpg": ("a.png", _square(8, 0, 0, 4))}
        pred = {"a.jpg": ("a.png", _square(4, 0, 0, 2))}           # half resolution
        code, res, out = run_seg(gt, pred, SEG_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertEqual(res["predictions_format"]["resized_to_ground_truth"], 1)
        self.assertAlmostEqual(res["metrics"]["dice"], 1.0)

    def test_empty_in_both_counts_as_correct_or_is_skipped(self):
        gt = {"a.jpg": ("a.png", _square(4, 0, 0, 2)), "b.jpg": ("b.png", _square(4, 0, 0, 0))}
        pred = {"a.jpg": ("a.png", _square(4, 0, 1, 2)), "b.jpg": ("b.png", _square(4, 0, 0, 0))}
        code, res, out = run_seg(gt, pred, SEG_CLASSES)
        self.assertEqual(code, 0, out)
        self.assertAlmostEqual(res["metrics"]["dice"], (0.5 + 1.0) / 2)
        self.assertEqual(res["per_class"]["lesion"]["images_empty_in_both"], 1)
        code, res, out = run_seg(gt, pred, SEG_CLASSES, spec_extra={"empty_score": "skip"})
        self.assertEqual(code, 0, out)
        self.assertAlmostEqual(res["metrics"]["dice"], 0.5)

    def test_multiclass_score_array_and_ignore_index(self):
        import numpy as np
        truth = np.array([[0, 1, 2, 255]], dtype=np.uint8)            # 255 = not scored
        scores = np.zeros((3, 1, 4), dtype=np.float32)
        for x, c in enumerate([0, 1, 2, 0]):
            scores[c, 0, x] = 5.0
        gt = {"a.jpg": ("a.png", truth)}
        pred = {"a.jpg": ("a.npy", scores)}                           # [C, H, W] -> argmax
        code, res, out = run_seg(gt, pred, ["bg", "x", "y"], spec_extra={"ignore_index": 255})
        self.assertEqual(code, 0, out)
        self.assertAlmostEqual(res["metrics"]["dice"], 1.0)
        self.assertAlmostEqual(res["metrics"]["pixel_accuracy"], 1.0)
        self.assertEqual(set(res["per_class"]), {"x", "y"})

    def test_missing_prediction_is_invalid(self):
        gt = {"a.jpg": ("a.png", _square(4, 0, 0, 2)), "b.jpg": ("b.png", _square(4, 0, 0, 2))}
        pred = {"a.jpg": ("a.png", _square(4, 0, 0, 2))}
        code, _, out = run_seg(gt, pred, SEG_CLASSES)
        self.assertEqual(code, 12, out)
        self.assertIn("have no predicted mask", out)

    def test_class_outside_class_names_is_invalid(self):
        gt = {"a.jpg": ("a.png", _square(4, 0, 0, 2))}
        pred = {"a.jpg": ("a.png", _square(4, 0, 0, 2, value=3))}
        code, _, out = run_seg(gt, pred, SEG_CLASSES)
        self.assertEqual(code, 12, out)
        self.assertIn("outside", out)

    def test_unreadable_predicted_mask_is_invalid(self):
        gt = {"a.jpg": ("a.png", _square(4, 0, 0, 2))}
        code, _, out = run_seg(gt, {}, SEG_CLASSES, pred_rows=[["a.jpg", "masks/missing.png"]])
        self.assertEqual(code, 12, out)
        self.assertIn("not found", out)

    def test_reference_adaptor_end_to_end(self):
        import numpy as np
        d = Path(tempfile.mkdtemp())
        logits = np.full((2, 2, 8, 8), -3.0, dtype=np.float32)        # [N, C, H, W]
        logits[:, 0] = 3.0                                             # background everywhere...
        logits[0, 1, 2:6, 2:6] = 9.0                                   # ...except a lesion in image 0
        (d / "raw").mkdir()
        np.savez(d / "raw" / "raw_outputs.npz", ids=np.array(["imgs/a.jpg", "imgs/b.jpg"]), logits=logits)
        (d / "spec.json").write_text(json.dumps({"class_names": SEG_CLASSES}))
        proc = subprocess.run([sys.executable, str(SEG_ADAPTOR), "--raw-dir", str(d / "raw"),
                               "--spec", str(d / "spec.json"), "--output-dir", str(d / "predictions")],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(d / "ground_truth.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "mask"])
            for name, m in (("a.png", _square(16, 4, 4, 8, value=255)), ("b.png", _square(16, 0, 0, 0))):
                _save_mask(d / "gt" / name, m)                         # ground truth at 2x the model size
                w.writerow([f"imgs/{name[0]}.jpg", f"gt/{name}"])
        proc = subprocess.run([sys.executable, str(SEG), "--predictions", str(d / "predictions" / "predictions.csv"),
                               "--ground-truth", str(d / "ground_truth.csv"), "--spec", str(d / "spec.json"),
                               "--results", str(d / "results.json")], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        res = json.loads((d / "results.json").read_text())
        self.assertAlmostEqual(res["metrics"]["dice"], 1.0)
        self.assertEqual(res["predictions_format"]["resized_to_ground_truth"], 2)


# ── object detection ─────────────────────────────────────────────────────────────

DET = HERE / "object_detection" / "evaluate.py"
GT_HDR = ["file", "x_min", "y_min", "x_max", "y_max", "class"]
PRED_HDR = ["file", "x_min", "y_min", "x_max", "y_max", "score", "class"]


def run_det(gt_rows, pred_header, pred_rows, class_names):
    d = Path(tempfile.mkdtemp())
    for name, header, rows in (("ground_truth.csv", GT_HDR, gt_rows), ("predictions.csv", pred_header, pred_rows)):
        with open(d / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
    (d / "spec.json").write_text(json.dumps({"class_names": class_names}))
    proc = subprocess.run(
        [sys.executable, str(DET), "--predictions", str(d / "predictions.csv"),
         "--ground-truth", str(d / "ground_truth.csv"), "--spec", str(d / "spec.json"),
         "--results", str(d / "results.json")],
        capture_output=True, text=True)
    results = json.loads((d / "results.json").read_text()) if proc.returncode == 0 else None
    return proc.returncode, results, proc.stdout + proc.stderr


KIDNEY = ["kidney"]
DET_GT = [["scans/a.png", 10, 10, 110, 110, 0], ["scans/b.png", 50, 50, 150, 100, 0], ["scans/c.png", "", "", "", "", ""]]


class DetectionEvaluator(unittest.TestCase):
    def test_perfect(self):
        preds = [[r[0], *r[1:5], 0.9, 0] for r in DET_GT[:2]]
        code, res, out = run_det(DET_GT, PRED_HDR, preds, KIDNEY)
        self.assertEqual(code, 0, out)
        for k in ("map", "ap50", "ap75", "map_giou", "recall_50", "mean_iou"):
            self.assertAlmostEqual(res["metrics"][k], 1.0, msg=k)
        self.assertEqual(res["num_samples"], 3)
        self.assertEqual(res["num_gt_boxes"], 2)

    def test_iou_thresholds(self):
        # a: 100x100 box shifted by 12 px -> IoU = 88*100 / (2*10000 - 8800) = 0.7857:
        # matches at 0.50..0.75 (6 of 10 thresholds), not at 0.80+. b is found exactly.
        preds = [["a.png", 22, 10, 122, 110, 0.9, 0], ["b.png", 50, 50, 150, 100, 0.8, 0]]
        code, res, out = run_det(DET_GT, PRED_HDR, preds, KIDNEY)
        self.assertEqual(code, 0, out)
        m = res["metrics"]
        self.assertAlmostEqual(m["ap50"], 1.0)
        self.assertAlmostEqual(m["ap75"], 1.0)
        # thresholds >= 0.80: only b matches, after the higher-scored miss: precision 0.5 up to
        # recall 0.5, then nothing -> 101-point AP = 51 * 0.5 / 101 at each of those 4 thresholds
        self.assertAlmostEqual(m["map"], (6 * 1.0 + 4 * (51 * 0.5 / 101)) / 10, places=6)
        self.assertAlmostEqual(m["mean_iou"], (88 * 100 / 11200 + 1.0) / 2, places=6)

    def test_false_positive_on_empty_image_lowers_ap(self):
        preds = [["a.png", 10, 10, 110, 110, 0.5, 0], ["b.png", 50, 50, 150, 100, 0.4, 0],
                 ["c.png", 0, 0, 30, 30, 0.95, 0]]                      # c has no objects
        code, res, out = run_det(DET_GT, PRED_HDR, preds, KIDNEY)
        self.assertEqual(code, 0, out)
        self.assertLess(res["metrics"]["ap50"], 1.0)
        self.assertAlmostEqual(res["metrics"]["recall_50"], 1.0)

    def test_images_without_predictions_score_zero_not_error(self):
        code, res, out = run_det(DET_GT, PRED_HDR, [["a.png", 10, 10, 110, 110, 0.9, 0]], KIDNEY)
        self.assertEqual(code, 0, out)
        self.assertAlmostEqual(res["metrics"]["recall_50"], 0.5)
        code, res, out = run_det(DET_GT, PRED_HDR, [["c.png", "", "", "", "", "", ""]], KIDNEY)
        self.assertEqual(code, 0, out)
        self.assertEqual(res["metrics"]["map"], 0.0)

    def test_xywh_names_and_no_class_column_single_class(self):
        preds = [["a", 10, 10, 100, 100, 0.9], ["b", 50, 50, 100, 50, 0.8]]
        code, res, out = run_det(DET_GT, ["image", "x", "y", "w", "h", "confidence"], preds, KIDNEY)
        self.assertEqual(code, 0, out)
        self.assertAlmostEqual(res["metrics"]["map"], 1.0)
        self.assertEqual(res["predictions_format"]["box_format"], "xywh")

    def test_multiclass_and_class_names(self):
        gt = [["a.png", 0, 0, 50, 50, 0], ["a.png", 60, 60, 100, 100, 1]]
        # the cyst is predicted as a kidney, with a higher score than the real kidney
        preds = [["a.png", 0, 0, 50, 50, 0.9, "kidney"], ["a.png", 60, 60, 100, 100, 0.95, "kidney"]]
        code, res, out = run_det(gt, PRED_HDR, preds, ["kidney", "cyst"])
        self.assertEqual(code, 0, out)
        # kidney: the false positive ranks first -> precision 0.5 at recall 1 -> AP 0.5
        self.assertAlmostEqual(res["per_class"]["kidney"]["ap50"], 0.5)
        self.assertAlmostEqual(res["per_class"]["cyst"]["ap50"], 0.0)
        self.assertAlmostEqual(res["metrics"]["ap50"], 0.25)

    def test_invalid_predictions(self):
        for rows, why in (([["a.png", 50, 10, 10, 110, 0.9, 0]], "x_max < x_min"),
                          ([["a.png", 10, 10, 110, 110, "high", 0]], "not a number"),
                          ([["a.png", 10, 10, 110, 110, 0.9, 7]], "neither an index"),
                          ([["zzz.png", 10, 10, 110, 110, 0.9, 0]], "no prediction file id matches")):
            code, _, out = run_det(DET_GT, PRED_HDR, rows, KIDNEY)
            self.assertEqual(code, 12, out)
            self.assertIn(why, out)

    def test_normalised_boxes_flagged(self):
        code, res, out = run_det(DET_GT, PRED_HDR, [["a.png", 0.1, 0.1, 0.5, 0.5, 0.9, 0]], KIDNEY)
        self.assertEqual(code, 0, out)
        self.assertTrue(res["predictions_format"]["looks_normalised"])

    def test_reference_adaptor_both_layouts_end_to_end(self):
        """infer.py --per-sample output (built by hand) -> reference adaptor -> evaluator.
        The model ran at 100x100; the images are 200x100 (a) and 400x200 (b)."""
        import numpy as np
        adaptor = HERE.parent / "tools" / "reference" / "adaptor_boxes_detection.py"

        def ragged(per_image):
            flat = np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in per_image])
            shapes = np.array([np.asarray(a).shape for a in per_image], dtype=np.int64)
            offs = np.concatenate([[0], np.cumsum([np.asarray(a).size for a in per_image])]).astype(np.int64)
            return flat, shapes, offs

        common = {"ids": np.array(["scans/a.png", "scans/b.png", "scans/c.png"]),
                  "orig_hw": np.array([[100, 200], [200, 400], [100, 100]]), "input_hw": np.array([100, 100])}
        # in model pixels: a's kidney at x 5..55, y 10..90 -> original x 10..110, y 10..90;
        # b's at x 12.5..37.5, y 25..50 -> original x 50..150, y 50..100
        dets = [[[5, 10, 55, 90, 0.9, 0]], [[12.5, 25, 37.5, 50, 0.8, 0]], np.zeros((0, 6))]
        layouts = {
            "one [K,6] output": dict(zip(("det", "det__shape", "det__offset"), ragged(dets))),
            "boxes/scores/labels": {
                **dict(zip(("boxes", "boxes__shape", "boxes__offset"), ragged([np.asarray(d)[:, :4] for d in dets]))),
                **dict(zip(("scores", "scores__shape", "scores__offset"), ragged([np.asarray(d)[:, 4] for d in dets]))),
                **dict(zip(("labels", "labels__shape", "labels__offset"), ragged([np.asarray(d)[:, 5] for d in dets]))),
            },
        }
        for name, arrays in layouts.items():
            d = Path(tempfile.mkdtemp())
            (d / "raw").mkdir()
            np.savez(d / "raw" / "raw_outputs.npz", **common, **arrays)
            (d / "spec.json").write_text(json.dumps({"class_names": KIDNEY}))
            proc = subprocess.run([sys.executable, str(adaptor), "--raw-dir", str(d / "raw"), "--spec",
                                   str(d / "spec.json"), "--output-dir", str(d / "pred")], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, name + proc.stdout + proc.stderr)
            with open(d / "gt.csv", "w", newline="") as f:
                csv.writer(f).writerows([GT_HDR, ["scans/a.png", 10, 10, 110, 90, 0],
                                         ["scans/b.png", 50, 50, 150, 100, 0], ["scans/c.png", "", "", "", "", ""]])
            proc = subprocess.run([sys.executable, str(DET), "--predictions", str(d / "pred" / "predictions.csv"),
                                   "--ground-truth", str(d / "gt.csv"), "--spec", str(d / "spec.json"),
                                   "--results", str(d / "res.json")], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, name + proc.stdout + proc.stderr)
            self.assertAlmostEqual(json.loads((d / "res.json").read_text())["metrics"]["map"], 1.0, msg=name)


if __name__ == "__main__":
    unittest.main()
