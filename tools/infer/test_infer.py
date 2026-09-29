"""
Tests for the platform inference step (needs numpy, onnx, onnxruntime, opencv,
Pillow, torch — the image's packages).

    python3 -m unittest tools/infer/test_infer.py

Tiny models are built on the fly, so no model files are needed.
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

HERE = Path(__file__).resolve().parent
INFER = HERE / "infer.py"
_spec = importlib.util.spec_from_file_location("tanuh_infer", INFER)
infer = importlib.util.module_from_spec(_spec)
sys.modules["tanuh_infer"] = infer
_spec.loader.exec_module(infer)


def tiny_onnx(path, shape, elem=TensorProto.FLOAT, resize=None):
    """Model with one input of `shape`; output = per-sample mean over all but the batch axis."""
    x = helper.make_tensor_value_info("x", elem, shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    nodes = [helper.make_node("Cast", ["x"], ["xf"], to=TensorProto.FLOAT),
             helper.make_node("ReduceMean", ["xf", "axes"], ["y"], keepdims=0)]
    axes = helper.make_tensor("axes", TensorProto.INT64, [3], [1, 2, 3])
    model = helper.make_model(helper.make_graph(nodes, "g", [x], [y], [axes]),
                              opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    if resize:
        model.metadata_props.append(onnx.StringStringEntryProto(key="tanuh.resize", value=resize))
    onnx.save(model, str(path))


def write_image(path, value=128, size=(40, 30)):
    import cv2
    cv2.imwrite(str(path), np.full((size[1], size[0], 3), value, dtype=np.uint8))


class OnnxSpec(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def load(self):
        return infer.OnnxRuntime(self.dir)

    def test_nchw_float_with_dynamic_batch(self):
        tiny_onnx(self.dir / "model.onnx", ["batch", 3, 16, 12], resize="bicubic")
        s = self.load().spec
        self.assertEqual((s.channels, s.height, s.width, s.layout, s.dtype, s.resize, s.fixed_batch),
                         (3, 16, 12, "NCHW", "float32", "bicubic", None))

    def test_nhwc_uint8_fixed_batch_default_resize(self):
        tiny_onnx(self.dir / "model.onnx", [4, 16, 16, 1], elem=TensorProto.UINT8)
        s = self.load().spec
        self.assertEqual((s.channels, s.layout, s.dtype, s.resize, s.fixed_batch), (1, "NHWC", "uint8", "bilinear", 4))

    def test_dynamic_height_width_rejected(self):
        tiny_onnx(self.dir / "model.onnx", ["batch", 3, "h", "w"])
        with self.assertRaises(infer.ModelError):
            self.load()

    def test_unknown_resize_rejected(self):
        tiny_onnx(self.dir / "model.onnx", ["batch", 3, 8, 8], resize="lanczos")
        with self.assertRaises(infer.ModelError):
            self.load()

    def test_external_weights_missing_rejected(self):
        # A real model saved with external data, then its weights removed.
        w = helper.make_tensor("w", TensorProto.FLOAT, [1, 3, 1, 1], np.ones(3, np.float32).tobytes(), raw=True)
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["b", 3, 8, 8])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
        g = helper.make_graph([helper.make_node("Mul", ["x", "w"], ["y"])], "g", [x], [y], [w])
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
        m.ir_version = 10
        onnx.save_model(m, str(self.dir / "model.onnx"), save_as_external_data=True,
                        location="other_name.data", size_threshold=0)
        with self.assertRaises(infer.ModelError):
            self.load()
        # Uploaded under the platform's name, it loads whatever the exporter called it.
        (self.dir / "other_name.data").rename(self.dir / "model.onnx.data")
        self.assertEqual(self.load().spec.height, 8)

    def test_garbage_file_is_a_model_error(self):
        (self.dir / "model.onnx").write_bytes(b"not an onnx model")
        with self.assertRaises(infer.ModelError):
            self.load()


class TorchScriptSpec(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def save(self, extra=None):
        import torch

        class M(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=(1, 2, 3)).unsqueeze(1)

        files = {"tanuh.json": json.dumps(extra)} if extra is not None else {}
        torch.jit.save(torch.jit.script(M()), str(self.dir / "model.pt"), _extra_files=files)

    def test_spec_read_without_loading(self):
        self.save({"input_shape": [3, 32, 24], "resize": "bicubic"})
        s = infer.read_torchscript_spec(self.dir / "model.pt")
        self.assertEqual((s.channels, s.height, s.width, s.layout, s.resize), (3, 32, 24, "NCHW", "bicubic"))

    def test_missing_tanuh_json_rejected(self):
        self.save()
        with self.assertRaises(infer.ModelError):
            infer.read_torchscript_spec(self.dir / "model.pt")

    def test_bad_input_shape_rejected(self):
        self.save({"input_shape": [3, "h", 24]})
        with self.assertRaises(infer.ModelError):
            infer.read_torchscript_spec(self.dir / "model.pt")

    def test_non_torchscript_file_rejected(self):
        (self.dir / "model.pt").write_bytes(b"pickled state dict, not TorchScript")
        with self.assertRaises(infer.ModelError):
            infer.read_torchscript_spec(self.dir / "model.pt")

    def test_runs_and_names_outputs(self):
        self.save({"input_shape": [3, 8, 8]})
        rt = infer.TorchScriptRuntime(self.dir)
        out = rt.run(np.zeros((2, 3, 8, 8), np.float32))
        self.assertEqual(list(out), ["output"])
        self.assertEqual(out["output"].shape, (2, 1))


class HuggingFaceValidation(unittest.TestCase):
    def make(self, files, config=None):
        d = Path(tempfile.mkdtemp()) / "hf"
        d.mkdir()
        (d / "config.json").write_text(json.dumps(config or {"model_type": "mobilevitv2"}))
        for name in files:
            (d / name).write_text("x")
        return d

    def test_complete_package_passes(self):
        infer.validate_hf_dir(self.make(["model.safetensors", "preprocessor_config.json"]))

    def test_rejections(self):
        cases = {
            "pickle weights": (["pytorch_model.bin", "preprocessor_config.json", "model.safetensors"], None),
            "python code": (["model.safetensors", "preprocessor_config.json", "modeling_evil.py"], None),
            "no safetensors": (["preprocessor_config.json"], None),
            "no preprocessor": (["model.safetensors"], None),
            "remote code": (["model.safetensors", "preprocessor_config.json"],
                            {"model_type": "x", "auto_map": {"AutoModel": "evil.Model"}}),
        }
        for name, (files, cfg) in cases.items():
            with self.subTest(name):
                with self.assertRaises(infer.ModelError):
                    infer.validate_hf_dir(self.make(files, cfg))


class InputContract(unittest.TestCase):
    def test_grey_nhwc_uint8(self):
        spec = infer.InputSpec(1, 10, 20, "NHWC", "uint8", "bilinear", None)
        t = infer.to_tensor(np.full((50, 60, 3), 200, np.uint8), spec)
        self.assertEqual((t.shape, t.dtype), ((10, 20, 1), np.uint8))

    def test_rgb_nchw_float_in_unit_range(self):
        spec = infer.InputSpec(3, 8, 8, "NCHW", "float32", "bicubic", None)
        t = infer.to_tensor(np.full((30, 30, 3), 255, np.uint8), spec)
        self.assertEqual(t.shape, (3, 8, 8))
        self.assertAlmostEqual(float(t.max()), 1.0)

    def test_unsupported_file_type(self):
        with self.assertRaises(infer.DecodeError):
            infer.decode(Path("scan.nii"))

    def test_output_names_are_made_safe(self):
        m = infer.safe_output_names(["ids", "out/logits:0", "out_logits_0"])
        self.assertNotIn("ids", m.values())
        self.assertEqual(len(set(m.values())), 3)


class EndToEnd(unittest.TestCase):
    """Run infer.py as the TEE does: fixed-batch padding, output order, exit codes."""

    def run_infer(self, model_shape, images):
        d = Path(tempfile.mkdtemp())
        (d / "model").mkdir()
        tiny_onnx(d / "model" / "model.onnx", model_shape)
        paths = []
        for i, value in enumerate(images):
            p = d / f"img{i}.jpg"
            if value is None:
                p.write_bytes(b"corrupt")
            else:
                write_image(p, value)
            paths.append(str(p))
        (d / "inputs.txt").write_text("\n".join(paths))
        proc = subprocess.run([sys.executable, str(INFER), "--format", "onnx", "--model-dir", str(d / "model"),
                               "--inputs", str(d / "inputs.txt"), "--output-dir", str(d / "raw")],
                              capture_output=True, text=True)
        return proc, d

    def test_fixed_batch_pads_and_keeps_order(self):
        proc, d = self.run_infer([4, 3, 8, 8], [0, 51, 102, 153, 204])  # 5 inputs, batch fixed at 4
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        npz = np.load(d / "raw" / "raw_outputs.npz")
        self.assertEqual([str(x) for x in npz["ids"]], [f"img{i}.jpg" for i in range(5)])
        np.testing.assert_allclose(npz["y"], [0, 0.2, 0.4, 0.6, 0.8], atol=0.01)
        meta = json.loads((d / "raw" / "meta.json").read_text())
        self.assertEqual(meta["input_spec"]["fixed_batch"], 4)

    def test_corrupt_input_exits_14(self):
        proc, _ = self.run_infer(["b", 3, 8, 8], [10, None])
        self.assertEqual(proc.returncode, 14, proc.stdout + proc.stderr)

    def test_bad_model_exits_13(self):
        proc, _ = self.run_infer(["b", 3, "h", "w"], [10])
        self.assertEqual(proc.returncode, 13, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
