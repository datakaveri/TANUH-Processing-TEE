#!/usr/bin/env python3
"""
Check a model locally before submitting it to the TANUH benchmark.

Runs the exact loading and input-preparation code the Processing TEE uses
(tools/infer/infer.py), then one dummy batch, and prints what your adaptor
will receive.

    python3 check_model.py --format onnx        --model model.onnx [--weights model.onnx.data]
    python3 check_model.py --format torchscript --model model.pt [--input-spec input_spec.json]
    python3 check_model.py --format huggingface --model model.zip
    python3 check_model.py ... --image sample.jpg     (use a real image instead of zeros)

--input-spec is the input_spec.json you will upload with an ONNX or TorchScript
model, e.g. {"input_size": [224, 224], "resize": "bicubic"}. A TorchScript model
needs it unless its input size is embedded in the .pt.

Exit 0 when the platform can run the model; non-zero with the reason otherwise.
Install the platform's pinned packages first so the result matches the TEE
(numpy<2, onnxruntime 1.19.2, torch 2.5.1, transformers 4.46.3).
"""

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

INFER = Path(__file__).resolve().parent / "infer" / "infer.py"


def load_infer():
    spec = importlib.util.spec_from_file_location("tanuh_infer", INFER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tanuh_infer"] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


HF_NEEDED = ("config.json", "preprocessor_config.json")


def _junk(name):
    base = name.rsplit("/", 1)[-1]
    return "__MACOSX/" in f"/{name}" or base in (".DS_Store", "Thumbs.db") or base.startswith("._")


def stage_hf_zip(model, dest):
    """What the Processing TEE does with a Hugging Face zip (internal/modelpkg):
    take the shallowest folder holding config.json and keep only config.json,
    preprocessor_config.json and safetensors weights; drop auto_map."""
    with zipfile.ZipFile(model) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/") and not _junk(n)]
        roots = sorted({n.rsplit("/", 1)[0] if "/" in n else "" for n in names if n.rsplit("/", 1)[-1] == "config.json"},
                       key=lambda d: d.count("/") + (d != ""))
        if not roots:
            raise SystemExit("FAIL the zip has no config.json; zip the folder save_pretrained() wrote")
        root = roots[0]
        hf = dest / "hf"
        hf.mkdir()
        kept = 0
        for n in names:
            folder, base = (n.rsplit("/", 1) if "/" in n else ("", n))
            if folder == root and (base in HF_NEEDED or base.endswith(".safetensors")
                                   or base.endswith(".safetensors.index.json")):
                (hf / base).write_bytes(zf.read(n))
                kept += 1
        print(f"OK   using {kept} file(s) from {root or 'the zip root'!r}; other files are ignored")
    for cfg in HF_NEEDED:
        p = hf / cfg
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            if data.pop("auto_map", None) is not None:
                print(f"NOTE auto_map removed from {cfg}: only built-in transformers classes are used")
                p.write_text(json.dumps(data, indent=2))


def stage_model(fmt, model, weights, input_spec, dest):
    """Lay the files out exactly as the Processing TEE does."""
    if fmt == "onnx":
        shutil.copy(model, dest / "model.onnx")
        if weights:
            shutil.copy(weights, dest / "model.onnx.data")
    elif fmt == "torchscript":
        shutil.copy(model, dest / "model.pt")
    else:
        if input_spec:
            raise SystemExit("FAIL --input-spec is for ONNX and TorchScript; Hugging Face uses preprocessor_config.json")
        stage_hf_zip(model, dest)
    if input_spec:
        shutil.copy(input_spec, dest / "input_spec.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True, choices=["onnx", "torchscript", "huggingface"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", help="ONNX external weights (.onnx.data)")
    ap.add_argument("--input-spec", help="input_spec.json you will upload with the model (ONNX / TorchScript)")
    ap.add_argument("--image", help="optional sample input (JPEG/PNG/TIFF/BMP/WebP or DICOM)")
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    infer = load_infer()
    with tempfile.TemporaryDirectory() as tmp:
        mdir = Path(tmp)
        stage_model(args.format, Path(args.model), args.weights and Path(args.weights),
                    args.input_spec and Path(args.input_spec), mdir)
        try:
            runtime = infer.RUNTIMES[args.format](mdir)
            print(f"OK   loaded with {runtime.name} {runtime.version} on {runtime.device}")
            if runtime.spec:
                print(f"OK   platform will feed: {asdict(runtime.spec)} (from {runtime.spec_source})")
            else:
                print("OK   inputs prepared by the model's own preprocessor_config.json")
            img = infer.decode(args.image) if args.image else np.zeros((512, 512, 3), dtype=np.uint8)
            n = runtime.spec.fixed_batch if runtime.spec and runtime.spec.fixed_batch else args.batch_size
            if runtime.spec is None:
                out = runtime.run([img] * n)
            else:
                out = runtime.run(np.stack([infer.to_tensor(img, runtime.spec)] * n))
        except infer.ModelError as exc:
            print(f"FAIL {exc}")
            return 13
        except infer.DecodeError as exc:
            print(f"FAIL sample image: {exc}")
            return 14
        for name, arr in out.items():
            print(f"OK   output {name!r}: shape {tuple(np.asarray(arr).shape)} (first axis = batch of {n})")
    print("PASS the platform can run this model; your adaptor reads these outputs from raw_outputs.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
