#!/usr/bin/env python3
"""
Platform inference step (stage 1 of a Processing TEE job).

Runs a submitted model over a dataset's input files and writes the raw model
outputs. It never reads ground truth and never computes metrics — the
model provider's adaptor (stage 2) and the bucket evaluator (stage 3) do that.

    python3 infer.py --format onnx|torchscript|huggingface \
                     --model-dir DIR --inputs inputs.txt --output-dir raw/ \
                     [--batch-size 8]

  --model-dir   ONNX:         model.onnx (+ model.onnx.data when the model has external weights)
                TorchScript:  model.pt   (saved with torch.jit.save, carrying extra/tanuh.json)
                Hugging Face: hf/        (config.json, *.safetensors, preprocessor_config.json)
  --inputs      one input file path per line (data files, in ground-truth order)
  --output-dir  receives raw_outputs.npz (`ids` + one array per model output) and meta.json

Platform input contract (ONNX and TorchScript) — the platform turns every file
into exactly the tensor the model declares, so no preprocessing script exists:
  1. decode:   images (.jpg/.jpeg/.png) -> RGB uint8; DICOM (.dcm) -> middle frame,
               MONOCHROME1 inverted, min-max to 0..255, grey -> 3 channels
  2. resize:   to the model's H x W; "bilinear" (default, Pillow) or "bicubic" (OpenCV)
  3. channels: 1 (grey) or 3 (RGB) and layout NCHW or NHWC, as the model declares
  4. dtype:    float -> [0,1]; uint8 -> 0..255
Anything model-specific (normalisation, colour balancing, channel order) must be
inside the model. Hugging Face models use their own preprocessor_config.json instead.

Where the declaration comes from:
  ONNX         the model's input signature + metadata_props key "tanuh.resize"
  TorchScript  JSON embedded at save time:
                 torch.jit.save(m, f, _extra_files={"tanuh.json": json.dumps(
                     {"input_shape": [3, 256, 256], "layout": "NCHW",
                      "dtype": "float32", "resize": "bicubic"})})
  Hugging Face preprocessor_config.json (applied by transformers' AutoImageProcessor)

Exit codes (the Processing TEE maps them to leaderboard error categories):
  0   raw outputs written
  11  CUDA / GPU runtime failure (environment)
  13  model failed to load or run, or its declared input is unsupported (model)
  14  an input file could not be decoded (dataset)
  other non-zero  platform error
"""

import argparse
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

EXIT_CUDA = 11
EXIT_MODEL = 13
EXIT_DECODE = 14

DEFAULT_BATCH = 8
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DICOM_EXTS = {".dcm"}
RESIZE_METHODS = ("bilinear", "bicubic")
CUDA_KEYWORDS = ("cuda", "cudnn", "cublas", "nvidia", "out of memory", "gpu")
FORMATS = ("onnx", "torchscript", "huggingface")

# Hugging Face packages must be pure data: config + safetensors weights. Anything
# that can carry executable code (pickles, Python modules) is refused.
HF_FORBIDDEN_SUFFIXES = {".py", ".pyc", ".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".h5", ".joblib"}


class ModelError(Exception):
    """The submitted model could not be loaded/run, or declares an unsupported input."""


class DecodeError(Exception):
    """An input file could not be decoded."""


def log(msg):
    print(f"[infer] {msg}", flush=True)


def is_cuda_error(exc):
    text = str(exc).lower()
    return any(k in text for k in CUDA_KEYWORDS)


# ── decoding ───────────────────────────────────────────────────────────────────

def decode_image(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise DecodeError(f"cannot decode image {Path(path).name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_dicom(path):
    import pydicom
    from pydicom.uid import ImplicitVRLittleEndian
    ds = pydicom.dcmread(str(path), force=True)
    if not hasattr(ds, "file_meta"):
        ds.file_meta = pydicom.dataset.FileMetaDataset()
    if getattr(ds.file_meta, "TransferSyntaxUID", None) is None:
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] > 1 and arr.shape[-1] != 3:
        arr = arr[arr.shape[0] // 2]
    if "MONOCHROME1" in str(getattr(ds, "PhotometricInterpretation", "")).upper():
        arr = arr.max() - arr
    arr -= arr.min()
    mx = arr.max()
    if mx > 0:
        arr /= mx
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr[..., :3]


def decode(path):
    """Return the file as uint8 (H, W, 3) RGB, or raise DecodeError."""
    ext = Path(path).suffix.lower()
    try:
        if ext in IMAGE_EXTS:
            return decode_image(path)
        if ext in DICOM_EXTS:
            return decode_dicom(path)
    except DecodeError:
        raise
    except Exception as exc:
        raise DecodeError(f"cannot decode {Path(path).name}: {exc}") from exc
    raise DecodeError(f"unsupported input file type {ext!r} ({Path(path).name})")


# ── input contract ─────────────────────────────────────────────────────────────

@dataclass
class InputSpec:
    channels: int              # 1 or 3
    height: int
    width: int
    layout: str                # "NCHW" or "NHWC"
    dtype: str                 # "float32", "float16" or "uint8"
    resize: str                # "bilinear" or "bicubic"
    fixed_batch: int | None    # set when the model only accepts one batch size

    def validate(self):
        if self.channels not in (1, 3):
            raise ModelError(f"input must have 1 or 3 channels, declared {self.channels}")
        if self.height <= 0 or self.width <= 0:
            raise ModelError("input height and width must be fixed positive sizes")
        if self.layout not in ("NCHW", "NHWC"):
            raise ModelError(f"layout must be NCHW or NHWC, declared {self.layout!r}")
        if self.dtype not in ("float32", "float16", "uint8"):
            raise ModelError(f"input dtype must be float32, float16 or uint8, declared {self.dtype!r}")
        if self.resize not in RESIZE_METHODS:
            raise ModelError(f"resize must be one of {RESIZE_METHODS}, declared {self.resize!r}")
        return self


def resize(img, width, height, method):
    if method == "bilinear":
        from PIL import Image
        mode = "L" if img.ndim == 2 else "RGB"
        return np.asarray(Image.fromarray(img, mode).resize((width, height), Image.BILINEAR))
    import cv2
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)


def to_tensor(img_rgb, spec):
    """uint8 (H,W,3) RGB -> one sample in the model's declared input layout."""
    img = img_rgb
    if spec.channels == 1:
        import cv2
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = resize(img, spec.width, spec.height, spec.resize)
    if img.ndim == 2:
        img = img[:, :, None]
    if spec.dtype == "uint8":
        arr = img.astype(np.uint8)
    else:
        arr = img.astype(np.float32) / 255.0
        if spec.dtype == "float16":
            arr = arr.astype(np.float16)
    if spec.layout == "NCHW":
        arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(arr)


# ── runtimes ───────────────────────────────────────────────────────────────────

def load_onnx_model_bytes(model_path, weights_path):
    """Serialize a self-contained model, pointing every external tensor at the
    uploaded weights file whatever name the exporter recorded."""
    import onnx
    from onnx.external_data_helper import _get_all_tensors, load_external_data_for_model
    model = onnx.load(str(model_path), load_external_data=False)
    ext = [t for t in _get_all_tensors(model)
           if t.HasField("data_location") and t.data_location == onnx.TensorProto.EXTERNAL]
    if not ext:
        return model_path.read_bytes()
    if weights_path is None or not weights_path.exists():
        raise ModelError("the ONNX model declares external weights but no weights file was uploaded")
    for t in ext:
        for entry in t.external_data:
            if entry.key == "location":
                entry.value = weights_path.name
    load_external_data_for_model(model, str(weights_path.parent))
    return model.SerializeToString()


class OnnxRuntime:
    name = "onnxruntime"

    def __init__(self, model_dir):
        import onnxruntime as ort
        model_path = model_dir / "model.onnx"
        if not model_path.exists():
            raise ModelError("model.onnx not found")
        weights = model_dir / "model.onnx.data"
        try:
            model_bytes = load_onnx_model_bytes(model_path, weights if weights.exists() else None)
            available = set(ort.get_available_providers())
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available] \
                or ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_bytes, providers=providers)
        except ModelError:
            raise
        except Exception as exc:
            if is_cuda_error(exc):
                raise
            raise ModelError(f"ONNX model failed to load: {exc}") from exc
        self.version = ort.__version__
        self.providers = self.session.get_providers()
        self.device = "cuda" if "CUDAExecutionProvider" in self.providers else "cpu"
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ModelError(f"model must have exactly one input, found {len(inputs)}")
        self.input_name = inputs[0].name
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        self.spec = self._spec(inputs[0], meta.get("tanuh.resize", "bilinear"))

    @staticmethod
    def _spec(inp, resize_method):
        shape = list(inp.shape)
        if len(shape) != 4:
            raise ModelError(f"input must be 4-D (batch, ...), declared {shape}")
        dtype = {"tensor(float)": "float32", "tensor(float16)": "float16",
                 "tensor(uint8)": "uint8"}.get(inp.type)
        if dtype is None:
            raise ModelError(f"unsupported input type {inp.type}")
        dims = [d if isinstance(d, int) else None for d in shape]
        if dims[1] in (1, 3) and dims[2] and dims[3]:
            layout, c, h, w = "NCHW", dims[1], dims[2], dims[3]
        elif dims[3] in (1, 3) and dims[1] and dims[2]:
            layout, c, h, w = "NHWC", dims[3], dims[1], dims[2]
        else:
            raise ModelError(f"input must be NCHW or NHWC with a fixed height and width, declared {shape}")
        fixed = dims[0] if dims[0] and dims[0] > 0 else None
        return InputSpec(c, h, w, layout, dtype, resize_method, fixed).validate()

    def run(self, batch):
        outs = self.session.run(None, {self.input_name: batch})
        return {o.name: np.asarray(v) for o, v in zip(self.session.get_outputs(), outs)}


def read_torchscript_spec(model_path):
    """Read extra/tanuh.json from the archive without executing anything."""
    try:
        with zipfile.ZipFile(model_path) as zf:
            names = zf.namelist()
            if not any("/code/" in n for n in names) or not any(n.endswith("constants.pkl") for n in names):
                raise ModelError("model.pt is not a TorchScript archive (save it with torch.jit.save)")
            spec_entries = [n for n in names if n.endswith("/extra/tanuh.json")]
            if not spec_entries:
                raise ModelError("model.pt has no tanuh.json; save it with "
                                 "torch.jit.save(m, f, _extra_files={'tanuh.json': ...})")
            raw = json.loads(zf.read(spec_entries[0]).decode("utf-8"))
    except zipfile.BadZipFile as exc:
        raise ModelError("model.pt is not a TorchScript archive") from exc
    except json.JSONDecodeError as exc:
        raise ModelError(f"tanuh.json is not valid JSON: {exc}") from exc
    shape = raw.get("input_shape")
    if not (isinstance(shape, list) and len(shape) == 3 and all(isinstance(v, int) for v in shape)):
        raise ModelError("tanuh.json input_shape must be [C, H, W] integers")
    layout = raw.get("layout", "NCHW")
    c, h, w = shape if layout == "NCHW" else (shape[2], shape[0], shape[1])
    return InputSpec(c, h, w, layout, raw.get("dtype", "float32"),
                     raw.get("resize", "bilinear"), None).validate()


def flatten_torch_outputs(out):
    import torch
    if isinstance(out, torch.Tensor):
        items = {"output": out}
    elif isinstance(out, (tuple, list)):
        items = {f"output_{i}": t for i, t in enumerate(out)}
    elif isinstance(out, dict):
        items = {str(k): v for k, v in out.items()}
    else:
        raise ModelError(f"model returned {type(out).__name__}; expected a tensor, tuple or dict of tensors")
    result = {}
    for k, v in items.items():
        if not isinstance(v, torch.Tensor):
            raise ModelError(f"output {k!r} is {type(v).__name__}, expected a tensor")
        result[k] = v.detach().float().cpu().numpy()
    return result


class TorchScriptRuntime:
    name = "torch.jit"

    def __init__(self, model_dir):
        import torch
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            raise ModelError("model.pt not found")
        self.spec = read_torchscript_spec(model_path)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = torch.jit.load(str(model_path), map_location=self.device)
            self.model.eval()
        except Exception as exc:
            if is_cuda_error(exc):
                raise
            raise ModelError(f"TorchScript model failed to load: {exc}") from exc
        self.version = torch.__version__
        self.providers = [f"torch:{self.device}"]

    def run(self, batch):
        with self.torch.inference_mode():
            out = self.model(self.torch.from_numpy(batch).to(self.device))
        return flatten_torch_outputs(out)


def validate_hf_dir(hf_dir):
    if not hf_dir.is_dir():
        raise ModelError("Hugging Face model folder not found")
    files = [p for p in hf_dir.rglob("*") if p.is_file()]
    bad = sorted(p.name for p in files if p.suffix.lower() in HF_FORBIDDEN_SUFFIXES)
    if bad:
        raise ModelError(f"Hugging Face package contains files that can carry code: {bad[:5]}")
    for required in ("config.json", "preprocessor_config.json"):
        if not (hf_dir / required).is_file():
            raise ModelError(f"Hugging Face package is missing {required}")
    if not any(p.suffix == ".safetensors" for p in files):
        raise ModelError("Hugging Face package has no *.safetensors weights")
    try:
        config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelError(f"config.json is not valid JSON: {exc}") from exc
    if "auto_map" in config:
        raise ModelError("config.json uses auto_map (custom model code), which is not allowed")
    return config


class HuggingFaceRuntime:
    name = "transformers"

    def __init__(self, model_dir):
        hf_dir = model_dir / "hf"
        config = validate_hf_dir(hf_dir)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                str(hf_dir), local_files_only=True, trust_remote_code=False)
            self.model = AutoModelForImageClassification.from_pretrained(
                str(hf_dir), local_files_only=True, use_safetensors=True,
                trust_remote_code=False, torch_dtype=torch.float32)
            self.model.to(self.device).eval()
        except Exception as exc:
            if is_cuda_error(exc):
                raise
            raise ModelError(f"Hugging Face model failed to load "
                             f"(model_type={config.get('model_type')!r}): {exc}") from exc
        self.version = f"{transformers.__version__} (torch {torch.__version__})"
        self.providers = [f"torch:{self.device}"]
        self.spec = None  # the model's own image processor prepares inputs

    def run(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        with self.torch.inference_mode():
            out = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
        return {"logits": out.logits.detach().float().cpu().numpy()}


RUNTIMES = {"onnx": OnnxRuntime, "torchscript": TorchScriptRuntime, "huggingface": HuggingFaceRuntime}


# ── main loop ──────────────────────────────────────────────────────────────────

def safe_output_names(names):
    """npz member names must be plain; keep a mapping back to the model's names."""
    mapping, used = {}, {"ids"}
    for n in names:
        s = re.sub(r"[^A-Za-z0-9_.-]", "_", n) or "output"
        base, i = s, 1
        while s in used:
            s = f"{base}_{i}"
            i += 1
        used.add(s)
        mapping[n] = s
    return mapping


def run_inference(runtime, paths, batch_size):
    """Decode, prepare and run every input; returns {output_name: [N, ...] array}."""
    fixed = runtime.spec.fixed_batch if runtime.spec else None
    step = fixed or batch_size
    parts = {}
    for off in range(0, len(paths), step):
        chunk = paths[off: off + step]
        images = [decode(p) for p in chunk]
        try:
            if runtime.spec is None:
                out = runtime.run(images)
            else:
                samples = [to_tensor(img, runtime.spec) for img in images]
                if fixed and len(samples) < fixed:
                    samples += [np.zeros_like(samples[0])] * (fixed - len(samples))
                out = runtime.run(np.stack(samples, axis=0))
        except (ModelError, DecodeError):
            raise
        except Exception as exc:
            if is_cuda_error(exc):
                raise
            raise ModelError(f"model failed on a batch: {exc}") from exc
        for name, arr in out.items():
            arr = np.asarray(arr)
            if arr.ndim == 0 or arr.shape[0] < len(chunk):
                raise ModelError(f"output {name!r} has shape {arr.shape}; the first axis must be the batch")
            parts.setdefault(name, []).append(arr[: len(chunk)])
        if off // step % 10 == 0:
            log(f"  {min(off + step, len(paths))}/{len(paths)} inputs done")
    try:
        return {name: np.concatenate(chunks, axis=0) for name, chunks in parts.items()}
    except ValueError as exc:
        raise ModelError(f"model output shape changes between batches: {exc}") from exc


def main():
    ap = argparse.ArgumentParser(description="TANUH platform inference (stage 1)")
    ap.add_argument("--format", required=True, choices=FORMATS)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = ap.parse_args()

    started = time.perf_counter()
    paths = [Path(line.strip()) for line in Path(args.inputs).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if not paths:
        log("no inputs listed")
        return 1
    log(f"format={args.format} inputs={len(paths)}")

    try:
        runtime = RUNTIMES[args.format](Path(args.model_dir))
        log(f"runtime={runtime.name} {runtime.version} device={runtime.device} providers={runtime.providers}")
        if runtime.spec:
            log(f"input spec: {asdict(runtime.spec)}")
        outputs = run_inference(runtime, paths, max(1, args.batch_size))
    except DecodeError as exc:
        log(f"DATASET DECODE ERROR: {exc}")
        return EXIT_DECODE
    except ModelError as exc:
        log(f"MODEL ERROR: {exc}")
        return EXIT_MODEL
    except Exception as exc:
        if is_cuda_error(exc):
            log(f"CUDA ERROR: {exc}")
            return EXIT_CUDA
        raise

    names = safe_output_names(list(outputs))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = np.array([p.name for p in paths])
    np.savez(out_dir / "raw_outputs.npz", ids=ids, **{names[k]: v for k, v in outputs.items()})
    meta = {
        "format": args.format,
        "runtime": runtime.name,
        "runtime_version": runtime.version,
        "device": runtime.device,
        "providers": runtime.providers,
        "num_inputs": len(paths),
        "input_spec": asdict(runtime.spec) if runtime.spec else None,
        "outputs": {names[k]: {"model_name": k, "shape": list(v.shape)} for k, v in outputs.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"wrote raw outputs {[(names[k], list(v.shape)) for k, v in outputs.items()]} "
        f"in {meta['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
