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
                TorchScript:  model.pt   (saved with torch.jit.save)
                Hugging Face: hf/        (config.json, *.safetensors, preprocessor_config.json)
                ONNX / TorchScript may also hold input_spec.json, uploaded next to the model
  --inputs      one "<id>\t<path>" line per sample, in ground-truth order; <id> is
                the sample's path in the dataset (a line without a tab: id = file name)
  --output-dir  receives raw_outputs.npz (`ids` + one array per model output) and meta.json

Platform input contract (ONNX and TorchScript) — the platform turns every file
into exactly the tensor the model declares, so no preprocessing script exists:
  1. decode:   by content, not extension. DICOM -> middle frame, MONOCHROME1
               inverted, min-max to 0..255, grey -> 3 channels; images (JPEG, PNG,
               TIFF, BMP, WebP, GIF) -> RGB uint8
  2. resize:   to the model's H x W; "bilinear" (default, Pillow) or "bicubic" (OpenCV)
  3. channels: 1 (grey) or 3 (RGB) and layout NCHW or NHWC, as the model declares
  4. dtype:    float -> [0,1]; uint8 -> 0..255
Anything model-specific (normalisation, colour balancing, channel order) must be
inside the model. Hugging Face models use their own preprocessor_config.json instead.

Where the declaration comes from (later wins):
  ONNX         the model's input signature, metadata_props key "tanuh.resize",
               then input_spec.json (resize; H x W for a model with dynamic size)
  TorchScript  JSON embedded at save time as extra/tanuh.json or extra/input_spec.json,
                 torch.jit.save(m, f, _extra_files={"tanuh.json": json.dumps(
                     {"input_shape": [3, 256, 256], "resize": "bicubic"})})
               then input_spec.json. At least the input size must come from one of them.
  Hugging Face preprocessor_config.json (applied by transformers' AutoImageProcessor)
The spec's keys are flexible: input_shape / shape / input_size / image_size / size
(an int, [H, W], [C, H, W], [N, C, H, W] or {"height", "width"}), height, width,
channels, layout / data_format, dtype, resize / interpolation / resample.

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
DICOM_EXTS = {".dcm", ".dicom", ".dic"}
IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"II*\x00", b"MM\x00*", b"BM", b"GIF8")
RESIZE_METHODS = ("bilinear", "bicubic")
SPEC_FILE = "input_spec.json"                       # uploaded next to the model
EMBEDDED_SPEC_NAMES = ("tanuh.json", "input_spec.json")  # TorchScript extra/ files
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
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:  # formats OpenCV does not read (GIF, some TIFF variants)
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"))
    except Exception as exc:
        raise DecodeError(f"cannot decode image {Path(path).name}") from exc


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


def file_kind(path):
    """"dicom", "image" or "unknown", from the file's first bytes (the
    extension only decides for a DICOM file without its "DICM" preamble)."""
    with open(path, "rb") as f:
        head = f.read(132)
    if len(head) >= 132 and head[128:132] == b"DICM":
        return "dicom"
    if head.startswith(IMAGE_MAGIC) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return "image"
    if Path(path).suffix.lower() in DICOM_EXTS:
        return "dicom"
    return "unknown"


def decode(path):
    """Return the file as uint8 (H, W, 3) RGB, or raise DecodeError."""
    try:
        kind = file_kind(path)
        if kind == "dicom":
            return decode_dicom(path)
        if kind == "image":
            return decode_image(path)
        try:  # unknown: an image format without a known signature, else a DICOM without preamble
            return decode_image(path)
        except DecodeError:
            return decode_dicom(path)
    except DecodeError:
        raise
    except Exception as exc:
        raise DecodeError(f"cannot decode {Path(path).name}: {exc}") from exc


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


RESIZE_ALIASES = {"bilinear": "bilinear", "linear": "bilinear", "inter_linear": "bilinear", "2": "bilinear",
                  "bicubic": "bicubic", "cubic": "bicubic", "inter_cubic": "bicubic", "3": "bicubic"}
DTYPE_ALIASES = {"float32": "float32", "float": "float32", "fp32": "float32", "f32": "float32",
                 "float16": "float16", "half": "float16", "fp16": "float16", "f16": "float16",
                 "uint8": "uint8", "u8": "uint8"}
LAYOUT_ALIASES = {"nchw": "NCHW", "chw": "NCHW", "channels_first": "NCHW",
                  "nhwc": "NHWC", "hwc": "NHWC", "channels_last": "NHWC"}
SHAPE_KEYS = ("input_shape", "shape", "input_size", "image_size", "img_size", "size", "resolution")
# Preprocessing the platform does not apply: silently ignoring these would feed
# the model differently from what its author expects.
UNSUPPORTED_SPEC_KEYS = {"mean", "std", "normalize", "normalization", "image_mean", "image_std",
                         "channel_order", "color_order", "colour_order", "bgr"}


def _norm_key(k):
    return re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_")


def _positive_int(v, what, source):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or int(v) <= 0:
        raise ModelError(f"{source}: {what} must be a positive whole number, got {v!r}")
    return int(v)


def parse_input_spec(raw, source):
    """Read a flexible input declaration into {channels, height, width,
    layout, dtype, resize}, holding only what it states."""
    if not isinstance(raw, dict):
        raise ModelError(f"{source} must be a JSON object")
    keys = {_norm_key(k): v for k, v in raw.items()}
    unsupported = sorted(k for k in keys if k in UNSUPPORTED_SPEC_KEYS)
    if unsupported:
        raise ModelError(f"{source}: {unsupported} are not applied by the platform; "
                         f"put that preprocessing inside the model")
    out = {}
    layout = keys.get("layout", keys.get("data_format"))
    if layout is not None:
        out["layout"] = LAYOUT_ALIASES.get(str(layout).lower())
        if out["layout"] is None:
            raise ModelError(f"{source}: layout must be NCHW or NHWC, got {layout!r}")

    shape = next((keys[k] for k in SHAPE_KEYS if k in keys), None)
    if isinstance(shape, (int, float)) and not isinstance(shape, bool):
        out["height"] = out["width"] = _positive_int(shape, "size", source)
    elif isinstance(shape, dict):
        dims = {_norm_key(k): v for k, v in shape.items()}
        h, w = dims.get("height", dims.get("h")), dims.get("width", dims.get("w"))
        if h is None or w is None:
            raise ModelError(f"{source}: size object needs height and width, got {shape!r}")
        out["height"], out["width"] = _positive_int(h, "height", source), _positive_int(w, "width", source)
    elif isinstance(shape, (list, tuple)):
        dims = list(shape)
        if len(dims) == 4:
            dims = dims[1:]  # [N, ...]: the batch size is not part of the spec
        dims = [_positive_int(d, "every dimension", source) for d in dims]
        if len(dims) == 3:
            lay = out.get("layout")
            if lay is None:  # channels are the 1-or-3 end
                lay = "NHWC" if dims[2] in (1, 3) and dims[0] not in (1, 3) else "NCHW"
                out["layout"] = lay
            c, h, w = dims if lay == "NCHW" else (dims[2], dims[0], dims[1])
            out.update(channels=c, height=h, width=w)
        elif len(dims) == 2:
            out["height"], out["width"] = dims
        elif len(dims) == 1:
            out["height"] = out["width"] = dims[0]
        else:
            raise ModelError(f"{source}: input shape must be [H, W], [C, H, W] or [N, C, H, W], got {shape!r}")
    elif shape is not None:
        raise ModelError(f"{source}: cannot read input size {shape!r}")
    for key in ("height", "width"):
        if key in keys:
            out[key] = _positive_int(keys[key], key, source)
    for key in ("channels", "num_channels", "in_channels"):
        if key in keys:
            out["channels"] = _positive_int(keys[key], "channels", source)
    if "dtype" in keys:
        out["dtype"] = DTYPE_ALIASES.get(str(keys["dtype"]).lower())
        if out["dtype"] is None:
            raise ModelError(f"{source}: dtype must be float32, float16 or uint8, got {keys['dtype']!r}")
    resize_value = next((keys[k] for k in ("resize", "interpolation", "resample", "resize_method") if k in keys), None)
    if resize_value is not None:
        out["resize"] = RESIZE_ALIASES.get(_norm_key(resize_value))
        if out["resize"] is None:
            raise ModelError(f"{source}: resize must be bilinear or bicubic, got {resize_value!r}")
    return out


def read_uploaded_spec(model_dir):
    """input_spec.json uploaded next to the model, or {} when there is none."""
    path = model_dir / SPEC_FILE
    if not path.exists():
        return {}, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelError(f"{SPEC_FILE} is not valid JSON: {exc}") from exc
    return parse_input_spec(raw, SPEC_FILE), SPEC_FILE


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
        declared = parse_input_spec({"resize": meta["tanuh.resize"]}, "metadata tanuh.resize") \
            if "tanuh.resize" in meta else {}
        uploaded, uploaded_source = read_uploaded_spec(model_dir)
        self.spec = self._spec(inputs[0], {**declared, **uploaded})
        self.spec_source = " + ".join(["model input signature"] + (["metadata tanuh.resize"] if declared else [])
                                      + ([uploaded_source] if uploaded_source else []))

    @staticmethod
    def _spec(inp, given):
        """The model's input signature, completed (dynamic sizes) or refined
        (resize) by the declared spec. A spec that contradicts a fixed
        dimension of the model is an error, not a silent override."""
        shape = list(inp.shape)
        if len(shape) != 4:
            raise ModelError(f"input must be 4-D (batch, ...), declared {shape}")
        dtype = {"tensor(float)": "float32", "tensor(float16)": "float16",
                 "tensor(uint8)": "uint8"}.get(inp.type)
        if dtype is None:
            raise ModelError(f"unsupported input type {inp.type}")
        if given.get("dtype") not in (None, dtype):
            raise ModelError(f"the input spec says dtype={given['dtype']} but the model's input is {dtype}")
        dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
        if given.get("layout"):
            layout = given["layout"]
        elif dims[3] in (1, 3) and dims[1] not in (1, 3):
            layout = "NHWC"
        else:
            layout = "NCHW"
        c, h, w = (dims[1], dims[2], dims[3]) if layout == "NCHW" else (dims[3], dims[1], dims[2])
        for name, model_value in (("channels", c), ("height", h), ("width", w)):
            if model_value is not None and given.get(name) not in (None, model_value):
                raise ModelError(f"the input spec says {name}={given[name]} but the model's input is {shape}")
        c = c or given.get("channels", 3)
        h, w = h or given.get("height"), w or given.get("width")
        if not h or not w:
            raise ModelError(f"the model's input {shape} has a dynamic height/width; give the size in "
                             f"{SPEC_FILE} (e.g. {{\"input_size\": [224, 224]}})")
        fixed = dims[0]
        return InputSpec(c, h, w, layout, dtype, given.get("resize") or "bilinear", fixed).validate()

    def run(self, batch):
        outs = self.session.run(None, {self.input_name: batch})
        return {o.name: np.asarray(v) for o, v in zip(self.session.get_outputs(), outs)}


def read_embedded_spec(model_path):
    """The input declaration saved inside a TorchScript archive
    (extra/tanuh.json or extra/input_spec.json), read without executing
    anything; {} when there is none."""
    try:
        with zipfile.ZipFile(model_path) as zf:
            names = zf.namelist()
            if not any("/code/" in n for n in names) or not any(n.endswith("constants.pkl") for n in names):
                raise ModelError("model.pt is not a TorchScript archive (save it with torch.jit.save)")
            for wanted in EMBEDDED_SPEC_NAMES:
                entries = [n for n in names if n.endswith("/extra/" + wanted)]
                if entries:
                    try:
                        raw = json.loads(zf.read(entries[0]).decode("utf-8-sig"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise ModelError(f"embedded {wanted} is not valid JSON: {exc}") from exc
                    return parse_input_spec(raw, f"embedded {wanted}"), f"embedded {wanted}"
    except zipfile.BadZipFile as exc:
        raise ModelError("model.pt is not a TorchScript archive (save it with torch.jit.save)") from exc
    return {}, None


def torchscript_spec(model_dir, model_path):
    """TorchScript files do not record their input size, so it comes from an
    embedded declaration and/or input_spec.json uploaded next to the model
    (the uploaded one wins). Everything but the size has a default."""
    embedded, embedded_source = read_embedded_spec(model_path)
    uploaded, uploaded_source = read_uploaded_spec(model_dir)
    spec = {"channels": 3, "layout": "NCHW", "dtype": "float32", "resize": "bilinear", **embedded, **uploaded}
    if "height" not in spec or "width" not in spec:
        raise ModelError("the TorchScript model's input size is unknown: give it on the submit form "
                         f"(uploaded as {SPEC_FILE}, e.g. {{\"input_size\": [224, 224]}}) or embed it with "
                         "torch.jit.save(m, f, _extra_files={'tanuh.json': '{\"input_size\": [224, 224]}'})")
    source = " + ".join(s for s in (embedded_source, uploaded_source) if s)
    return InputSpec(spec["channels"], spec["height"], spec["width"], spec["layout"], spec["dtype"],
                     spec["resize"], None).validate(), source


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
        self.spec, self.spec_source = torchscript_spec(model_dir, model_path)
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
        self.spec_source = "preprocessor_config.json"

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


def read_inputs(path):
    """(ids, paths) from "<id>\\t<path>" lines; a line without a tab is a bare
    path whose id is its file name."""
    ids, paths = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample_id, sep, file_path = line.partition("\t")
        if not sep:
            file_path = sample_id.strip()
            sample_id = Path(file_path).name
        ids.append(sample_id)
        paths.append(Path(file_path))
    return ids, paths


def main():
    ap = argparse.ArgumentParser(description="TANUH platform inference (stage 1)")
    ap.add_argument("--format", required=True, choices=FORMATS)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    args = ap.parse_args()

    started = time.perf_counter()
    ids, paths = read_inputs(Path(args.inputs))
    if not paths:
        log("no inputs listed")
        return 1
    log(f"format={args.format} inputs={len(paths)}")

    try:
        runtime = RUNTIMES[args.format](Path(args.model_dir))
        log(f"runtime={runtime.name} {runtime.version} device={runtime.device} providers={runtime.providers}")
        if runtime.spec:
            log(f"input spec: {asdict(runtime.spec)} (from {runtime.spec_source})")
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
    np.savez(out_dir / "raw_outputs.npz", ids=np.array(ids), **{names[k]: v for k, v in outputs.items()})
    meta = {
        "format": args.format,
        "runtime": runtime.name,
        "runtime_version": runtime.version,
        "device": runtime.device,
        "providers": runtime.providers,
        "num_inputs": len(paths),
        "input_spec": asdict(runtime.spec) if runtime.spec else None,
        "input_spec_source": runtime.spec_source,
        "outputs": {names[k]: {"model_name": k, "shape": list(v.shape)} for k, v in outputs.items()},
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"wrote raw outputs {[(names[k], list(v.shape)) for k, v in outputs.items()]} "
        f"in {meta['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
