#!/usr/bin/env python3
"""
Build a platform-ready TorchScript model: input preparation inside the model,
plus the tanuh.json input declaration the platform reads.

    python3 wrap_torchscript.py --mode ocs --src mvit2_fold5_2_latest_traced.pt --dst ocs.pt
    python3 wrap_torchscript.py --mode none --src my_model.pt --dst my_model_tanuh.pt \
        --input-shape 3 224 224 --resize bilinear

TorchScript files do not record their input shape, so a platform-ready .pt
carries a small JSON declaration saved alongside the model:

    torch.jit.save(model, "model.pt", _extra_files={"tanuh.json": json.dumps({
        "input_shape": [3, 256, 256], "layout": "NCHW", "dtype": "float32", "resize": "bicubic"})})

The platform then feeds RGB input resized to that H x W and scaled to [0,1].

Modes:
  ocs       OCS colour balancing (same steps as wrap_onnx.py --mode ocs, computed in
            the same precision as the previous OCS evaluator), then RGB -> BGR.
            Defaults: input 3x256x256, resize bicubic.
  imagenet  (x - mean) / std with ImageNet statistics. Default resize bilinear.
  none      no preparation; only adds tanuh.json (for models that already
            normalise internally).
The wrapped model is scripted (torch.jit.script), so the original .pt is called unchanged.
"""

import argparse
import json
from pathlib import Path

import torch


class OcsPreprocess(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor):
        t = torch.round(x * 255.0)                                   # exact 0..255 values
        m = t.to(torch.float64).mean(dim=(2, 3), keepdim=True)       # per-channel mean, float64
        scale = torch.where(m == 0, torch.ones_like(m), 128.0 / m).to(torch.float32)
        s = torch.clamp(t * scale, 0.0, 255.0)
        r = torch.floor(torch.round(s * 10.0) / 10.0)
        return self.inner((r / 255.0).flip(1))                       # RGB -> BGR


class ImagenetPreprocess(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        return self.inner((x - self.mean) / self.std)


DEFAULTS = {"ocs": ([3, 256, 256], "bicubic"), "imagenet": ([3, 224, 224], "bilinear"), "none": (None, "bilinear")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=sorted(DEFAULTS))
    ap.add_argument("--src", required=True, help="an existing TorchScript .pt")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--input-shape", type=int, nargs=3, metavar=("C", "H", "W"))
    ap.add_argument("--resize", choices=["bilinear", "bicubic"])
    args = ap.parse_args()

    default_shape, default_resize = DEFAULTS[args.mode]
    shape = args.input_shape or default_shape
    if shape is None:
        ap.error("--input-shape is required with --mode none")
    inner = torch.jit.load(args.src, map_location="cpu").eval()
    model = {"ocs": OcsPreprocess, "imagenet": ImagenetPreprocess}.get(args.mode)
    scripted = torch.jit.script(model(inner).eval()) if model else inner

    spec = {"input_shape": list(shape), "layout": "NCHW", "dtype": "float32",
            "resize": args.resize or default_resize}
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(dst), _extra_files={"tanuh.json": json.dumps(spec)})
    with torch.inference_mode():
        out = scripted(torch.rand(1, *shape))
    print(f"wrote {dst} with tanuh.json={spec}; smoke-test output shape {tuple(out.shape)}")


if __name__ == "__main__":
    main()
