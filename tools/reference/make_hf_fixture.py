#!/usr/bin/env python3
"""
Build a small Hugging Face image-classification package for testing the
platform's Hugging Face path. Nothing is downloaded: the model is built from a
config with random weights, so it exercises loading and inference, not scoring.

    python3 make_hf_fixture.py --out hf_fixture.zip [--num-labels 2]

A platform-ready Hugging Face package is a zip of a save_pretrained() folder:
    config.json               built-in transformers architecture (no auto_map / custom code)
    model.safetensors         weights in safetensors format (no pickle files)
    preprocessor_config.json  the image processor settings the platform applies
"""

import argparse
import tempfile
import zipfile
from pathlib import Path

import torch
from transformers import MobileViTImageProcessor, MobileViTV2Config, MobileViTV2ForImageClassification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-labels", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(0)
    config = MobileViTV2Config(num_labels=args.num_labels, image_size=256, width_multiplier=0.5)
    model = MobileViTV2ForImageClassification(config).eval()
    processor = MobileViTImageProcessor(
        do_resize=True, size={"shortest_edge": 256}, do_center_crop=True,
        crop_size={"height": 256, "width": 256}, do_rescale=True, do_flip_channel_order=True)

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "hf"
        model.save_pretrained(folder, safe_serialization=True)
        processor.save_pretrained(folder)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(folder.iterdir()):
                zf.write(f, f.name)
        names = sorted(p.name for p in folder.iterdir())
    print(f"wrote {out}: {names}")


if __name__ == "__main__":
    main()
