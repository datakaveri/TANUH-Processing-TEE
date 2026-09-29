#!/usr/bin/env python3
"""
Build a platform-ready ONNX model by putting its input preparation inside the graph.

The platform feeds every model a standard input (RGB, resized to the model's
H x W, scaled to [0,1]). A model trained with extra preprocessing must do that
preprocessing itself. This tool prepends it to an existing ONNX graph and sets
the `tanuh.resize` metadata key:

    python3 wrap_onnx.py --mode imagenet --src resnet34_density_trained.onnx --dst bcd.onnx
    python3 wrap_onnx.py --mode ocs      --src mobilevitv2_ocs_trained.onnx  --dst ocs.onnx

Modes:
  imagenet  (x - mean) / std with ImageNet statistics             resize: bilinear
  ocs       colour balancing used to train the OCS model:         resize: bicubic
              v = round(x*255); scale each channel so its mean is 128
              (mean in float64); clip 0..255; floor(round(v,1)); /255;
              then RGB -> BGR channel order
            This reproduces the previous OCS evaluator's built-in preprocessing
            step for step, so scores match the old flow.

The output keeps external weights in <dst>.data, like the originals.
Requires the `onnx` package (and `onnxruntime` for the --check smoke test).
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GraphBuilder:
    def __init__(self, prefix):
        self.prefix = prefix
        self.nodes, self.inits = [], []
        self.n = 0

    def const(self, value, dtype):
        name = f"{self.prefix}c{len(self.inits)}"
        self.inits.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(self, op_type, inputs, out=None, **attrs):
        self.n += 1
        out = out or f"{self.prefix}t{self.n}"
        self.nodes.append(helper.make_node(op_type, inputs, [out], **attrs))
        return out


def build_imagenet(b, x, out):
    mean = b.const(np.array(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1), np.float32)
    std = b.const(np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1), np.float32)
    b.op("Div", [b.op("Sub", [x, mean]), std], out=out)


def build_ocs(b, x, out):
    f = np.float32
    t = b.op("Round", [b.op("Mul", [x, b.const(255.0, f)])])                  # back to exact 0..255 values
    td = b.op("Cast", [t], to=TensorProto.DOUBLE)
    mean = b.op("ReduceMean", [td, b.const([2, 3], np.int64)], keepdims=1)    # per channel, float64
    scale = b.op("Div", [b.const(128.0, np.float64), mean])
    scale = b.op("Where", [b.op("Equal", [mean, b.const(0.0, np.float64)]),
                           b.const(1.0, np.float64), scale])
    scaled = b.op("Mul", [t, b.op("Cast", [scale], to=TensorProto.FLOAT)])
    clipped = b.op("Clip", [scaled, b.const(0.0, f), b.const(255.0, f)])
    ten = b.const(10.0, f)
    rounded = b.op("Floor", [b.op("Div", [b.op("Round", [b.op("Mul", [clipped, ten])]), ten])])
    unit = b.op("Div", [rounded, b.const(255.0, f)])
    b.op("Gather", [unit, b.const([2, 1, 0], np.int64)], out=out, axis=1)     # RGB -> BGR


MODES = {"imagenet": (build_imagenet, "bilinear"), "ocs": (build_ocs, "bicubic")}


def wrap(src, dst, mode):
    builder_fn, resize = MODES[mode]
    model = onnx.load(str(src))  # also loads external weights from src's folder
    graph = model.graph
    x = graph.input[0].name
    x_prepared = f"{x}__tanuh_prepared"
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == x:
                node.input[i] = x_prepared
    b = GraphBuilder("tanuh_pre_")
    builder_fn(b, x, x_prepared)
    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(b.nodes + existing)
    graph.initializer.extend(b.inits)

    props = {p.key: p for p in model.metadata_props}
    if "tanuh.resize" in props:
        props["tanuh.resize"].value = resize
    else:
        model.metadata_props.append(onnx.StringStringEntryProto(key="tanuh.resize", value=resize))

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = dst.with_name(dst.name + ".data")
    if data.exists():
        data.unlink()
    onnx.save_model(model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data.name, size_threshold=1024)
    onnx.checker.check_model(str(dst))
    return dst, data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    dst, data = wrap(Path(args.src), Path(args.dst), args.mode)
    print(f"wrote {dst} (+ {data.name}), tanuh.resize={MODES[args.mode][1]}")


if __name__ == "__main__":
    main()
