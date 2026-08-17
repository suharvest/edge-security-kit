#!/usr/bin/env python3
"""Convert a YOLOv8n ONNX to RKNN for RK3588 / RK3576.

Runs only on x86_64: RKNN Toolkit 2 has no aarch64 wheel. The toolkit version
must match the runtime the board actually loads -- radxa's
``/usr/lib/librknnrt.so`` reports ``2.3.2`` even though the file is named
``.so.2.3.0``, so 2.3.2 is what this converts with.

Normalization is pushed into the model (``mean_values=0``, ``std_values=255``),
so the runtime is fed raw uint8 NHWC RGB and no float conversion happens on the
CPU.

Two ONNX graphs are supported and the difference matters more than it looks:

``ultralytics``
    The stock export. One ``(1, 84, 8400)`` output with DFL already applied
    inside the graph. Portable, but the DFL block is a softmax over a
    transposed 64-channel tensor that RKNPU2 cannot place, so it lands on the
    CPU-ish fallback path and drags the whole graph.

``zoo``
    ``airockchip/rknn_model_zoo``'s export. Nine outputs -- per stride, a
    64-channel box distribution, an 80-channel class map, and a 1-channel
    class-score sum -- with DFL and the final concat removed. Every remaining
    op is one RKNPU2 has an int8 kernel for. The removed maths is done in NumPy
    by ``esk_rknn.rknn_yolo``, which is why the score sum is exported: it lets
    the decoder reject almost every anchor before the expensive DFL.

``--dtype int8`` requires ``--dataset``: a text file of image paths, one per
line, that the toolkit runs the float graph over to record activation ranges.
See ``../calibration/`` -- the ranges *are* the quantization, and a generic
calibration set is the usual reason an int8 detector loses small targets.
"""
from __future__ import annotations

import argparse
import hashlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--platform", choices=("rk3576", "rk3588"), default="rk3588")
    ap.add_argument("--dtype", choices=("fp16", "int8"), default="fp16")
    ap.add_argument("--dataset", help="calibration path list, required for int8")
    ap.add_argument(
        "--optimization-level", type=int, default=3, choices=(0, 1, 2, 3)
    )
    args = ap.parse_args()

    quantize = args.dtype == "int8"
    if quantize and not args.dataset:
        print("--dtype int8 requires --dataset", file=sys.stderr)
        return 2

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    config: dict = {
        "mean_values": [[0, 0, 0]],
        "std_values": [[255, 255, 255]],
        "target_platform": args.platform,
        "optimization_level": args.optimization_level,
    }
    if quantize:
        # Per-channel asymmetric int8. The detection head's channels have wildly
        # different dynamic ranges -- a per-tensor scale sized for the box
        # regression leaves the class logits with a handful of levels.
        config["quantized_dtype"] = "w8a8"
        config["quantized_algorithm"] = "normal"
        config["quantized_method"] = "channel"
    rknn.config(**config)

    if rknn.load_onnx(model=args.onnx):
        print("load_onnx failed", file=sys.stderr)
        return 1

    build_kwargs: dict = {"do_quantization": quantize}
    if quantize:
        build_kwargs["dataset"] = args.dataset
    if rknn.build(**build_kwargs):
        print("build failed", file=sys.stderr)
        return 1
    if rknn.export_rknn(args.out):
        print("export_rknn failed", file=sys.stderr)
        return 1
    rknn.release()

    digest = hashlib.sha256()
    with open(args.out, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    print(f"RKNN_SHA256 {digest.hexdigest()}  {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
