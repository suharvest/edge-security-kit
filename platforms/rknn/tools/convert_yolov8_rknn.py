#!/usr/bin/env python3
"""Convert the stock YOLOv8n ONNX to a fp16 RKNN for RK3588.

Runs only on x86_64: RKNN Toolkit 2 has no aarch64 wheel. The toolkit version
must match the runtime the board actually loads -- radxa's
``/usr/lib/librknnrt.so`` reports ``2.3.2`` even though the file is named
``.so.2.3.0``, so 2.3.2 is what this converts with.

Normalization is pushed into the model (``mean_values=0``, ``std_values=255``),
so the runtime is fed raw uint8 NHWC RGB and no float conversion happens on the
CPU. The head is left intact: the single (1, 84, 8400) output is decoded by the
same NumPy postprocess the generic platform uses, which keeps one decoder
implementation across platforms.
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
    ap.add_argument(
        "--optimization-level", type=int, default=3, choices=(0, 1, 2, 3)
    )
    args = ap.parse_args()

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform=args.platform,
        optimization_level=args.optimization_level,
    )
    if rknn.load_onnx(model=args.onnx):
        print("load_onnx failed", file=sys.stderr)
        return 1
    # fp16 only. w8a8 on a detection head needs a calibration set that matches
    # the deployment scene; without one the score head drifts and the tracker
    # sees flicker instead of a person.
    if rknn.build(do_quantization=False):
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
