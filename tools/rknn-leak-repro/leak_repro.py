#!/usr/bin/env python3
"""Minimal reproduction: rknn_toolkit_lite2 grows the heap once per inference.

Run one graph through ``rknnlite.api.RKNNLite`` in a steady loop and sample
``VmRSS`` from ``/proc/self/status`` on a fixed interval. On RV1126B with
rknn_toolkit_lite2 2.3.2 / librknnrt 2.3.2 this reports a growth of tens of
kB per ``inference()`` call that never levels off and never comes back.

Two misreadings this script is built to rule out
------------------------------------------------

1. *"You forgot to release the outputs."*
   ``RKNNLite.inference()`` owns the whole output path. There is no
   caller-visible output handle to release: the method fetches the tensors,
   converts them, and is documented to free the runtime buffers before it
   returns. The only release call available to a caller is ``RKNNLite.release()``,
   which tears down the entire context and is therefore not something an
   inference loop can call per frame. This script calls it exactly once, at
   the very end, in a ``finally`` block.

2. *"You are re-initialising the model every frame."*
   ``load_rknn()`` and ``init_runtime()`` are each called exactly once, before
   the loop starts, and never again. Grep this file: each name appears once as
   a call. The measured loop body contains a single RKNN call --
   ``rknn.inference(...)`` -- and nothing else.

The input buffer is allocated once, outside the loop, and never rewritten, so
nothing on the Python side grows by construction. The returned list is dropped
immediately with ``del``; holding it would make Python the obvious suspect.

Requires ``/dev/rknpu``, which is root-only on reCamera Pro. A non-root run
fails inside ``init_runtime()`` with

    failed to open rknpu module, need to insmod rknpu dirver!

which reads like a missing kernel module and is in fact a permission denial.
See README.md.

Dependencies: numpy, rknn_toolkit_lite2. Nothing else -- this file is meant to
be copied onto any RV1126B / RK3588 board on its own.

Usage:
    python3 leak_repro.py --model yolov8n.rknn --seconds 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# Sampling. Kept byte-for-byte identical to ctypes_control.py: same warmup,
# same interval, same kB/inference arithmetic. If the two differed, the two
# numbers could not be compared, which is the entire point of the pair.
# --------------------------------------------------------------------------


def rss_kb():
    """Resident set size in kB -- the number the OOM killer acts on."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return None


def heap_bytes():
    """Size of the anonymous ``[heap]`` VMA.

    Reported alongside RSS so the growth can be attributed: RSS climbing while
    ``[heap]`` stays put would mean something else (a new mapping, file-backed
    pages) rather than unfreed ``malloc``.
    """
    total = 0
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                if line.rstrip().endswith("[heap]"):
                    lo, hi = line.split()[0].split("-")
                    total += int(hi, 16) - int(lo, 16)
    except OSError:
        return None
    return total


def free_mb():
    """``MemAvailable`` in MiB.

    ``MemFree`` would be the wrong guard input: most of what a board like this
    holds is reclaimable page cache, so ``MemFree`` reads alarmingly low while
    the system is healthy. ``MemAvailable`` is the kernel's own estimate of
    what a new allocation could actually obtain.
    """
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    return None


def take_sample(iterations, started):
    return {
        "t_s": round(time.monotonic() - started, 2),
        "iterations": iterations,
        "rss_kb": rss_kb(),
        "heap_bytes": heap_bytes(),
        "free_mb": round(free_mb() or 0.0, 1),
    }


def summarize(samples, extra=None):
    """kB per inference from the first to the last post-warmup sample."""
    out = dict(extra or {})
    out["samples"] = samples
    if len(samples) >= 2:
        first, last = samples[0], samples[-1]
        d_kb = last["rss_kb"] - first["rss_kb"]
        d_it = last["iterations"] - first["iterations"]
        d_s = last["t_s"] - first["t_s"]
        out["rss_delta_kb"] = d_kb
        out["iterations_measured"] = d_it
        if d_it > 0:
            out["kb_per_inference"] = round(d_kb / d_it, 4)
        if d_s > 0:
            out["mb_per_min"] = round(d_kb / 1024.0 / (d_s / 60.0), 3)
    return out


def print_report(title, result):
    print()
    print(f"=== {title} ===")
    print(f"{'t_s':>9} {'iters':>9} {'rss_kb':>10} {'heap_MB':>9} {'free_MB':>9}")
    for s in result["samples"]:
        heap = "" if s["heap_bytes"] is None else f"{s['heap_bytes'] / 1e6:.1f}"
        print(f"{s['t_s']:>9.2f} {s['iterations']:>9} {s['rss_kb']:>10} "
              f"{heap:>9} {s['free_mb']:>9.1f}")
    print()
    for key in ("iterations_measured", "rss_delta_kb", "kb_per_inference",
                "mb_per_min", "stopped_by", "infer_ms_p50"):
        if key in result:
            print(f"{key:>22}: {result[key]}")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def query_model(rknn):
    """Ask the loaded model for its shape rather than assuming 640x640.

    ``RKNNLite`` exposes no query API of its own, but the Cython runtime it
    wraps does: ``get_in_out_num()`` returns ``(n_input, n_output)`` and
    ``get_tensor_attr(index, is_input=True)`` returns one ``rknn_tensor_attr``.
    Both are used here so the script adapts to whatever graph it is handed.
    ``--input-shape`` is the escape hatch if a future runtime renames them.

    Returns ``(input_shape_nhwc, n_input, n_output)``.
    """
    runtime = getattr(rknn, "rknn_runtime", None)
    if runtime is None or not hasattr(runtime, "get_tensor_attr"):
        raise RuntimeError("this rknnlite build exposes no get_tensor_attr; "
                           "pass --input-shape N,H,W,C")
    n_input, n_output = runtime.get_in_out_num()
    attr = runtime.get_tensor_attr(0)
    shape = [int(attr.dims[i]) for i in range(int(attr.n_dims))]
    # rknnlite takes NHWC uint8 at the API boundary. A graph queried as NCHW
    # still has to be fed NHWC, so transpose the queried dims rather than
    # hard-coding a square 640.
    if int(attr.fmt) == 0 and len(shape) == 4:  # 0 == RKNN_TENSOR_NCHW
        n, c, h, w = shape
        shape = [n, h, w, c]
    return tuple(shape), int(n_input), int(n_output)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="path to a .rknn model")
    ap.add_argument("--seconds", type=float, default=300.0,
                    help="measured duration, excluding warmup (default 300)")
    ap.add_argument("--sample-every", type=float, default=15.0,
                    help="seconds between RSS samples (default 15)")
    ap.add_argument("--warmup", type=int, default=20,
                    help="inferences run before measurement starts (default 20)")
    ap.add_argument("--max-iterations", type=int, default=0,
                    help="stop after this many measured inferences (0 = no cap)")
    ap.add_argument("--min-free-mb", type=float, default=250.0,
                    help="abort if MemAvailable drops below this (default 250)")
    ap.add_argument("--input-shape", default="",
                    help="override the queried input shape, e.g. 1,640,640,3")
    ap.add_argument("--json", default="", help="also write the result as JSON here")
    args = ap.parse_args(argv)

    from rknnlite.api import RKNNLite

    rknn = RKNNLite()

    # --- one-time setup. Both calls happen here and nowhere else. -----------
    if rknn.load_rknn(args.model) != 0:
        print(f"load_rknn failed for {args.model}", file=sys.stderr)
        return 1
    if rknn.init_runtime() != 0:
        print("init_runtime failed -- if the log above says 'failed to open "
              "rknpu module, need to insmod rknpu dirver!' this is a permission "
              "denial on /dev/rknpu, not a missing driver. See README.md.",
              file=sys.stderr)
        return 1
    # --- end of setup. Nothing below re-initialises anything. ---------------

    n_input = n_output = None
    if args.input_shape:
        shape = tuple(int(x) for x in args.input_shape.split(","))
    else:
        shape, n_input, n_output = query_model(rknn)

    # Allocated once, never rewritten. Not all zero, so the graph does work.
    frame = np.zeros(shape, dtype=np.uint8)
    frame[..., 1] = 128

    infer_ms = []
    result = {
        "script": "leak_repro.py",
        "path": "rknnlite.api.RKNNLite.inference",
        "model": args.model,
        "input_shape": list(shape),
        "n_input": n_input,
        "n_output": n_output,
        "warmup": args.warmup,
        "sample_every_s": args.sample_every,
    }
    print(f"model {args.model}\n  input {tuple(shape)} uint8 NHWC, "
          f"{n_input} input(s), {n_output} output(s)")

    try:
        for _ in range(args.warmup):
            outputs = rknn.inference(inputs=[frame])
            if outputs is not None and "out_shapes" not in result:
                result["out_shapes"] = [list(np.asarray(o).shape) for o in outputs]
            del outputs

        started = time.monotonic()
        deadline = started + args.seconds
        next_sample = started
        iterations = 0
        samples = []

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_sample:
                samples.append(take_sample(iterations, started))
                next_sample = now + args.sample_every

            t0 = time.perf_counter()
            outputs = rknn.inference(inputs=[frame])
            infer_ms.append((time.perf_counter() - t0) * 1000.0)
            # Dropped immediately: holding the list would make the Python side
            # the obvious suspect and defeat the point of the measurement.
            del outputs

            iterations += 1
            if args.max_iterations and iterations >= args.max_iterations:
                result["stopped_by"] = "max_iterations"
                break
            avail = free_mb()
            if avail is not None and avail < args.min_free_mb:
                result["stopped_by"] = "min_free_mb"
                result["stopped_free_mb"] = round(avail, 1)
                break
            if len(infer_ms) > 20000:
                del infer_ms[:10000]

        samples.append(take_sample(iterations, started))
        if infer_ms:
            ordered = sorted(infer_ms)
            result["infer_ms_p50"] = round(ordered[len(ordered) // 2], 3)
        result = summarize(samples, result)
    finally:
        # The only release call rknnlite offers a caller. It tears down the
        # whole context, so it cannot be a per-frame operation -- see the
        # module docstring, point 1.
        rknn.release()

    print_report("rknnlite baseline", result)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
