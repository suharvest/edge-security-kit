#!/usr/bin/env python3
"""Control for leak_repro.py: the same API sequence, without rknn_toolkit_lite2.

``rknnlite.api.RKNNLite.inference()`` is a thin Python shell over a closed
Cython extension (``rknn_runtime.cpython-*-aarch64-linux-gnu.so``) which in turn
calls ``librknnrt.so``. Sampling from outside the process cannot say which of
those two layers drops the ``free``.

This script removes the middle layer. It drives ``librknnrt`` directly over
ctypes, performing the API sequence the Cython extension performs -- its
``.dynstr`` names exactly these entry points, and carries the string
``Release outputs failed, ret code:`` --

    rknn_init                                   once
    rknn_query  IN_OUT_NUM / INPUT_ATTR / OUTPUT_ATTR / SDK_VERSION
    per inference:
        rknn_inputs_set
        rknn_run
        rknn_outputs_get      want_float = 1
        rknn_outputs_release

with the same warmup, the same sampling interval and the same kB/inference
arithmetic as leak_repro.py, so the two numbers are directly comparable.

Positive control (``--omit-release``)
-------------------------------------

Every ``rknn_outputs_release`` is skipped and nothing else changes. A harness
that reports "flat" proves nothing unless it can be shown to detect a leak of
the size in question; this mode is that demonstration. It is *deliberately*
broken code -- do not copy it into anything.

It is also dangerous to run unbounded: each iteration retains the full
dequantized output set (about 4.9 MB for a 9-output yolov8n head), which
exhausts a 1.27 GB headroom in roughly ten seconds and takes the board off the
network. So this mode is capped by construction: ``--max-iterations`` defaults
to 40 (about 196 MB) and ``MemAvailable`` is checked against ``--min-free-mb``
(default 250) after *every* iteration -- at 4.9 MB a call, checking every 32nd
iteration would leave a 157 MB blind spot, which is most of the headroom the
guard exists to defend.

Every ``restype``/``argtypes`` below is load-bearing. ctypes types an
unprototyped return as C ``int``, which on aarch64 truncates a returned pointer
to 32 bits; a dereference of the truncated value takes the interpreter down.

Dependencies: ctypes (stdlib) and numpy. Nothing else -- this file is meant to
be copied onto any RV1126B / RK3588 board on its own.

Usage:
    python3 ctypes_control.py --model yolov8n.rknn --seconds 300
    python3 ctypes_control.py --model yolov8n.rknn --omit-release
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import time

import numpy as np

DEFAULT_LIB = "/usr/lib/librknnrt.so"

RKNN_SUCC = 0
RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256

# rknn_query_cmd, in the header's declared order. SDK_VERSION is 5, not 3:
# 3 is PERF_DETAIL, whose struct is a pointer plus a length, so querying it
# into an RknnSdkVersion returns RKNN_SUCC and fills the buffer with garbage
# instead of failing.
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_QUERY_PERF_DETAIL = 3
RKNN_QUERY_PERF_RUN = 4
RKNN_QUERY_SDK_VERSION = 5

# rknn_tensor_type / rknn_tensor_format
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1

# aarch64 is LP64, so the header's non-__arm__ branch applies.
rknn_context = ctypes.c_uint64


class RknnTensorAttr(ctypes.Structure):
    """``rknn_tensor_attr``, field order and types verbatim from the header.

    ``fl`` (int8) sitting in front of ``zp`` (int32) is the one place a
    hand-packed layout goes wrong; ctypes inserts the same three padding bytes
    the C compiler does, so this must NOT be declared ``_pack_``-ed.
    """

    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class RknnInputOutputNum(ctypes.Structure):
    _fields_ = [("n_input", ctypes.c_uint32), ("n_output", ctypes.c_uint32)]


class RknnInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class RknnOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


class RknnSdkVersion(ctypes.Structure):
    _fields_ = [("api_version", ctypes.c_char * 256),
                ("drv_version", ctypes.c_char * 256)]


def load_lib(path):
    """dlopen ``librknnrt`` and prototype every entry point used below."""
    lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

    lib.rknn_init.restype = ctypes.c_int
    lib.rknn_init.argtypes = [ctypes.POINTER(rknn_context), ctypes.c_void_p,
                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]

    lib.rknn_destroy.restype = ctypes.c_int
    lib.rknn_destroy.argtypes = [rknn_context]

    lib.rknn_query.restype = ctypes.c_int
    lib.rknn_query.argtypes = [rknn_context, ctypes.c_int, ctypes.c_void_p,
                               ctypes.c_uint32]

    lib.rknn_inputs_set.restype = ctypes.c_int
    lib.rknn_inputs_set.argtypes = [rknn_context, ctypes.c_uint32,
                                    ctypes.POINTER(RknnInput)]

    lib.rknn_run.restype = ctypes.c_int
    lib.rknn_run.argtypes = [rknn_context, ctypes.c_void_p]

    lib.rknn_outputs_get.restype = ctypes.c_int
    lib.rknn_outputs_get.argtypes = [rknn_context, ctypes.c_uint32,
                                     ctypes.POINTER(RknnOutput), ctypes.c_void_p]

    lib.rknn_outputs_release.restype = ctypes.c_int
    lib.rknn_outputs_release.argtypes = [rknn_context, ctypes.c_uint32,
                                         ctypes.POINTER(RknnOutput)]
    return lib


# --------------------------------------------------------------------------
# Sampling. Byte-for-byte identical to leak_repro.py -- if the two differed the
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
    """Size of the anonymous ``[heap]`` VMA."""
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
    """``MemAvailable`` in MiB -- what the abort guard watches."""
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
    for key in ("gc_every", "iterations_measured", "rss_delta_kb", "kb_per_inference",
                "mb_per_min", "stopped_by", "stopped_free_mb", "infer_ms_p50"):
        if key in result:
            print(f"{key:>22}: {result[key]}")


# --------------------------------------------------------------------------


class CtypesRknn:
    """One ``rknn_context``, driven straight from Python."""

    def __init__(self, lib, model_path):
        self.lib = lib
        self.ctx = rknn_context(0)
        self.released = False

        with open(model_path, "rb") as fh:
            blob = fh.read()
        # Held for the object's life. The runtime frees its own copy right
        # after rknn_init, so this is belt-and-braces -- but a freed buffer the
        # runtime did retain a pointer into is not a failure mode worth
        # discovering halfway through a leak run.
        self._blob = ctypes.create_string_buffer(blob, len(blob))
        ret = lib.rknn_init(ctypes.byref(self.ctx), self._blob, len(blob), 0, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(
                f"rknn_init failed: ret={ret}. If the log says 'failed to open "
                f"rknpu module, need to insmod rknpu dirver!', this is a "
                f"permission denial on /dev/rknpu, not a missing driver.")

        ver = RknnSdkVersion()
        self.sdk = {}
        if lib.rknn_query(self.ctx, RKNN_QUERY_SDK_VERSION, ctypes.byref(ver),
                          ctypes.sizeof(ver)) == RKNN_SUCC:
            self.sdk = {"api": ver.api_version.decode("utf-8", "replace"),
                        "drv": ver.drv_version.decode("utf-8", "replace")}

        io = RknnInputOutputNum()
        ret = lib.rknn_query(self.ctx, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(io),
                             ctypes.sizeof(io))
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_query(IN_OUT_NUM) failed: ret={ret}")
        self.n_input = int(io.n_input)
        self.n_output = int(io.n_output)

        self.input_attrs = (RknnTensorAttr * self.n_input)()
        for i in range(self.n_input):
            self.input_attrs[i].index = i
            ret = lib.rknn_query(self.ctx, RKNN_QUERY_INPUT_ATTR,
                                 ctypes.byref(self.input_attrs[i]),
                                 ctypes.sizeof(RknnTensorAttr))
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_query(INPUT_ATTR {i}) failed: ret={ret}")

        self.output_attrs = (RknnTensorAttr * self.n_output)()
        for i in range(self.n_output):
            self.output_attrs[i].index = i
            ret = lib.rknn_query(self.ctx, RKNN_QUERY_OUTPUT_ATTR,
                                 ctypes.byref(self.output_attrs[i]),
                                 ctypes.sizeof(RknnTensorAttr))
            if ret != RKNN_SUCC:
                raise RuntimeError(f"rknn_query(OUTPUT_ATTR {i}) failed: ret={ret}")

        # Reused for the life of the object, so the harness allocates nothing
        # per call and cannot itself be the source of growth.
        self._inputs = (RknnInput * self.n_input)()
        self._outputs = (RknnOutput * self.n_output)()
        self._out_shapes = [
            tuple(self.output_attrs[i].dims[d]
                  for d in range(self.output_attrs[i].n_dims))
            for i in range(self.n_output)
        ]

    def input_shape(self):
        """NHWC uint8 shape to feed, derived from the queried attrs."""
        attr = self.input_attrs[0]
        dims = [int(attr.dims[i]) for i in range(attr.n_dims)]
        if attr.fmt == RKNN_TENSOR_NCHW and len(dims) == 4:
            n, c, h, w = dims
            dims = [n, h, w, c]
        return tuple(dims)

    def infer(self, arr, omit_release=False):
        lib = self.lib
        inp = self._inputs[0]
        inp.index = 0
        inp.buf = arr.ctypes.data_as(ctypes.c_void_p)
        inp.size = arr.nbytes
        inp.pass_through = 0
        inp.type = RKNN_TENSOR_UINT8
        inp.fmt = RKNN_TENSOR_NHWC
        ret = lib.rknn_inputs_set(self.ctx, self.n_input, self._inputs)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_inputs_set failed: ret={ret}")

        ret = lib.rknn_run(self.ctx, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_run failed: ret={ret}")

        for i in range(self.n_output):
            self._outputs[i].want_float = 1
            self._outputs[i].is_prealloc = 0
            self._outputs[i].index = i
            self._outputs[i].buf = None
            self._outputs[i].size = 0
        ret = lib.rknn_outputs_get(self.ctx, self.n_output, self._outputs, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_outputs_get failed: ret={ret}")

        try:
            out = []
            for i in range(self.n_output):
                o = self._outputs[i]
                # Copy *before* release. The header is explicit that
                # rknn_outputs_release frees buf; a np.frombuffer view on the
                # raw pointer would alias freed memory.
                a = np.frombuffer(ctypes.string_at(o.buf, o.size), dtype=np.float32)
                shape = self._out_shapes[i]
                if a.size == int(np.prod(shape)):
                    a = a.reshape(shape)
                out.append(a)
        finally:
            # In a finally block on purpose: an exception between _get and
            # _release would move the leak onto a different path and make the
            # measurement mean something else.
            if not omit_release:
                ret = lib.rknn_outputs_release(self.ctx, self.n_output,
                                               self._outputs)
                if ret != RKNN_SUCC:
                    raise RuntimeError(f"rknn_outputs_release failed: ret={ret}")
        return out

    def release(self):
        if self.released:
            return
        self.released = True
        if self.ctx.value:
            self.lib.rknn_destroy(self.ctx)
            self.ctx = rknn_context(0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="path to a .rknn model")
    ap.add_argument("--lib", default=DEFAULT_LIB, help=f"default {DEFAULT_LIB}")
    ap.add_argument("--seconds", type=float, default=300.0,
                    help="measured duration, excluding warmup (default 300)")
    ap.add_argument("--sample-every", type=float, default=15.0,
                    help="seconds between RSS samples (default 15)")
    ap.add_argument("--warmup", type=int, default=20,
                    help="inferences run before measurement starts (default 20)")
    ap.add_argument("--omit-release", action="store_true",
                    help="POSITIVE CONTROL: skip rknn_outputs_release. Retains "
                         "the whole output set per iteration; capped, see below.")
    ap.add_argument("--max-iterations", type=int, default=-1,
                    help="stop after this many measured inferences. Default: no "
                         "cap normally, 40 with --omit-release.")
    ap.add_argument("--gc-every", type=int, default=0,
                    help="run gc.collect() every N measured inferences "
                         "(0 = never, the default). Forces Python to reclaim "
                         "anything a reference cycle is holding, so growth that "
                         "survives this cannot be a Python-side object pile-up.")
    ap.add_argument("--min-free-mb", type=float, default=250.0,
                    help="abort if MemAvailable drops below this (default 250)")
    ap.add_argument("--json", default="", help="also write the result as JSON here")
    args = ap.parse_args(argv)

    if args.max_iterations < 0:
        # The cap is not advice: at ~4.9 MB retained per iteration an
        # unbounded run exhausts this board's headroom in about ten seconds.
        args.max_iterations = 40 if args.omit_release else 0

    lib = load_lib(args.lib)
    model = CtypesRknn(lib, args.model)
    shape = model.input_shape()

    # Allocated once, never rewritten. Not all zero, so the graph does work.
    frame = np.ascontiguousarray(np.zeros(shape, dtype=np.uint8))
    frame[..., 1] = 128

    label = "ctypes + omit release (POSITIVE CONTROL)" if args.omit_release \
        else "ctypes control"
    result = {
        "script": "ctypes_control.py",
        "path": "librknnrt via ctypes",
        "omit_release": bool(args.omit_release),
        "lib": args.lib,
        "model": args.model,
        "sdk": model.sdk,
        "input_shape": list(shape),
        "n_input": model.n_input,
        "n_output": model.n_output,
        "out_shapes": [list(s) for s in model._out_shapes],
        "warmup": args.warmup,
        "sample_every_s": args.sample_every,
        "gc_every": args.gc_every,
        "max_iterations": args.max_iterations,
        "min_free_mb": args.min_free_mb,
    }

    print(f"model {args.model}\n  input {tuple(shape)} uint8 NHWC, "
          f"{model.n_input} input(s), {model.n_output} output(s)\n"
          f"  librknnrt {model.sdk.get('api', '?')} driver {model.sdk.get('drv', '?')}")

    infer_ms = []
    try:
        for _ in range(args.warmup):
            # Warmup always releases, even in the positive-control run: the
            # control has to start from the same baseline as the normal one.
            del_ = model.infer(frame, omit_release=False)
            del del_

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
            outputs = model.infer(frame, omit_release=args.omit_release)
            infer_ms.append((time.perf_counter() - t0) * 1000.0)
            del outputs

            iterations += 1
            # Identical placement in both scripts: after the call and after
            # the outputs have been dropped, before any early exit.
            if args.gc_every and iterations % args.gc_every == 0:
                gc.collect()
            if args.max_iterations and iterations >= args.max_iterations:
                result["stopped_by"] = "max_iterations"
                break
            # Checked every iteration. At 4.9 MB a call, checking every 32nd
            # would leave a 157 MB blind spot -- most of the headroom this
            # guard exists to defend.
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
        model.release()

    print_report(label, result)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
