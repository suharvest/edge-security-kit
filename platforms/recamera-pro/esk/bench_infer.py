"""A bare ``inference()`` loop, to cut the RSS question in half.

The 12-13 MB/min RSS growth (README, "RSS growth: where it is, and where it is
not") was narrowed by elimination to "malloc'd memory a C extension holds and
never frees", with the RKNN runtime as the only native component common to both
frame paths. Elimination is not proof: the pipeline also runs NumPy head decode,
a tracker, an MQTT client, a JPEG encoder and an HTTP server, and any of those
could hold native memory too.

This module removes all of them. It calls ``model.infer()`` on a **constant**
pre-allocated array in a tight loop -- no frame source, no letterbox, no decode,
no tracker, no publisher, no preview server -- and samples RSS on a fixed
interval. If RSS climbs here, everything except the RKNN runtime has been
excluded by construction. If it does not climb, the runtime is exonerated and
the evidence chain has to continue into our own code.

Two controls make the answer harder to fake:

``mode="idle"``
    the same loop, the same cadence, the same sampling, **no inference call**.
    A leak that shows up here is the harness, not the runtime.

``model_path=<other .rknn>``
    the same loop against a different graph. The shipped ``fall-detection`` app
    ran 12 h at a flat 263 MB through this very same ``kit.runtime.engine``
    wrapper, so if the runtime leaks it must be output-shape dependent -- this
    app's ``rknn_model_zoo`` head returns nine tensors per frame, that one's pose
    head returns fewer. Running both graphs in one harness turns that from a
    story into a measurement.

Samples go to a JSON file rather than stdout because appmgr sends app stdout to
a root-owned ``logs/app.log`` that the fleet account cannot read.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np


def _rss_kb():
    """RSS from ``/proc/self/status``; the number the OOM killer acts on."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def _heap_bytes():
    """Size of the anonymous ``[heap]`` VMA -- where the growth was localized.

    ``rss_kb`` moving without this moving would mean the growth changed
    character (file-backed, or a fresh mapping) rather than continuing.
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


def _mapping_count():
    try:
        with open("/proc/self/maps") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return None


def sample(iterations: int, started: float) -> dict:
    return {
        "t_s": round(time.monotonic() - started, 2),
        "iterations": iterations,
        "rss_kb": _rss_kb(),
        "heap_bytes": _heap_bytes(),
        "mapping_count": _mapping_count(),
        "wall": time.strftime("%H:%M:%S"),
    }


def load_model(path: str):
    """Open an ``.rknn`` through the kit's own engine wrapper.

    Deliberately the same class the app uses (``kit.runtime.engine.RknnModel``)
    rather than a fresh ``RKNNLite``: the question is whether *this* call path
    leaks, so substituting a different one would answer a different question.
    """
    from kit.runtime.engine import RknnModel

    return RknnModel(path)


def run_bench(
    model=None,
    *,
    model_path: str = "",
    input_shape=(1, 640, 640, 3),
    seconds: float = 600.0,
    sample_every_s: float = 30.0,
    out_path: str = "/userdata/esk-fixture/bench.json",
    mode: str = "infer",
    target_fps: float = 0.0,
    label: str = "",
) -> dict:
    """Run the loop and write a sampled RSS curve to ``out_path``.

    ``model`` is the already-loaded handle (the app's ``self.models.det``);
    ``model_path`` loads a second graph instead, which is how the
    output-shape-dependence hypothesis gets tested. ``target_fps`` throttles to
    a realistic rate so per-inference arithmetic is comparable with the live
    pipeline; 0 runs flat out, which reaches the same iteration count sooner.
    """
    owned = None
    if model is None:
        if not model_path:
            raise ValueError("run_bench needs either a model handle or a model_path")
        owned = model = load_model(model_path)

    # One buffer, allocated once, never rewritten. Any growth therefore cannot
    # be input churn -- there is exactly one input object for the whole run.
    frame = np.zeros(input_shape, dtype=np.uint8)
    frame[..., 1] = 128  # not all-zero, so the graph does real work

    started = time.monotonic()
    deadline = started + float(seconds)
    next_sample = started
    interval = 1.0 / target_fps if target_fps > 0 else 0.0

    samples = []
    out_shapes = None
    infer_ms = []
    iterations = 0
    result = {
        "mode": mode,
        "label": label or mode,
        "model_path": model_path or getattr(model, "path", ""),
        "input_shape": list(input_shape),
        "seconds_requested": seconds,
        "sample_every_s": sample_every_s,
        "target_fps": target_fps,
        "pid": os.getpid(),
        "samples": samples,
    }

    def flush():
        result["iterations"] = iterations
        result["out_shapes"] = out_shapes
        if infer_ms:
            ordered = sorted(infer_ms)
            result["infer_ms_p50"] = round(ordered[len(ordered) // 2], 3)
            result["infer_ms_p95"] = round(
                ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 3
            )
        if len(samples) >= 2:
            first, last = samples[0], samples[-1]
            span_min = (last["t_s"] - first["t_s"]) / 60.0
            if span_min > 0 and first["rss_kb"] and last["rss_kb"]:
                delta_kb = last["rss_kb"] - first["rss_kb"]
                result["rss_slope_kb_per_min"] = round(delta_kb / span_min, 1)
                result["rss_slope_mb_per_min"] = round(delta_kb / 1024.0 / span_min, 3)
                done = last["iterations"] - first["iterations"]
                if done > 0:
                    result["kb_per_inference"] = round(delta_kb / done, 4)
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(result, fh, indent=1)
        os.replace(tmp, out_path)
        # World-readable: appmgr apps are root, the reader is uid 1000.
        try:
            os.chmod(out_path, 0o644)
        except OSError:
            pass

    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_sample:
                samples.append(sample(iterations, started))
                next_sample = now + sample_every_s
                flush()

            if mode == "infer":
                t0 = time.perf_counter()
                outputs = model.infer(frame)
                infer_ms.append((time.perf_counter() - t0) * 1000.0)
                if out_shapes is None and outputs is not None:
                    out_shapes = [list(np.asarray(o).shape) for o in outputs]
                # Dropped immediately. Holding the list would make Python the
                # obvious suspect and defeat the point of the experiment.
                del outputs
            elif mode == "idle":
                time.sleep(0.001)
            else:
                raise ValueError(f"unknown bench mode {mode!r}")

            iterations += 1
            if len(infer_ms) > 20000:
                del infer_ms[:10000]
            if interval:
                slack = interval - (time.monotonic() - now)
                if slack > 0:
                    time.sleep(slack)
    finally:
        samples.append(sample(iterations, started))
        flush()
        if owned is not None:
            owned.release()
    return result
