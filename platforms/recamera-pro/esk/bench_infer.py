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

``mode="ctypes_get"`` / ``"ctypes_run"`` / ``"ctypes_iomem"`` / ``"ctypes_leak"``
    the same loop with ``rknn_toolkit_lite2`` taken out of the call path
    entirely -- ``librknnrt`` driven straight from Python. Same graph, same
    constant input, same sampler; only the API sequence differs. That is what
    splits "the Cython extension drops the free" from "``librknnrt`` does", a
    question no amount of sampling from outside the process can answer. See
    ``esk/ctypes_infer`` for what each of the four sequences is and which
    hypothesis each one kills. ``ctypes_leak`` is the positive control.

``model_path=<other .rknn>``
    the same loop against a different graph. The shipped ``fall-detection`` app
    ran 12 h at a flat 263 MB through this very same ``kit.runtime.engine``
    wrapper, so if the runtime leaks it must be output-shape dependent -- this
    app's ``rknn_model_zoo`` head returns nine tensors per frame, that one's pose
    head returns fewer. Running both graphs in one harness turns that from a
    story into a measurement.

Samples go to a JSON file rather than stdout because appmgr sends app stdout to
a root-owned ``logs/app.log`` that the fleet account cannot read. For the same
reason ``run_bench`` records any exception into that JSON before re-raising:
otherwise a crash in the loop is indistinguishable from a run that did nothing.

Every mode here has to run *as an appmgr app*, which is the only reason this is
wired into ``app.py`` rather than being a standalone script. ``/dev/rknpu`` is
mode 0600 root:root, the fleet account is uid 1000 with no sudo, and appmgr runs
apps as root -- a script run over SSH gets "failed to open rknpu module, need to
insmod rknpu driver" from ``rknn_init``, which reads like a missing kernel module
and is in fact a permission denial.
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


def _free_mb():
    """``MemAvailable`` in MiB -- the headroom the abort guard watches.

    ``MemFree`` would be the wrong number: most of what this board is holding is
    reclaimable page cache, so ``MemFree`` reads alarmingly low while the box is
    perfectly healthy. ``MemAvailable`` is the kernel's own estimate of what a
    new allocation could actually get.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


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
        "free_mb": _free_mb(),
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
    max_iterations: int = 0,
    min_free_mb: float = 250.0,
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
    describe = None
    if mode.startswith("ctypes_"):
        # The ctypes variants exist to split "which native layer drops the
        # free" -- they must NOT reuse the rknnlite handle, because that handle
        # is the thing under suspicion. Each opens its own rknn_context against
        # the same graph file. See esk/ctypes_infer for what the four do.
        from esk.ctypes_infer import CtypesRknnModel

        path = model_path or getattr(model, "path", "")
        if not path:
            raise ValueError(f"bench mode {mode!r} needs a model path")
        model_path = path
        owned = model = CtypesRknnModel(path, mode=mode[len("ctypes_"):])
        describe = model.describe()
    elif model is None:
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
        "max_iterations": max_iterations,
        "min_free_mb": min_free_mb,
        "pid": os.getpid(),
        "describe": describe,
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

            if mode == "infer" or mode.startswith("ctypes_"):
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
            if max_iterations and iterations >= max_iterations:
                result["stopped_by"] = "max_iterations"
                break
            # Headroom guard. The ctypes_leak control retains ~4.9 MB of output
            # per inference by design; unthrottled that took this board off the
            # network in ten seconds and cost a reboot. A control only has to
            # make the leak *visible*, and by the time headroom is this low it
            # already has -- so stop, and leave the samples behind rather than
            # letting the OOM killer decide what survives. Checked every
            # iteration, not every N: at 4.9 MB a call even N=32 is 157 MB of
            # blind spot, which is most of the headroom this guard is defending.
            # A /proc/meminfo read is tens of microseconds against a ~36 ms
            # inference, so the sampling cost is not the constraint here.
            free_mb = _free_mb()
            if free_mb is not None and free_mb < min_free_mb:
                result["stopped_by"] = "min_free_mb"
                result["stopped_free_mb"] = round(free_mb, 1)
                break
            if len(infer_ms) > 20000:
                del infer_ms[:10000]
            if interval:
                slack = interval - (time.monotonic() - now)
                if slack > 0:
                    time.sleep(slack)
    except BaseException as exc:  # noqa: BLE001 -- recorded, then re-raised
        # Under appmgr the app's stdout is a root-owned ``logs/app.log`` the
        # fleet account cannot read, so a crash in here would surface only as
        # ``"iterations": 0`` with no reason attached -- which is exactly how
        # the first ctypes run failed. ``out_path`` is the one artifact that is
        # world-readable, so the failure has to land in it. ``flush()`` runs
        # after this block (``finally``), so the fields are already set by then.
        import traceback

        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        samples.append(sample(iterations, started))
        flush()
        if owned is not None:
            owned.release()
    return result
