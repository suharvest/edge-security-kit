"""Does the ctypes path return the *same numbers* as rknnlite?

The leak evidence says the ctypes ``get`` sequence is flat where
``RKNNLite.inference`` grows 43.8 kB per call. That is an argument for swapping
the implementation, but only if the swap is invisible to everything downstream.
``kit.runtime.engine.RknnModel`` is shared by every installed app, so "close
enough" is not a usable standard here: a half-LSB drift that moves one anchor
past ``conf_threshold`` changes the detection count, and the app that notices
first will be one nobody re-tested.

So this module asserts equality rather than assuming it. Both engines are
opened against the *same* ``.rknn`` file at the same time, fed the *same*
``uint8`` canvas, and their nine output tensors are compared element by element:
shape, dtype, whether the bytes are identical, and -- if not -- the largest
absolute and relative difference over the whole tensor.

Tensor equality is necessary but not sufficient, because the number that
actually ships is a box. The comparison therefore runs ``decode_zoo_head`` over
both output sets and diffs the post-NMS detections: count, per-box corner
distance in ``frame_norm`` units, and score delta. A run where the tensors
differ in the last mantissa bit but every box matches to 1e-6 is a different
verdict from one where a detection appears or disappears, and the JSON keeps
them apart.

Inputs are real frames off the truth clip plus four synthetic edge cases
(all-black, all-white, mid-grey, deterministic noise). The synthetic ones matter
because a real frame exercises a narrow slice of the activation range; a
saturated input is where a mis-sized output buffer or a wrong dequant scale
would show up, and it costs one extra inference to cover.

Runs as an appmgr app for the same reason ``bench_infer`` does: ``/dev/rknpu``
is 0600 root:root and there is no sudo on this board.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np


def _canvases_from_video(path: str, count: int, input_size: int) -> list:
    """Decode `count` real frames and letterbox each the way the app does."""
    from esk.file_source import FfmpegFileSource
    from esk.letterbox import LetterboxTransform, pad_into_canvas

    src = FfmpegFileSource(path, loop=False, realtime=False)
    tf = LetterboxTransform.for_source(src.w, src.h, input_size)
    out = []
    try:
        for frame in src.frames():
            rgb = frame.data
            if (tf.scaled_w, tf.scaled_h) == (rgb.shape[1], rgb.shape[0]):
                scaled = rgb
            else:
                from PIL import Image

                scaled = np.asarray(
                    Image.fromarray(rgb).resize(
                        (tf.scaled_w, tf.scaled_h), Image.BILINEAR
                    ),
                    dtype=np.uint8,
                )
            out.append(pad_into_canvas(scaled, tf.dst))
            if len(out) >= count:
                break
    finally:
        src.close()
    return out, tf


def _synthetic(input_size: int) -> list:
    """Saturated and noise inputs -- where a numeric defect is loudest."""
    shape = (input_size, input_size, 3)
    rng = np.random.RandomState(20260819)  # fixed: the run must be repeatable
    return [
        ("black", np.zeros(shape, dtype=np.uint8)),
        ("white", np.full(shape, 255, dtype=np.uint8)),
        ("grey128", np.full(shape, 128, dtype=np.uint8)),
        ("noise", rng.randint(0, 256, size=shape).astype(np.uint8)),
    ]


def _compare_tensors(a_list, b_list) -> dict:
    """Per-tensor diff between two output sets. `a` is rknnlite, `b` is ctypes."""
    if len(a_list) != len(b_list):
        return {"count_mismatch": [len(a_list), len(b_list)]}
    tensors = []
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        a = np.asarray(a)
        b = np.asarray(b)
        rec = {
            "index": i,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "dtype_a": str(a.dtype),
            "dtype_b": str(b.dtype),
        }
        if a.shape != b.shape:
            # Compare flat instead of giving up: a reshape-only difference is a
            # different (and much easier) defect than wrong values, and the
            # caller needs to be able to tell which one it is looking at.
            rec["shape_equal"] = False
            fa, fb = a.ravel(), b.ravel()
            if fa.size != fb.size:
                rec["size_mismatch"] = [int(fa.size), int(fb.size)]
                tensors.append(rec)
                continue
        else:
            rec["shape_equal"] = True
            fa, fb = a.ravel(), b.ravel()
        fa = fa.astype(np.float64)
        fb = fb.astype(np.float64)
        rec["bit_identical"] = bool(
            a.dtype == b.dtype
            and a.tobytes() == b.tobytes()
        )
        diff = np.abs(fa - fb)
        rec["max_abs_err"] = float(diff.max()) if diff.size else 0.0
        denom = np.maximum(np.abs(fa), np.abs(fb))
        # Relative error is undefined where both sides are zero; those elements
        # are already covered by max_abs_err == 0 and would otherwise inject a
        # 0/0 nan that hides a real maximum elsewhere in the tensor.
        nz = denom > 0
        rec["max_rel_err"] = float((diff[nz] / denom[nz]).max()) if nz.any() else 0.0
        rec["n_differing"] = int((diff > 0).sum())
        rec["n_elems"] = int(fa.size)
        tensors.append(rec)
    return {"tensors": tensors}


def _compare_dets(da, db) -> dict:
    """Diff two post-NMS detection lists, pairing by sorted box order."""

    def key(d):
        return (round(d.box[0], 6), round(d.box[1], 6))

    da = sorted(da, key=key)
    db = sorted(db, key=key)
    rec = {"n_a": len(da), "n_b": len(db), "count_equal": len(da) == len(db)}
    if not rec["count_equal"]:
        rec["boxes_a"] = [[round(v, 6) for v in d.box] for d in da]
        rec["boxes_b"] = [[round(v, 6) for v in d.box] for d in db]
        return rec
    max_box = 0.0
    max_score = 0.0
    for x, y in zip(da, db):
        max_box = max(max_box, max(abs(p - q) for p, q in zip(x.box, y.box)))
        max_score = max(max_score, abs(x.score - y.score))
    rec["max_box_delta"] = max_box
    rec["max_score_delta"] = max_score
    return rec


def run_equiv(
    *,
    model_path: str,
    video_path: str = "",
    out_path: str = "/userdata/esk-fixture/equiv.json",
    n_frames: int = 60,
    input_size: int = 640,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
) -> dict:
    """Run both engines over the same inputs and write the diff to `out_path`."""
    # Both halves come from ``kit.runtime.engine`` on purpose: what has to be
    # equivalent is the code that actually ships, not a private copy of it.
    from kit.runtime.ctypes_rknn import CtypesRknnModel
    from kit.runtime.engine import RknnLiteModel

    from esk.letterbox import LetterboxTransform
    from esk.zoo_head import decode_zoo_head

    result = {
        "model_path": model_path,
        "video_path": video_path,
        "n_frames_requested": n_frames,
        "input_size": input_size,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "pid": os.getpid(),
        "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": [],
    }

    def flush():
        tmp = out_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(result, fh, indent=1)
        os.replace(tmp, out_path)
        try:
            os.chmod(out_path, 0o644)
        except OSError:
            pass

    inputs = []
    tf = None
    if video_path:
        try:
            frames, tf = _canvases_from_video(video_path, n_frames, input_size)
            inputs.extend((f"frame{i:03d}", c) for i, c in enumerate(frames))
        except Exception as exc:  # noqa: BLE001 -- recorded, run continues
            result["video_error"] = f"{type(exc).__name__}: {exc}"
    if tf is None:
        tf = LetterboxTransform.for_source(input_size, input_size, input_size)
    inputs.extend(_synthetic(input_size))
    result["n_inputs"] = len(inputs)

    lite = None
    ct = None
    try:
        lite = RknnLiteModel(model_path)
        ct = CtypesRknnModel(model_path)
        result["describe_ctypes"] = ct.describe()

        worst = {
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "all_bit_identical": True,
            "all_shapes_equal": True,
            "det_count_mismatches": 0,
            "max_box_delta": 0.0,
            "max_score_delta": 0.0,
        }
        for name, canvas in inputs:
            batched = np.ascontiguousarray(canvas[None])
            out_lite = [np.asarray(o) for o in lite.infer(batched)]
            out_ct = [np.asarray(o) for o in ct.infer(batched)]
            cmp = _compare_tensors(out_lite, out_ct)
            case = {"input": name, **cmp}
            for rec in cmp.get("tensors", []):
                worst["max_abs_err"] = max(worst["max_abs_err"],
                                           rec.get("max_abs_err", 0.0))
                worst["max_rel_err"] = max(worst["max_rel_err"],
                                           rec.get("max_rel_err", 0.0))
                if not rec.get("bit_identical"):
                    worst["all_bit_identical"] = False
                if not rec.get("shape_equal"):
                    worst["all_shapes_equal"] = False
            try:
                da = decode_zoo_head(out_lite, tf, conf_threshold,
                                     iou_threshold, input_size)
                db = decode_zoo_head(out_ct, tf, conf_threshold,
                                     iou_threshold, input_size)
                det = _compare_dets(da, db)
                case["detections"] = det
                if not det["count_equal"]:
                    worst["det_count_mismatches"] += 1
                else:
                    worst["max_box_delta"] = max(worst["max_box_delta"],
                                                 det["max_box_delta"])
                    worst["max_score_delta"] = max(worst["max_score_delta"],
                                                   det["max_score_delta"])
            except Exception as exc:  # noqa: BLE001
                case["decode_error"] = f"{type(exc).__name__}: {exc}"
            result["cases"].append(case)

        worst["equivalent"] = (
            worst["all_shapes_equal"]
            and worst["det_count_mismatches"] == 0
            and worst["max_box_delta"] == 0.0
            and worst["max_score_delta"] == 0.0
        )
        result["summary"] = worst
    except BaseException as exc:  # noqa: BLE001 -- recorded, then re-raised
        import traceback

        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        for handle in (lite, ct):
            if handle is not None:
                try:
                    handle.release()
                except Exception:
                    pass
        flush()
    return result
