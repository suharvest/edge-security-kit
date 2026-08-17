#!/usr/bin/env python3
"""Draw two models' detections on the same frame, for visual diffing.

An AP table says how much was lost; it does not say what was lost. This runs
two ``.rknn`` files over identical pixels and overlays both results, so the
question "did int8 drop the far one or just move the near one by two pixels"
has a picture attached.

Green is the reference (fp16), magenta is the candidate (int8), and boxes the
two agree on at IoU >= 0.7 are drawn thin -- the eye should be drawn to the
disagreements, which are the only part that carries information.

Runs on the board; RKNN Lite needs the NPU.

    ./compare_annotate.py --reference models/yolov8n_zoo_fp16.rk3588.rknn \
        --candidate models/yolov8n_zoo_int8.rk3588.rknn \
        --conf 0.35 --out-dir /tmp/cmp frame1.jpg frame2.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esk_rknn.letterbox import LetterboxTransform, pad_into_canvas  # noqa: E402
from esk_rknn.rknn_yolo import RKNNPersonDetector  # noqa: E402

REFERENCE_BGR = (80, 220, 80)
CANDIDATE_BGR = (220, 60, 220)


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def run(model: RKNNPersonDetector, image: np.ndarray, size: int):
    height, width = image.shape[:2]
    tf = LetterboxTransform.for_source(width, height, size)
    scaled = cv2.resize(image, (tf.scaled_w, tf.scaled_h), interpolation=cv2.INTER_LINEAR)
    canvas = pad_into_canvas(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB), size)
    return model.detect(canvas, tf)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--input-size", type=int, default=640)
    ap.add_argument("--match-iou", type=float, default=0.7)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Loaded one at a time: two RKNN contexts alive at once is a needless way to
    # find out what this board does when the NPU runs out of context memory.
    frames = {p: cv2.imread(p) for p in args.images}
    for path, image in frames.items():
        if image is None:
            raise SystemExit(f"unreadable: {path}")

    results = {}
    for role, path in (("reference", args.reference), ("candidate", args.candidate)):
        model = RKNNPersonDetector(path, conf_threshold=args.conf,
                                   iou_threshold=args.iou, input_size=args.input_size)
        results[role] = {p: run(model, im, args.input_size) for p, im in frames.items()}
        model.close()

    for path, image in frames.items():
        height, width = image.shape[:2]
        canvas = image.copy()
        ref = results["reference"][path]
        cand = results["candidate"][path]
        matched_r, matched_c = set(), set()
        for i, r in enumerate(ref):
            for j, c in enumerate(cand):
                if j not in matched_c and iou(r.box, c.box) >= args.match_iou:
                    matched_r.add(i)
                    matched_c.add(j)
                    break
        for role, dets, matched, colour in (
            ("fp16", ref, matched_r, REFERENCE_BGR),
            ("int8", cand, matched_c, CANDIDATE_BGR),
        ):
            for k, det in enumerate(dets):
                x1, y1, x2, y2 = det.box
                pt1 = (int(x1 * width), int(y1 * height))
                pt2 = (int(x2 * width), int(y2 * height))
                cv2.rectangle(canvas, pt1, pt2, colour, 1 if k in matched else 3)
                cv2.putText(canvas, f"{role} {det.score:.2f}",
                            (pt1[0], max(12, pt1[1] - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        banner = (f"fp16={len(ref)} int8={len(cand)} "
                  f"agree@IoU{args.match_iou}={len(matched_r)} conf>={args.conf}")
        cv2.putText(canvas, banner, (8, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        out = args.out_dir / f"cmp_{Path(path).stem}.jpg"
        cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"{out}  {banner}")
        for role, dets in (("fp16", ref), ("int8", cand)):
            for det in sorted(dets, key=lambda d: -d.score):
                print(f"    {role} score={det.score:.3f} "
                      f"box={[round(v, 4) for v in det.box]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
