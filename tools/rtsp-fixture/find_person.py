#!/usr/bin/env python3
"""Scan source_720p.mp4 for the clearest standing person and cut a patch.

Writes:
  person_patch.png   the crop (BGR)
  person_patch.json  {src_frame, score, patch_w, patch_h, box_in_patch}
  probe_report.json  detection probe of the patch pasted on the synthetic bg
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Paths and endpoints are overridable so this runs outside the machine it was
# written on. Defaults match the layout the acceptance run used.
FIX = Path(os.environ.get("ESK_FIXTURE_DIR", Path.home() / "edge-security-fixture"))
GENERIC = os.environ.get(
    "ESK_GENERIC_DIR", str(Path(__file__).resolve().parents[2] / "platforms" / "generic")
)
sys.path.insert(0, GENERIC)
from esk_generic.yolo import PersonDetector  # noqa: E402

SRC = FIX / "source_720p.mp4"
MODEL = os.environ.get("ESK_MODEL", str(Path(GENERIC) / "models" / "yolov8n.onnx"))

W, H = 1280, 720


def make_background(w: int = W, h: int = H) -> np.ndarray:
    """Textured grey background: no person, but not a flat plane either.

    A flat frame compresses to nothing and gives x264 no work; the grid and
    noise keep the bitrate realistic without introducing person-like shapes.
    """
    rng = np.random.default_rng(1337)
    bg = np.zeros((h, w, 3), np.uint8)
    # vertical gradient 70..130
    col = np.linspace(70, 130, h, dtype=np.float32)
    bg[:] = col[:, None, None]
    # floor line + a couple of wall seams, purely cosmetic
    cv2.line(bg, (0, int(h * 0.78)), (w, int(h * 0.78)), (150, 150, 150), 2)
    for x in range(0, w, 160):
        cv2.line(bg, (x, 0), (x, int(h * 0.78)), (95, 95, 95), 1)
    noise = rng.normal(0, 4, (h, w, 3))
    return np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main() -> int:
    det = PersonDetector(MODEL, conf_threshold=0.35, iou_threshold=0.45)
    cap = cv2.VideoCapture(str(SRC))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"source frames={total} size={cap.get(3)}x{cap.get(4)} fps={cap.get(5)}")

    best = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 15 == 0:  # ~1 Hz
            dets = det(frame)
            if len(dets) == 1:
                d = dets[0]
                x1, y1, x2, y2 = d.box
                bw, bh = x2 - x1, y2 - y1
                # prefer tall (standing) and confident
                if bh > bw:
                    rank = d.score + bh
                    if best is None or rank > best["rank"]:
                        best = {"rank": rank, "idx": idx, "score": d.score,
                                "box": [x1, y1, x2, y2], "frame": frame.copy()}
        idx += 1
    cap.release()
    if best is None:
        print("no single tall person found", file=sys.stderr)
        return 1
    print(f"best frame={best['idx']} score={best['score']:.3f} box={best['box']}")

    frame = best["frame"]
    x1, y1, x2, y2 = best["box"]
    px1, py1 = int(x1 * W), int(y1 * H)
    px2, py2 = int(x2 * W), int(y2 * H)
    m = 8  # margin so the silhouette is not cut at the box edge
    px1, py1 = max(0, px1 - m), max(0, py1 - m)
    px2, py2 = min(W, px2 + m), min(H, py2 + m)
    patch = frame[py1:py2, px1:px2].copy()
    cv2.imwrite(str(FIX / "person_patch.png"), patch)
    cv2.imwrite(str(FIX / "person_source_frame.jpg"), frame)
    meta = {"src_frame": best["idx"], "src_score": best["score"],
            "src_box_norm": best["box"],
            "patch_w": patch.shape[1], "patch_h": patch.shape[0]}
    (FIX / "person_patch.json").write_text(json.dumps(meta, indent=2))
    print("patch:", meta)

    # --- probe: paste the patch on the synthetic background and re-detect ---
    bg = make_background()
    cv2.imwrite(str(FIX / "background.png"), bg)
    report = {"patch": meta, "background_only": None, "positions": []}
    d0 = det(bg)
    report["background_only"] = [{"score": d.score, "box": d.box} for d in d0]
    print(f"background-only detections: {len(d0)}")

    ph, pw = patch.shape[:2]
    for cx, cy in [(0.20, 0.60), (0.40, 0.60), (0.50, 0.60), (0.70, 0.60), (0.85, 0.60)]:
        canvas = bg.copy()
        ox = int(cx * W - pw / 2)
        oy = int(cy * H - ph / 2)
        ox = max(0, min(W - pw, ox))
        oy = max(0, min(H - ph, oy))
        canvas[oy:oy + ph, ox:ox + pw] = patch
        dets = det(canvas)
        entry = {"want_cx": cx, "want_cy": cy,
                 "paste_cx": (ox + pw / 2) / W, "paste_cy": (oy + ph / 2) / H,
                 "n": len(dets),
                 "dets": [{"score": round(d.score, 3),
                           "cx": round((d.box[0] + d.box[2]) / 2, 4),
                           "cy": round((d.box[1] + d.box[3]) / 2, 4)}
                          for d in dets]}
        report["positions"].append(entry)
        print(entry)
        cv2.imwrite(str(FIX / f"probe_{int(cx*100)}.jpg"), canvas)

    (FIX / "probe_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
