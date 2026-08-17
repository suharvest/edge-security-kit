#!/usr/bin/env python3
"""Compose the controlled ground-truth video and export its trajectory table.

1280x720, FPS from argv[1] (default 5), 31.000 s, one person patch translated
along a hardcoded trajectory defined in SECONDS so the wall-clock schedule is
identical at any frame rate. The loop is seamless (last phase returns to the
start x), so ffmpeg -stream_loop -1 produces a continuous track.

The frame rate is a parameter because the generic CPU detector on this host
sustains ~6.2 fps (inference p50 138 ms). Publishing faster than the consumer
makes the OpenCV RTSP reader accumulate and then fast-forward, which skips
trajectory segments and starves the tracker past track_max_lost_s. 5 fps is
under the measured capacity, so every published frame is consumed and the
observed trajectory equals the designed one.

Rule geometry this trajectory is designed against (all frame_norm):
  line   start=(0.50, 0.05) end=(0.50, 0.95)
         side(p) = (0)*(py-0.05) - (0.90)*(px-0.50) = -0.90*(px-0.50)
         => px < 0.5 -> side +1 ; px > 0.5 -> side -1
         contracts/MQTT.md: forward = side>0 -> side<0  =>  LEFT-to-RIGHT is forward
  zone   rectangle x in [0.05, 0.30], y in [0.35, 0.90], dwell_seconds = 10
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2

# Paths and endpoints are overridable so this runs outside the machine it was
# written on. Defaults match the layout the acceptance run used.
FIX = Path(os.environ.get("ESK_FIXTURE_DIR", Path.home() / "edge-security-fixture"))
GENERIC = os.environ.get(
    "ESK_GENERIC_DIR", str(Path(__file__).resolve().parents[2] / "platforms" / "generic")
)
sys.path.insert(0, GENERIC)
from esk_generic.yolo import PersonDetector  # noqa: E402

W, H = 1280, 720
FPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODEL = os.environ.get("ESK_MODEL", str(Path(GENERIC) / "models" / "yolov8n.onnx"))

LINE = {"start": [0.50, 0.05], "end": [0.50, 0.95]}
ZONE_X = (0.05, 0.30)
ZONE_Y = (0.35, 0.90)
ZONE_POINTS = [[ZONE_X[0], ZONE_Y[0]], [ZONE_X[1], ZONE_Y[0]],
               [ZONE_X[1], ZONE_Y[1]], [ZONE_X[0], ZONE_Y[1]]]
DWELL_SECONDS = 10.0
DURATION_S = 31.0

# (t_start_s, t_end_s, cx_at_start, cx_at_end, label) -- the schedule in seconds
PHASES_S = [
    (0.0, 2.0, 0.40, 0.40, "settle-left-of-line"),
    (2.0, 5.0, 0.40, 0.70, "walk-right-CROSS-forward"),
    (5.0, 7.0, 0.70, 0.70, "stand-right"),
    (7.0, 10.0, 0.70, 0.40, "walk-left-CROSS-backward"),
    (10.0, 12.0, 0.40, 0.40, "stand-mid"),
    (12.0, 14.0, 0.40, 0.20, "walk-into-zone"),
    (14.0, 29.0, 0.20, 0.20, "dwell-in-zone-15s"),
    (29.0, 31.0, 0.20, 0.40, "walk-out-of-zone"),
]
N_FRAMES = int(round(DURATION_S * FPS))


def trajectory() -> tuple[list[float], list[dict]]:
    """cx per frame plus the phase table resolved to frame indices."""
    cx = [0.0] * N_FRAMES
    table = []
    for t0, t1, a, b, label in PHASES_S:
        f0 = int(round(t0 * FPS))
        f1 = int(round(t1 * FPS)) - 1  # last frame belonging to this phase
        span = (t1 - t0) * FPS
        for f in range(f0, min(f1 + 1, N_FRAMES)):
            cx[f] = a + (b - a) * (f - f0) / span
        table.append({"f0": f0, "f1": f1, "t0_s": t0, "t1_s": t1,
                      "cx0": a, "cx1": b, "label": label})
    return cx, table


def side(px: float) -> int:
    s, e = LINE["start"], LINE["end"]
    cross = (e[0] - s[0]) * (0.5 - s[1]) - (e[1] - s[1]) * (px - s[0])
    return 1 if cross > 0 else (-1 if cross < 0 else 0)


def in_zone(px: float, py: float) -> bool:
    return ZONE_X[0] <= px <= ZONE_X[1] and ZONE_Y[0] <= py <= ZONE_Y[1]


def main() -> int:
    patch = cv2.imread(str(FIX / "person_patch.png"))
    bg = cv2.imread(str(FIX / "background.png"))
    ph, pw = patch.shape[:2]
    # The patch is 662 px tall in a 720 px frame: there is no vertical freedom,
    # so cy is a constant for the whole clip.
    oy = max(0, min(H - ph, int(0.60 * H - ph / 2)))
    cy_paste = (oy + ph / 2) / H

    cx_want, phase_table = trajectory()
    # The paste offset is an integer, so the realized cx is quantized. The truth
    # table records the REALIZED value, which is what the detector can see.
    ox = [max(0, min(W - pw, int(round(c * W - pw / 2)))) for c in cx_want]
    cx_real = [(o + pw / 2) / W for o in ox]

    raw = FIX / "truth_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in range(N_FRAMES):
        canvas = bg.copy()
        canvas[oy:oy + ph, ox[f]:ox[f] + pw] = patch
        cv2.putText(canvas, f"f{f:04d} t={f / FPS:6.3f}s", (W - 330, H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(canvas)
    writer.release()

    # --- ground truth events, derived from the realized trajectory ----------
    events = []
    for f in range(1, N_FRAMES):
        s_prev, s_curr = side(cx_real[f - 1]), side(cx_real[f])
        if s_prev > 0 and s_curr < 0:
            direction = "forward"
        elif s_prev < 0 and s_curr > 0:
            direction = "backward"
        else:
            continue
        frac = (0.50 - cx_real[f - 1]) / (cx_real[f] - cx_real[f - 1])
        events.append({"event_type": "line_cross", "direction": direction,
                       "detect_frame": f, "detect_t_s": round(f / FPS, 4),
                       "exact_cross_t_s": round((f - 1 + frac) / FPS, 4),
                       "cx_prev": round(cx_real[f - 1], 5),
                       "cx_curr": round(cx_real[f], 5)})

    inside = [in_zone(c, cy_paste) for c in cx_real]
    for f in range(N_FRAMES):
        was = inside[f - 1] if f > 0 else False
        if inside[f] and not was:
            events.append({"event_type": "zone_enter", "detect_frame": f,
                           "detect_t_s": round(f / FPS, 4), "cx": round(cx_real[f], 5)})
            events.append({"event_type": "loitering", "dwell_seconds": DWELL_SECONDS,
                           "detect_frame": f + int(DWELL_SECONDS * FPS),
                           "detect_t_s": round(f / FPS + DWELL_SECONDS, 4),
                           "entered_frame": f,
                           "note": "first frame at which dwell_s >= dwell_seconds"})
        if was and not inside[f]:
            events.append({"event_type": "zone_exit", "detect_frame": f,
                           "detect_t_s": round(f / FPS, 4), "cx": round(cx_real[f], 5)})
    events.sort(key=lambda e: e["detect_frame"])

    truth = {
        "video": "truth.mp4", "width": W, "height": H, "fps": FPS,
        "n_frames": N_FRAMES, "period_s": round(N_FRAMES / FPS, 4),
        "patch": {"w": pw, "h": ph, "oy": oy},
        "geometry": {"line": LINE, "zone_points": ZONE_POINTS,
                     "dwell_seconds": DWELL_SECONDS,
                     "direction_convention":
                         "side(p) = -0.90*(px-0.50); px<0.5 -> +1. "
                         "forward = +1 -> -1 = left-to-right"},
        "phases": phase_table,
        "cy_const": round(cy_paste, 5),
        "cx_by_frame": [round(c, 5) for c in cx_real],
        "events": events,
    }
    (FIX / "truth.json").write_text(json.dumps(truth, indent=2))
    print(json.dumps({k: v for k, v in truth.items() if k != "cx_by_frame"}, indent=2))

    out = FIX / "truth.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(raw), "-an", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "20", "-pix_fmt", "yuv420p",
                    "-g", str(2 * FPS), "-keyint_min", str(2 * FPS), "-bf", "0",
                    str(out)], check=True)

    # --- verify the encoded file still yields one person per frame ----------
    det = PersonDetector(MODEL, conf_threshold=0.35, iou_threshold=0.45)
    cap = cv2.VideoCapture(str(out))
    event_frames = {e["detect_frame"] for e in events}
    probe = sorted(event_frames | {f - 1 for f in event_frames} | {0, N_FRAMES - 1,
                   N_FRAMES // 2})
    probe = [f for f in probe if 0 <= f < N_FRAMES]
    checks = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f in probe:
            dets = det(frame)
            best = max(dets, key=lambda d: d.score) if dets else None
            dcx = (best.box[0] + best.box[2]) / 2 if best else None
            checks.append({"frame": f, "n": len(dets),
                           "truth_cx": round(cx_real[f], 5),
                           "det_cx": round(dcx, 5) if dcx else None,
                           "det_cy": round((best.box[1] + best.box[3]) / 2, 5) if best else None,
                           "err_cx": round(abs(dcx - cx_real[f]), 5) if dcx else None,
                           "in_zone_truth": in_zone(cx_real[f], cy_paste),
                           "in_zone_detected": in_zone(dcx, (best.box[1] + best.box[3]) / 2)
                           if best else None,
                           "score": round(best.score, 3) if best else None})
        f += 1
    cap.release()
    (FIX / "truth_probe.json").write_text(json.dumps(checks, indent=2))
    print("\nencoded-video probe:")
    for c in checks:
        print(" ", c)
    bad = [c for c in checks
           if c["n"] != 1 or (c["score"] or 0) < 0.5 or (c["err_cx"] or 1) > 0.01
           or c["in_zone_truth"] != c["in_zone_detected"]]
    print("\nFAILED PROBES:", bad if bad else "none")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
