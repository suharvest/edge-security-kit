"""Per-frame tracker trace. Diagnostic only; off unless ``ESK_TRACE_DIR`` is set.

Written to answer one question with data instead of argument: when a *stationary*
person is handed a new ``track_id``, is it because the detector stopped
producing a box (miss), or because the box moved enough that IoU association
failed (drift)? Those two have opposite fixes, and the published detections
alone cannot tell them apart -- a frame with no box and a frame whose box failed
to associate both look like "a new id appeared".

So the trace records, per frame:

* every detection the tracker was given (``dets``), and
* every detection that *would* have existed at a lower confidence
  (``cand``, gated by ``ESK_TRACE_CONF``), which is what makes a miss legible as
  "the box was still there at 0.28" rather than as an absence, and
* the tracker's live tracks **before** ``update()`` ran, with the best IoU each
  one achieves against each detection. Computed here rather than read back out
  of the tracker, because after ``update()`` the boxes have already been
  overwritten.

Frames are kept in a small half-resolution ring and flushed to disk only around
an id change, so a long run costs one JSONL line per frame and nothing else.
"""

from __future__ import annotations

import collections
import json
import os
import time

import numpy as np


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


def _round_box(box):
    return [round(float(v), 5) for v in box]


class Tracer:
    def __init__(
        self,
        out_dir: str,
        ring: int = 8,
        after: int = 4,
        max_events: int = 8,
        scale: int = 2,
    ) -> None:
        self.dir = out_dir
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl = open(os.path.join(self.dir, "trace.jsonl"), "a", buffering=1)
        self.ring = collections.deque(maxlen=ring)
        self.after = after
        self.max_events = max_events
        self.scale = max(1, int(scale))
        self.events = 0
        self._pending = []          # [(event_name, frames_left, [paths])]
        self._prev_ids: set[int] = set()
        self._prev_t = None

    # -- called immediately BEFORE tracker.update() -------------------------
    @staticmethod
    def snapshot(tracker):
        return [
            (tid, list(tr.box), float(tr.score), float(tr.last_seen), int(tr.missed))
            for tid, tr in sorted(tracker.tracks.items())
        ]

    # -- called immediately AFTER tracker.update() -------------------------
    def frame(self, *, frame_id, now, tracker, before, dets, cand, tracked, rgb):
        live_ids = {tid for tid, *_ in before}
        out_ids = {tr.track_id for tr, _ in tracked}
        new_ids = sorted(out_ids - live_ids)

        # Best IoU each detection reaches against any pre-update track, and the
        # symmetric view: the best any track reaches against any detection. A
        # near-1.0 value on a frame that still minted a new id means the track
        # was already gone when the detection arrived -- i.e. a miss, not drift.
        assoc = []
        for di, det in enumerate(dets):
            best = (0.0, None)
            for tid, box, *_ in before:
                v = _iou(box, det.box)
                if v > best[0]:
                    best = (v, tid)
            assoc.append([di, round(best[0], 4), best[1]])

        record = {
            "f": int(frame_id),
            "t": round(float(now), 4),
            "dt": None if self._prev_t is None else round(float(now) - self._prev_t, 4),
            "wall": round(time.time(), 3),
            "thr": round(float(tracker.threshold), 4),
            "lost": round(float(tracker.max_lost_sec), 4),
            "dets": [[*_round_box(d.box), round(float(d.score), 4)] for d in dets],
            "cand": [[*_round_box(d.box), round(float(d.score), 4)] for d in cand],
            "before": [
                [tid, *_round_box(box), round(sc, 4), round(float(now) - ls, 4), missed]
                for tid, box, sc, ls, missed in before
            ],
            "assoc": assoc,
            "trk": [
                [tr.track_id, *_round_box(det.box), round(float(det.score), 4)]
                for tr, det in tracked
            ],
            "expired": list(tracker.expired_ids),
            "new": new_ids,
            "next_id": tracker.next_id,
        }
        self.jsonl.write(json.dumps(record) + "\n")
        self._prev_t = float(now)
        self._prev_ids = out_ids

        small = None
        if rgb is not None:
            small = np.ascontiguousarray(rgb[:: self.scale, :: self.scale])
        self.ring.append((int(frame_id), small))

        # flush frames still owed to an event opened a few frames ago
        for item in list(self._pending):
            item["left"] -= 1
            if small is not None:
                self._save(item["name"], int(frame_id), small)
            if item["left"] <= 0:
                self._pending.remove(item)

        if (new_ids or tracker.expired_ids) and self.events < self.max_events:
            self.events += 1
            name = f"ev{self.events:02d}_f{int(frame_id)}"
            for fid, img in self.ring:
                if img is not None:
                    self._save(name, fid, img)
            self._pending.append({"name": name, "left": self.after})
            with open(os.path.join(self.dir, "events.jsonl"), "a") as fh:
                fh.write(
                    json.dumps(
                        {"name": name, "frame": int(frame_id), "new": new_ids,
                         "expired": list(tracker.expired_ids)}
                    )
                    + "\n"
                )

    def _save(self, name, frame_id, img) -> None:
        d = os.path.join(self.dir, name)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{frame_id:06d}.npy")
        if not os.path.exists(path):
            np.save(path, img)

    def close(self) -> None:
        try:
            self.jsonl.close()
        except Exception:
            pass
