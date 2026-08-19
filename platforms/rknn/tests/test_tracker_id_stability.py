"""ID stability against a REAL captured detection sequence.

``test_tracker.py`` checks the tracker's contract on four hand-built frames.
That is enough to catch an inverted comparison and nothing else: every box in
it is where the next box expects it to be, which is exactly the condition the
field does not satisfy.

This file replays ``fixtures/tracker_live_recamera_pro.json`` -- 701 consecutive
frames (49.8 s at 14.2 fps) captured off a reCamera Pro's NPU by
``platforms/recamera-pro/esk/trace.py`` while it watched a static indoor scene
with no person in it. Recorded at confidence 0.05 so the score of a *dropped*
detection is in the record rather than inferred from its absence, which is what
lets these assertions separate the two failure modes:

  * the detector stopped producing a box (score fell under `conf_threshold`),
    the track aged past `track_max_lost_s`, and the next box got a new id; or
  * the box was still there and association failed -- IoU below `track_iou`.

Measured on this capture: **all** of the first, **none** of the second. That is
the property worth freezing. Loosening `track_iou` to buy id stability is the
tempting change and the wrong one; it cannot help here (there is nothing to
associate on the frames that matter) and it makes two adjacent people swap ids,
which is a far harder bug to see. If someone makes that change,
``test_what_each_knob_would_actually_do`` shows it changes nothing: 15 ids at
``track_iou`` 0.20 and 15 at 0.05.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass

import pytest

from esk_rknn.tracker import IoUTracker

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "tracker_live_recamera_pro.json"


@dataclass
class Det:
    box: list[float]
    score: float


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-9)


@pytest.fixture(scope="module")
def capture():
    return json.loads(FIXTURE.read_text())


def _replay(capture, *, conf=None, track_iou=None, max_lost=None):
    """Run the capture through the tracker; return (tracker, events).

    ``events`` is one entry per minted id: ``(t, new_id, prior_id_at_this_spot,
    seconds_since_that_id_was_last_published, iou_between_the_two)``. A prior of
    ``None`` means nothing had ever been published at that location.
    """
    conf = capture["conf_threshold"] if conf is None else conf
    track_iou = capture["track_iou"] if track_iou is None else track_iou
    max_lost = capture["track_max_lost_s"] if max_lost is None else max_lost

    tracker = IoUTracker(track_iou, max_lost)
    history: dict[int, tuple[list[float], float]] = {}
    events = []
    for frame in capture["frames"]:
        now = frame["t"]
        dets = [Det(d[:4], d[4]) for d in frame["dets"] if d[4] >= conf]
        live = set(tracker.tracks)
        tracked = tracker.update(dets, now)
        for track, _ in tracked:
            if track.track_id in live:
                continue
            prior = [
                (seen, tid, box)
                for tid, (box, seen) in history.items()
                if tid != track.track_id and _iou(box, track.box) >= 0.5
            ]
            if prior:
                seen, tid, box = max(prior)
                events.append((now, track.track_id, tid, now - seen, _iou(box, track.box)))
            else:
                events.append((now, track.track_id, None, None, None))
        for track, _ in tracked:
            history[track.track_id] = (list(track.box), now)
    return tracker, events


def test_the_capture_is_the_one_the_diagnosis_was_written_against(capture):
    """Pin the fixture. Every number below describes THIS recording; silently
    swapping it for another would turn the assertions into noise."""
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest().startswith(
        _EXPECTED_SHA_PREFIX
    )
    assert len(capture["frames"]) == 701
    assert capture["conf_threshold"] == 0.35


def test_no_id_change_is_caused_by_association_failure(capture):
    """The invariant. A new id is only ever legitimate when the previous track
    at that spot had genuinely aged out; if one is minted while that track was
    still alive, association failed and the tracker is at fault."""
    _, events = _replay(capture)
    max_lost = capture["track_max_lost_s"]
    premature = [e for e in events if e[3] is not None and e[3] <= max_lost]
    assert premature == [], (
        f"{len(premature)} id(s) were minted at a location whose previous track "
        f"was still inside max_lost={max_lost}s -- that is an association "
        f"failure, not a detection dropout: {premature[:5]}"
    )


def test_reborn_boxes_sit_almost_exactly_on_the_dead_ones(capture):
    """Why loosening `track_iou` cannot help: on every id change the new box
    overlaps the box the expired track died on by ~0.95. There is no IoU
    threshold that would have kept these ids -- on the frames in between there
    was no detection to associate at all."""
    _, events = _replay(capture)
    overlaps = sorted(e[4] for e in events if e[4] is not None)
    assert len(overlaps) == 14
    assert min(overlaps) > 0.9          # measured 0.922
    assert overlaps[len(overlaps) // 2] > 0.95   # measured 0.959


def test_id_churn_matches_the_captured_baseline(capture):
    """The number this whole investigation is about: 49.8 s of a scene with
    nobody in it mints 15 ids -- one per 3.3 s -- and the hub reads each as a new
    person entering the zone with a dwell timer starting from zero."""
    tracker, events = _replay(capture)
    assert tracker.next_id - 1 == 15
    assert len(events) == 15
    dropouts = [e for e in events if e[3] is not None]
    assert len(dropouts) == 14
    assert all(e[3] > capture["track_max_lost_s"] for e in dropouts)
    # The shortest gap is 1.09 s against a 0.75 s budget: these are not near
    # misses that a slightly longer coast would have caught for free.
    assert min(e[3] for e in dropouts) > 1.0


@pytest.mark.parametrize(
    "kwargs, at_most",
    [
        # Loosening association buys nothing -- there is no box to associate.
        ({"track_iou": 0.05}, 15),
        # Coasting longer does help, because the gaps really are gaps. It buys
        # id stability with up to `max_lost` seconds of dwell credited to a
        # person who was not visible.
        ({"max_lost": 3.0}, 5),
        # Raising confidence removes the objects entirely: nothing in this
        # capture ever scores 0.5, because none of it is a person.
        ({"conf": 0.5}, 0),
    ],
)
def test_what_each_knob_would_actually_do(capture, kwargs, at_most):
    tracker, _ = _replay(capture, **kwargs)
    assert tracker.next_id - 1 <= at_most


def test_every_platform_ships_the_same_tracker():
    """rknn / jetson / recamera-pro ship byte-identical copies of this file, so
    that ID semantics cannot drift between platforms. The assertions above only
    execute one of them; this makes drift an error rather than something a
    reader is expected to notice. (``platforms/generic`` carries its own
    variant and is deliberately not in this set.)"""
    here = pathlib.Path(__file__).resolve().parents[3]
    copies = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            here / "platforms" / "rknn" / "esk_rknn" / "tracker.py",
            here / "platforms" / "jetson" / "esk_jetson" / "tracker.py",
            here / "platforms" / "recamera-pro" / "esk" / "tracker.py",
        ]
    }
    assert len(set(copies.values())) == 1, {str(k): v[:12] for k, v in copies.items()}


_EXPECTED_SHA_PREFIX = "1090dbc0b0b714bf"
