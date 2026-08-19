#!/usr/bin/env python3
"""Regression tests for the parts of verify_e2e.py that decide pass/fail.

Runs anywhere: `python3 test_verify_e2e.py` or `pytest`. No broker, no device.

The data below is a verbatim excerpt of a real reCamera Pro / RK3576 run
(2026-08-19, e2e-out/{alerts,observed-events}.json, alert ids 507-523). It is
kept because that run is what exposed the matcher: the hub emitted one extra
`zone_enter` mid-clip, and the old in-order "nearest unused" matcher turned that
single anomaly into three failures, two of them off by exactly one clip period
(31.0 s) -- the same shape as the "leading zone_enter artefact" documented in the
RK3588, RK3576 and Jetson platform READMEs.

A fix that only made failures disappear would be a fix that disabled the
assertions, so every property here is tested in both directions: the recorded
run must produce exactly one complaint, and deliberately corrupted variants of
it must still produce complaints.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("ESK_FIXTURE_DIR", str(HERE))
sys.path.insert(0, str(HERE))

import verify_e2e as V  # noqa: E402

TOL_MS = V.TOL_S * 1000.0
T0 = 1787110358551.0  # first truth instant of the recorded run, for readability

# --- recorded run -----------------------------------------------------------
# Truth zone entries derived from the tapped detections, one per clip loop.
Z_WANT = [{"event_type": "zone_enter", "truth_ts_ms": T0 + ms, "frame_id": f,
           "track_id": 1}
          for ms, f in ((10205.0, 1481), (41213.0, 1636), (72221.0, 1791))]
# What the hub actually raised. 510 is the anomaly: it lands 16.0 s after the
# first entry, at the instant the subject walks OUT of the zone.
Z_GOT = [{"id": i, "received_ms": T0 + ms, "track_id": 1,
          "rule_name": "zone-dwell", "snapshot_state": "received"}
         for i, ms in ((507, 10215.0), (510, 26221.0), (515, 41226.0),
                       (523, 72232.0))]


def _old_greedy(z_got, z_want):
    """The matcher this file replaced, kept to prove the cascade was real."""
    used, pairs = set(), []
    for a in sorted(z_got, key=lambda x: x["id"]):
        cands = [(i, e) for i, e in enumerate(z_want)
                 if i not in used and e["track_id"] == a["track_id"]]
        if not cands:
            continue
        i, e = min(cands, key=lambda p: abs(p[1]["truth_ts_ms"] - a["received_ms"]))
        used.add(i)
        pairs.append((a, e))
    return pairs


def test_old_matcher_cascaded_one_anomaly_into_three_failures():
    errs = [round((a["received_ms"] - e["truth_ts_ms"]) / 1000.0, 3)
            for a, e in _old_greedy(Z_GOT, Z_WANT)]
    # 507 paired correctly, then every later alert took the NEXT loop's instant.
    assert errs == [0.01, -14.992, -30.995]
    assert len([e for e in errs if abs(e) > V.TOL_S]) == 2
    # ...and 523 was left with nothing, for a third failure.
    assert len(_old_greedy(Z_GOT, Z_WANT)) == len(Z_WANT) < len(Z_GOT)


def test_recorded_run_reports_the_anomaly_once_and_pairs_the_rest():
    pairs, extra, missing = V.assign_nearest(Z_GOT, Z_WANT, TOL_MS)
    assert [a["id"] for a, _ in pairs] == [507, 515, 523]
    assert all(abs(a["received_ms"] - e["truth_ts_ms"]) <= 20 for a, e in pairs)
    assert [a["id"] for a in extra] == [510]     # the hub's extra zone_enter
    assert missing == []                          # nothing went unanswered
    # The anomaly is still ~16 s from anything real -- it is not being absorbed.
    assert abs(V.nearest_gap_s(extra[0], Z_WANT)) > 14.0


def test_assertions_still_bite_when_every_alert_is_late():
    late = [dict(a, received_ms=a["received_ms"] + 5000.0) for a in Z_GOT]
    pairs, extra, missing = V.assign_nearest(late, Z_WANT, TOL_MS)
    assert pairs == []
    assert len(extra) == 4 and len(missing) == 3


def test_assertions_still_bite_when_an_alert_is_missing():
    dropped = [a for a in Z_GOT if a["id"] != 515]
    pairs, extra, missing = V.assign_nearest(dropped, Z_WANT, TOL_MS)
    assert [a["id"] for a, _ in pairs] == [507, 523]
    assert [a["id"] for a in extra] == [510]
    assert [e["truth_ts_ms"] for e in missing] == [T0 + 41213.0]


def test_a_track_id_mismatch_is_never_matched_away():
    other = [dict(a, track_id=99) for a in Z_GOT]
    pairs, extra, missing = V.assign_nearest(other, Z_WANT, TOL_MS)
    assert pairs == [] and len(extra) == 4 and len(missing) == 3


def test_observation_edge_is_the_last_detection_not_the_tap_close():
    # The tap stayed open 8 s past the last frame the stream published.
    rows = [{"ts_ms": T0 + ms} for ms in (0.0, 40000.0, 74400.0)]
    t_end = T0 + 82400.0
    edge = V.assertable_until(rows, t_end)
    assert edge == T0 + 74400.0 - TOL_MS
    # A loitering instant is entry + dwell_seconds. The last loop's escalation
    # falls past the stream, so it cannot be demanded...
    assert T0 + 82221.0 > edge
    # ...while its zone_enter, and every earlier loop, still are.
    assert T0 + 72221.0 <= edge
    assert T0 + 51213.0 <= edge


def test_tap_close_still_wins_when_it_is_the_earlier_edge():
    rows = [{"ts_ms": T0 + ms} for ms in (0.0, 90000.0)]
    t_end = T0 + 60000.0
    assert V.assertable_until(rows, t_end) == t_end - TOL_MS


def test_edge_helper_survives_a_stream_that_published_nothing():
    t_end = T0 + 1000.0
    assert V.assertable_until([], t_end) == t_end - TOL_MS


def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except AssertionError as exc:
            bad += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{len(tests) - bad}/{len(tests)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())
