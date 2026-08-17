"""zone_enter / loitering positive and negative cases (HUB_SPEC §2)."""

from __future__ import annotations

from conftest import DEVICE, STREAM, detection, rules_body

from edge_hub.clock import FakeClock
from edge_hub.rules.engine import RuleEngine

INSIDE = (0.75, 0.55)
OUTSIDE = (0.2, 0.55)


def make_engine(clock: FakeClock, **overrides) -> RuleEngine:
    """Zone-only rules, so a walk across the frame cannot trip a line rule."""
    overrides.setdefault("lines", [])
    body = rules_body(**overrides)
    return RuleEngine(lambda d, s: body if (d, s) == (DEVICE, STREAM) else None, clock=clock)


def test_zone_enter_fires_once_on_entry(clock):
    engine = make_engine(clock)
    assert engine.on_detections(detection(*OUTSIDE, frame_id=1)) == []
    fired = engine.on_detections(detection(*INSIDE, frame_id=2))
    assert [c.event_type for c in fired] == ["zone_enter"]
    assert fired[0].rule_name == "bay"
    # Staying inside does not re-enter.
    assert engine.on_detections(detection(*INSIDE, frame_id=3)) == []


def test_zone_enter_refires_after_leaving(clock):
    engine = make_engine(clock)
    engine.on_detections(detection(*INSIDE, frame_id=1))
    engine.on_detections(detection(*OUTSIDE, frame_id=2))
    fired = engine.on_detections(detection(*INSIDE, frame_id=3))
    assert [c.event_type for c in fired] == ["zone_enter"]


def test_zone_never_fires_for_a_track_outside(clock):
    engine = make_engine(clock)
    for frame in range(1, 6):
        assert engine.on_detections(detection(*OUTSIDE, frame_id=frame)) == []


def test_zone_disabled_by_feature_flag(clock):
    engine = make_engine(clock, features={"zone_detection": False, "loitering": False,
                                          "line_crossing": True})
    assert engine.on_detections(detection(*INSIDE, frame_id=1)) == []


def test_loitering_fires_after_dwell_on_the_hub_clock(clock):
    engine = make_engine(clock, zones=[{
        "id": "bay", "name": "bay",
        "points": [[0.6, 0.3], [0.9, 0.3], [0.9, 0.8], [0.6, 0.8]],
        "dwell_seconds": 4,
    }])
    fired = engine.on_detections(detection(*INSIDE, frame_id=1))
    assert [c.event_type for c in fired] == ["zone_enter"]

    # 3 s inside: still under the threshold. The device timestamp is unchanged
    # on purpose — only the hub clock counts (HUB_SPEC §2.1).
    clock.advance(3_000)
    assert engine.on_detections(detection(*INSIDE, frame_id=2)) == []

    clock.advance(1_100)
    fired = engine.on_detections(detection(*INSIDE, frame_id=3))
    assert [c.event_type for c in fired] == ["loitering"]
    assert fired[0].dwell_s >= 4.0


def test_loitering_timer_restarts_after_leaving_the_zone(clock):
    engine = make_engine(clock, zones=[{
        "id": "bay", "name": "bay",
        "points": [[0.6, 0.3], [0.9, 0.3], [0.9, 0.8], [0.6, 0.8]],
        "dwell_seconds": 4,
    }])
    engine.on_detections(detection(*INSIDE, frame_id=1))
    clock.advance(3_500)
    engine.on_detections(detection(*OUTSIDE, frame_id=2))
    clock.advance(1_000)
    fired = engine.on_detections(detection(*INSIDE, frame_id=3))
    # Re-entry resets dwell: zone_enter again, no loitering.
    assert [c.event_type for c in fired] == ["zone_enter"]
    clock.advance(2_000)
    assert engine.on_detections(detection(*INSIDE, frame_id=4)) == []


def test_loitering_needs_the_feature_flag(clock):
    engine = make_engine(
        clock,
        zones=[{"id": "bay", "name": "bay",
                "points": [[0.6, 0.3], [0.9, 0.3], [0.9, 0.8], [0.6, 0.8]],
                "dwell_seconds": 1}],
        features={"zone_detection": True, "loitering": False, "line_crossing": False},
    )
    engine.on_detections(detection(*INSIDE, frame_id=1))
    clock.advance(5_000)
    assert engine.on_detections(detection(*INSIDE, frame_id=2)) == []


def test_track_state_expires_after_track_expiry(clock):
    engine = make_engine(clock, track_expiry_s=5)
    engine.on_detections(detection(*INSIDE, frame_id=1))
    assert engine.track_count(DEVICE, STREAM) == 1
    # The track vanishes from detections for longer than track_expiry_s.
    clock.advance(6_000)
    engine.on_detections(
        detection(0.0, 0.0, frame_id=2,
                  detections=[{"track_id": 9, "class": "person", "score": 0.5,
                               "bbox": [0.1, 0.1, 0.05, 0.1]}])
    )
    assert engine.track_count(DEVICE, STREAM) == 1
    # The expired track re-enters from scratch, so zone_enter fires again.
    fired = engine.on_detections(detection(*INSIDE, frame_id=3))
    assert [c.event_type for c in fired] == ["zone_enter"]
