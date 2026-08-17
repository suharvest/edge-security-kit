"""The four HUB_SPEC §2.1 boundary clauses."""

from __future__ import annotations

from conftest import DEVICE, LINE, STREAM, detection, rules_body

from edge_hub.rules.engine import DROP_OUT_OF_ORDER, RuleEngine

INSIDE = (0.75, 0.55)
OUTSIDE = (0.2, 0.55)
LEFT = (0.3, 0.5)
RIGHT = (0.7, 0.5)


def make_engine(clock, **overrides) -> RuleEngine:
    """Zone-only unless a test asks for lines: the zone sits on the right half of
    the frame, so a walk into it would otherwise also cross the gate line."""
    overrides.setdefault("lines", [])
    body = rules_body(**overrides)
    return RuleEngine(lambda d, s: body if (d, s) == (DEVICE, STREAM) else None, clock=clock)


# -- untracked targets --------------------------------------------------
def test_track_id_zero_never_participates_in_rules(clock):
    engine = make_engine(clock)
    untracked = [{"track_id": 0, "class": "person", "score": 0.9,
                  "bbox": [INSIDE[0], INSIDE[1], 0.08, 0.2]}]
    assert engine.on_detections(detection(0, 0, frame_id=1, detections=untracked)) == []
    assert engine.on_detections(detection(0, 0, frame_id=2, detections=untracked)) == []
    assert engine.track_count(DEVICE, STREAM) == 0


def test_untracked_detection_does_not_break_a_tracked_one(clock):
    engine = make_engine(clock)
    mixed = [
        {"track_id": 0, "class": "person", "score": 0.4, "bbox": [0.05, 0.05, 0.04, 0.1]},
        {"track_id": 7, "class": "person", "score": 0.9,
         "bbox": [INSIDE[0], INSIDE[1], 0.08, 0.2]},
    ]
    fired = engine.on_detections(detection(0, 0, frame_id=1, detections=mixed))
    assert [(c.event_type, c.track_id) for c in fired] == [("zone_enter", 7)]


# -- out-of-order frames ------------------------------------------------
def test_out_of_order_frame_is_dropped(clock):
    engine = make_engine(clock)
    engine.on_detections(detection(*OUTSIDE, frame_id=10))
    # A late duplicate and an older frame both get dropped before rule eval.
    assert engine.on_detections(detection(*INSIDE, frame_id=10)) == []
    assert engine.on_detections(detection(*INSIDE, frame_id=4)) == []
    assert engine.stats[DROP_OUT_OF_ORDER] == 2
    # The next in-order frame is evaluated normally.
    assert [c.event_type for c in engine.on_detections(detection(*INSIDE, frame_id=11))] == [
        "zone_enter"
    ]


def test_frame_id_watermark_is_per_stream(clock):
    body = rules_body(lines=[])
    engine = RuleEngine(lambda d, s: body, clock=clock)
    engine.on_detections(detection(*OUTSIDE, frame_id=100, stream_id="cam-0"))
    # cam-1 has its own counter; frame 1 is not "out of order" there.
    fired = engine.on_detections(detection(*INSIDE, frame_id=1, stream_id="cam-1"))
    assert [c.event_type for c in fired] == ["zone_enter"]
    assert engine.stats[DROP_OUT_OF_ORDER] == 0


# -- generation reset ---------------------------------------------------
def test_new_session_resets_zone_state_and_frame_watermark(clock):
    engine = make_engine(clock)
    fired = engine.on_detections(detection(*INSIDE, frame_id=50, session_id="s1"))
    assert [c.event_type for c in fired] == ["zone_enter"]
    # Restarted detector: session changes, frame_id restarts at 1.
    fired = engine.on_detections(detection(*INSIDE, frame_id=1, session_id="s2"))
    assert [c.event_type for c in fired] == ["zone_enter"]
    assert engine.stats[DROP_OUT_OF_ORDER] == 0


def test_new_session_resets_the_line_chain_across_all_streams(clock):
    body = rules_body(zones=[])
    engine = RuleEngine(lambda d, s: body, clock=clock)
    engine.on_detections(detection(*LEFT, frame_id=1, stream_id="cam-1", session_id="s1"))
    engine.on_detections(detection(*LEFT, frame_id=1, stream_id="cam-2", session_id="s1"))
    clock.advance(200)
    # New generation: the stored left-side origin on both streams is gone, so the
    # first frame of s2 cannot be paired into a crossing.
    assert engine.on_detections(
        detection(*RIGHT, frame_id=1, stream_id="cam-1", session_id="s2")
    ) == []
    assert engine.on_detections(
        detection(*RIGHT, frame_id=1, stream_id="cam-2", session_id="s2")
    ) == []


def test_same_session_keeps_state(clock):
    engine = make_engine(clock, zones=[], lines=[dict(LINE)])
    engine.on_detections(detection(*LEFT, frame_id=1, session_id="s1"))
    clock.advance(200)
    fired = engine.on_detections(detection(*RIGHT, frame_id=2, session_id="s1"))
    assert [c.direction for c in fired] == ["forward"]


# -- no rules configured ------------------------------------------------
def test_stream_without_rules_produces_nothing(clock):
    engine = make_engine(clock)
    assert engine.on_detections(detection(*INSIDE, frame_id=1, stream_id="cam-9")) == []
