"""line_cross: both directions, the `any` default, and the chain-break clause."""

from __future__ import annotations

from conftest import DEVICE, STREAM, detection, rules_body

from edge_hub.clock import FakeClock
from edge_hub.rules.engine import RuleEngine

LEFT = (0.3, 0.5)
RIGHT = (0.7, 0.5)
FAR_LEFT = (0.1, 0.5)


def make_engine(clock: FakeClock, direction: str = "any", **overrides) -> RuleEngine:
    body = rules_body(
        lines=[{"id": "gate", "name": "gate", "start": [0.5, 0.1], "end": [0.5, 0.9],
                "direction": direction}],
        zones=[],
        **overrides,
    )
    return RuleEngine(lambda d, s: body if (d, s) == (DEVICE, STREAM) else None, clock=clock)


def walk(engine: RuleEngine, points, step_ms: float = 200.0, clock=None):
    """Feed a sequence of centroids, returning the candidates from the last one."""
    fired = []
    for index, (x, y) in enumerate(points, start=1):
        if clock is not None and index > 1:
            clock.advance(step_ms)
        fired = engine.on_detections(detection(x, y, frame_id=index))
    return fired


def test_first_frame_cannot_cross(clock):
    engine = make_engine(clock)
    assert engine.on_detections(detection(*RIGHT, frame_id=1)) == []


def test_forward_crossing_left_to_right(clock):
    engine = make_engine(clock)
    fired = walk(engine, [LEFT, RIGHT], clock=clock)
    assert [(c.event_type, c.direction) for c in fired] == [("line_cross", "forward")]


def test_backward_crossing_right_to_left(clock):
    engine = make_engine(clock)
    fired = walk(engine, [RIGHT, LEFT], clock=clock)
    assert [(c.event_type, c.direction) for c in fired] == [("line_cross", "backward")]


def test_any_accepts_both_directions(clock):
    engine = make_engine(clock, direction="any")
    assert walk(engine, [LEFT, RIGHT], clock=clock)[0].direction == "forward"
    clock.advance(200)
    assert engine.on_detections(detection(*LEFT, frame_id=3))[0].direction == "backward"


def test_forward_only_rule_ignores_a_backward_crossing(clock):
    engine = make_engine(clock, direction="forward")
    assert walk(engine, [RIGHT, LEFT], clock=clock) == []
    # ...and still reports the forward one.
    clock.advance(200)
    fired = engine.on_detections(detection(*RIGHT, frame_id=3))
    assert [c.direction for c in fired] == ["forward"]


def test_backward_only_rule_ignores_a_forward_crossing(clock):
    engine = make_engine(clock, direction="backward")
    assert walk(engine, [LEFT, RIGHT], clock=clock) == []
    clock.advance(200)
    assert [c.direction for c in engine.on_detections(detection(*LEFT, frame_id=3))] == [
        "backward"
    ]


def test_movement_on_one_side_never_crosses(clock):
    engine = make_engine(clock)
    assert walk(engine, [FAR_LEFT, LEFT, (0.45, 0.5)], clock=clock) == []


def test_segment_beyond_the_line_endpoints_does_not_cross(clock):
    engine = make_engine(clock)
    # y = 0.95 is past the line's end at y = 0.9.
    assert walk(engine, [(0.3, 0.95), (0.7, 0.95)], clock=clock) == []


def test_chain_break_discards_the_pair(clock):
    engine = make_engine(clock, line_chain_gap_ms=1000)
    engine.on_detections(detection(*LEFT, frame_id=1))
    # A QoS0 gap longer than line_chain_gap_ms: the long chord is not a crossing.
    clock.advance(1_500)
    assert engine.on_detections(detection(*RIGHT, frame_id=2)) == []
    # The chain re-anchors on the current point, so the next real crossing fires.
    clock.advance(200)
    fired = engine.on_detections(detection(*LEFT, frame_id=3))
    assert [c.direction for c in fired] == ["backward"]


def test_chain_gap_exactly_at_the_threshold_still_counts(clock):
    engine = make_engine(clock, line_chain_gap_ms=1000)
    engine.on_detections(detection(*LEFT, frame_id=1))
    clock.advance(1_000)
    fired = engine.on_detections(detection(*RIGHT, frame_id=2))
    assert [c.direction for c in fired] == ["forward"]


def test_line_crossing_feature_flag_off(clock):
    engine = make_engine(clock)
    body = rules_body(
        lines=[{"id": "gate", "name": "gate", "start": [0.5, 0.1], "end": [0.5, 0.9]}],
        zones=[],
        features={"zone_detection": False, "loitering": False, "line_crossing": False},
    )
    engine = RuleEngine(lambda d, s: body, clock=clock)
    assert walk(engine, [LEFT, RIGHT], clock=clock) == []
