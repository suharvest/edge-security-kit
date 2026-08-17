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


# --- side == 0: the centroid lands exactly on the line ----------------------
#
# contracts/MQTT.md `direction`: compare against the track's last *non-zero*
# side. Detector centroids are quantized (the RK3588 preprocessor scales 1280
# down to 640, so cx snaps to multiples of 1/640 and x = 0.5 is exactly
# 320/640), which puts a line drawn down the middle of the frame on side 0 for
# every frame of the traverse.

ON_LINE = (0.5, 0.5)
#: 1/640 grid values straddling the mid-frame line at 320/640 = 0.5
G = [x / 640.0 for x in (312, 316, 320, 320, 324, 328)]


def walk_all(engine: RuleEngine, points, step_ms: float = 200.0, clock=None):
    """Feed a sequence of centroids, returning every candidate they produced."""
    seen = []
    for index, (x, y) in enumerate(points, start=1):
        if clock is not None and index > 1:
            clock.advance(step_ms)
        seen.extend(engine.on_detections(detection(x, y, frame_id=index)))
    return seen


def test_a_frame_on_the_line_is_not_yet_a_crossing(clock):
    engine = make_engine(clock)
    assert walk_all(engine, [LEFT, ON_LINE], clock=clock) == []


def test_crossing_through_a_frame_on_the_line_fires_once_forward(clock):
    engine = make_engine(clock)
    fired = walk_all(engine, [LEFT, ON_LINE, RIGHT], clock=clock)
    assert [(c.event_type, c.direction) for c in fired] == [("line_cross", "forward")]


def test_several_frames_on_the_line_still_resolve_the_crossing(clock):
    engine = make_engine(clock)
    fired = walk_all(engine, [LEFT, ON_LINE, ON_LINE, ON_LINE, RIGHT], clock=clock)
    assert [c.direction for c in fired] == ["forward"]


def test_quantized_midline_walk_fires_exactly_one_forward(clock):
    """The RK3588 case: cx on the 1/640 grid, line drawn at x = 0.5."""
    engine = make_engine(clock)
    fired = walk_all(engine, [(x, 0.5) for x in G], clock=clock)
    assert [(c.rule_id, c.direction) for c in fired] == [("gate", "forward")]


def test_jitter_across_the_line_does_not_refire(clock):
    """side +1, 0, +1, 0, +1 -- never leaves the left side, never fires."""
    engine = make_engine(clock)
    fired = walk_all(
        engine, [LEFT, ON_LINE, LEFT, ON_LINE, (0.48, 0.5), ON_LINE], clock=clock
    )
    assert fired == []


def test_dwelling_exactly_on_the_line_does_not_fire(clock):
    engine = make_engine(clock)
    assert walk_all(engine, [LEFT] + [ON_LINE] * 6, clock=clock) == []


def test_track_starting_on_the_line_does_not_cross(clock):
    """No last non-zero side yet, so leaving the line only seeds one."""
    engine = make_engine(clock)
    assert walk_all(engine, [ON_LINE, RIGHT, (0.8, 0.5)], clock=clock) == []


def test_track_starting_on_the_line_still_reports_a_later_traverse(clock):
    engine = make_engine(clock)
    fired = walk_all(engine, [ON_LINE, RIGHT, ON_LINE, LEFT], clock=clock)
    assert [c.direction for c in fired] == ["backward"]


def test_direction_filter_applies_to_an_on_line_crossing(clock):
    engine = make_engine(clock, direction="backward")
    assert walk_all(engine, [LEFT, ON_LINE, RIGHT], clock=clock) == []
    clock.advance(200)
    fired = engine.on_detections(detection(*LEFT, frame_id=99))
    assert [c.direction for c in fired] == ["backward"]


def test_on_line_excursion_does_not_survive_a_chain_break(clock):
    """A QoS0 gap re-anchors: the pre-gap side must not decide a crossing."""
    engine = make_engine(clock, line_chain_gap_ms=1000)
    engine.on_detections(detection(*LEFT, frame_id=1))
    clock.advance(200)
    assert engine.on_detections(detection(*ON_LINE, frame_id=2)) == []
    clock.advance(1_500)
    assert engine.on_detections(detection(*RIGHT, frame_id=3)) == []
