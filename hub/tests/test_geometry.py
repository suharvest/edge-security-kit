"""Geometry primitives migrated from behavior_demo.py:50-68 (HUB_SPEC §2)."""

from __future__ import annotations

from edge_hub.rules.geometry import (
    crossing_direction,
    point_in_polygon,
    segments_intersect,
    side,
)

SQUARE = [(0.6, 0.3), (0.9, 0.3), (0.9, 0.8), (0.6, 0.8)]


def test_point_in_polygon_inside_and_outside():
    assert point_in_polygon((0.75, 0.55), SQUARE) is True
    assert point_in_polygon((0.5, 0.55), SQUARE) is False
    assert point_in_polygon((0.75, 0.95), SQUARE) is False


def test_point_in_polygon_degenerate_polygon_is_never_inside():
    assert point_in_polygon((0.5, 0.5), [(0.0, 0.0), (1.0, 1.0)]) is False


def test_point_in_polygon_concave_notch():
    # An L shape: the notch at (0.8, 0.8) is outside despite being in the bbox.
    poly = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.4), (0.4, 0.4), (0.4, 0.9), (0.1, 0.9)]
    assert point_in_polygon((0.25, 0.75), poly) is True
    assert point_in_polygon((0.8, 0.8), poly) is False


def test_segments_intersect():
    assert segments_intersect((0.4, 0.5), (0.6, 0.5), (0.5, 0.1), (0.5, 0.9)) is True
    assert segments_intersect((0.1, 0.5), (0.4, 0.5), (0.5, 0.1), (0.5, 0.9)) is False


def test_side_sign_convention():
    start, end = (0.5, 0.1), (0.5, 0.9)
    # Left of a downward arrow is positive, right is negative.
    assert side(start, end, (0.2, 0.5)) == 1
    assert side(start, end, (0.8, 0.5)) == -1
    assert side(start, end, (0.5, 0.5)) == 0


def test_crossing_direction_matches_contract():
    start, end = (0.5, 0.1), (0.5, 0.9)
    assert crossing_direction(start, end, (0.4, 0.5), (0.6, 0.5)) == "forward"
    assert crossing_direction(start, end, (0.6, 0.5), (0.4, 0.5)) == "backward"
    # Landing exactly on the line is not a decidable crossing.
    assert crossing_direction(start, end, (0.4, 0.5), (0.5, 0.5)) is None
