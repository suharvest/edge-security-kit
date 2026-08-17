"""Rule engine package (HUB_SPEC §2)."""

from .engine import Candidate, RuleEngine
from .geometry import ccw, crossing_direction, point_in_polygon, segments_intersect, side

__all__ = [
    "Candidate",
    "RuleEngine",
    "ccw",
    "crossing_direction",
    "point_in_polygon",
    "segments_intersect",
    "side",
]
