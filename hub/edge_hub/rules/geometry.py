"""Pure-scalar geometry primitives.

Migrated from the upstream Industrial-security-demo
(`app/behavior_demo.py:50-68`) per HUB_SPEC §2. Two changes:

* inputs are normalized ``frame_norm`` coordinates in [0, 1] instead of pixels,
  so the upstream ``norm_px`` pixel conversion is deleted outright;
* :func:`side` is new, and provides the sign convention the line-crossing
  direction test needs (contracts/MQTT.md `direction`).

No numpy: every operation here is scalar arithmetic (HUB_SPEC §8 image budget).
"""

from __future__ import annotations

Point = tuple[float, float]


def point_in_polygon(pt: Point, poly: list[Point]) -> bool:
    """Ray-casting point-in-polygon test (upstream behavior_demo.py:50-60)."""
    if len(poly) < 3:
        return False
    x, y = pt
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / max((yj - yi), 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def ccw(a: Point, b: Point, c: Point) -> bool:
    """Counter-clockwise orientation test (upstream behavior_demo.py:63-64)."""
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """True when segment a-b properly straddles segment c-d (upstream :67-68)."""
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def side(start: Point, end: Point, p: Point) -> int:
    """Sign of the 2-D cross product ``(end-start) x (p-start)``.

    contracts/MQTT.md: a crossing is ``forward`` when the centroid moves from
    ``side > 0`` to ``side < 0``, ``backward`` for the reverse. Returns -1, 0
    or 1; 0 means the point lies exactly on the (infinite) line.
    """
    cross = (end[0] - start[0]) * (p[1] - start[1]) - (end[1] - start[1]) * (
        p[0] - start[0]
    )
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def direction_from_sides(prev_side: int, curr_side: int) -> str | None:
    """Classify a sign flip between two sides.

    ``forward`` is ``+1 -> -1``, ``backward`` is ``-1 -> +1``. Any pair
    involving a zero returns ``None``: a point exactly on the line carries no
    side information. contracts/MQTT.md `direction` therefore requires callers
    to compare the current side against the track's *last non-zero* side rather
    than against the immediately preceding frame.
    """
    if prev_side > 0 and curr_side < 0:
        return "forward"
    if prev_side < 0 and curr_side > 0:
        return "backward"
    return None


def crossing_direction(start: Point, end: Point, prev: Point, curr: Point) -> str | None:
    """Classify a crossing of the directed segment ``start -> end``.

    Stateless two-point form, for callers that already hold a decided pair.
    Returns ``None`` when either point lies on the line — the stateful
    last-non-zero-side rule in :mod:`edge_hub.rules.line` is what resolves that
    case for live tracks.
    """
    return direction_from_sides(side(start, end, prev), side(start, end, curr))
