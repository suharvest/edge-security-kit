"""zone_enter + loitering (upstream multi_camera_manager.py:521-540).

Migrated verbatim in structure — enter/leave bookkeeping through
``zone_entered_at``, dwell measured against that instant — with two spec-driven
changes:

* the instant is the hub's monotonic receive clock, not the device clock
  (HUB_SPEC §2.1);
* coordinates are already normalized, so no ``norm_px`` step.
"""

from __future__ import annotations

from typing import Any

from .geometry import Point, point_in_polygon
from .state import TrackState


def zone_key(zone: dict[str, Any], index: int) -> str:
    return str(zone.get("id") or zone.get("name") or f"zone{index}")


def zone_points(zone: dict[str, Any]) -> list[Point]:
    return [(float(p[0]), float(p[1])) for p in zone.get("points") or []]


def evaluate_zones(
    track: TrackState,
    centroid: Point,
    zones: list[dict[str, Any]],
    features: dict[str, Any],
    mono_ms: float,
) -> list[dict[str, Any]]:
    """Return zero or more candidate dicts for one detection of one track.

    Candidate shape: ``{event_type, rule_name, rule_id, dwell_s?}``.
    """
    out: list[dict[str, Any]] = []
    if not features.get("zone_detection", True):
        return out
    loitering_on = bool(features.get("loitering", True))

    for index, zone in enumerate(zones):
        pts = zone_points(zone)
        if len(pts) < 3:
            continue
        key = zone_key(zone, index)
        name = str(zone.get("name") or key)
        inside = point_in_polygon(centroid, pts)

        if inside and key not in track.zone_entered_at:
            track.zone_entered_at[key] = mono_ms
            out.append(
                {"event_type": "zone_enter", "rule_name": name, "rule_id": key}
            )
        if not inside and key in track.zone_entered_at:
            del track.zone_entered_at[key]
        if inside and loitering_on and key in track.zone_entered_at:
            dwell_s = (mono_ms - track.zone_entered_at[key]) / 1000.0
            if dwell_s >= float(zone.get("dwell_seconds", 10)):
                out.append(
                    {
                        "event_type": "loitering",
                        "rule_name": name,
                        "rule_id": key,
                        "dwell_s": round(dwell_s, 2),
                    }
                )
    return out
