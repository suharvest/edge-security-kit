"""line_cross (upstream multi_camera_manager.py:542-554) plus direction.

Upstream tested only "does the prev->curr segment intersect the line". HUB_SPEC
§2 adds:

* direction via the ``side()`` sign flip, per the contracts/MQTT.md convention;
* a per-rule ``direction: any | forward | backward`` filter (default ``any``);
* the ``line_chain_gap_ms`` break (HUB_SPEC §2.1): if the two centroids arrived
  more than N ms apart the pair is not a decidable crossing — QoS0 frame loss
  would otherwise draw a long chord across the whole scene.
"""

from __future__ import annotations

from typing import Any

from .geometry import Point, crossing_direction, segments_intersect
from .state import TrackState

CHAIN_BROKEN = "chain_broken"


def line_key(line: dict[str, Any], index: int) -> str:
    return str(line.get("id") or line.get("name") or f"line{index}")


def evaluate_lines(
    track: TrackState,
    centroid: Point,
    lines: list[dict[str, Any]],
    features: dict[str, Any],
    mono_ms: float,
    line_chain_gap_ms: float,
) -> list[dict[str, Any]]:
    """Return candidate dicts for line crossings, and advance the line chain.

    Always updates ``track.last_point`` / ``last_point_mono`` before returning,
    so a broken chain re-anchors on the current centroid.
    """
    out: list[dict[str, Any]] = []
    if not features.get("line_crossing", True):
        # Still keep the chain fresh so toggling the feature on mid-session does
        # not evaluate a stale, arbitrarily old pair.
        track.reset_line_chain(centroid, mono_ms)
        return out

    prev = track.last_point
    prev_mono = track.last_point_mono
    track.reset_line_chain(centroid, mono_ms)

    if prev is None or prev_mono is None:
        return out
    if mono_ms - prev_mono > line_chain_gap_ms:
        return out

    for index, line in enumerate(lines):
        start_raw = line.get("start")
        end_raw = line.get("end")
        if not start_raw or not end_raw:
            continue
        start = (float(start_raw[0]), float(start_raw[1]))
        end = (float(end_raw[0]), float(end_raw[1]))
        if not segments_intersect(prev, centroid, start, end):
            continue
        direction = crossing_direction(start, end, prev, centroid)
        if direction is None:
            continue
        wanted = str(line.get("direction") or "any")
        if wanted != "any" and wanted != direction:
            continue
        out.append(
            {
                "event_type": "line_cross",
                "rule_name": str(line.get("name") or line_key(line, index)),
                "rule_id": line_key(line, index),
                "direction": direction,
            }
        )
    return out
