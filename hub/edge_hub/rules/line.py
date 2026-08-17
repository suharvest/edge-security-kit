"""line_cross (upstream multi_camera_manager.py:542-554) plus direction.

Upstream tested only "does the prev->curr segment intersect the line". HUB_SPEC
§2 adds:

* direction via the ``side()`` sign flip, per the contracts/MQTT.md convention;
* a per-rule ``direction: any | forward | backward`` filter (default ``any``);
* the ``line_chain_gap_ms`` break (HUB_SPEC §2.1): if the two centroids arrived
  more than N ms apart the pair is not a decidable crossing — QoS0 frame loss
  would otherwise draw a long chord across the whole scene;
* the ``side == 0`` rule (contracts/MQTT.md `direction`): the comparison is
  against the track's *last non-zero* side, not against the previous frame.

The last clause is not a corner case. Detector centroids arrive on a coarse
quantization grid — the RK3588 preprocessor scales 1280 px down to 640 before
inference, so ``cx`` snaps to multiples of 1/640 and ``x = 0.5`` is exactly
320/640. A line drawn down the middle of the frame, which is the most natural
thing a user does, then sees ``side == 0`` on every frame. Comparing frame to
frame, that line can never fire. Carrying the last non-zero side across the
on-line frames makes the crossing decidable while still refusing to fire for a
track that merely sits on the line and jitters.
"""

from __future__ import annotations

from typing import Any

from .geometry import Point, direction_from_sides, segments_intersect, side
from .state import TrackState

CHAIN_BROKEN = "chain_broken"


def line_key(line: dict[str, Any], index: int) -> str:
    return str(line.get("id") or line.get("name") or f"line{index}")


def _parse(line: dict[str, Any]) -> tuple[Point, Point] | None:
    start_raw = line.get("start")
    end_raw = line.get("end")
    if not start_raw or not end_raw:
        return None
    return (
        (float(start_raw[0]), float(start_raw[1])),
        (float(end_raw[0]), float(end_raw[1])),
    )


def _reanchor(track: TrackState, centroid: Point, lines: list[dict[str, Any]]) -> None:
    """Forget every remembered side and re-seed from the current centroid.

    Used when the chain is not decidable (first frame of a track, a QoS0 gap
    longer than ``line_chain_gap_ms``, the feature toggled off): the track is
    treated as newly observed on whichever side it is on now. A centroid on the
    line seeds nothing, so a track that *starts* on the line and walks off it
    does not report a crossing.
    """
    track.forget_line_sides()
    for index, line in enumerate(lines):
        parsed = _parse(line)
        if parsed is None:
            continue
        start, end = parsed
        track.note_line_side(line_key(line, index), side(start, end, centroid), centroid)


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
        _reanchor(track, centroid, lines)
        return out

    prev_mono = track.last_point_mono
    chain_ok = (
        track.last_point is not None
        and prev_mono is not None
        and mono_ms - prev_mono <= line_chain_gap_ms
    )
    track.reset_line_chain(centroid, mono_ms)

    if not chain_ok:
        _reanchor(track, centroid, lines)
        return out

    for index, line in enumerate(lines):
        parsed = _parse(line)
        if parsed is None:
            continue
        start, end = parsed
        key = line_key(line, index)
        curr_side = side(start, end, centroid)
        if curr_side == 0:
            # On the line: no side information. Leave the remembered side and
            # its anchor point alone — the track still counts as being on the
            # side it last had.
            continue

        prev_side = track.line_side.get(key)
        anchor = track.line_anchor.get(key)
        track.note_line_side(key, curr_side, centroid)
        if prev_side is None or anchor is None:
            continue

        direction = direction_from_sides(prev_side, curr_side)
        if direction is None:
            continue
        # The remembered side is about the infinite line; the rule is a finite
        # segment, so the anchor -> centroid chord still has to hit it.
        if not segments_intersect(anchor, centroid, start, end):
            continue
        wanted = str(line.get("direction") or "any")
        if wanted != "any" and wanted != direction:
            continue
        out.append(
            {
                "event_type": "line_cross",
                "rule_name": str(line.get("name") or key),
                "rule_id": key,
                "direction": direction,
            }
        )
    return out
