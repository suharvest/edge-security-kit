"""Validation for a rules body posted to PUT /api/rules/{device}/{stream}.

Rules are hub-side configuration, not a wire contract, so they are checked here
rather than by the MQTT JSON Schema. The structure is the upstream
``demo_config.json`` rules section (zones/lines field names, normalized
coordinates) so old configs load unchanged — HUB_SPEC §9 upgrade path — with the
new ``direction`` field added.
"""

from __future__ import annotations

from typing import Any

DIRECTIONS = ("any", "forward", "backward")
FEATURE_KEYS = ("zone_detection", "loitering", "line_crossing")


class RulesError(ValueError):
    pass


def _point(value: Any, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RulesError(f"{where}: expected [x, y]")
    out = []
    for axis, raw in zip("xy", value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RulesError(f"{where}.{axis}: expected a number")
        if not 0.0 <= float(raw) <= 1.0:
            raise RulesError(f"{where}.{axis}: normalized coordinate out of [0, 1]")
        out.append(float(raw))
    return out


def validate_rules_body(body: Any) -> dict[str, Any]:
    """Return a normalized copy of ``body`` or raise :class:`RulesError`."""
    if not isinstance(body, dict):
        raise RulesError("rules body must be an object")

    zones_in = body.get("zones") or []
    lines_in = body.get("lines") or []
    if not isinstance(zones_in, list) or not isinstance(lines_in, list):
        raise RulesError("zones and lines must be arrays")

    zones: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, zone in enumerate(zones_in):
        if not isinstance(zone, dict):
            raise RulesError(f"zones[{index}]: expected an object")
        points = zone.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise RulesError(f"zones[{index}].points: need at least 3 points")
        norm = [_point(p, f"zones[{index}].points[{i}]") for i, p in enumerate(points)]
        rule_id = str(zone.get("id") or zone.get("name") or f"zone{index}")
        if rule_id in seen:
            raise RulesError(f"duplicate rule id {rule_id!r}")
        seen.add(rule_id)
        dwell = zone.get("dwell_seconds", 10)
        if isinstance(dwell, bool) or not isinstance(dwell, (int, float)) or dwell < 0:
            raise RulesError(f"zones[{index}].dwell_seconds: expected a non-negative number")
        out = {**zone, "id": rule_id, "name": str(zone.get("name") or rule_id),
               "points": norm, "dwell_seconds": float(dwell)}
        zones.append(out)

    lines: list[dict[str, Any]] = []
    for index, line in enumerate(lines_in):
        if not isinstance(line, dict):
            raise RulesError(f"lines[{index}]: expected an object")
        start = _point(line.get("start"), f"lines[{index}].start")
        end = _point(line.get("end"), f"lines[{index}].end")
        if start == end:
            raise RulesError(f"lines[{index}]: start and end must differ")
        direction = str(line.get("direction") or "any")
        if direction not in DIRECTIONS:
            raise RulesError(
                f"lines[{index}].direction: expected one of {', '.join(DIRECTIONS)}"
            )
        rule_id = str(line.get("id") or line.get("name") or f"line{index}")
        if rule_id in seen:
            raise RulesError(f"duplicate rule id {rule_id!r}")
        seen.add(rule_id)
        lines.append(
            {**line, "id": rule_id, "name": str(line.get("name") or rule_id),
             "start": start, "end": end, "direction": direction}
        )

    features_in = body.get("features") or {}
    if not isinstance(features_in, dict):
        raise RulesError("features must be an object")
    features = {key: bool(features_in.get(key, True)) for key in FEATURE_KEYS}
    for key, value in features_in.items():
        if key not in features:
            features[key] = bool(value)

    cooldown = body.get("cooldown", 30)
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) or cooldown < 0:
        raise RulesError("cooldown: expected a non-negative number of seconds")

    out_body: dict[str, Any] = {
        **{k: v for k, v in body.items()
           if k not in ("zones", "lines", "features", "cooldown")},
        "zones": zones,
        "lines": lines,
        "features": features,
        "cooldown": float(cooldown),
    }
    for optional, minimum in (("stream_rate_limit_s", 0), ("track_expiry_s", 0),
                              ("line_chain_gap_ms", 0)):
        if optional in body:
            value = body[optional]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
                raise RulesError(f"{optional}: expected a number >= {minimum}")
            out_body[optional] = float(value)
    return out_body


def find_rule(body: dict[str, Any], rule_id: str) -> tuple[str, dict[str, Any]] | None:
    """Locate a rule by id/name; returns ``(kind, rule)`` where kind is zone|line."""
    for zone in body.get("zones") or []:
        if rule_id in (zone.get("id"), zone.get("name")):
            return "zone", zone
    for line in body.get("lines") or []:
        if rule_id in (line.get("id"), line.get("name")):
            return "line", line
    return None
