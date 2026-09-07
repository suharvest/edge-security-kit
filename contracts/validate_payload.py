#!/usr/bin/env python3
"""Dependency-free validator for sensecraft edge-security MQTT contract v1.

Checks a payload file against the required surface of
``mqtt-detection.schema.json`` — the three message kinds discriminated by
``schema`` — without needing jsonschema installed. Platforms run this over their
conformance fixtures in host-only tests (see MQTT.md "Conformance").

    python3 contracts/validate_payload.py path/to/payload.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DETECTION = "sensecraft.detection/1"
STATUS = "sensecraft.status/1"
EVENT = "sensecraft.event/1"
COMMAND = "sensecraft.command/1"
ACK = "sensecraft.ack/1"

DETECTION_REQUIRED = {
    "schema", "timestamp", "session_id", "frame_id", "device_id", "stream_id",
    "coordinate_space", "frame", "inference_time_ms", "detections", "health",
}
STATUS_REQUIRED = {"schema", "timestamp", "session_id", "device_id", "online"}
EVENT_REQUIRED = {
    "schema", "timestamp", "session_id", "event_id", "device_id", "stream_id",
    "event_type", "rule_name", "track_id", "bbox",
}
DETECTION_ITEM_REQUIRED = {"track_id", "class", "score", "bbox"}
HEALTH_REQUIRED = {"fps", "decode", "backend", "fallback_active"}
EVENT_TYPES = {"zone_enter", "loitering", "line_cross"}
STREAM_STATES = {"running", "reconnecting", "stopped"}
DECODE_PATHS = {"hw", "sw"}
COMMAND_REQUIRED = {"schema", "timestamp", "request_id", "device_id", "command", "params"}
ACK_REQUIRED = {
    "schema", "timestamp", "session_id", "device_id", "request_id", "command", "ok",
}
COMMAND_NAMES = {"set_conf_threshold", "add_stream", "remove_stream"}
#: Per-command required params. The runtime control surface is small on purpose:
#: anything that needs a restart belongs in the config file, not here.
COMMAND_PARAMS = {
    "set_conf_threshold": {"stream_id", "conf_threshold"},
    "add_stream": {"stream_id", "source"},
    "remove_stream": {"stream_id"},
}


def _require(obj: object, keys: set[str], where: str) -> dict:
    if not isinstance(obj, dict):
        raise ValueError(f"{where}: expected an object")
    missing = keys - obj.keys()
    if missing:
        raise ValueError(f"{where}: missing {sorted(missing)}")
    return obj


def _epoch_ms(value: object, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: must be Unix epoch milliseconds (integer >= 0)")


def _number(value: object, where: str, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number")
    if low is not None and value < low:
        raise ValueError(f"{where}: must be >= {low}")
    if high is not None and value > high:
        raise ValueError(f"{where}: must be <= {high}")
    return float(value)


def _nonempty_str(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: expected a non-empty string")
    return value


def _bbox(value: object, where: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{where}: bbox must be [cx, cy, w, h]")
    for index, item in enumerate(value):
        _number(item, f"{where}[{index}]", 0, 1)


def _int_min(value: object, where: str, low: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: expected an integer")
    if value < low:
        raise ValueError(f"{where}: must be >= {low}")
    return value


def _health(value: object, where: str) -> None:
    health = _require(value, HEALTH_REQUIRED, where)
    _number(health["fps"], f"{where}.fps", 0)
    if health["decode"] not in DECODE_PATHS:
        raise ValueError(f"{where}.decode: expected hw or sw")
    _nonempty_str(health["backend"], f"{where}.backend")
    if not isinstance(health["fallback_active"], bool):
        raise ValueError(f"{where}.fallback_active: expected a boolean")


def validate_detection(payload: dict) -> None:
    _require(payload, DETECTION_REQUIRED, "payload")
    _epoch_ms(payload["timestamp"], "payload.timestamp")
    _nonempty_str(payload["session_id"], "payload.session_id")
    _int_min(payload["frame_id"], "payload.frame_id", 0)
    _nonempty_str(payload["device_id"], "payload.device_id")
    _nonempty_str(payload["stream_id"], "payload.stream_id")
    if payload["coordinate_space"] != "frame_norm":
        raise ValueError("payload.coordinate_space: the only permitted value is frame_norm")
    frame = _require(payload["frame"], {"w", "h"}, "payload.frame")
    if set(frame) - {"w", "h"}:
        raise ValueError("payload.frame: no additional properties allowed")
    _int_min(frame["w"], "payload.frame.w", 1)
    _int_min(frame["h"], "payload.frame.h", 1)
    _number(payload["inference_time_ms"], "payload.inference_time_ms", 0)
    if "pipeline_ms" in payload:
        _number(payload["pipeline_ms"], "payload.pipeline_ms", 0)
    if not isinstance(payload["detections"], list):
        raise ValueError("payload.detections: expected an array")
    for index, det in enumerate(payload["detections"]):
        where = f"payload.detections[{index}]"
        _require(det, DETECTION_ITEM_REQUIRED, where)
        _int_min(det["track_id"], f"{where}.track_id", 0)
        _nonempty_str(det["class"], f"{where}.class")
        _number(det["score"], f"{where}.score", 0, 1)
        _bbox(det["bbox"], f"{where}.bbox")
    _health(payload["health"], "payload.health")


def validate_status(payload: dict) -> None:
    _require(payload, STATUS_REQUIRED, "payload")
    _epoch_ms(payload["timestamp"], "payload.timestamp")
    _nonempty_str(payload["session_id"], "payload.session_id")
    _nonempty_str(payload["device_id"], "payload.device_id")
    if not isinstance(payload["online"], bool):
        raise ValueError("payload.online: expected a boolean")
    streams = payload.get("streams")
    if streams is not None:
        if not isinstance(streams, list):
            raise ValueError("payload.streams: expected an array")
        for index, stream in enumerate(streams):
            where = f"payload.streams[{index}]"
            _require(stream, {"stream_id", "state"}, where)
            _nonempty_str(stream["stream_id"], f"{where}.stream_id")
            if stream["state"] not in STREAM_STATES:
                raise ValueError(f"{where}.state: expected one of {sorted(STREAM_STATES)}")
            if "fps" in stream:
                _number(stream["fps"], f"{where}.fps", 0)
            if "decode" in stream and stream["decode"] not in DECODE_PATHS:
                raise ValueError(f"{where}.decode: expected hw or sw")
    if "uptime_s" in payload:
        _number(payload["uptime_s"], "payload.uptime_s", 0)
    if payload["online"] is False and streams:
        raise ValueError("payload.streams: the LWT payload (online false) omits streams")


def validate_event(payload: dict) -> None:
    _require(payload, EVENT_REQUIRED, "payload")
    _epoch_ms(payload["timestamp"], "payload.timestamp")
    _nonempty_str(payload["session_id"], "payload.session_id")
    _nonempty_str(payload["event_id"], "payload.event_id")
    _nonempty_str(payload["device_id"], "payload.device_id")
    _nonempty_str(payload["stream_id"], "payload.stream_id")
    _nonempty_str(payload["rule_name"], "payload.rule_name")
    if payload["event_type"] not in EVENT_TYPES:
        raise ValueError(f"payload.event_type: expected one of {sorted(EVENT_TYPES)}")
    _int_min(payload["track_id"], "payload.track_id", 0)
    _bbox(payload["bbox"], "payload.bbox")
    if payload["event_type"] == "line_cross":
        if payload.get("direction") not in ("forward", "backward"):
            raise ValueError("payload.direction: line_cross requires forward or backward")
    elif "direction" in payload and payload["direction"] not in ("forward", "backward"):
        raise ValueError("payload.direction: expected forward or backward")
    if payload["event_type"] == "loitering":
        _number(payload.get("dwell_s"), "payload.dwell_s", 0)
    elif "dwell_s" in payload:
        _number(payload["dwell_s"], "payload.dwell_s", 0)
    if "score" in payload:
        _number(payload["score"], "payload.score", 0, 1)
    if "class" in payload:
        _nonempty_str(payload["class"], "payload.class")


def validate_command(payload: dict) -> None:
    _require(payload, COMMAND_REQUIRED, "payload")
    _epoch_ms(payload["timestamp"], "payload.timestamp")
    _nonempty_str(payload["request_id"], "payload.request_id")
    _nonempty_str(payload["device_id"], "payload.device_id")
    command = payload["command"]
    if command not in COMMAND_NAMES:
        raise ValueError(f"payload.command: expected one of {sorted(COMMAND_NAMES)}")
    params = _require(payload["params"], COMMAND_PARAMS[command], f"payload.params ({command})")
    _nonempty_str(params["stream_id"], "payload.params.stream_id")
    if command == "set_conf_threshold":
        _number(params["conf_threshold"], "payload.params.conf_threshold", 0, 1)
    if command == "add_stream":
        _nonempty_str(params["source"], "payload.params.source")
        transport = params.get("rtsp_transport")
        if transport is not None and transport not in ("tcp", "udp"):
            raise ValueError("payload.params.rtsp_transport: expected tcp or udp")


def validate_ack(payload: dict) -> None:
    _require(payload, ACK_REQUIRED, "payload")
    _epoch_ms(payload["timestamp"], "payload.timestamp")
    _nonempty_str(payload["session_id"], "payload.session_id")
    _nonempty_str(payload["device_id"], "payload.device_id")
    _nonempty_str(payload["request_id"], "payload.request_id")
    if payload["command"] not in COMMAND_NAMES:
        raise ValueError(f"payload.command: expected one of {sorted(COMMAND_NAMES)}")
    if not isinstance(payload["ok"], bool):
        raise ValueError("payload.ok: expected a boolean")
    # A failure that carries no reason is unactionable at the console, so the
    # field is required in exactly the case where it means something.
    if payload["ok"] is False:
        _nonempty_str(payload.get("error"), "payload.error")
    if "applied" in payload and not isinstance(payload["applied"], dict):
        raise ValueError("payload.applied: expected an object")


VALIDATORS = {
    DETECTION: validate_detection,
    STATUS: validate_status,
    EVENT: validate_event,
    COMMAND: validate_command,
    ACK: validate_ack,
}


def validate(payload: object) -> str:
    """Validate one payload; returns the message kind it matched."""
    if not isinstance(payload, dict):
        raise ValueError("payload: expected a JSON object")
    kind = payload.get("schema")
    if kind not in VALIDATORS:
        raise ValueError(
            f"payload.schema: expected one of {sorted(VALIDATORS)}, got {kind!r}"
        )
    VALIDATORS[kind](payload)
    return kind


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} PAYLOAD.json")
    try:
        kind = validate(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    except ValueError as exc:
        raise SystemExit(f"MQTT contract v1 FAILED: {exc}") from None
    print(f"MQTT contract v1 passed ({kind})")


if __name__ == "__main__":
    main()
