#!/usr/bin/env python3
"""Capture conformance fixtures from a real hub run (HUB_SPEC §10).

Drives the rule engine with synthetic detections that cross a line, enter a
zone and loiter in it, then writes the payloads the hub actually republished on
``events/<stream_id>`` into ``contracts/fixtures/``. Also emits one
device-origin event as a single-box publisher would send it.

The point is that the fixtures are captured output, not hand-written JSON: if a
field name or its type drifts, the fixture drifts with it and the schema check
in CI notices.

    uv run python tools/capture_fixtures.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "hub"))
sys.path.insert(0, str(REPO / "hub" / "tests"))

OUT = REPO / "contracts" / "fixtures"

DEVICE = "generic-01"
STREAM = "cam-0"
SESSION = "1786977934162"
# A vertical line at x=0.5 and a zone on the right half, matching the shapes the
# integration run used.
RULES = {
    "zones": [
        {
            "id": "loading_bay",
            "name": "loading_bay",
            "points": [[0.6, 0.3], [0.9, 0.3], [0.9, 0.8], [0.6, 0.8]],
            "dwell_seconds": 5,
        }
    ],
    "lines": [
        {"id": "gate", "name": "gate", "start": [0.5, 0.1], "end": [0.5, 0.9],
         "direction": "any"},
    ],
    "features": {"zone_detection": True, "loitering": True, "line_crossing": True},
    "cooldown": 30,
}


def detection(x: float, y: float, frame_id: int, ts_ms: int, track_id: int = 1) -> dict:
    return {
        "schema": "sensecraft.detection/1",
        "timestamp": ts_ms,
        "session_id": SESSION,
        "frame_id": frame_id,
        "device_id": DEVICE,
        "stream_id": STREAM,
        "coordinate_space": "frame_norm",
        "frame": {"w": 1280, "h": 720},
        "inference_time_ms": 34.29,
        "detections": [
            {"track_id": track_id, "class": "person", "score": 0.9117,
             "bbox": [x, y, 0.14, 0.65]},
        ],
        "health": {"fps": 15.0, "decode": "sw",
                   "backend": "onnxruntime-1.28.0-cpu", "fallback_active": False},
    }


async def main() -> int:
    from edge_hub.app import Hub
    from edge_hub.clock import FakeClock

    OUT.mkdir(parents=True, exist_ok=True)
    captured: list[tuple[str, bytes]] = []

    with tempfile.TemporaryDirectory() as tmp:
        clock = FakeClock()
        hub = Hub(Path(tmp) / "data", clock=clock, env={}, web_dir=None)

        async def publisher(topic: str, payload: bytes, qos: int = 1) -> None:
            captured.append((topic, payload))

        hub.alerts.publisher = publisher
        hub.storage.put_rules(DEVICE, STREAM, RULES, clock.wall_ms())

        frame = 0
        ts = 1786977945994

        async def feed(x: float, y: float, track_id: int = 1) -> None:
            nonlocal frame, ts
            frame += 1
            ts += 67
            clock.advance(67)
            await hub.on_detections(detection(x, y, frame, ts, track_id))

        # Left of the gate, outside the zone.
        await feed(0.30, 0.55)
        # Cross the gate left -> right: line_cross.
        await feed(0.70, 0.55)
        # Enter the zone: zone_enter.
        await feed(0.75, 0.55)
        # Hold inside past dwell_seconds=5: loitering.
        for _ in range(3):
            clock.advance(2_000)
            await feed(0.75, 0.55)

        # A single-box device publishing its own verdict (origin defaults to
        # device when absent — the fixture keeps it explicit).
        device_event = {
            "schema": "sensecraft.event/1",
            "origin": "device",
            "timestamp": ts + 100,
            "session_id": SESSION,
            "event_id": f"{DEVICE}-{STREAM}-{SESSION}-device-1",
            "device_id": "recamera-02",
            "stream_id": "cam-0",
            "event_type": "zone_enter",
            "rule_name": "loading_bay",
            "track_id": 4,
            "bbox": [0.71, 0.52, 0.13, 0.61],
            "class": "person",
            "score": 0.86,
        }

        await hub.shutdown()

    by_type: dict[str, dict] = {}
    for topic, payload in captured:
        if "/events/" not in topic:
            continue
        body = json.loads(payload)
        by_type.setdefault(body["event_type"], body)

    missing = {"zone_enter", "line_cross", "loitering"} - set(by_type)
    if missing:
        print(f"FAILED to capture: {sorted(missing)}", file=sys.stderr)
        print(f"captured topics: {[t for t, _ in captured]}", file=sys.stderr)
        return 1

    written = []
    for event_type, body in sorted(by_type.items()):
        path = OUT / f"event-{event_type.replace('_', '-')}-hub.json"
        path.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n")
        written.append(path)
    path = OUT / "event-zone-enter-device.json"
    path.write_text(json.dumps(device_event, indent=2) + "\n")
    written.append(path)

    for path in written:
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
