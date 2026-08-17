#!/usr/bin/env python3
"""Synthetic detector for end-to-end hub self-test (no real device needed).

Publishes contract-conformant `sensecraft.status/1` (retained, with LWT) and
`sensecraft.detection/1` messages that deliberately produce all three rule
events, and answers `cmd/snapshot` with a tiny valid JPEG so the §3.1 evidence
round trip completes.

    uv run python tools/fake_detector.py --host localhost --seed-rules

Scenario (all coordinates already frame_norm, per contracts/MQTT.md):

* track 1 walks left -> right across the vertical line x=0.5  => line_cross forward
* track 2 walks right -> left across the same line            => line_cross backward
* track 3 stands still inside the zone                        => zone_enter, then loitering
* track 0 (untracked) is emitted every frame and must never fire a rule
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import time
import urllib.error
import urllib.request

# 1x1 baseline JPEG: enough to satisfy the magic-byte + size check.
TINY_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
    "Hh0aHBwcJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPDs0NDP/wAALCAABAAEBAREA/8QAFAAB"
    "AQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAA"
    "AAAAAAAAAP/aAAwDAQACEQMRAD8AmAA//9k="
)

RULES_BODY = {
    "zones": [
        {
            "id": "loading_bay",
            "name": "loading_bay",
            "points": [[0.60, 0.30], [0.92, 0.30], [0.92, 0.80], [0.60, 0.80]],
            "dwell_seconds": 3,
        }
    ],
    "lines": [
        {"id": "gate", "name": "gate", "start": [0.5, 0.10], "end": [0.5, 0.90],
         "direction": "any"}
    ],
    "features": {"zone_detection": True, "loitering": True, "line_crossing": True},
    "cooldown": 5,
    "stream_rate_limit_s": 0,
}


def seed_rules(hub_url: str, username: str, password: str, device_id: str, stream_id: str) -> None:
    """Log in to the hub and PUT the scenario rules for this device/stream."""
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(),
    )
    login = urllib.request.Request(
        f"{hub_url}/api/auth/login",
        data=json.dumps({"username": username, "password": password}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(login, timeout=10) as response:
        print("login:", response.status, response.read().decode())
    put = urllib.request.Request(
        f"{hub_url}/api/rules/{device_id}/{stream_id}",
        data=json.dumps(RULES_BODY).encode(),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with opener.open(put, timeout=10) as response:
        print("put rules:", response.status, response.read().decode()[:200])


def status_payload(device_id: str, session_id: str, started: float, online: bool = True) -> dict:
    payload = {
        "schema": "sensecraft.status/1",
        "timestamp": int(time.time() * 1000),
        "session_id": session_id,
        "device_id": device_id,
        "online": online,
        "versions": {"app": "fake-detector/0.1", "model": "synthetic"},
        "uptime_s": round(time.time() - started, 1),
    }
    if online:
        payload["streams"] = [
            {
                "stream_id": "cam-0",
                "state": "running",
                "fps": 15.0,
                "decode": "hw",
                "preview_url": "http://127.0.0.1:8080/preview/cam-0.jpg",
                "live_url": "http://127.0.0.1:8080/live/cam-0",
            }
        ]
    return payload


def positions(step: int, total: int) -> list[dict]:
    """Three tracked persons plus one untracked detection."""
    phase = step / max(total - 1, 1)
    left_to_right = 0.15 + 0.70 * phase
    right_to_left = 0.85 - 0.70 * phase
    return [
        {"track_id": 1, "class": "person", "score": 0.91,
         "bbox": [round(left_to_right, 4), 0.55, 0.08, 0.22]},
        {"track_id": 2, "class": "person", "score": 0.88,
         "bbox": [round(right_to_left, 4), 0.65, 0.08, 0.22]},
        {"track_id": 3, "class": "person", "score": 0.83,
         "bbox": [0.75, round(0.55 + 0.01 * math.sin(step / 3), 4), 0.09, 0.24]},
        {"track_id": 0, "class": "person", "score": 0.41,
         "bbox": [0.05, 0.05, 0.04, 0.09]},
    ]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--device-id", default="fake-01")
    parser.add_argument("--stream-id", default="cam-0")
    parser.add_argument("--prefix", default="sensecraft/security")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--hub-url", default="http://127.0.0.1:8090")
    parser.add_argument("--seed-rules", action="store_true",
                        help="PUT the scenario rules to the hub before publishing")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="admin")
    args = parser.parse_args()

    if args.seed_rules:
        try:
            seed_rules(args.hub_url, args.username, args.password,
                       args.device_id, args.stream_id)
        except urllib.error.URLError as exc:
            print(f"could not seed rules ({exc}); continuing")

    import aiomqtt

    session_id = str(int(time.time() * 1000))
    started = time.time()
    base = f"{args.prefix}/{args.device_id}"
    total = max(int(args.fps * args.seconds), 2)
    interval = 1.0 / args.fps
    snapshots_sent = 0

    async with aiomqtt.Client(
        hostname=args.host,
        port=args.port,
        identifier=f"fake-detector-{session_id}",
        will=aiomqtt.Will(
            topic=f"{base}/status",
            payload=json.dumps(status_payload(args.device_id, session_id, started, online=False)),
            qos=1,
            retain=True,
        ),
    ) as client:
        await client.subscribe(f"{base}/cmd/snapshot", qos=1)
        await client.publish(
            f"{base}/status",
            json.dumps(status_payload(args.device_id, session_id, started)),
            qos=1,
            retain=True,
        )
        print(f"published retained status on {base}/status")

        async def answer_snapshots() -> None:
            nonlocal snapshots_sent
            async for message in client.messages:
                try:
                    request = json.loads(message.payload)
                except ValueError:
                    continue
                event_id = request.get("event_id")
                if not event_id:
                    continue
                await client.publish(f"{base}/snapshot/{event_id}", TINY_JPEG, qos=1)
                snapshots_sent += 1
                print(f"snapshot -> {event_id}")

        reader = asyncio.create_task(answer_snapshots())
        try:
            for step in range(total):
                payload = {
                    "schema": "sensecraft.detection/1",
                    "timestamp": int(time.time() * 1000),
                    "session_id": session_id,
                    "frame_id": step,
                    "device_id": args.device_id,
                    "stream_id": args.stream_id,
                    "coordinate_space": "frame_norm",
                    "frame": {"w": 1920, "h": 1080},
                    "inference_time_ms": 11.4,
                    "pipeline_ms": 28.0,
                    "detections": positions(step, total),
                    "health": {
                        "fps": args.fps,
                        "decode": "hw",
                        "backend": "synthetic",
                        "fallback_active": False,
                    },
                }
                await client.publish(
                    f"{base}/detections/{args.stream_id}", json.dumps(payload), qos=0
                )
                await asyncio.sleep(interval)
            # Give the hub time to request and receive the evidence.
            await asyncio.sleep(2.0)
        finally:
            reader.cancel()
        print(f"sent {total} detection frames, answered {snapshots_sent} snapshot requests")


if __name__ == "__main__":
    asyncio.run(main())
