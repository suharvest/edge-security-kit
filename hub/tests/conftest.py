"""Shared fixtures and payload builders."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "contracts"))

from edge_hub.alert_manager import AlertManager  # noqa: E402
from edge_hub.app import Hub  # noqa: E402
from edge_hub.clock import FakeClock  # noqa: E402
from edge_hub.rules.engine import RuleEngine  # noqa: E402
from edge_hub.storage import Storage  # noqa: E402

DEVICE = "jetson-01"
STREAM = "cam-0"
SESSION = "sess-1"

#: A vertical line at x=0.5 pointing downwards; left of it is side>0 (forward
#: means left -> right), per contracts/MQTT.md.
LINE = {"id": "gate", "name": "gate", "start": [0.5, 0.1], "end": [0.5, 0.9],
        "direction": "any"}
#: An axis-aligned box on the right half of the frame.
ZONE = {"id": "bay", "name": "bay",
        "points": [[0.6, 0.3], [0.9, 0.3], [0.9, 0.8], [0.6, 0.8]],
        "dwell_seconds": 10}
FEATURES = {"zone_detection": True, "loitering": True, "line_crossing": True}


def rules_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "zones": [dict(ZONE)],
        "lines": [dict(LINE)],
        "features": dict(FEATURES),
        "cooldown": 30,
        # Isolate the four-tuple cooldown unless a test opts into the stream
        # rate limit explicitly.
        "stream_rate_limit_s": 0,
    }
    body.update(overrides)
    return body


def detection(
    x: float,
    y: float,
    frame_id: int = 1,
    track_id: int = 1,
    device_id: str = DEVICE,
    stream_id: str = STREAM,
    session_id: str = SESSION,
    detections: list[dict[str, Any]] | None = None,
    timestamp: int = 1_755_400_000_000,
) -> dict[str, Any]:
    """Build a valid ``sensecraft.detection/1`` payload."""
    if detections is None:
        detections = [
            {"track_id": track_id, "class": "person", "score": 0.9,
             "bbox": [x, y, 0.08, 0.2]}
        ]
    return {
        "schema": "sensecraft.detection/1",
        "timestamp": timestamp,
        "session_id": session_id,
        "frame_id": frame_id,
        "device_id": device_id,
        "stream_id": stream_id,
        "coordinate_space": "frame_norm",
        "frame": {"w": 1920, "h": 1080},
        "inference_time_ms": 10.0,
        "detections": detections,
        "health": {"fps": 15.0, "decode": "hw", "backend": "synthetic",
                   "fallback_active": False},
    }


def status(
    device_id: str = DEVICE,
    session_id: str = SESSION,
    online: bool = True,
    streams: list[dict[str, Any]] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "sensecraft.status/1",
        "timestamp": 1_755_400_000_000,
        "session_id": session_id,
        "device_id": device_id,
        "online": online,
        "versions": {"app": "test/0.1", "model": "synthetic"},
        "uptime_s": 12.0,
    }
    if online:
        payload["streams"] = streams if streams is not None else [
            {"stream_id": STREAM, "state": "running", "fps": 15.0, "decode": "hw"}
        ]
    if mode:
        payload["mode"] = mode
    return payload


def device_event(
    event_id: str = "reCam-cam0-s1-1",
    event_type: str = "zone_enter",
    device_id: str = "recamera-01",
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "sensecraft.event/1",
        "timestamp": 1_755_400_000_000,
        "session_id": "s1",
        "event_id": event_id,
        "device_id": device_id,
        "stream_id": "cam0",
        "event_type": event_type,
        "rule_name": "door",
        "track_id": 4,
        "bbox": [0.5, 0.5, 0.1, 0.2],
    }
    if event_type == "line_cross":
        payload["direction"] = "forward"
    if event_type == "loitering":
        payload["dwell_s"] = 12.0
    payload.update(overrides)
    return payload


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    store = Storage(tmp_path / "data")
    yield store
    store.close()


@pytest.fixture
def engine(clock: FakeClock) -> RuleEngine:
    body = rules_body()
    return RuleEngine(lambda d, s: body if (d, s) == (DEVICE, STREAM) else None,
                      clock=clock)


@pytest.fixture
def alerts(storage: Storage, clock: FakeClock) -> AlertManager:
    published: list[tuple[str, bytes, int]] = []
    pushed: list[dict[str, Any]] = []

    async def publisher(topic: str, payload: bytes, qos: int = 1) -> None:
        published.append((topic, payload, qos))

    async def broadcaster(message: dict[str, Any]) -> None:
        pushed.append(message)

    manager = AlertManager(
        storage,
        clock=clock,
        publisher=publisher,
        broadcaster=broadcaster,
        snapshot_timeout_s=60.0,
    )
    manager.published = published  # type: ignore[attr-defined]
    manager.pushed = pushed  # type: ignore[attr-defined]
    return manager


@pytest.fixture
def hub(tmp_path: Path, clock: FakeClock) -> Hub:
    instance = Hub(tmp_path / "data", clock=clock, env={}, web_dir=None)
    instance.auth.ensure_default_account(password="admin")
    yield instance
    instance.storage.close()
