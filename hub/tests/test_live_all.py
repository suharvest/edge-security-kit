"""``GET /api/live`` — every stream's last detections in one response.

The video wall draws overlay boxes on up to nine tiles. Nine per-stream
requests per tick cost nine round trips over data that lives in one dict, and
the tiles then disagree by a request latency. This endpoint is the batch form,
and it must stay identical in content to the per-stream one it replaces.
"""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer
from conftest import DEVICE, STREAM, detection

from edge_hub.app import Hub


@pytest.fixture
async def client(tmp_path, clock):
    hub = Hub(tmp_path / "data", clock=clock, env={}, web_dir=None)
    hub.auth.ensure_default_account(password="admin")
    server = TestServer(hub.api.build())
    test_client = TestClient(server)
    await test_client.start_server()
    test_client.hub = hub
    await test_client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    try:
        yield test_client
    finally:
        await test_client.close()
        await hub.shutdown()


async def test_empty_before_any_detection(client):
    body = await (await client.get("/api/live")).json()
    assert body["streams"] == []
    assert body["count"] == 0


async def test_every_stream_appears_once_with_its_latest_frame(client):
    registry = client.hub.registry
    registry.note_detections(detection(0.2, 0.2, frame_id=1))
    registry.note_detections(detection(0.3, 0.3, frame_id=2))
    registry.note_detections(detection(0.4, 0.4, frame_id=1, stream_id="cam-1"))
    registry.note_detections(detection(0.5, 0.5, frame_id=1, device_id="rk3588-02"))

    body = await (await client.get("/api/live")).json()
    assert body["count"] == 3
    keys = [(s["device_id"], s["stream_id"]) for s in body["streams"]]
    assert keys == [(DEVICE, STREAM), (DEVICE, "cam-1"), ("rk3588-02", STREAM)]
    first = next(s for s in body["streams"] if s["stream_id"] == STREAM
                 and s["device_id"] == DEVICE)
    assert first["payload"]["frame_id"] == 2


async def test_batch_and_per_stream_answer_the_same_thing(client):
    """The wall must not see a different scene from the rule editor."""
    client.hub.registry.note_detections(detection(0.6, 0.4, frame_id=7))
    batch = await (await client.get("/api/live")).json()
    single = await (await client.get(f"/api/live/{DEVICE}/{STREAM}")).json()
    entry = batch["streams"][0]
    assert entry["payload"] == single["payload"]
    assert entry["received_ms"] == single["received_ms"]


async def test_live_needs_a_session(tmp_path, clock):
    hub = Hub(tmp_path / "data", clock=clock, env={}, web_dir=None)
    hub.auth.ensure_default_account(password="admin")
    server = TestServer(hub.api.build())
    anon = TestClient(server)
    await anon.start_server()
    try:
        assert (await anon.get("/api/live")).status == 401
    finally:
        await anon.close()
        await hub.shutdown()
