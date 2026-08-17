"""Reconnect catch-up over the real WS + REST pair (FRONTEND_SPEC §7).

The frontend only sees `alert.new` for alerts produced while its socket is open.
Anything the hub raises during a disconnect is recovered on reconnect with
`GET /api/alerts?after_id=<highest id already rendered>`. This test drives that
exact sequence against a running server and asserts the gap is filled — the path
the browser acceptance run could not observe directly.
"""

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer
from conftest import DEVICE, STREAM, detection, rules_body

from edge_hub.app import Hub


@pytest.fixture
async def client(tmp_path, clock):
    hub = Hub(tmp_path / "data", clock=clock, env={}, web_dir=None)
    hub.auth.ensure_default_account(password="admin")

    async def publisher(topic, payload, qos=1):
        return None

    hub.alerts.publisher = publisher
    test_client = TestClient(TestServer(hub.api.build()))
    await test_client.start_server()
    test_client.hub = hub
    try:
        yield test_client
    finally:
        await test_client.close()
        await hub.shutdown()


async def walk_into_zone(hub, track_id: int) -> None:
    """One track crossing from outside the zone to inside it -> one zone_enter."""
    await hub.on_detections(
        detection(0.2, 0.55, frame_id=track_id * 2 - 1, track_id=track_id)
    )
    await hub.on_detections(
        detection(0.75, 0.55, frame_id=track_id * 2, track_id=track_id)
    )


async def drain_alert_new(ws, count: int) -> list[dict]:
    """Read `count` alert.new frames, ignoring anything else on the socket."""
    out: list[dict] = []
    while len(out) < count:
        message = json.loads(await ws.receive_str())
        if message.get("type") == "alert.new":
            out.append(message["alert"])
    return out


async def test_alerts_raised_while_the_socket_is_down_are_recovered_by_after_id(client):
    hub = client.hub
    assert (
        await client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    ).status == 200
    hub.storage.put_rules(DEVICE, STREAM, rules_body(lines=[]), hub.clock.wall_ms())

    # ---- connected: the frontend renders what the socket pushes ----
    ws = await client.ws_connect("/ws")
    await walk_into_zone(hub, track_id=1)
    live = await drain_alert_new(ws, 1)
    rendered = [a["id"] for a in live]
    last_alert_id = max(rendered)

    # ---- disconnected: the hub keeps working, the browser hears nothing ----
    await ws.close()
    assert not hub.api.websockets or all(w.closed for w in hub.api.websockets)
    for track_id in (2, 3, 4):
        clock_before = hub.clock.wall_ms()
        await walk_into_zone(hub, track_id=track_id)
        assert hub.clock.wall_ms() == clock_before  # no wall-clock dependency here
    missed = [row["id"] for row in hub.storage.query_alerts()]
    assert len(missed) == 4, f"expected 4 stored alerts, got {missed}"

    # ---- reconnect: ws.js catchUp() issues exactly this request ----
    reconnected = await client.ws_connect("/ws")
    try:
        response = await client.get(f"/api/alerts?after_id={last_alert_id}&limit=500")
        assert response.status == 200
        body = await response.json()
        recovered = [a["id"] for a in body["alerts"]]

        # Every alert the socket missed, and nothing it already showed.
        assert last_alert_id not in recovered
        assert set(recovered) == set(missed) - set(rendered)
        assert len(recovered) == 3
        # ws.js upserts in the delivered order; ascending ids keep the list stable.
        assert recovered == sorted(recovered)
        # The recovered rows are whole alerts, not id stubs — the list renders them
        # without a second round trip.
        for alert in body["alerts"]:
            assert alert["device_id"] == DEVICE
            assert alert["stream_id"] == STREAM
            assert alert["event_type"] == "zone_enter"
            assert alert["state"] == "new"

        # ---- and the live path is healthy again after the catch-up ----
        await walk_into_zone(hub, track_id=5)
        after = await drain_alert_new(reconnected, 1)
        assert after[0]["id"] > max(recovered)
    finally:
        await reconnected.close()


async def test_after_id_returns_nothing_when_the_socket_missed_nothing(client):
    hub = client.hub
    await client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    hub.storage.put_rules(DEVICE, STREAM, rules_body(lines=[]), hub.clock.wall_ms())

    ws = await client.ws_connect("/ws")
    try:
        await walk_into_zone(hub, track_id=1)
        seen = await drain_alert_new(ws, 1)
    finally:
        await ws.close()

    body = await (await client.get(f"/api/alerts?after_id={seen[0]['id']}")).json()
    assert body["alerts"] == []
    assert body["count"] == 0
