"""REST, WS and auth surface (HUB_SPEC §4, §5, §7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from conftest import DEVICE, STREAM, detection, device_event, rules_body, status

from edge_hub.app import Hub

VALID_RULES = {
    "zones": [{"id": "bay", "name": "bay",
               "points": [[0.6, 0.3], [0.9, 0.3], [0.9, 0.8], [0.6, 0.8]],
               "dwell_seconds": 5}],
    "lines": [{"id": "gate", "name": "gate", "start": [0.5, 0.1], "end": [0.5, 0.9],
               "direction": "forward"}],
    "features": {"zone_detection": True, "loitering": True, "line_crossing": True},
    "cooldown": 10,
}


@pytest.fixture
async def client(tmp_path, clock):
    hub = Hub(tmp_path / "data", clock=clock, env={}, web_dir=None)
    hub.auth.ensure_default_account(password="admin")
    # Capture what the hub would publish; there is no broker in these tests.
    captured: list[tuple[str, bytes, int]] = []

    async def publisher(topic: str, payload: bytes, qos: int = 1) -> None:
        captured.append((topic, payload, qos))

    hub.alerts.publisher = publisher
    hub._captured_publishes = captured
    server = TestServer(hub.api.build())
    test_client = TestClient(server)
    await test_client.start_server()
    test_client.hub = hub
    try:
        yield test_client
    finally:
        await test_client.close()
        await hub.shutdown()


async def login(client) -> None:
    response = await client.post("/api/auth/login",
                                json={"username": "admin", "password": "admin"})
    assert response.status == 200, await response.text()


async def seed_alert(client, **overrides) -> dict:
    """Push one detection through the whole chain to get a stored alert.

    Zone-only rules, so walking into the zone produces exactly one alert.
    """
    hub = client.hub
    overrides.setdefault("lines", [])
    hub.storage.put_rules(DEVICE, STREAM, rules_body(**overrides), hub.clock.wall_ms())
    await hub.on_detections(detection(0.2, 0.55, frame_id=1))
    await hub.on_detections(detection(0.75, 0.55, frame_id=2))
    rows = hub.storage.query_alerts()
    assert rows, "expected the zone rule to fire"
    return rows[0]


# -- auth ---------------------------------------------------------------
async def test_health_needs_no_session(client):
    response = await client.get("/api/health")
    assert response.status == 200
    body = await response.json()
    assert body["ok"] is True
    assert body["mqtt_connected"] is False
    assert body["schema_failures"] == 0


async def test_every_other_endpoint_returns_401_without_a_session(client):
    for path in ("/api/alerts", "/api/devices", "/api/rules", "/api/config"):
        response = await client.get(path)
        assert response.status == 401, path
        assert (await response.json())["error"] == "unauthenticated"


async def test_login_sets_an_httponly_strict_cookie(client):
    response = await client.post("/api/auth/login",
                                json={"username": "admin", "password": "admin"})
    assert response.status == 200
    assert (await response.json())["must_change"] is True
    cookie = response.headers.get("Set-Cookie", "")
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie


async def test_bad_credentials_are_rejected(client):
    response = await client.post("/api/auth/login",
                                json={"username": "admin", "password": "wrong"})
    assert response.status == 401


async def test_logout_invalidates_the_session(client):
    await login(client)
    assert (await client.get("/api/alerts")).status == 200
    assert (await client.post("/api/auth/logout")).status == 200
    assert (await client.get("/api/alerts")).status == 401


async def test_password_change_clears_must_change(client):
    await login(client)
    response = await client.post(
        "/api/auth/password",
        json={"old_password": "admin", "new_password": "hunter2hunter2"},
    )
    assert response.status == 200
    assert (await client.get("/api/auth/session")).status == 200
    assert (await (await client.get("/api/auth/session")).json())["must_change"] is False
    bad = await client.post("/api/auth/password",
                            json={"old_password": "admin", "new_password": "x" * 12})
    assert bad.status == 400


async def test_short_new_password_is_rejected(client):
    await login(client)
    response = await client.post("/api/auth/password",
                                 json={"old_password": "admin", "new_password": "short"})
    assert response.status == 400


def test_initial_password_is_generated_not_fixed(tmp_path, clock):
    """§7: a first start with no HUB_ADMIN_PASSWORD generates a random secret.

    Adjudicated against a fixed admin/admin default — a shipped credential on a
    security product's own console is the failure mode being avoided here.
    """
    from edge_hub.auth import AuthManager
    from edge_hub.storage import Storage

    secrets_seen = set()
    for index in range(2):
        storage = Storage(tmp_path / f"gen{index}")
        auth = AuthManager(storage, clock=clock)
        username, plaintext = auth.ensure_default_account()
        assert username == "admin"
        assert plaintext is not None and len(plaintext) >= 16
        assert plaintext not in ("admin", "password")
        assert auth.must_change("admin") is True
        # The generated secret actually authenticates.
        assert auth.login("admin", plaintext) is not None
        secrets_seen.add(plaintext)
        # A second call on a populated table must not rotate the password.
        assert auth.ensure_default_account()[1] is None
        storage.close()
    assert len(secrets_seen) == 2, "generated passwords must not repeat"


def test_explicit_password_overrides_generation(tmp_path, clock):
    from edge_hub.auth import AuthManager
    from edge_hub.storage import Storage

    storage = Storage(tmp_path / "explicit")
    auth = AuthManager(storage, clock=clock)
    assert auth.ensure_default_account(password="from-env-1234")[1] == "from-env-1234"
    assert auth.login("admin", "from-env-1234") is not None
    storage.close()


# -- alerts -------------------------------------------------------------
async def test_alert_list_and_single_fetch(client):
    await login(client)
    alert = await seed_alert(client)
    listing = await (await client.get("/api/alerts")).json()
    assert [a["id"] for a in listing["alerts"]] == [alert["id"]]
    single = await client.get(f"/api/alerts/{alert['id']}")
    assert (await single.json())["alert"]["rule_name"] == "bay"
    assert (await client.get("/api/alerts/9999")).status == 404


async def test_after_id_gap_fill_is_ascending(client):
    await login(client)
    hub = client.hub
    hub.storage.put_rules(
        DEVICE, STREAM, rules_body(cooldown=0, lines=[]), hub.clock.wall_ms()
    )
    for frame in range(1, 7):
        inside = frame % 2 == 0
        await hub.on_detections(
            detection(0.75 if inside else 0.2, 0.55, frame_id=frame)
        )
    ids = [a["id"] for a in (await (await client.get("/api/alerts")).json())["alerts"]]
    assert len(ids) >= 3
    cursor = min(ids)
    rows = (await (await client.get(f"/api/alerts?after_id={cursor}")).json())["alerts"]
    assert [a["id"] for a in rows] == sorted(a["id"] for a in rows)
    assert all(a["id"] > cursor for a in rows)


async def test_ack_then_repeat_is_409(client):
    await login(client)
    alert = await seed_alert(client)
    first = await client.post(f"/api/alerts/{alert['id']}/ack")
    assert first.status == 200
    assert (await first.json())["alert"]["state"] == "acked"
    repeat = await client.post(f"/api/alerts/{alert['id']}/ack")
    assert repeat.status == 409
    assert (await repeat.json())["state"] == "acked"


async def test_acked_to_dismissed_reclassification_is_allowed(client):
    await login(client)
    alert = await seed_alert(client)
    await client.post(f"/api/alerts/{alert['id']}/ack")
    response = await client.post(f"/api/alerts/{alert['id']}/dismiss")
    assert response.status == 200
    assert (await response.json())["alert"]["state"] == "dismissed"
    back = await client.post(f"/api/alerts/{alert['id']}/ack")
    assert back.status == 200


async def test_transition_on_a_missing_alert_is_404(client):
    await login(client)
    assert (await client.post("/api/alerts/4242/ack")).status == 404


async def test_batch_returns_per_item_status(client):
    await login(client)
    alert = await seed_alert(client)
    await client.post(f"/api/alerts/{alert['id']}/ack")
    response = await client.post("/api/alerts/ack", json={"ids": [alert["id"], 9999]})
    results = (await response.json())["results"]
    assert {r["id"]: r["status"] for r in results} == {alert["id"]: 409, 9999: 404}


async def test_batch_limit_is_enforced(client):
    await login(client)
    response = await client.post("/api/alerts/dismiss", json={"ids": list(range(501))})
    assert response.status == 400
    assert (await client.post("/api/alerts/dismiss", json={"ids": "nope"})).status == 400


async def test_csv_export_has_a_bom_and_the_documented_columns(client):
    await login(client)
    await seed_alert(client)
    response = await client.get("/api/alerts/export.csv")
    assert response.status == 200
    raw = await response.read()
    assert raw.startswith(b"\xef\xbb\xbf")
    header = raw.decode("utf-8-sig").splitlines()[0]
    assert header.split(",")[:5] == ["id", "ts_ms", "device_id", "stream_id", "event_type"]


async def test_alerts_and_csv_export_both_filter_by_rule_name(client):
    """§4: ``rule_name`` is a server-side filter on both endpoints.

    They share one parser, so an export taken from a filtered list covers the
    same rows as the list did. Filtering only in the browser produced a CSV
    containing every rule.
    """
    await login(client)
    await seed_alert(client)
    stored = client.hub.storage.query_alerts()
    assert len(stored) == 1
    name = stored[0]["rule_name"]

    unfiltered = await (await client.get("/api/alerts")).json()
    match = await (await client.get(f"/api/alerts?rule_name={name}")).json()
    miss = await (await client.get("/api/alerts?rule_name=no-such-rule")).json()
    assert unfiltered["count"] == 1
    assert match["count"] == 1
    assert miss["count"] == 0

    csv_match = await (await client.get(f"/api/alerts/export.csv?rule_name={name}")).read()
    csv_miss = await (await client.get("/api/alerts/export.csv?rule_name=no-such-rule")).read()
    assert len(csv_match.decode("utf-8-sig").strip().splitlines()) == 2  # header + row
    assert len(csv_miss.decode("utf-8-sig").strip().splitlines()) == 1   # header only


async def test_alert_payload_names_the_timestamp_once(client):
    """§6 calls the column ``ts_ms``; the wire uses that name and no alias."""
    await login(client)
    await seed_alert(client)
    body = await (await client.get("/api/alerts")).json()
    alert = body["alerts"][0]
    assert "ts_ms" in alert
    assert "ts" not in alert


async def test_snapshot_endpoint_serves_the_file_and_404s_otherwise(client):
    await login(client)
    alert = await seed_alert(client)
    assert (await client.get(f"/api/alerts/{alert['id']}/snapshot.jpg")).status == 404
    await client.hub.alerts.attach_snapshot(alert["event_id"], b"\xff\xd8\xff\xd9")
    response = await client.get(f"/api/alerts/{alert['id']}/snapshot.jpg")
    assert response.status == 200
    assert response.headers["Content-Type"] == "image/jpeg"
    assert await response.read() == b"\xff\xd8\xff\xd9"


async def test_alert_stats_group_by_rule(client):
    await login(client)
    alert = await seed_alert(client)
    await client.post(f"/api/alerts/{alert['id']}/dismiss")
    stats = (await (await client.get("/api/alerts/stats")).json())["by_rule"]
    assert stats[0]["dismissed"] == 1


# -- rules --------------------------------------------------------------
async def test_put_rules_returns_rev_and_persist_time(client):
    await login(client)
    response = await client.put(f"/api/rules/{DEVICE}/{STREAM}", json=VALID_RULES)
    assert response.status == 200
    body = await response.json()
    assert body["rev"] == 1 and body["persisted_ms"] > 0
    again = await client.put(f"/api/rules/{DEVICE}/{STREAM}", json=VALID_RULES)
    assert (await again.json())["rev"] == 2
    tree = (await (await client.get("/api/rules")).json())["rules"]
    assert tree[DEVICE][STREAM]["rev"] == 2
    single = await client.get(f"/api/rules/{DEVICE}/{STREAM}")
    assert (await single.json())["body"]["cooldown"] == 10


async def test_put_rules_rejects_bad_geometry(client):
    await login(client)
    for bad in (
        {"zones": [{"points": [[0.1, 0.1], [0.2, 0.2]]}]},
        {"zones": [{"points": [[0.1, 0.1], [0.2, 0.2], [1.4, 0.3]]}]},
        {"lines": [{"start": [0.1, 0.1], "end": [0.1, 0.1]}]},
        {"lines": [{"start": [0.1, 0.1], "end": [0.9, 0.9], "direction": "sideways"}]},
        {"cooldown": -1},
    ):
        response = await client.put(f"/api/rules/{DEVICE}/{STREAM}", json=bad)
        assert response.status == 400, bad
        assert "error" in await response.json()


async def test_rules_take_effect_immediately(client):
    await login(client)
    await client.put(f"/api/rules/{DEVICE}/{STREAM}", json=VALID_RULES)
    hub = client.hub
    await hub.on_detections(detection(0.75, 0.55, frame_id=1))
    assert hub.storage.query_alerts()[0]["rule_name"] == "bay"


async def test_missing_rules_returns_404(client):
    await login(client)
    assert (await client.get("/api/rules/nobody/nostream")).status == 404


async def test_simulate_produces_a_flagged_alert(client):
    await login(client)
    await client.put(f"/api/rules/{DEVICE}/{STREAM}", json=VALID_RULES)
    response = await client.post(f"/api/rules/{DEVICE}/{STREAM}/simulate",
                                json={"rule_id": "gate"})
    assert response.status == 200
    alert = (await response.json())["alert"]
    assert alert["simulated"] is True
    assert alert["event_type"] == "line_cross"
    assert alert["direction"] == "forward"
    assert client.hub.storage.get_alert(alert["id"]) is not None
    missing = await client.post(f"/api/rules/{DEVICE}/{STREAM}/simulate",
                                json={"rule_id": "ghost"})
    assert missing.status == 404


# -- devices ------------------------------------------------------------
async def test_devices_pass_through_preview_and_live_urls(client):
    await login(client)
    await client.hub.on_status(
        status(streams=[{"stream_id": STREAM, "state": "running", "fps": 14.2,
                         "decode": "sw", "fallback_active": True,
                         "preview_url": "http://cam/preview.jpg",
                         "live_url": "http://cam/live"}])
    )
    devices = (await (await client.get("/api/devices")).json())["devices"]
    assert devices[0]["online"] is True
    assert devices[0]["fallback_active"] is True
    assert devices[0]["streams"][0]["preview_url"] == "http://cam/preview.jpg"
    assert devices[0]["streams"][0]["live_url"] == "http://cam/live"


async def test_lwt_marks_the_device_offline(client):
    await login(client)
    hub = client.hub
    await hub.on_status(status())
    await hub.on_status(status(online=False))
    devices = (await (await client.get("/api/devices")).json())["devices"]
    assert devices[0]["online"] is False
    # The offline instant is the hub's receipt time, not the LWT timestamp.
    assert devices[0]["last_seen_ms"] == hub.clock.wall_ms()


async def test_device_config_export_and_restore(client):
    await login(client)
    await client.put(f"/api/rules/{DEVICE}/{STREAM}", json=VALID_RULES)
    exported = await (await client.get(f"/api/devices/{DEVICE}/config")).json()
    assert exported["streams"][STREAM]["cooldown"] == 10
    restored = await client.put(f"/api/devices/{DEVICE}/config", json=exported)
    assert restored.status == 200
    assert (await restored.json())["saved"][0]["rev"] == 2
    bad = await client.put(f"/api/devices/{DEVICE}/config",
                           json={"streams": {STREAM: {"cooldown": -5}}})
    assert bad.status == 400


async def test_live_returns_the_last_detections(client):
    await login(client)
    assert (await client.get(f"/api/live/{DEVICE}/{STREAM}")).status == 404
    await client.hub.on_detections(detection(0.4, 0.4, frame_id=1))
    body = await (await client.get(f"/api/live/{DEVICE}/{STREAM}")).json()
    assert body["payload"]["frame_id"] == 1
    assert body["payload"]["detections"][0]["track_id"] == 1


# -- hub config ---------------------------------------------------------
async def test_config_get_and_put(client):
    await login(client)
    before = await (await client.get("/api/config")).json()
    assert before["config"]["retention_days"] == 30
    assert "mqtt_password" not in before["config"]
    response = await client.put("/api/config",
                                json={"retention_days": 7, "mqtt_host": "broker"})
    assert response.status == 200
    body = await response.json()
    assert body["rev"] == 1
    assert body["config"]["retention_days"] == 7
    assert body["restart_required"] == ["mqtt_host"]
    assert client.hub.config["retention_days"] == 7


async def test_config_rejects_unknown_keys(client):
    await login(client)
    response = await client.put("/api/config", json={"nope": 1})
    assert response.status == 400


async def test_env_override_wins_over_stored_config(tmp_path, clock):
    hub = Hub(tmp_path / "data", clock=clock, env={"MQTT_HOST": "from-env"})
    try:
        hub.storage.put_hub_config({"mqtt_host": "from-file"}, clock.wall_ms())
        reloaded = Hub(tmp_path / "data", clock=clock, env={"MQTT_HOST": "from-env"})
        assert reloaded.config["mqtt_host"] == "from-env"
        reloaded.storage.close()
        plain = Hub(tmp_path / "data", clock=clock, env={})
        assert plain.config["mqtt_host"] == "from-file"
        plain.storage.close()
    finally:
        hub.storage.close()


# -- WS -----------------------------------------------------------------
async def test_ws_requires_a_session(client):
    with pytest.raises(Exception):
        await client.ws_connect("/ws")


async def test_ws_receives_alert_new_then_alert_update(client):
    await login(client)
    hub = client.hub
    hub.storage.put_rules(DEVICE, STREAM, rules_body(lines=[]), hub.clock.wall_ms())
    ws = await client.ws_connect("/ws")
    try:
        await hub.on_detections(detection(0.75, 0.55, frame_id=1))
        first = json.loads(await ws.receive_str())
        assert first["type"] == "alert.new"
        assert first["alert"]["snapshot_state"] == "pending"
        await hub.alerts.attach_snapshot(first["alert"]["event_id"], b"\xff\xd8\xff\xd9")
        second = json.loads(await ws.receive_str())
        assert second["type"] == "alert.update"
        assert second["alert"]["snapshot_state"] == "received"
        assert second["alert"]["snapshot_url"].endswith("/snapshot.jpg")
    finally:
        await ws.close()


async def test_ws_receives_device_status_and_answers_ping(client):
    await login(client)
    ws = await client.ws_connect("/ws")
    try:
        await client.hub.on_status(status())
        message = json.loads(await ws.receive_str())
        assert message["type"] == "device.status"
        assert message["device"]["device_id"] == DEVICE
        await ws.send_str("ping")
        assert await ws.receive_str() == "pong"
    finally:
        await ws.close()


# -- static hosting -----------------------------------------------------
async def test_placeholder_page_when_web_dist_is_absent(client):
    response = await client.get("/")
    assert response.status == 200
    assert "frontend bundle is not installed" in await response.text()


async def test_built_frontend_is_served_when_present(tmp_path, clock):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<h1>hub ui</h1>")
    (dist / "app.js").write_text("console.log(1)")
    hub = Hub(tmp_path / "data", clock=clock, env={}, web_dir=dist)
    hub.auth.ensure_default_account(password="admin")
    server = TestServer(hub.api.build())
    test_client = TestClient(server)
    await test_client.start_server()
    try:
        assert "hub ui" in await (await test_client.get("/")).text()
        assert "console.log" in await (await test_client.get("/app.js")).text()
        # SPA fallback: a client-side route resolves to index.html.
        assert "hub ui" in await (await test_client.get("/devices")).text()
        # A traversal attempt cannot escape the dist directory: it falls back to
        # index.html rather than serving the file next to it.
        (tmp_path / "secret.txt").write_text("do-not-serve")
        escaped = await (await test_client.get("/../secret.txt")).text()
        assert "do-not-serve" not in escaped
    finally:
        await test_client.close()
        await hub.shutdown()


# -- single-box passthrough --------------------------------------------
async def test_single_box_events_are_trusted_without_hub_rules(client):
    await login(client)
    hub = client.hub
    await hub.on_event(device_event(event_type="loitering", event_id="rc-1"))
    rows = hub.storage.query_alerts()
    assert [r["event_type"] for r in rows] == ["loitering"]
    assert rows[0]["snapshot_state"] == "none"
    devices = (await (await client.get("/api/devices")).json())["devices"]
    assert devices[0]["mode"] == "single_box"


async def test_hub_does_not_ingest_its_own_republished_events(client):
    """Regression: the hub republishes verdicts on events/<stream> and is
    subscribed there. Ingesting the echo used to reclassify the device as
    single-box, after which its detections were no longer judged at all."""
    await login(client)
    hub = client.hub
    hub.storage.put_rules(
        DEVICE, STREAM, rules_body(lines=[], cooldown=0), hub.clock.wall_ms()
    )
    await hub.on_detections(detection(0.75, 0.55, frame_id=1))
    echo = json.loads(
        next(p for t, p, _ in _published(hub) if "/events/" in t)
    )
    assert echo["origin"] == "hub"
    await hub.on_event(echo)
    assert len(hub.storage.query_alerts()) == 1
    assert hub.registry.mode(DEVICE) == "hub"
    # Rule evaluation still works for the next frame.
    await hub.on_detections(detection(0.2, 0.55, frame_id=2))
    await hub.on_detections(detection(0.75, 0.55, frame_id=3))
    assert len(hub.storage.query_alerts()) == 2


async def test_echo_without_the_marker_is_still_deduplicated(client):
    await login(client)
    hub = client.hub
    hub.storage.put_rules(DEVICE, STREAM, rules_body(lines=[]), hub.clock.wall_ms())
    await hub.on_detections(detection(0.75, 0.55, frame_id=1))
    echo = json.loads(next(p for t, p, _ in _published(hub) if "/events/" in t))
    echo.pop("origin")
    await hub.on_event(echo)
    assert len(hub.storage.query_alerts()) == 1
    assert hub.registry.mode(DEVICE) == "hub"


def _published(hub) -> list[tuple[str, bytes, int]]:
    return getattr(hub, "_captured_publishes", [])


async def test_single_box_device_detections_are_not_double_judged(client):
    await login(client)
    hub = client.hub
    hub.storage.put_rules(DEVICE, STREAM, rules_body(lines=[]), hub.clock.wall_ms())
    await hub.on_status(status(mode="single_box"))
    await hub.on_detections(detection(0.75, 0.55, frame_id=1))
    assert hub.storage.query_alerts() == []
