"""Runtime control downlink and its REST surface (contracts/MQTT.md).

The property under test throughout is that the console is never told a change
is live when it is not. Every path that cannot prove the detector applied the
command has to come out as something other than 200.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer
from conftest import DEVICE, STREAM, status

from edge_hub.app import Hub
from edge_hub.control import ControlPlane, ControlRejected, ControlTimeout


def ack_for(command_payload: dict, ok: bool = True, **extra) -> dict:
    payload = {
        "schema": "sensecraft.ack/1",
        "timestamp": 1_755_400_000_100,
        "session_id": "sess-1",
        "device_id": command_payload["device_id"],
        "request_id": command_payload["request_id"],
        "command": command_payload["command"],
        "ok": ok,
    }
    payload.update(extra)
    return payload


class Bus:
    """A detector that answers on the wire, with a scripted verdict."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict, int]] = []
        self.control: ControlPlane | None = None
        self.reply = True          # True | False | None (stay silent)
        self.error = "no"
        self.applied: dict | None = None
        self.delay_s = 0.0

    async def publish(self, topic: str, payload: bytes, qos: int = 1) -> None:
        message = json.loads(payload.decode("utf-8"))
        self.published.append((topic, message, qos))
        if self.reply is None or self.control is None:
            return

        async def answer() -> None:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            extra: dict = {}
            if self.reply is False:
                extra["error"] = self.error
            if self.applied is not None:
                extra["applied"] = self.applied
            await self.control.on_ack(ack_for(message, ok=bool(self.reply), **extra))

        asyncio.get_running_loop().create_task(answer())


# -- the plane itself --------------------------------------------------------

async def test_send_returns_what_the_detector_actually_applied():
    bus = Bus()
    plane = ControlPlane(bus.publish)
    bus.control = plane
    bus.applied = {"stream_id": STREAM, "conf_threshold": 0.4, "persisted": True}

    applied = await plane.send(
        DEVICE, "set_conf_threshold", {"stream_id": STREAM, "conf_threshold": 0.42}
    )

    # The detector clamped 0.42 to 0.4. The caller is told 0.4, not its request:
    # a slider that snapped back to what was asked would show a value the site
    # is not running at.
    assert applied["conf_threshold"] == 0.4
    topic, message, qos = bus.published[0]
    assert topic == "sensecraft/security/jetson-01/cmd/control"
    assert qos == 1
    assert message["schema"] == "sensecraft.command/1"
    assert message["request_id"].startswith("hub-")


async def test_silence_is_a_timeout_not_a_success():
    bus = Bus()
    bus.reply = None
    plane = ControlPlane(bus.publish, timeout_s=0.05)
    bus.control = plane
    with pytest.raises(ControlTimeout):
        await plane.send(DEVICE, "remove_stream", {"stream_id": STREAM})
    assert plane.timeouts == 1
    assert plane.acked == 0


async def test_rejection_carries_the_device_reason():
    bus = Bus()
    bus.reply = False
    bus.error = "single-stream runtime: add_stream is not supported"
    plane = ControlPlane(bus.publish)
    bus.control = plane
    with pytest.raises(ControlRejected) as caught:
        await plane.send(DEVICE, "add_stream", {"stream_id": "cam-9", "source": "rtsp://x"})
    assert "single-stream" in str(caught.value)


async def test_each_command_gets_its_own_request_id():
    bus = Bus()
    plane = ControlPlane(bus.publish)
    bus.control = plane
    await plane.send(DEVICE, "remove_stream", {"stream_id": "a"})
    await plane.send(DEVICE, "remove_stream", {"stream_id": "b"})
    ids = {message["request_id"] for _, message, _ in bus.published}
    assert len(ids) == 2


async def test_a_late_ack_lands_on_nobody_rather_than_the_next_request():
    """A timed-out request must not be settled by an ack that arrives later.

    Without the check, the next command's wait could be resolved by the
    previous command's answer and the console would report the wrong outcome.
    """
    bus = Bus()
    bus.reply = None
    plane = ControlPlane(bus.publish, timeout_s=0.05)
    bus.control = plane
    with pytest.raises(ControlTimeout):
        await plane.send(DEVICE, "remove_stream", {"stream_id": "a"})
    _, first, _ = bus.published[0]
    await plane.on_ack(ack_for(first))
    assert plane.orphan_acks == 1


# -- the REST surface --------------------------------------------------------

@pytest.fixture
async def client(tmp_path, clock):
    hub = Hub(tmp_path / "data", clock=clock, env={}, web_dir=None)
    hub.auth.ensure_default_account(password="admin")
    bus = Bus()
    bus.control = hub.control
    hub.control.publisher = bus.publish
    hub.control.timeout_s = 0.1
    server = TestServer(hub.api.build())
    test_client = TestClient(server)
    await test_client.start_server()
    test_client.hub = hub
    test_client.bus = bus
    await test_client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    await hub.registry.on_status(status())
    try:
        yield test_client
    finally:
        await test_client.close()
        await hub.shutdown()


async def test_put_conf_threshold_reports_the_applied_value(client):
    client.bus.applied = {"stream_id": STREAM, "conf_threshold": 0.42, "persisted": True}
    response = await client.put(
        f"/api/devices/{DEVICE}/streams/{STREAM}/conf", json={"conf_threshold": 0.42}
    )
    assert response.status == 200
    body = await response.json()
    assert body == {"ok": True, "applied": {"stream_id": STREAM, "conf_threshold": 0.42,
                                            "persisted": True}}
    _, message, _ = client.bus.published[0]
    assert message["command"] == "set_conf_threshold"
    assert message["params"] == {"stream_id": STREAM, "conf_threshold": 0.42}


@pytest.mark.parametrize("value", [-0.1, 1.5, "0.4", None, True])
async def test_conf_threshold_outside_the_unit_interval_never_reaches_the_bus(client, value):
    response = await client.put(
        f"/api/devices/{DEVICE}/streams/{STREAM}/conf", json={"conf_threshold": value}
    )
    assert response.status == 400
    assert client.bus.published == []


async def test_a_silent_device_answers_504_and_is_audited_as_such(client):
    client.bus.reply = None
    response = await client.put(
        f"/api/devices/{DEVICE}/streams/{STREAM}/conf", json={"conf_threshold": 0.5}
    )
    assert response.status == 504
    rows = client.hub.storage.query_audit()
    assert [r["action"] for r in rows] == ["set_conf_threshold"]
    assert rows[0]["outcome"] == "timeout"
    assert rows[0]["actor"] == "admin"


async def test_a_refusal_answers_409_with_the_device_reason(client):
    client.bus.reply = False
    client.bus.error = "conf_threshold below the model's calibration floor"
    response = await client.put(
        f"/api/devices/{DEVICE}/streams/{STREAM}/conf", json={"conf_threshold": 0.01}
    )
    assert response.status == 409
    body = await response.json()
    assert "calibration floor" in body["error"]
    assert client.hub.storage.query_audit()[0]["outcome"] == "rejected"


async def test_add_stream_returns_201_and_records_who_added_it(client):
    client.bus.applied = {"stream_id": "cam-2", "source": "rtsp://cam2/live"}
    response = await client.post(
        f"/api/devices/{DEVICE}/streams",
        json={"stream_id": "cam-2", "source": "rtsp://cam2/live", "name": "North gate"},
    )
    assert response.status == 201, await response.text()
    _, message, _ = client.bus.published[0]
    assert message["command"] == "add_stream"
    assert message["params"]["name"] == "North gate"
    row = client.hub.storage.query_audit()[0]
    assert (row["action"], row["outcome"], row["stream_id"]) == ("add_stream", "ok", "cam-2")


async def test_add_stream_refuses_an_id_the_device_already_publishes(client):
    """Two capture loops on one stream_id look like packet loss, not a typo."""
    response = await client.post(
        f"/api/devices/{DEVICE}/streams", json={"stream_id": STREAM, "source": "rtsp://x"}
    )
    assert response.status == 409
    assert client.bus.published == []


@pytest.mark.parametrize("body", [
    {"source": "rtsp://x"},
    {"stream_id": "cam-2"},
    {"stream_id": "cam-2", "source": "rtsp://x", "rtsp_transport": "sctp"},
])
async def test_add_stream_validates_before_publishing(client, body):
    response = await client.post(f"/api/devices/{DEVICE}/streams", json=body)
    assert response.status == 400
    assert client.bus.published == []


async def test_add_stream_to_an_unknown_device_is_404(client):
    response = await client.post(
        "/api/devices/nope/streams", json={"stream_id": "cam-2", "source": "rtsp://x"}
    )
    assert response.status == 404


async def test_delete_stream_sends_remove_stream(client):
    response = await client.delete(f"/api/devices/{DEVICE}/streams/{STREAM}")
    assert response.status == 200
    _, message, _ = client.bus.published[0]
    assert message["command"] == "remove_stream"
    assert message["params"] == {"stream_id": STREAM}


async def test_audit_is_readable_and_newest_first(client):
    await client.put(f"/api/devices/{DEVICE}/streams/{STREAM}/conf",
                     json={"conf_threshold": 0.3})
    await client.put(f"/api/devices/{DEVICE}/streams/{STREAM}/conf",
                     json={"conf_threshold": 0.6})
    response = await client.get("/api/audit?limit=10")
    body = await response.json()
    assert body["count"] == 2
    assert body["audit"][0]["detail"]["params"]["conf_threshold"] == 0.6
    assert body["audit"][1]["detail"]["params"]["conf_threshold"] == 0.3


async def test_control_endpoints_need_a_session(tmp_path, clock):
    hub = Hub(tmp_path / "data", clock=clock, env={}, web_dir=None)
    hub.auth.ensure_default_account(password="admin")
    server = TestServer(hub.api.build())
    anon = TestClient(server)
    await anon.start_server()
    try:
        for method, path in (
            ("put", f"/api/devices/{DEVICE}/streams/{STREAM}/conf"),
            ("post", f"/api/devices/{DEVICE}/streams"),
            ("delete", f"/api/devices/{DEVICE}/streams/{STREAM}"),
            ("get", "/api/audit"),
        ):
            response = await getattr(anon, method)(path, json={} if method != "get" else None)
            assert response.status == 401, path
    finally:
        await anon.close()
        await hub.shutdown()
