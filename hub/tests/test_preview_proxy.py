"""Single-frame preview proxy (HUB_SPEC §4).

`status.streams[].preview_url` is a device-local address. A browser on any other
host cannot fetch it, which left the rule canvas with a grey backdrop on every
real deployment. The hub fetches one JPEG server-side, caches it for a moment and
serves it same-origin. One still frame per request — not a video stream.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from conftest import DEVICE, STREAM, status

from edge_hub.app import Hub
from edge_hub.http_api import PREVIEW_CACHE_MS

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


class FakeDevice:
    """A stand-in for the detector's preview HTTP server."""

    def __init__(self) -> None:
        self.hits = 0
        self.status = 200
        self.body = JPEG
        self.delay_s = 0.0
        self.content_type = "image/jpeg"

    async def handler(self, request: web.Request) -> web.StreamResponse:
        self.hits += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.status != 200:
            return web.Response(status=self.status, text="nope")
        return web.Response(body=self.body, content_type=self.content_type)


@pytest.fixture
async def device():
    fake = FakeDevice()
    app = web.Application()
    app.router.add_get("/preview/{tail}", fake.handler)
    server = TestServer(app)
    await server.start_server()
    fake.url = str(server.make_url(f"/preview/{STREAM}.jpg"))
    try:
        yield fake
    finally:
        await server.close()


@pytest.fixture
async def client(tmp_path, clock):
    hub = Hub(tmp_path / "data", clock=clock, env={}, web_dir=None)
    hub.auth.ensure_default_account(password="admin")
    test_client = TestClient(TestServer(hub.api.build()))
    await test_client.start_server()
    test_client.hub = hub
    await test_client.post(
        "/api/auth/login", json={"username": "admin", "password": "admin"}
    )
    try:
        yield test_client
    finally:
        await test_client.close()
        await hub.shutdown()


def preview_path(device_id: str = DEVICE, stream_id: str = STREAM) -> str:
    return f"/api/devices/{device_id}/streams/{stream_id}/preview.jpg"


async def announce(hub, device, **stream_overrides) -> None:
    stream = {
        "stream_id": STREAM,
        "state": "running",
        "fps": 15.0,
        "decode": "hw",
        "preview_url": device.url,
    }
    stream.update(stream_overrides)
    await hub.on_status(status(streams=[stream]))


async def test_needs_a_session(client):
    client.session.cookie_jar.clear()
    assert (await client.get(preview_path())).status == 401


async def test_returns_the_device_jpeg(client, device):
    await announce(client.hub, device)
    response = await client.get(preview_path())
    assert response.status == 200, await response.text()
    assert response.headers["Content-Type"] == "image/jpeg"
    body = await response.read()
    assert body == JPEG
    assert body[:2] == b"\xff\xd8"
    assert device.hits == 1


async def test_the_cache_collapses_concurrent_operators_to_one_device_fetch(
    client, device, clock
):
    await announce(client.hub, device)
    assert (await client.get(preview_path())).status == 200
    assert (await client.get(preview_path())).status == 200
    assert device.hits == 1, "second request inside the window must not touch the device"

    clock.advance(PREVIEW_CACHE_MS + 1)
    assert (await client.get(preview_path())).status == 200
    assert device.hits == 2


async def test_unknown_device_and_stream_are_404(client, device):
    await announce(client.hub, device)
    assert (await client.get(preview_path(device_id="nope"))).status == 404
    assert (await client.get(preview_path(stream_id="nope"))).status == 404


async def test_a_stream_without_preview_url_is_404(client, device):
    await client.hub.on_status(
        status(streams=[{"stream_id": STREAM, "state": "running", "decode": "hw"}])
    )
    response = await client.get(preview_path())
    assert response.status == 404
    assert "preview_url" in (await response.json())["error"]


async def test_an_unreachable_device_is_502_with_a_reason(client):
    # Port 1 on loopback: nothing listens, the connection is refused immediately.
    await client.hub.on_status(
        status(
            streams=[
                {
                    "stream_id": STREAM,
                    "state": "running",
                    "decode": "hw",
                    "preview_url": "http://127.0.0.1:1/preview/cam-0.jpg",
                }
            ]
        )
    )
    response = await client.get(preview_path())
    assert response.status == 502
    body = await response.json()
    assert "unreachable" in body["error"]
    assert body["preview_url"].startswith("http://127.0.0.1:1/")


async def test_a_device_error_status_is_502(client, device):
    device.status = 503
    await announce(client.hub, device)
    response = await client.get(preview_path())
    assert response.status == 502
    assert "503" in (await response.json())["error"]


async def test_a_non_jpeg_body_is_502_rather_than_a_broken_image(client, device):
    device.body = b"<html>login page</html>"
    device.content_type = "text/html"
    await announce(client.hub, device)
    response = await client.get(preview_path())
    assert response.status == 502
    assert "JPEG" in (await response.json())["error"]


async def test_a_failure_does_not_poison_the_cache(client, device):
    device.status = 500
    await announce(client.hub, device)
    assert (await client.get(preview_path())).status == 502
    device.status = 200
    response = await client.get(preview_path())
    assert response.status == 200
    assert await response.read() == JPEG
