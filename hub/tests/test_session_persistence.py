"""Sessions survive a hub restart (HUB_SPEC §7).

The hub is restarted for upgrades and after crashes. Sessions that lived only in
process memory logged every operator out on each restart; these tests pin the
persisted behaviour and the idle deadline that still has to expire them.
"""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer

from edge_hub.app import Hub
from edge_hub.auth import COOKIE_NAME

IDLE_MS = 7 * 86_400_000


async def serve(data_dir, clock):
    hub = Hub(data_dir, clock=clock, env={}, web_dir=None)
    hub.auth.ensure_default_account(password="admin")
    client = TestClient(TestServer(hub.api.build()))
    await client.start_server()
    client.hub = hub
    return client


async def shutdown(client) -> None:
    await client.close()
    await client.hub.shutdown()


async def test_a_session_survives_a_restart(tmp_path, clock):
    data_dir = tmp_path / "data"
    first = await serve(data_dir, clock)
    try:
        response = await first.post(
            "/api/auth/login", json={"username": "admin", "password": "admin"}
        )
        assert response.status == 200
        token = response.cookies[COOKIE_NAME].value
        assert (await first.get("/api/devices")).status == 200
    finally:
        await shutdown(first)

    # A brand-new process against the same data directory: the cookie the browser
    # still holds must keep working.
    second = await serve(data_dir, clock)
    try:
        response = await second.get("/api/devices", cookies={COOKIE_NAME: token})
        assert response.status == 200, await response.text()
        who = await (
            await second.get("/api/auth/session", cookies={COOKIE_NAME: token})
        ).json()
        assert who["username"] == "admin"
    finally:
        await shutdown(second)


async def test_an_idle_expired_cookie_is_rejected_after_a_restart(tmp_path, clock):
    data_dir = tmp_path / "data"
    first = await serve(data_dir, clock)
    try:
        response = await first.post(
            "/api/auth/login", json={"username": "admin", "password": "admin"}
        )
        token = response.cookies[COOKIE_NAME].value
    finally:
        await shutdown(first)

    clock.advance(IDLE_MS + 1000)
    second = await serve(data_dir, clock)
    try:
        response = await second.get("/api/devices", cookies={COOKIE_NAME: token})
        assert response.status == 401
        # The start-up sweep drops the row rather than leaving it to rot.
        assert second.hub.storage.count_sessions() == 0
    finally:
        await shutdown(second)


async def test_logout_and_password_change_still_invalidate_across_a_restart(
    tmp_path, clock
):
    data_dir = tmp_path / "data"
    client = await serve(data_dir, clock)
    try:
        keep = await client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin"}
        )
        keep_token = keep.cookies[COOKIE_NAME].value
        other = await client.post(
            "/api/auth/login", json={"username": "admin", "password": "admin"}
        )
        other_token = other.cookies[COOKIE_NAME].value
        assert keep_token != other_token

        changed = await client.post(
            "/api/auth/password",
            json={"old_password": "admin", "new_password": "another-secret"},
            cookies={COOKIE_NAME: keep_token},
        )
        assert changed.status == 200
    finally:
        await shutdown(client)

    restarted = await serve(data_dir, clock)
    try:
        # §4: rotating the password kills every session except the one that did it,
        # and a restart must not resurrect the killed ones.
        assert (
            await restarted.get("/api/devices", cookies={COOKIE_NAME: other_token})
        ).status == 401
        assert (
            await restarted.get("/api/devices", cookies={COOKIE_NAME: keep_token})
        ).status == 200

        out = await restarted.post(
            "/api/auth/logout", cookies={COOKIE_NAME: keep_token}
        )
        assert out.status == 200
    finally:
        await shutdown(restarted)

    final = await serve(data_dir, clock)
    try:
        assert (
            await final.get("/api/devices", cookies={COOKIE_NAME: keep_token})
        ).status == 401
    finally:
        await shutdown(final)
