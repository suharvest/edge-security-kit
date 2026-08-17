#!/usr/bin/env python3
"""Observe the WS reconnect catch-up against a running hub (FRONTEND_SPEC §7).

`hub/tests/test_ws_catchup.py` pins the behaviour in isolation. This script does
the same thing to a live stack, because the browser acceptance run could not show
the path executing: a socket that never drops never exercises it, and forcing a
drop by hand rarely lines up with an alert being raised.

It mimics what `web/dist/js/ws.js` does, in order:

  1. open `/ws`, track the highest alert id rendered (WS pushes + the REST head)
  2. close the socket and wait -- the detectors keep firing, the client is deaf
  3. reopen `/ws`, then `GET /alerts?after_id=<last seen>&limit=500` (catchUp())
  4. assert the returned rows are exactly the gap: nothing already seen, ascending
     by id, and whole rows rather than id stubs

Exit code 0 means the gap was filled.

    uv run python tools/rtsp-fixture/verify_ws_catchup.py --gap 75

The gap must be long enough for the stack to raise at least one alert; with the
truth fixture 60-90 s is comfortable. An empty gap is reported as a failure --
"nothing came back" and "nothing happened" look identical from here, so the run
is not evidence unless alerts actually fired.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

DEFAULT_HUB = os.environ.get("ESK_HUB_URL", "http://127.0.0.1:18080")


def read_password(args: argparse.Namespace) -> str:
    if args.password:
        return args.password
    if args.password_file:
        return Path(args.password_file).read_text(encoding="utf-8").strip()
    env = os.environ.get("HUB_ADMIN_PASSWORD")
    if env:
        return env
    raise SystemExit(
        "no password: pass --password-file, --password, or set HUB_ADMIN_PASSWORD"
    )


async def run(args: argparse.Namespace) -> int:
    password = read_password(args)
    # aiohttp's cookie jar drops cookies set by a bare-IP host unless it is
    # unsafe; the hub is normally reached as 127.0.0.1 or a LAN address.
    jar = aiohttp.CookieJar(unsafe=True)
    async with aiohttp.ClientSession(cookie_jar=jar) as session:
        login = await session.post(
            f"{args.hub}/api/auth/login",
            json={"username": args.username, "password": password},
        )
        if login.status != 200:
            print(f"login failed: HTTP {login.status} {await login.text()}")
            return 2

        head = await (await session.get(f"{args.hub}/api/alerts?limit=1")).json()
        last_seen = head["alerts"][0]["id"] if head["alerts"] else 0
        print(f"[t0] highest alert id already rendered = {last_seen}")

        ws = await session.ws_connect(f"{args.hub}/ws")
        print("[t0] WS open")
        live: list[int] = []
        try:
            async with asyncio.timeout(args.listen):
                async for message in ws:
                    payload = json.loads(message.data)
                    if payload.get("type") == "alert.new":
                        live.append(payload["alert"]["id"])
                        last_seen = max(last_seen, payload["alert"]["id"])
        except TimeoutError:
            pass
        await ws.close()
        print(f"[t0+{args.listen:g}s] alert.new while connected: {live or 'none'}")
        print(
            f"[t0+{args.listen:g}s] WS CLOSED, last_seen={last_seen}; "
            f"waiting {args.gap:g}s with the hub still running"
        )

        started = time.monotonic()
        await asyncio.sleep(args.gap)
        print(f"[t0+{args.listen + time.monotonic() - started:.0f}s] reconnecting")

        reconnected = await session.ws_connect(f"{args.hub}/ws")
        response = await session.get(
            f"{args.hub}/api/alerts?after_id={last_seen}&limit=500"
        )
        body = await response.json()
        await reconnected.close()

        recovered = [alert["id"] for alert in body["alerts"]]
        print(
            f"[reconnect] GET /api/alerts?after_id={last_seen}&limit=500 -> "
            f"HTTP {response.status}, {len(recovered)} row(s)"
        )
        print(f"[reconnect] recovered ids: {recovered}")

        problems: list[str] = []
        if response.status != 200:
            problems.append(f"catch-up request returned HTTP {response.status}")
        if not recovered:
            problems.append(
                "nothing recovered -- no alert fired during the gap (lengthen "
                "--gap) or after_id is broken"
            )
        if any(alert_id <= last_seen for alert_id in recovered):
            problems.append("recovered an id the client had already rendered")
        if recovered != sorted(recovered):
            problems.append("rows are not ascending by id")
        required = ("device_id", "stream_id", "event_type", "rule_name", "state", "ts_ms")
        for alert in body["alerts"]:
            missing = [key for key in required if key not in alert]
            if missing:
                problems.append(f"alert {alert.get('id')} is a stub, missing {missing}")

        for problem in problems:
            print(f"FAIL: {problem}")
        print("RESULT:", "FAIL" if problems else "PASS -- the gap was filled by after_id")
        return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", default=DEFAULT_HUB)
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password")
    parser.add_argument("--password-file")
    parser.add_argument("--listen", type=float, default=8.0,
                        help="seconds to stay connected before dropping the socket")
    parser.add_argument("--gap", type=float, default=75.0,
                        help="seconds to stay disconnected")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
