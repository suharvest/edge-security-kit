"""REST + WS + static hosting (HUB_SPEC §4, §5, §7).

Everything under ``/api`` requires the session cookie except ``/api/health`` and
``/api/auth/login``. ``/ws`` refuses the upgrade without a valid session. The
built frontend is served from ``web/dist``; while that directory does not exist
the hub serves a placeholder page instead of failing to start.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import WSMsgType, web

from .auth import COOKIE_NAME, Session
from .control import ControlRejected, ControlTimeout
from .rules_schema import RulesError, find_rule, validate_rules_body

log = logging.getLogger("edge_hub.http")

MAX_BATCH = 500
#: §4 single-frame preview proxy. A device preview_url is a device-LOCAL address
#: (http://127.0.0.1:8099/...), so a browser on another host can never fetch it.
#: The hub fetches one JPEG server-side and caches it briefly. This is a still
#: frame on demand, not a video stream: no continuous pull, no fan-out, no
#: transcode, and the cache collapses N operators to one device request per
#: window. See HUB_SPEC §4 and the "not doing" list.
PREVIEW_TIMEOUT_S = 2.0
PREVIEW_CACHE_MS = 1500
PREVIEW_MAX_BYTES = 4 * 1024 * 1024
#: typed request key holding the authenticated session
SESSION_KEY: "web.RequestKey[Session]" = web.RequestKey("session")
PLACEHOLDER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>edge-security hub</title>
<style>body{font:16px/1.6 system-ui,sans-serif;margin:4rem auto;max-width:34rem;padding:0 1rem}
code{background:#eee;padding:.1em .35em;border-radius:3px}</style></head>
<body><h1>edge-security hub</h1>
<p>The API is up. The frontend bundle is not installed: no <code>web/dist/index.html</code>
was found.</p>
<p>Build it and mount it there, or check <code>GET /api/health</code> in the meantime.</p>
</body></html>
"""

#: ``ts_ms`` rather than ``ts``: one name for the device timestamp everywhere,
#: matching the §6 column and the WS payload.
EXPORT_COLUMNS = [
    "id", "ts_ms", "device_id", "stream_id", "event_type", "rule_name",
    "track_id", "score", "state", "acted_by", "acted_at",
]


def _int(value: str | None, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise web.HTTPBadRequest(text=json.dumps({"error": f"not an integer: {value}"}),
                                 content_type="application/json") from None


def alert_filters(query: Any) -> dict[str, Any]:
    """Shared filter parsing for /alerts and /alerts/export.csv (HUB_SPEC §4).

    Both endpoints parse the same query string through this one function, so the
    CSV export can never disagree with the list it was exported from.
    ``rule_name`` is part of that set: the workbench filters by rule, and a
    server that ignored the parameter would hand back a CSV covering every rule.
    """
    limit = _int(query.get("limit"), 50) or 50
    return {
        "state": query.get("state") or None,
        "device_id": query.get("device_id") or None,
        "stream_id": query.get("stream_id") or None,
        "event_type": query.get("event_type") or None,
        "rule_name": query.get("rule_name") or None,
        "date_from": _int(query.get("date_from")),
        "date_to": _int(query.get("date_to")),
        "after_id": _int(query.get("after_id")),
        "limit": max(1, min(limit, 1000)),
        "offset": max(0, _int(query.get("offset"), 0) or 0),
    }


def json_response(payload: Any, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status, dumps=lambda o: json.dumps(o, ensure_ascii=False))


def error(message: str, status: int) -> web.Response:
    return json_response({"error": message}, status=status)


class HttpApi:
    """Builds the aiohttp application. ``hub`` is an :class:`edge_hub.app.Hub`."""

    def __init__(self, hub: Any, web_dir: Path | None = None) -> None:
        self.hub = hub
        self.web_dir = web_dir
        self.websockets: set[web.WebSocketResponse] = set()
        #: (device_id, stream_id) -> (fetched_ms, jpeg bytes)
        self._preview_cache: dict[tuple[str, str], tuple[int, bytes]] = {}

    # -- app wiring ------------------------------------------------------
    def build(self) -> web.Application:
        app = web.Application(middlewares=[self.auth_middleware])
        api = web.Application()
        api.add_routes(
            [
                web.get("/health", self.health),
                web.post("/auth/login", self.login),
                web.post("/auth/logout", self.logout),
                web.post("/auth/password", self.change_password),
                web.get("/auth/session", self.session_info),
                web.get("/alerts", self.list_alerts),
                web.get("/alerts/export.csv", self.export_alerts),
                web.post("/alerts/ack", self.batch_ack),
                web.post("/alerts/dismiss", self.batch_dismiss),
                web.get("/alerts/stats", self.alert_stats),
                web.get("/alerts/{id}", self.get_alert),
                web.get("/alerts/{id}/snapshot.jpg", self.snapshot),
                web.post("/alerts/{id}/ack", self.ack),
                web.post("/alerts/{id}/dismiss", self.dismiss),
                web.get("/devices", self.list_devices),
                web.get("/devices/{device_id}/config", self.export_device_config),
                web.get(
                    "/devices/{device_id}/streams/{stream_id}/preview.jpg",
                    self.stream_preview,
                ),
                web.put("/devices/{device_id}/config", self.import_device_config),
                web.put(
                    "/devices/{device_id}/streams/{stream_id}/conf",
                    self.set_conf_threshold,
                ),
                web.post("/devices/{device_id}/streams", self.add_stream),
                web.delete("/devices/{device_id}/streams/{stream_id}", self.remove_stream),
                web.get("/audit", self.list_audit),
                web.get("/rules", self.all_rules),
                web.get("/rules/{device_id}/{stream_id}", self.get_rules),
                web.put("/rules/{device_id}/{stream_id}", self.put_rules),
                web.post("/rules/{device_id}/{stream_id}/simulate", self.simulate),
                web.get("/live", self.live_all),
                web.get("/live/{device_id}/{stream_id}", self.live),
                web.get("/config", self.get_config),
                web.put("/config", self.put_config),
            ]
        )
        app.add_subapp("/api", api)
        app.router.add_get("/ws", self.websocket)
        self._add_static(app)
        return app

    def _add_static(self, app: web.Application) -> None:
        app.router.add_get("/{tail:.*}", self.static_handler)

    # -- auth ------------------------------------------------------------
    @web.middleware
    async def auth_middleware(self, request: web.Request, handler: Any) -> web.StreamResponse:
        path = request.path
        open_paths = ("/api/health", "/api/auth/login")
        needs_auth = path.startswith("/api") or path == "/ws"
        if needs_auth and path not in open_paths:
            session = self.hub.auth.resolve(request.cookies.get(COOKIE_NAME))
            if session is None:
                return error("unauthenticated", 401)
            request[SESSION_KEY] = session
        return await handler(request)

    async def login(self, request: web.Request) -> web.Response:
        body = await self._json_body(request)
        session = self.hub.auth.login(
            str(body.get("username", "")), str(body.get("password", ""))
        )
        if session is None:
            return error("invalid credentials", 401)
        response = json_response(
            {
                "username": session.username,
                "must_change": self.hub.auth.must_change(session.username),
            }
        )
        response.set_cookie(
            COOKIE_NAME,
            session.token,
            httponly=True,
            samesite="Strict",
            path="/",
            max_age=int(self.hub.auth.idle_ms // 1000),
        )
        return response

    async def logout(self, request: web.Request) -> web.Response:
        self.hub.auth.logout(request.cookies.get(COOKIE_NAME))
        response = json_response({"ok": True})
        response.del_cookie(COOKIE_NAME, path="/")
        return response

    async def change_password(self, request: web.Request) -> web.Response:
        body = await self._json_body(request)
        session = request[SESSION_KEY]
        problem = self.hub.auth.change_password(
            session.username,
            str(body.get("old_password", "")),
            str(body.get("new_password", "")),
            keep_token=session.token,
        )
        if problem:
            return error(problem, 400)
        return json_response({"ok": True, "must_change": False})

    async def session_info(self, request: web.Request) -> web.Response:
        session = request[SESSION_KEY]
        return json_response(
            {
                "username": session.username,
                "must_change": self.hub.auth.must_change(session.username),
            }
        )

    # -- health ----------------------------------------------------------
    async def health(self, request: web.Request) -> web.Response:
        return json_response(self.hub.health())

    # -- alerts ----------------------------------------------------------
    async def list_alerts(self, request: web.Request) -> web.Response:
        filters = alert_filters(request.query)
        rows = self.hub.storage.query_alerts(**filters)
        return json_response({"alerts": rows, "count": len(rows), "filters": filters})

    async def get_alert(self, request: web.Request) -> web.Response:
        alert_id = _int(request.match_info["id"])
        alert = self.hub.storage.get_alert(int(alert_id or 0))
        if alert is None:
            return error("alert not found", 404)
        return json_response({"alert": alert})

    async def alert_stats(self, request: web.Request) -> web.Response:
        return json_response({"by_rule": self.hub.storage.dismissed_by_rule()})

    async def export_alerts(self, request: web.Request) -> web.Response:
        filters = alert_filters(request.query)
        filters["limit"] = 100_000  # export ignores the page limit
        rows = self.hub.storage.query_alerts(**filters)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(EXPORT_COLUMNS)
        for row in rows:
            writer.writerow([row.get(column) for column in EXPORT_COLUMNS])
        # UTF-8 BOM so Excel reads the CN columns correctly (HUB_SPEC §4).
        payload = "﻿" + buffer.getvalue()
        return web.Response(
            body=payload.encode("utf-8"),
            content_type="text/csv",
            charset="utf-8",
            headers={"Content-Disposition": 'attachment; filename="alerts.csv"'},
        )

    async def snapshot(self, request: web.Request) -> web.StreamResponse:
        alert_id = int(_int(request.match_info["id"]) or 0)
        path = self.hub.storage.alert_snapshot_path(alert_id)
        if not path or not Path(path).is_file():
            return error("snapshot not found", 404)
        return web.FileResponse(path, headers={"Content-Type": "image/jpeg"})

    async def ack(self, request: web.Request) -> web.Response:
        return await self._single_transition(request, "acked")

    async def dismiss(self, request: web.Request) -> web.Response:
        return await self._single_transition(request, "dismissed")

    async def _single_transition(self, request: web.Request, target: str) -> web.Response:
        alert_id = int(_int(request.match_info["id"]) or 0)
        result, alert = await self.hub.alerts.transition(
            alert_id, target, request[SESSION_KEY].username
        )
        if result == "not_found":
            return error("alert not found", 404)
        if result == "conflict":
            return json_response(
                {
                    "error": f"illegal transition to {target}",
                    "state": (alert or {}).get("state"),
                },
                status=409,
            )
        return json_response({"alert": alert})

    async def batch_ack(self, request: web.Request) -> web.Response:
        return await self._batch(request, "acked")

    async def batch_dismiss(self, request: web.Request) -> web.Response:
        return await self._batch(request, "dismissed")

    async def _batch(self, request: web.Request, target: str) -> web.Response:
        body = await self._json_body(request)
        ids = body.get("ids")
        if not isinstance(ids, list):
            return error("ids must be an array", 400)
        if len(ids) > MAX_BATCH:
            return error(f"batch limit is {MAX_BATCH}", 400)
        results = []
        for raw in ids:
            try:
                alert_id = int(raw)
            except (TypeError, ValueError):
                results.append({"id": raw, "status": 400, "error": "not an integer"})
                continue
            result, alert = await self.hub.alerts.transition(
                alert_id, target, request[SESSION_KEY].username
            )
            status = {"ok": 200, "conflict": 409, "not_found": 404}[result]
            entry: dict[str, Any] = {"id": alert_id, "status": status}
            if result == "ok":
                entry["alert"] = alert
            elif result == "conflict":
                entry["state"] = (alert or {}).get("state")
            results.append(entry)
        return json_response({"results": results})

    # -- devices ---------------------------------------------------------
    async def list_devices(self, request: web.Request) -> web.Response:
        return json_response({"devices": self.hub.registry.list_devices()})

    async def export_device_config(self, request: web.Request) -> web.Response:
        device_id = request.match_info["device_id"]
        tree = self.hub.storage.all_rules().get(device_id, {})
        device = next(
            (d for d in self.hub.registry.list_devices() if d["device_id"] == device_id),
            None,
        )
        return json_response(
            {
                "device_id": device_id,
                "streams": {sid: entry["body"] for sid, entry in tree.items()},
                "revs": {sid: entry["rev"] for sid, entry in tree.items()},
                "camera": {"streams": (device or {}).get("streams", [])},
            }
        )

    async def import_device_config(self, request: web.Request) -> web.Response:
        device_id = request.match_info["device_id"]
        body = await self._json_body(request)
        streams = body.get("streams")
        if not isinstance(streams, dict) or not streams:
            return error("streams must be a non-empty object", 400)
        validated: dict[str, Any] = {}
        for stream_id, raw in streams.items():
            try:
                validated[str(stream_id)] = validate_rules_body(raw)
            except RulesError as exc:
                return error(f"{stream_id}: {exc}", 400)
        results = self.hub.storage.put_device_rules(
            device_id, validated, self.hub.clock.wall_ms()
        )
        return json_response({"saved": results})

    async def stream_preview(self, request: web.Request) -> web.StreamResponse:
        """Proxy one still frame from the device preview endpoint (§4).

        The device advertises ``preview_url`` as a device-local address, which a
        remote browser cannot reach. The hub fetches it once per cache window and
        hands back the JPEG. A device that is down, slow or serving something that
        is not an image yields 502 with the reason, never a silent grey canvas.
        """
        device_id = request.match_info["device_id"]
        stream_id = request.match_info["stream_id"]
        key = (device_id, stream_id)
        now = self.hub.clock.wall_ms()

        cached = self._preview_cache.get(key)
        if cached is not None and now - cached[0] < PREVIEW_CACHE_MS:
            return self._jpeg_response(cached[1], age_ms=now - cached[0])

        device = next(
            (d for d in self.hub.registry.list_devices() if d["device_id"] == device_id),
            None,
        )
        if device is None:
            return error("device not found", 404)
        stream = next(
            (
                s
                for s in (device.get("streams") or [])
                if str(s.get("stream_id")) == stream_id
            ),
            None,
        )
        if stream is None:
            return error("stream not found", 404)
        preview_url = stream.get("preview_url")
        if not preview_url:
            return error("stream reports no preview_url", 404)

        body = await self._fetch_preview(str(preview_url))
        if isinstance(body, str):
            self._preview_cache.pop(key, None)
            return json_response(
                {"error": body, "preview_url": preview_url}, status=502
            )
        self._preview_cache[key] = (now, body)
        return self._jpeg_response(body, age_ms=0)

    @staticmethod
    async def _fetch_preview(preview_url: str) -> bytes | str:
        """Return JPEG bytes, or an error string describing why not."""
        timeout = aiohttp.ClientTimeout(total=PREVIEW_TIMEOUT_S)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(preview_url) as response:
                    if response.status != 200:
                        return f"device returned HTTP {response.status}"
                    raw = await response.content.read(PREVIEW_MAX_BYTES + 1)
        except aiohttp.ClientError as exc:
            return f"device unreachable: {type(exc).__name__}: {exc}"
        except TimeoutError:
            return f"device did not answer within {PREVIEW_TIMEOUT_S:g}s"
        if len(raw) > PREVIEW_MAX_BYTES:
            return f"preview larger than {PREVIEW_MAX_BYTES} bytes"
        if not raw.startswith(b"\xff\xd8"):
            return "device did not return a JPEG"
        return raw

    @staticmethod
    def _jpeg_response(body: bytes, age_ms: int) -> web.Response:
        return web.Response(
            body=body,
            content_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Preview-Age-Ms": str(int(age_ms)),
            },
        )

    # -- runtime control (contracts/MQTT.md "Control downlink") -----------
    async def _control(
        self,
        request: web.Request,
        action: str,
        command: str,
        params: dict[str, Any],
        device_id: str,
        stream_id: str | None,
        ok_status: int = 200,
    ) -> web.Response:
        """One command, one audit row, one HTTP answer per outcome.

        The three outcomes are kept distinct all the way to the status code
        because an operator has to act differently on each: 200 the change is
        live, 409 the device refused and said why, 504 nothing is known about
        the device state and the console must not redraw as if it succeeded.
        """
        actor = request[SESSION_KEY].username
        now = self.hub.clock.wall_ms()
        control = getattr(self.hub, "control", None)
        if control is None:
            return error("control plane unavailable", 503)
        try:
            applied = await control.send(device_id, command, params)
        except ControlTimeout as exc:
            self.hub.storage.append_audit(
                now, actor, action, "timeout", device_id, stream_id,
                {"params": params, "error": str(exc)},
            )
            return json_response({"error": str(exc), "params": params}, status=504)
        except ControlRejected as exc:
            self.hub.storage.append_audit(
                now, actor, action, "rejected", device_id, stream_id,
                {"params": params, "error": str(exc)},
            )
            return json_response(
                {"error": str(exc), "params": params, "applied": exc.applied}, status=409
            )
        self.hub.storage.append_audit(
            now, actor, action, "ok", device_id, stream_id,
            {"params": params, "applied": applied},
        )
        return json_response({"ok": True, "applied": applied}, status=ok_status)

    async def set_conf_threshold(self, request: web.Request) -> web.Response:
        device_id = request.match_info["device_id"]
        stream_id = request.match_info["stream_id"]
        body = await self._json_body(request)
        raw = body.get("conf_threshold")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return error("conf_threshold must be a number", 400)
        value = float(raw)
        if not 0.0 <= value <= 1.0:
            return error("conf_threshold must be between 0 and 1", 400)
        return await self._control(
            request,
            "set_conf_threshold",
            "set_conf_threshold",
            {"stream_id": stream_id, "conf_threshold": round(value, 4)},
            device_id,
            stream_id,
        )

    async def add_stream(self, request: web.Request) -> web.Response:
        device_id = request.match_info["device_id"]
        body = await self._json_body(request)
        stream_id = str(body.get("stream_id") or "").strip()
        source = str(body.get("source") or "").strip()
        if not stream_id:
            return error("stream_id is required", 400)
        if not source:
            return error("source is required", 400)
        # Rejected here rather than at the detector: a stream_id collision would
        # make two capture loops publish to one topic, and the resulting mixed
        # frame_id sequence looks like packet loss rather than a config mistake.
        device = next(
            (d for d in self.hub.registry.list_devices() if d["device_id"] == device_id),
            None,
        )
        if device is None:
            return error("device not found", 404)
        if any(str(s.get("stream_id")) == stream_id for s in device.get("streams") or []):
            return error(f"stream {stream_id} already exists on {device_id}", 409)
        params: dict[str, Any] = {"stream_id": stream_id, "source": source}
        name = str(body.get("name") or "").strip()
        if name:
            params["name"] = name
        transport = str(body.get("rtsp_transport") or "").strip()
        if transport:
            if transport not in ("tcp", "udp"):
                return error("rtsp_transport must be tcp or udp", 400)
            params["rtsp_transport"] = transport
        return await self._control(
            request, "add_stream", "add_stream", params, device_id, stream_id,
            ok_status=201,
        )

    async def remove_stream(self, request: web.Request) -> web.Response:
        device_id = request.match_info["device_id"]
        stream_id = request.match_info["stream_id"]
        return await self._control(
            request, "remove_stream", "remove_stream", {"stream_id": stream_id},
            device_id, stream_id,
        )

    async def list_audit(self, request: web.Request) -> web.Response:
        rows = self.hub.storage.query_audit(
            limit=_int(request.query.get("limit"), 100) or 100,
            device_id=request.query.get("device_id") or None,
        )
        return json_response({"audit": rows, "count": len(rows)})

    # -- rules -----------------------------------------------------------
    async def all_rules(self, request: web.Request) -> web.Response:
        return json_response({"rules": self.hub.storage.all_rules()})

    async def get_rules(self, request: web.Request) -> web.Response:
        entry = self.hub.storage.get_rules(
            request.match_info["device_id"], request.match_info["stream_id"]
        )
        if entry is None:
            return error("no rules for this stream", 404)
        return json_response(entry)

    async def put_rules(self, request: web.Request) -> web.Response:
        device_id = request.match_info["device_id"]
        stream_id = request.match_info["stream_id"]
        raw = await self._json_body(request)
        body = raw.get("body") if isinstance(raw.get("body"), dict) else raw
        try:
            validated = validate_rules_body(body)
        except RulesError as exc:
            return error(str(exc), 400)
        result = self.hub.storage.put_rules(
            device_id, stream_id, validated, self.hub.clock.wall_ms()
        )
        return json_response({**result, "body": validated})

    async def simulate(self, request: web.Request) -> web.Response:
        device_id = request.match_info["device_id"]
        stream_id = request.match_info["stream_id"]
        body = await self._json_body(request)
        rule_id = str(body.get("rule_id", ""))
        alert = await self.hub.simulate(device_id, stream_id, rule_id)
        if alert is None:
            return error("rule not found for this stream", 404)
        return json_response({"alert": alert})

    async def live_all(self, request: web.Request) -> web.Response:
        """Every stream's last detections in one response (video wall overlay)."""
        streams = self.hub.registry.live_all()
        return json_response(
            {"streams": streams, "count": len(streams), "now_ms": self.hub.clock.wall_ms()}
        )

    async def live(self, request: web.Request) -> web.Response:
        entry = self.hub.registry.live(
            request.match_info["device_id"], request.match_info["stream_id"]
        )
        if entry is None:
            return error("no detections seen for this stream", 404)
        return json_response(entry)

    # -- hub config ------------------------------------------------------
    async def get_config(self, request: web.Request) -> web.Response:
        return json_response(self.hub.config_view())

    async def put_config(self, request: web.Request) -> web.Response:
        body = await self._json_body(request)
        try:
            result = self.hub.update_config(body)
        except ValueError as exc:
            return error(str(exc), 400)
        return json_response(result)

    # -- WS --------------------------------------------------------------
    async def websocket(self, request: web.Request) -> web.StreamResponse:
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self.websockets.add(ws)
        try:
            async for message in ws:
                # §5: no client uplink beyond keepalive.
                if message.type == WSMsgType.TEXT and message.data == "ping":
                    await ws.send_str("pong")
                elif message.type == WSMsgType.ERROR:
                    break
        finally:
            self.websockets.discard(ws)
        return ws

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self.websockets:
            return
        data = json.dumps(message, ensure_ascii=False)
        for ws in list(self.websockets):
            try:
                await ws.send_str(data)
            except (ConnectionResetError, RuntimeError):
                self.websockets.discard(ws)

    async def close_websockets(self) -> None:
        for ws in list(self.websockets):
            await ws.close()
        self.websockets.clear()

    # -- static ----------------------------------------------------------
    async def static_handler(self, request: web.Request) -> web.StreamResponse:
        tail = request.match_info.get("tail", "")
        index = self.web_dir / "index.html" if self.web_dir else None
        if self.web_dir and tail:
            candidate = (self.web_dir / tail).resolve()
            try:
                candidate.relative_to(self.web_dir.resolve())
            except ValueError:
                return error("not found", 404)
            if candidate.is_file():
                return web.FileResponse(candidate)
        if index is not None and index.is_file():
            # SPA fallback: /devices, /rules, /login are client-side routes.
            return web.FileResponse(index)
        return web.Response(text=PLACEHOLDER_HTML, content_type="text/html")

    # -- misc ------------------------------------------------------------
    @staticmethod
    async def _json_body(request: web.Request) -> dict[str, Any]:
        if not request.can_read_body:
            return {}
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            raise web.HTTPBadRequest(
                text=json.dumps({"error": "invalid JSON body"}),
                content_type="application/json",
            ) from None
        return body if isinstance(body, dict) else {}
