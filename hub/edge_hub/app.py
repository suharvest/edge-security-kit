"""Application wiring: one asyncio process holding all six modules (HUB_SPEC §1).

``Hub`` owns the object graph and the cross-module callbacks; it is constructed
without touching the network so tests can drive it directly (no broker, no
sockets). :func:`run` adds the MQTT client, the HTTP listener and the retention
timer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from typing import Any

from aiohttp import web

from . import config as config_module
from .alert_manager import ORIGIN_HUB, AlertManager
from .auth import AuthManager
from .clock import Clock
from .device_registry import DeviceRegistry
from .http_api import HttpApi
from .mqtt_ingest import MqttIngest
from .rules.engine import Candidate, RuleEngine
from .rules_schema import find_rule
from .storage import Storage
from .validation import ContractValidator

log = logging.getLogger("edge_hub")

RETENTION_INTERVAL_S = 3600.0

#: ``web_dir=AUTO`` discovers ``web/dist``; ``web_dir=None`` means "serve the
#: placeholder", which is what the tests want.
AUTO = "auto"


def default_web_dir() -> Path | None:
    """Locate the built frontend (``web/dist``); ``None`` when absent."""
    override = os.environ.get("HUB_WEB_DIR")
    if override:
        path = Path(override)
        return path if path.is_dir() else None
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "web" / "dist"
        if candidate.is_dir():
            return candidate
    return None


class Hub:
    def __init__(
        self,
        data_dir: str | os.PathLike[str],
        clock: Clock | None = None,
        env: dict[str, str] | None = None,
        web_dir: Path | None | str = AUTO,
        validator: ContractValidator | None = None,
    ) -> None:
        self.clock = clock or Clock()
        self.storage = Storage(data_dir)
        self.env = env
        stored = (self.storage.get_hub_config() or {}).get("body")
        self.config = config_module.resolve(stored, env)
        self.validator = validator or ContractValidator()

        self.auth = AuthManager(
            self.storage, clock=self.clock, idle_days=float(self.config["session_idle_days"])
        )
        resolved_web_dir = default_web_dir() if web_dir == AUTO else web_dir
        self.api = HttpApi(self, web_dir=resolved_web_dir)
        self.alerts = AlertManager(
            self.storage,
            clock=self.clock,
            publisher=self._publish,
            broadcaster=self.api.broadcast,
            topic_prefix=str(self.config["topic_prefix"]),
            snapshot_timeout_s=float(self.config["snapshot_timeout_s"]),
            default_cooldown_s=float(self.config["default_cooldown_s"]),
        )
        self.engine = RuleEngine(self.storage.rules_body, clock=self.clock)
        self.registry = DeviceRegistry(
            self.storage,
            clock=self.clock,
            broadcaster=self.api.broadcast,
            on_session_change=self._on_session_change,
        )
        self.ingest: MqttIngest | None = None
        self._retention_task: asyncio.Task[None] | None = None

    # -- MQTT glue -------------------------------------------------------
    def build_ingest(self) -> MqttIngest:
        self.ingest = MqttIngest(
            host=str(self.config["mqtt_host"]),
            port=int(self.config["mqtt_port"]),
            username=self.config["mqtt_username"],
            password=self.config["mqtt_password"],
            client_id=str(self.config["mqtt_client_id"]),
            topic_prefix=str(self.config["topic_prefix"]),
            validator=self.validator,
            on_detections=self.on_detections,
            on_status=self.on_status,
            on_event=self.on_event,
            on_snapshot=self.on_snapshot,
            max_snapshot_bytes=int(self.config["max_snapshot_bytes"]),
        )
        return self.ingest

    async def _publish(self, topic: str, payload: bytes, qos: int = 1) -> None:
        if self.ingest is not None:
            await self.ingest.publish(topic, payload, qos)

    async def _on_session_change(self, device_id: str, session_id: str) -> None:
        """Generation reset (HUB_SPEC §2.1) driven by the status topic.

        The detections path resets on its own too — this covers a device whose
        heartbeat lands before its first frame of the new session.
        """
        self.engine.reset_device(device_id)
        self.alerts.reset_session(device_id)

    # -- ingest handlers -------------------------------------------------
    async def on_detections(self, payload: dict[str, Any]) -> None:
        self.registry.note_detections(payload)
        device_id = str(payload["device_id"])
        # A single-box device judges its own rules; the hub must not double-judge.
        if self.registry.mode(device_id) == "single_box":
            return
        candidates = self.engine.on_detections(payload)
        if candidates:
            await self.alerts.handle_candidates(candidates, self.storage.rules_body)

    async def on_status(self, payload: dict[str, Any]) -> None:
        await self.registry.on_status(payload)

    async def on_event(self, payload: dict[str, Any]) -> None:
        """Single-box events arrive here — including the hub's own echo.

        The hub republishes its verdicts on ``events/<stream_id>`` for third-party
        integrations (contracts/MQTT.md Modes) and is subscribed to that same
        topic family. Both guards below drop the echo: the ``origin`` marker, and
        an event_id that is already stored. Without them a hub-mode device would
        be reclassified as single-box after its first alert and the hub would
        stop judging its rules.
        """
        device_id = str(payload["device_id"])
        if payload.get("origin") == ORIGIN_HUB:
            return
        if self.storage.get_alert_by_event_id(str(payload["event_id"])) is not None:
            return
        self.registry.note_single_box(device_id)
        await self.alerts.ingest_device_event(payload)

    async def on_snapshot(self, device_id: str, event_id: str, payload: bytes) -> None:
        await self.alerts.attach_snapshot(event_id, payload)

    # -- REST support ----------------------------------------------------
    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            # Exposed for diagnosis: two hubs on one broker must not share a
            # client id, or they disconnect each other and drop QoS 0
            # detections without logging an error.
            "mqtt_client_id": (
                self.ingest.client_id
                if self.ingest
                else str(self.config["mqtt_client_id"])
            ),
            "mqtt_connected": bool(self.ingest and self.ingest.connected),
            "mqtt_messages": self.ingest.messages if self.ingest else 0,
            "handler_errors": self.ingest.handler_errors if self.ingest else 0,
            "ws_clients": len(self.api.websockets),
            "suppressed_by_cooldown": self.alerts.suppressed,
            "rule_engine_drops": dict(self.engine.stats),
            **self.validator.stats(),
        }

    def config_view(self) -> dict[str, Any]:
        stored = self.storage.get_hub_config() or {"rev": 0, "body": {}}
        public = {k: v for k, v in self.config.items() if k != "mqtt_password"}
        public["mqtt_password_set"] = bool(self.config.get("mqtt_password"))
        return {"rev": stored["rev"], "config": public,
                "env_overrides": sorted(config_module.env_overrides(self.env))}

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        body = patch.get("config") if isinstance(patch.get("config"), dict) else patch
        unknown = [k for k in body if k not in config_module.DEFAULTS]
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
        stored = dict((self.storage.get_hub_config() or {}).get("body") or {})
        stored.update(body)
        before = dict(self.config)
        result = self.storage.put_hub_config(stored, self.clock.wall_ms())
        self.config = config_module.resolve(stored, self.env)
        self.auth.idle_ms = float(self.config["session_idle_days"]) * 86_400_000
        self.alerts.snapshot_timeout_s = float(self.config["snapshot_timeout_s"])
        self.alerts.default_cooldown_s = float(self.config["default_cooldown_s"])
        return {
            **result,
            "restart_required": config_module.restart_required(before, self.config),
            "config": self.config_view()["config"],
        }

    async def simulate(self, device_id: str, stream_id: str, rule_id: str) -> dict[str, Any] | None:
        """POST /rules/{d}/{s}/simulate — a real alert with meta.simulated (§4)."""
        body = self.storage.rules_body(device_id, stream_id)
        if not body:
            return None
        found = find_rule(body, rule_id)
        if found is None:
            return None
        kind, rule = found
        if kind == "zone":
            points = rule["points"]
            centroid = (
                sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points),
            )
            event_type = "zone_enter"
            direction = None
            dwell_s = None
        else:
            centroid = (
                (rule["start"][0] + rule["end"][0]) / 2,
                (rule["start"][1] + rule["end"][1]) / 2,
            )
            event_type = "line_cross"
            direction = rule["direction"] if rule["direction"] != "any" else "forward"
            dwell_s = None
        candidate = Candidate(
            device_id=device_id,
            stream_id=stream_id,
            session_id="simulate",
            event_type=event_type,
            rule_name=str(rule["name"]),
            rule_id=str(rule["id"]),
            track_id=0,
            bbox=[round(centroid[0], 4), round(centroid[1], 4), 0.1, 0.2],
            ts_ms=self.clock.wall_ms(),
            received_ms=self.clock.wall_ms(),
            received_mono=self.clock.mono_ms(),
            score=1.0,
            cls="person",
            direction=direction,
            dwell_s=dwell_s,
            meta={"simulated": True},
        )
        # Simulation bypasses cooldown on purpose: the installer presses the
        # button to prove the chain works, possibly twice in a row.
        return await self.alerts.fire(candidate, mode="single_box")

    # -- lifecycle -------------------------------------------------------
    async def start_background(self) -> None:
        await self.alerts.rearm_pending_snapshots()
        self._retention_task = asyncio.get_running_loop().create_task(self._retention_loop())

    async def _retention_loop(self) -> None:
        while True:
            try:
                cutoff = self.clock.wall_ms() - int(
                    float(self.config["retention_days"]) * 86_400_000
                )
                removed = self.storage.purge_older_than(cutoff)
                if removed:
                    log.info("retention: purged %d alerts", removed)
                self.alerts.sweep_parked()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("retention pass failed: %s", exc)
            await asyncio.sleep(RETENTION_INTERVAL_S)

    async def shutdown(self) -> None:
        if self._retention_task is not None:
            self._retention_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._retention_task
            self._retention_task = None
        await self.alerts.shutdown()
        if self.ingest is not None:
            await self.ingest.stop()
        await self.api.close_websockets()
        self.storage.close()


async def run(data_dir: str | None = None) -> None:
    logging.basicConfig(
        level=os.environ.get("HUB_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    hub = Hub(data_dir or os.environ.get("HUB_DATA_DIR", "/data"))
    username, plaintext = hub.auth.ensure_default_account(
        password=os.environ.get("HUB_ADMIN_PASSWORD")
    )
    if plaintext is not None:
        # §7: the generated secret goes to the log AND to a 0600 file, so an
        # operator who missed the startup output can still read it off the host
        # without a reset path existing.
        pw_path = Path(hub.storage.data_dir) / "initial-password.txt"
        try:
            with open(
                pw_path, "w", opener=lambda p, f: os.open(p, f | os.O_CREAT | os.O_TRUNC, 0o600)
            ) as handle:
                handle.write(f"{username}\n{plaintext}\n")
            os.chmod(pw_path, 0o600)
        except OSError as exc:  # noqa: BLE001 - the log copy is the fallback
            log.warning("could not write %s: %s", pw_path, exc)
        log.warning(
            "created initial account %s with a generated password "
            "(also written to %s, mode 0600) - must be changed on first login: %s",
            username,
            pw_path,
            plaintext,
        )
    app = hub.api.build()
    hub.build_ingest()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, str(hub.config["http_host"]), int(hub.config["http_port"]))
    await site.start()
    log.info("http listening on %s:%s", hub.config["http_host"], hub.config["http_port"])
    assert hub.ingest is not None
    hub.ingest.start()
    await hub.start_background()
    stop = asyncio.Event()
    try:
        await stop.wait()
    finally:
        await hub.shutdown()
        await runner.cleanup()
