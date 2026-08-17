"""Hub configuration: single ``config.json`` + environment overrides.

The persisted body lives in SQLite (``config_versions`` scope ``hub``) and is
mirrored to ``<data>/config.json`` atomically on every PUT (HUB_SPEC §4).
Environment variables win over the file for the transport settings a deployment
needs to set before the file exists (HUB_SPEC §8 compose ``MQTT_HOST``).
"""

from __future__ import annotations

import os
import re
import secrets
import socket
from typing import Any

CLIENT_ID_PREFIX = "edge-security-hub"


def _host_slug() -> str:
    try:
        host = socket.gethostname().split(".")[0]
    except OSError:  # pragma: no cover - gethostname failing is exotic
        host = ""
    slug = re.sub(r"[^A-Za-z0-9-]+", "-", host).strip("-").lower()[:24]
    return slug or "host"


def default_client_id() -> str:
    """A fresh MQTT client id: ``edge-security-hub-<host>-<4 hex>``.

    MQTT client ids are exclusive per broker: a second connection presenting an
    id already in use takes the session over and the first client is
    disconnected. Two hubs sharing a fixed ``edge-security-hub`` id therefore
    kick each other in a ~1 s reconnect loop, and since detections are published
    at QoS 0 the losing side just loses messages, with no error logged anywhere.
    The symptom is skewed rather than obvious: rules that need two consecutive
    frames (``line_cross``) fail about half the time while rules that need any
    single frame (``zone_enter``) keep working.

    Every call returns a different value; the process-wide default is resolved
    once, by :func:`process_client_id`.
    """
    return f"{CLIENT_ID_PREFIX}-{_host_slug()}-{secrets.token_hex(2)}"


_PROCESS_CLIENT_ID: str | None = None


def process_client_id() -> str:
    """The default client id for this process — generated once, then stable.

    Stability inside one process matters: :func:`resolve` runs again on every
    config PUT, and a value that changed each time would report a spurious
    ``restart_required`` for ``mqtt_client_id``.
    """
    global _PROCESS_CLIENT_ID
    if _PROCESS_CLIENT_ID is None:
        _PROCESS_CLIENT_ID = default_client_id()
    return _PROCESS_CLIENT_ID


DEFAULTS: dict[str, Any] = {
    "mqtt_host": "localhost",
    "mqtt_port": 1883,
    "mqtt_username": None,
    "mqtt_password": None,
    # ``None`` means "derive a unique one" — see :func:`process_client_id`.
    "mqtt_client_id": None,
    "topic_prefix": "sensecraft/security",
    "http_host": "0.0.0.0",
    "http_port": 8090,
    # HUB_SPEC §3: snapshot request timeout, seconds
    "snapshot_timeout_s": 60,
    # HUB_SPEC §6: alerts + snapshots retention, days
    "retention_days": 30,
    # HUB_SPEC §7: idle session expiry, days
    "session_idle_days": 7,
    "default_cooldown_s": 30,
    "track_expiry_s": 5,
    "line_chain_gap_ms": 1000,
    "max_snapshot_bytes": 200 * 1024,
}

#: Items that only take effect after a restart; echoed back by PUT /api/config.
RESTART_REQUIRED = ("mqtt_host", "mqtt_port", "mqtt_username", "mqtt_password",
                    "mqtt_client_id", "topic_prefix", "http_host", "http_port")

_ENV_MAP = {
    "MQTT_HOST": ("mqtt_host", str),
    "MQTT_PORT": ("mqtt_port", int),
    "MQTT_USERNAME": ("mqtt_username", str),
    "MQTT_PASSWORD": ("mqtt_password", str),
    "MQTT_CLIENT_ID": ("mqtt_client_id", str),
    "HUB_TOPIC_PREFIX": ("topic_prefix", str),
    "HUB_HTTP_HOST": ("http_host", str),
    "HUB_HTTP_PORT": ("http_port", int),
    "HUB_RETENTION_DAYS": ("retention_days", int),
    "HUB_SNAPSHOT_TIMEOUT_S": ("snapshot_timeout_s", int),
}


def env_overrides(env: dict[str, str] | None = None) -> dict[str, Any]:
    src = os.environ if env is None else env
    out: dict[str, Any] = {}
    for key, (field, caster) in _ENV_MAP.items():
        raw = src.get(key)
        if raw is None or raw == "":
            continue
        try:
            out[field] = caster(raw)
        except ValueError:
            continue
    return out


def resolve(stored: dict[str, Any] | None, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Merge defaults < stored config.json body < environment."""
    cfg = dict(DEFAULTS)
    cfg.update(stored or {})
    cfg.update(env_overrides(env))
    # An explicit id (config.json or MQTT_CLIENT_ID) always wins; otherwise
    # every hub process gets its own, so two hubs on one broker cannot collide.
    if not cfg.get("mqtt_client_id"):
        cfg["mqtt_client_id"] = process_client_id()
    return cfg


def restart_required(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return [k for k in RESTART_REQUIRED if before.get(k) != after.get(k)]
