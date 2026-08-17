"""Hub configuration: single ``config.json`` + environment overrides.

The persisted body lives in SQLite (``config_versions`` scope ``hub``) and is
mirrored to ``<data>/config.json`` atomically on every PUT (HUB_SPEC §4).
Environment variables win over the file for the transport settings a deployment
needs to set before the file exists (HUB_SPEC §8 compose ``MQTT_HOST``).
"""

from __future__ import annotations

import os
from typing import Any

DEFAULTS: dict[str, Any] = {
    "mqtt_host": "localhost",
    "mqtt_port": 1883,
    "mqtt_username": None,
    "mqtt_password": None,
    "mqtt_client_id": "edge-security-hub",
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
    return cfg


def restart_required(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return [k for k in RESTART_REQUIRED if before.get(k) != after.get(k)]
