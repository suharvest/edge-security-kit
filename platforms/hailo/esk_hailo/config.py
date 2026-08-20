"""Runtime configuration for the Hailo-8 / 8L detector.

Nothing here branches on Hailo-8 versus Hailo-8L. The two differ in how many
compute clusters the compiler had to fit the graph into, which is settled when
the HEF is built; at runtime the same code drives both, and a HEF built for the
other variant is rejected by ``configure()`` rather than run wrongly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class Config:
    # Identity -- both are also embedded in every payload (MQTT.md).
    device_id: str = "rpi5-hailo8-01"
    stream_id: str = "cam-0"

    # Source
    source: str = "rtsp://127.0.0.1:8557/edge-sec-truth"
    rtsp_transport: str = "tcp"
    # Which decode path this deployment considers primary. On a Raspberry Pi 5
    # this is "sw" and that is not a fallback: the Pi 5 dropped the H.264
    # hardware decoder its predecessor had (VideoCore VII decodes HEVC only),
    # so an H.264 RTSP stream is decoded by FFmpeg on the Cortex-A76 cores.
    # health.decode reports "sw" and health.fallback_active stays false,
    # because there is no faster path being missed.
    decode_primary: str = "sw"

    # Model
    model: str = "models/yolov8n.hef"
    input_size: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    infer_timeout_ms: int = 5000

    # Tracking
    track_iou_threshold: float = 0.2
    track_max_lost_s: float = 0.75

    # MQTT
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_username: str | None = None
    mqtt_password: str | None = None
    mqtt_keepalive: int = 30
    topic_prefix: str = "sensecraft/security"
    status_interval_s: float = 30.0

    # Preview HTTP endpoint (rule-canvas backdrop; CORS GET)
    preview_enabled: bool = True
    preview_bind: str = "0.0.0.0"
    preview_port: int = 8099
    # Host advertised in status.streams[].preview_url. Empty -> omit the field,
    # because the device cannot guess which address a browser can reach and a
    # wrong URL is worse than an absent one.
    preview_advertise_host: str = ""

    app_version: str = "0.1.0"

    @classmethod
    def load(cls, path: str | None) -> "Config":
        data: dict[str, Any] = {}
        if path:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
            if path.endswith(".json"):
                data = json.loads(text)
            else:
                import yaml

                data = yaml.safe_load(text) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**data)
        if path and not os.path.isabs(cfg.model):
            cfg.model = os.path.join(os.path.dirname(os.path.abspath(path)), cfg.model)
        return cfg
