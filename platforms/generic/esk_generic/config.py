"""Runtime configuration for the generic CPU detector."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class Config:
    # Identity -- both are also embedded in every payload (MQTT.md).
    device_id: str = "generic-01"
    stream_id: str = "cam-0"

    # Source
    source: str = "rtsp://127.0.0.1:8555/edge-sec-720p"
    rtsp_transport: str = "tcp"
    # Which decode path this deployment considers primary. The generic platform
    # is CPU-only, so "sw" is normal here and does not raise fallback_active.
    decode_primary: str = "sw"

    # Model
    model: str = "models/yolov8n.onnx"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    intra_threads: int = 0
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])

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
    # Host advertised in status.streams[].preview_url. Empty -> omit the field.
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
