"""Runtime configuration for the RKNPU2 detector (RK3588 / RK3576).

The chip is not a setting. Nothing in this module or in the pipeline branches
on it: the only per-chip input is which ``.rknn`` ``model`` points at, and the
runtime refuses a model built for the other SoC at ``init_runtime`` rather than
running it wrongly. ``config.example.yaml`` and ``config.rk3576.example.yaml``
differ in exactly that line and in ``device_id``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Any

from esk_core import config_streams


@dataclass
class Config:
    # Identity -- both are also embedded in every payload (MQTT.md).
    device_id: str = "rk3588-01"
    stream_id: str = "cam-0"

    # Source. `streams` is the multi-stream form; `stream_id`/`source` below are
    # the single-stream shorthand every existing config file uses and are folded
    # into `streams` at load time, so no deployment needs editing.
    streams: list[dict[str, Any]] = field(default_factory=list)

    # Source
    source: str = "rtsp://127.0.0.1:8556/edge-sec-truth"
    rtsp_transport: str = "tcp"
    rtsp_codec: str = "h264"
    rtsp_latency_ms: int = 100
    appsink_timeout_ms: int = 2000
    appsink_queue: int = 3
    # This platform's primary decode path is the Rockchip MPP hardware decoder.
    decode_primary: str = "hw"
    # Refuse to run on the CPU decoder rather than degrade silently. Set false
    # only for a deployment that prefers a slow stream to no stream; the
    # detector then reports decode: sw with fallback_active: true.
    require_hw_decode: bool = True

    # Model
    model: str = "models/yolov8n_fp16.rk3588.rknn"
    input_size: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    # RKNNLite core mask. None lets the runtime schedule. RK3588 has three NPU
    # cores and RK3576 two; on both, only Core0 is ever busy for one small model,
    # so a single-stream detector does not benefit from pinning.
    npu_core_mask: int | None = None

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

    #: Where this config was loaded from. Runtime changes are written back here
    #: so a threshold moved from the console survives a restart; None means the
    #: process was configured from flags and nothing is persisted.
    path: str | None = None
    #: Upper bound on streams in this process, 0 = no limit. The boards have a
    #: measured knee (see the top-level README's multi-stream ladder); refusing
    #: past it is kinder than accepting and degrading every existing stream.
    max_streams: int = 0

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
        data.pop("path", None)  # not an operator-settable key
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**data)
        cfg.path = path
        if path and not os.path.isabs(cfg.model):
            cfg.model = os.path.join(os.path.dirname(os.path.abspath(path)), cfg.model)
        return cfg

    def stream_configs(self) -> list[dict[str, Any]]:
        """The stream list, whichever form the file used."""
        return config_streams.stream_configs(self)

    def persist_streams(self, streams: list[dict[str, Any]]) -> bool:
        """Write the current stream list back, atomically. See esk_core."""
        return config_streams.persist_streams(self, streams)
