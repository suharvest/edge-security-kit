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

    # Source. `streams` is the multi-stream form; `source`/`stream_id` above are
    # the single-stream shorthand every existing config file uses and are folded
    # into `streams` at load time, so nothing that worked before needs editing.
    streams: list[dict[str, Any]] = field(default_factory=list)
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

    #: Where this config was loaded from. Runtime changes are written back here
    #: so a threshold moved from the console survives a restart; None means the
    #: process was configured from flags and nothing is persisted.
    path: str | None = None

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
        """The stream list, whichever form the file used.

        A file with no ``streams`` key describes one stream through the
        top-level ``stream_id``/``source`` pair. Folding the two forms into one
        here means the rest of the runtime never branches on which was written.
        """
        if self.streams:
            return [dict(entry) for entry in self.streams]
        return [{"stream_id": self.stream_id, "source": self.source,
                 "rtsp_transport": self.rtsp_transport}]

    def persist_streams(self, streams: list[dict[str, Any]]) -> bool:
        """Write the current stream list back to the config file.

        Returns whether it was persisted. A runtime change that only lives in
        memory is a change the next restart silently undoes, so the ack tells
        the caller which of the two it got (``applied.persisted``) rather than
        letting the console assume.

        The write is atomic: a power cut during it must not leave a detector
        with a truncated config file and no way to start.
        """
        if not self.path:
            return False
        try:
            import yaml
        except ImportError:  # pragma: no cover - yaml ships with the runtime
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = handle.read()
            data = (json.loads(raw) if self.path.endswith(".json")
                    else yaml.safe_load(raw)) or {}
            data["streams"] = streams
            # The shorthand keys would now contradict the list; drop them so the
            # file has exactly one description of what this detector watches.
            data.pop("source", None)
            data.pop("stream_id", None)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                if self.path.endswith(".json"):
                    json.dump(data, handle, indent=2, ensure_ascii=False)
                else:
                    yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
            return True
        except (OSError, ValueError) as exc:  # noqa: BLE001
            import logging

            logging.getLogger("esk.generic").warning(
                "could not persist config to %s: %s", self.path, exc
            )
            return False
