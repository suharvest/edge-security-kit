"""The Jetson half of :class:`esk_core.streams.StreamBackend`.

NVDEC through GStreamer, TensorRT for inference. The shared capture loop never
sees either. Same two behaviours as the Rockchip backend, for the same reasons:
``read()`` returning None is a pull timeout rather than a lost source, and a
missing hardware decoder is fatal rather than retried.
"""

from __future__ import annotations

import threading
from typing import Any

from esk_core import FatalSourceError, SourceLost, StreamConfig

from .video_source import HardwareDecodeUnavailable, open_source


class JetsonBackend:
    """One TensorRT context, several streams, serialized by the lock.

    A second context per stream costs the measured 208 MB apiece and, past the
    GPU's 236 inferences/s ceiling, buys nothing — see the multi-stream note in
    the top-level README. One context keeps that budget legible.
    """

    class_name = "person"

    def __init__(self, cfg: Any, model: Any) -> None:
        self.cfg = cfg
        self.model = model
        self.lock = threading.Lock()

    def open(self, config: StreamConfig) -> Any:
        extra = config.extra
        kwargs = dict(
            size=int(extra.get("input_size", self.cfg.input_size)),
            transport=config.rtsp_transport,
            require_hw=bool(extra.get("require_hw_decode", self.cfg.require_hw_decode)),
        )
        for name in ("rtsp_codec", "rtsp_latency_ms", "appsink_timeout_ms", "appsink_queue"):
            if hasattr(self.cfg, name):
                key = name.replace("rtsp_", "") if name == "rtsp_codec" else name
                kwargs[key] = extra.get(name, getattr(self.cfg, name))
        try:
            return open_source(config.source, **kwargs)
        except HardwareDecodeUnavailable as exc:
            raise FatalSourceError(str(exc)) from exc

    def read(self, source: Any) -> Any | None:
        try:
            return source.read()
        except HardwareDecodeUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - a bus error means reopen
            raise SourceLost(str(exc)) from exc

    def close(self, source: Any) -> None:
        source.close()

    def decode_path(self, source: Any) -> str:
        return source.decode_path

    def frame_of(self, bundle: Any) -> Any:
        return bundle.active_rgb()

    def detect(self, bundle: Any, conf_threshold: float) -> tuple[list, float, int, int]:
        with self.lock:
            detections = self.model.detect(
                bundle.canvas, bundle.transform, conf_threshold=conf_threshold
            )
            inference_ms = self.model.last_inference_ms
        return detections, inference_ms, bundle.transform.src_w, bundle.transform.src_h
