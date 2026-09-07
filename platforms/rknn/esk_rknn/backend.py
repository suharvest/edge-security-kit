"""The RKNPU2 half of :class:`esk_core.streams.StreamBackend`.

Everything Rockchip-shaped lives here: the GStreamer/MPP source, the RGA
letterbox it hands back in a :class:`FrameBundle`, and the NPU inference call.
The shared capture loop never sees any of it.

Two behaviours are carried across from the single-stream loop this replaces,
and both are load-bearing on this board:

* ``read()`` returning ``None`` is a pull timeout, not a lost source. A paused
  5 fps stream does it routinely, and tearing down a working MPP pipeline over
  it costs a full renegotiation each time.
* a missing hardware decoder is :class:`FatalSourceError`, not a retry. No
  amount of waiting installs ``libgstrockchipmpp.so``, and a stream that
  retries forever shows up as a grey tile with no reason attached.
"""

from __future__ import annotations

import threading
from typing import Any

from esk_core import FatalSourceError, SourceLost, StreamConfig
from .video_source import HardwareDecodeUnavailable, open_source


class RknnBackend:
    """One RKNN model, several streams. The lock is why that is safe.

    A second RKNN context per stream would double the weight memory and place
    itself on an NPU core the runtime picks independently — the multi-stream
    ladder in the top-level README is measured with one process per stream for
    exactly that reason, and this is the other arrangement. Serializing the
    inference call keeps the placement decision to one context.
    """

    class_name = "person"

    def __init__(self, cfg: Any, model: Any) -> None:
        self.cfg = cfg
        self.model = model
        self.lock = threading.Lock()

    # -- source ----------------------------------------------------------
    def open(self, config: StreamConfig) -> Any:
        extra = config.extra
        try:
            return open_source(
                config.source,
                size=int(extra.get("input_size", self.cfg.input_size)),
                transport=config.rtsp_transport,
                codec=str(extra.get("rtsp_codec", self.cfg.rtsp_codec)),
                require_hw=bool(extra.get("require_hw_decode", self.cfg.require_hw_decode)),
                latency_ms=int(extra.get("rtsp_latency_ms", self.cfg.rtsp_latency_ms)),
                appsink_timeout_ms=int(
                    extra.get("appsink_timeout_ms", self.cfg.appsink_timeout_ms)
                ),
                appsink_queue=int(extra.get("appsink_queue", self.cfg.appsink_queue)),
            )
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

    # -- inference -------------------------------------------------------
    def frame_of(self, bundle: Any) -> Any:
        return bundle.active_rgb()

    def detect(self, bundle: Any, conf_threshold: float) -> tuple[list, float, int, int]:
        with self.lock:
            detections = self.model.detect(
                bundle.canvas, bundle.transform, conf_threshold=conf_threshold
            )
            inference_ms = self.model.last_inference_ms
        return detections, inference_ms, bundle.transform.src_w, bundle.transform.src_h

