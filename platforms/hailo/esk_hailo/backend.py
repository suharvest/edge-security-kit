"""The Hailo half of :class:`esk_core.streams.StreamBackend`.

The Pi 5 has no H.264 decoder — VideoCore VII is HEVC-only — so decode is
ffmpeg on the CPU through OpenCV, and ``decode_primary`` is ``sw`` here rather
than being a fallback. That makes the per-stream CPU cost the thing that runs
out first on this board: the top-level README's figure of 8.7-13.0% of one core
buys decode as well as inference, and it is per stream.

A failed ``capture.read()`` is a lost source (unlike the GStreamer backends,
OpenCV has no "not yet" answer), so it raises rather than returning None.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from esk_core import SourceLost, StreamConfig

from .letterbox import letterbox


@dataclass
class FrameBundle:
    """One decoded frame plus the letterbox transform used to reach the model."""

    frame: np.ndarray
    canvas: np.ndarray
    transform: Any


class HailoBackend:
    """One HEF / VDevice, several streams, serialized by the lock.

    A second VDevice is not a cheap thing to hold on this part, and the teardown
    order matters (see the infer-wrapper teardown fix in the README). One device
    for the process keeps both problems to one instance.
    """

    class_name = "person"

    def __init__(self, cfg: Any, model: Any) -> None:
        self.cfg = cfg
        self.model = model
        self.lock = threading.Lock()

    def open(self, config: StreamConfig) -> Any:
        if config.source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{config.rtsp_transport}"
            )
        capture = cv2.VideoCapture(config.source, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"could not open {config.source}")
        return capture

    def read(self, source: Any) -> Any | None:
        ok, frame = source.read()
        if not ok or frame is None:
            # OpenCV has no "no frame yet" answer: a false here is the stream
            # having ended or dropped, which is a reopen.
            raise SourceLost("capture.read() returned no frame")
        canvas, transform = letterbox(frame[:, :, ::-1], self.model.input_size,
                                      self.model.input_size)
        return FrameBundle(frame=frame, canvas=canvas, transform=transform)

    def close(self, source: Any) -> None:
        source.release()

    def decode_path(self, source: Any) -> str:
        # There is no hardware H.264 decoder on this board to fall back from.
        return "sw"

    def frame_of(self, bundle: Any) -> Any:
        return bundle.frame

    def detect(self, bundle: Any, conf_threshold: float) -> tuple[list, float, int, int]:
        with self.lock:
            detections = self.model.detect(
                bundle.canvas, bundle.transform, conf_threshold=conf_threshold
            )
            inference_ms = self.model.last_inference_ms
        height, width = bundle.frame.shape[:2]
        return detections, inference_ms, width, height
