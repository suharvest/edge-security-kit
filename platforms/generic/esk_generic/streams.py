"""Per-stream configuration and the capture worker that runs one of them.

The detector used to be one process, one source, one ``stream_id``. Adding a
camera meant editing a config file and restarting, which is not something an
operator can do while standing in front of the camera — and it is the operation
``cmd/control``'s ``add_stream`` exists to make possible.

So the capture loop moved here, into a worker that owns exactly one stream's
state: its source, its tracker, its frame counter, its latest frame and its
confidence threshold. The supervisor in ``detector.py`` owns everything that is
genuinely shared — the MQTT client, the ONNX session, the status heartbeat.

Two things are deliberately *not* shared:

* **The tracker.** ``track_id`` is per-stream by contract. One tracker across
  two cameras would hand the same id to two different people.
* **The confidence threshold.** It is what an operator retunes per camera; a
  single value on the model session would move every camera at once.

Inference *is* shared, under a lock. One ONNX session per stream would multiply
both the memory and the thread pool, and the generic platform's own README
already documents what oversubscribing that pool costs.
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .preview import FrameStore
from .tracker import IoUTracker
from .yolo import PERSON_CLASS_NAME

LOG = logging.getLogger("esk.generic.stream")

#: How long ``add_stream`` waits for a source to open before answering. The ack
#: has to say whether the stream is actually running, so this wait is the price
#: of an honest answer rather than an optimistic one.
OPEN_TIMEOUT_S = 6.0


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class StreamConfig:
    """One camera. ``name`` is carried for the console and never used here."""

    stream_id: str
    source: str
    name: str = ""
    rtsp_transport: str = "tcp"
    conf_threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"stream_id": self.stream_id, "source": self.source}
        if self.name:
            out["name"] = self.name
        if self.rtsp_transport != "tcp":
            out["rtsp_transport"] = self.rtsp_transport
        if self.conf_threshold is not None:
            out["conf_threshold"] = round(float(self.conf_threshold), 4)
        return out


@dataclass
class StreamWorker:
    """One capture loop. Started by the supervisor, stopped by it or by SIGTERM."""

    config: StreamConfig
    conf_threshold: float
    infer: Callable[[np.ndarray, float], tuple[list[Any], float]]
    publish: Callable[[str, dict[str, Any]], None]
    tracker: IoUTracker
    decode_primary: str = "sw"
    frame_id: int = 0
    state: str = "stopped"
    store: FrameStore = field(default_factory=FrameStore)
    frame_times: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=60)
    )
    #: Set once the source has opened at least once, so add_stream can answer
    #: "running" rather than "asked to run".
    opened: threading.Event = field(default_factory=threading.Event)
    open_error: str = ""
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    published: int = 0

    # -- lifecycle -------------------------------------------------------
    @property
    def stream_id(self) -> str:
        return self.config.stream_id

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run, name=f"stream-{self.stream_id}", daemon=True
        )
        self.thread.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=join_timeout)
        self.state = "stopped"

    def wait_until_open(self, timeout_s: float = OPEN_TIMEOUT_S) -> bool:
        return self.opened.wait(timeout_s)

    # -- health ----------------------------------------------------------
    def measured_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return 0.0 if span <= 0 else (len(self.frame_times) - 1) / span

    def status_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "stream_id": self.stream_id,
            "state": self.state,
            "fps": round(self.measured_fps(), 2),
            "decode": "sw",
            # Not in the required surface, but the console's confidence slider
            # has to be able to render the value the stream is actually running
            # at rather than the value someone last typed into a config file.
            "conf_threshold": round(float(self.conf_threshold), 4),
        }
        if self.config.name:
            entry["name"] = self.config.name
        return entry

    # -- capture ---------------------------------------------------------
    def open_capture(self) -> cv2.VideoCapture:
        if self.config.source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.config.rtsp_transport}"
            )
        capture = cv2.VideoCapture(self.config.source, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        backoff = 1.0
        capture: cv2.VideoCapture | None = None
        try:
            while not self.stop_event.is_set():
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()
                    self.state = "reconnecting"
                    LOG.info("[%s] opening source %s", self.stream_id, self.config.source)
                    capture = self.open_capture()
                    if not capture.isOpened():
                        self.open_error = f"source did not open: {self.config.source}"
                        LOG.warning("[%s] %s, retrying in %.1fs",
                                    self.stream_id, self.open_error, backoff)
                        self.stop_event.wait(backoff)
                        backoff = min(backoff * 2, 15.0)
                        continue
                    backoff = 1.0
                    self.open_error = ""
                    self.state = "running"
                    self.opened.set()

                captured_at = time.monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    LOG.warning("[%s] frame read failed, reopening", self.stream_id)
                    capture.release()
                    capture = None
                    continue

                self.store.put(frame)
                self.frame_times.append(time.monotonic())
                self.publish(self.stream_id, self.build_payload(frame, captured_at))
                self.published += 1
        finally:
            if capture is not None:
                capture.release()
            self.state = "stopped"

    def build_payload(self, frame: np.ndarray, captured_at: float) -> dict[str, Any]:
        detections, inference_ms = self.infer(frame, self.conf_threshold)
        tracked = self.tracker.update(detections, time.monotonic())

        items = []
        for track, det in tracked:
            x1, y1, x2, y2 = det.box
            items.append(
                {
                    "track_id": track.track_id,
                    "class": PERSON_CLASS_NAME,
                    "score": round(det.score, 4),
                    "bbox": [
                        round((x1 + x2) / 2, 6),
                        round((y1 + y2) / 2, 6),
                        round(x2 - x1, 6),
                        round(y2 - y1, 6),
                    ],
                }
            )

        self.frame_id += 1
        height, width = frame.shape[:2]
        return {
            "timestamp": now_ms(),
            "frame_id": self.frame_id,
            "stream_id": self.stream_id,
            "coordinate_space": "frame_norm",
            "frame": {"w": int(width), "h": int(height)},
            "inference_time_ms": round(inference_ms, 3),
            "pipeline_ms": round((time.monotonic() - captured_at) * 1000.0, 3),
            "detections": items,
        }
