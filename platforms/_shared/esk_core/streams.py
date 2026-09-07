"""One capture loop per stream, with a platform-supplied backend behind it.

Every detector used to be one process, one source, one ``stream_id``. Adding a
camera meant editing a config file and restarting — not an operation available
to someone standing in front of the camera, and the thing ``cmd/control``'s
``add_stream`` exists to make possible.

So the capture loop lives here and owns exactly one stream's state: its source,
its tracker, its frame counter, its latest frame and its confidence threshold.
The supervisor owns what is genuinely shared — the MQTT client, the model, the
status heartbeat.

Two things are deliberately *not* shared between streams, on every platform:

* **The tracker.** ``track_id`` is per-stream by contract. One tracker across
  two cameras hands the same id to two different people.
* **The confidence threshold.** It is what an operator retunes per camera; a
  single value on the model session moves every camera when one slider moves.

Inference *is* shared, under the supervisor's lock. A second TensorRT context,
RKNN context or Hailo VDevice multiplies memory and, on the Rockchip boards,
NPU core placement — see the multi-stream ladder in the top-level README.

Nothing here imports cv2, TensorRT, RKNN or HailoRT. The platform supplies a
:class:`StreamBackend`; that is the whole seam.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

LOG = logging.getLogger("esk.core.stream")

#: How long ``add_stream`` waits for a source to open before answering. The ack
#: has to say whether the stream is actually running, so this wait is the price
#: of an honest answer rather than an optimistic one. A hardware decoder that
#: has to be negotiated (GStreamer NVDEC / MPP) takes longer to come up than an
#: OpenCV capture, hence the size of it.
OPEN_TIMEOUT_S = 8.0


def now_ms() -> int:
    """Epoch milliseconds. The contract forbids seconds or floats."""
    return int(time.time() * 1000)


class SourceLost(Exception):
    """The source needs reopening. Distinct from ``read()`` returning None.

    The distinction is not cosmetic. The Rockchip and Jetson GStreamer paths
    return None on a pull timeout, which is normal on a stream that has paused;
    treating that as a lost source tears down and rebuilds a working hardware
    pipeline every time a 5 fps camera hiccups. Only a bus error or a closed
    capture is a reopen, and only the backend can tell which it got.
    """


class FatalSourceError(Exception):
    """A source failure retrying cannot fix (a missing decoder plugin).

    Kept distinct from an ordinary open failure because the two need opposite
    handling: a camera that is rebooting deserves the backoff loop, and a
    GStreamer plugin that is not installed deserves an immediate refusal with
    the reason. Retrying the second one forever is how a misconfigured deploy
    turns into "the tile is grey and nobody knows why".
    """


class StreamBackend(Protocol):
    """What a platform must supply. Everything accelerator-shaped is here."""

    def open(self, config: "StreamConfig") -> Any:
        """Open the source, or raise. May raise :class:`FatalSourceError`."""

    def read(self, source: Any) -> Any | None:
        """One frame bundle, or None when no frame is ready yet.

        Raise :class:`SourceLost` when the source must be reopened. Returning
        None must mean "nothing this tick, the source is fine".
        """

    def close(self, source: Any) -> None:
        ...

    def detect(self, bundle: Any, conf_threshold: float) -> tuple[list, float, int, int]:
        """``(detections, inference_ms, frame_w, frame_h)``.

        Detections are the model's own objects, pre-tracking: the worker runs
        the tracker, because ``track_id`` is per-stream by contract and a
        backend shared between streams cannot hold that state. ``frame_w/h`` are
        the ORIGINAL frame, not the model input -- the published bbox is
        normalized against them after the letterbox pad is reversed.
        """

    def frame_of(self, bundle: Any) -> Any:
        """The displayable frame, for the preview endpoint and snapshots."""

    def decode_path(self, source: Any) -> str:
        """``hw`` or ``sw`` for this source, as the contract's health.decode."""


@dataclass
class StreamConfig:
    """One camera. ``name`` is carried for the console and never used here."""

    stream_id: str
    source: str
    name: str = ""
    rtsp_transport: str = "tcp"
    conf_threshold: float | None = None
    #: Platform-specific extras (codec, latency, require_hw_decode ...). Kept
    #: opaque so the base does not grow a union of every platform's options.
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"stream_id": self.stream_id, "source": self.source}
        if self.name:
            out["name"] = self.name
        if self.rtsp_transport != "tcp":
            out["rtsp_transport"] = self.rtsp_transport
        if self.conf_threshold is not None:
            out["conf_threshold"] = round(float(self.conf_threshold), 4)
        out.update(self.extra)
        return out


class StreamWorker:
    """One stream's capture thread. Started and stopped by the supervisor."""

    def __init__(
        self,
        config: StreamConfig,
        conf_threshold: float,
        backend: StreamBackend,
        publish: Callable[[str, dict[str, Any]], None],
        tracker: Any,
        decode_primary: str = "sw",
    ) -> None:
        self.config = config
        self.conf_threshold = conf_threshold
        self.backend = backend
        self.publish = publish
        self.tracker = tracker
        self.decode_primary = decode_primary

        self.frame_id = 0
        self.state = "stopped"
        self.decode = decode_primary
        self.published = 0
        self.source: Any = None
        self.frame: Any = None
        self.frame_times: collections.deque[float] = collections.deque(maxlen=60)
        self.inference_times: collections.deque[float] = collections.deque(maxlen=200)
        self._frame_lock = threading.Lock()
        self.opened = threading.Event()
        self.open_error = ""
        self.fatal: Exception | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

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
        """True once the stream has produced a frame. Early on a fatal failure.

        Waiting the full timeout on a missing decoder plugin would make every
        misconfigured add_stream take the maximum time to say the same thing.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.opened.is_set():
                return True
            if self.fatal is not None:
                return False
            time.sleep(0.05)
        return self.opened.is_set()

    # -- frame store -----------------------------------------------------
    def put_frame(self, frame: Any) -> None:
        with self._frame_lock:
            self.frame = frame

    def get_frame(self) -> Any:
        with self._frame_lock:
            frame = self.frame
        return None if frame is None else (frame.copy() if hasattr(frame, "copy") else frame)

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
            "decode": self.decode,
            # Not in the contract's required surface, but the console's slider
            # has to render the value the stream is actually running at rather
            # than the value someone last typed into a config file.
            "conf_threshold": round(float(self.conf_threshold), 4),
            "fallback_active": self.decode != self.decode_primary,
        }
        if self.config.name:
            entry["name"] = self.config.name
        return entry

    # -- capture ---------------------------------------------------------
    def _run(self) -> None:
        backoff = 1.0
        try:
            while not self.stop_event.is_set():
                if self.source is None:
                    self.state = "reconnecting"
                    LOG.info("[%s] opening source %s", self.stream_id, self.config.source)
                    try:
                        self.source = self.backend.open(self.config)
                    except FatalSourceError as exc:
                        # Not retried: no amount of waiting installs a plugin.
                        self.fatal = exc
                        self.open_error = str(exc)
                        LOG.error("[%s] %s", self.stream_id, exc)
                        return
                    except Exception as exc:  # noqa: BLE001 - transient by assumption
                        self.source = None
                        self.open_error = f"source did not open: {exc}"
                        LOG.warning("[%s] %s, retrying in %.1fs",
                                    self.stream_id, self.open_error, backoff)
                        self.stop_event.wait(backoff)
                        backoff = min(backoff * 2, 15.0)
                        continue
                    backoff = 1.0
                    self.open_error = ""
                    self.decode = self.backend.decode_path(self.source)
                    if self.decode != self.decode_primary:
                        LOG.warning("[%s] decode fell back to %s; health.decode=%s, "
                                    "fallback_active=true",
                                    self.stream_id, self.decode, self.decode)
                    self.state = "running"

                try:
                    bundle = self.backend.read(self.source)
                except SourceLost as exc:
                    LOG.warning("[%s] source lost (%s), reopening", self.stream_id, exc)
                    self._release()
                    continue
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("[%s] read raised %s, reopening", self.stream_id, exc)
                    self._release()
                    continue
                if bundle is None:
                    # No frame this tick. The source is fine -- a paused 5 fps
                    # stream does this -- so do NOT tear down a working hardware
                    # pipeline over it.
                    continue

                # The clock starts here, not before the read: read() blocks
                # until the next frame exists, so timing from before it folds
                # the idle wait for a 5 fps source into pipeline_ms and reports
                # a steady ~200 ms regardless of how fast the board is. That
                # number is load-bearing -- pipeline_ms exceeding the frame
                # period is the documented signal that the tracker is being
                # starved -- so inflating it disables the diagnostic.
                captured_at = time.monotonic()
                # "Open" means producing frames, not "the pipeline was built".
                # Measured on an RK3588 against an unreachable RTSP URL: the
                # GStreamer pipeline constructs and reports decode=hw before the
                # connection is attempted, so signalling here rather than after
                # open() is the difference between add_stream acking ok:true for
                # a camera that does not exist and acking the refusal. The
                # contract says ok:true means live; this is where that is true.
                self.opened.set()
                self.put_frame(self.backend.frame_of(bundle))
                self.frame_times.append(time.monotonic())
                self.publish(self.stream_id, self.build_payload(bundle, captured_at))
                self.published += 1
        finally:
            self._release()
            self.state = "stopped"

    def _release(self) -> None:
        if self.source is not None:
            try:
                self.backend.close(self.source)
            except Exception as exc:  # noqa: BLE001 - teardown must not mask the cause
                LOG.warning("[%s] closing the source raised %s", self.stream_id, exc)
            self.source = None

    def build_payload(self, bundle: Any, captured_at: float) -> dict[str, Any]:
        detections, inference_ms, width, height = self.backend.detect(
            bundle, self.conf_threshold
        )
        self.inference_times.append(inference_ms)
        tracked = self.tracker.update(detections, time.monotonic())
        items = detection_items(tracked, getattr(self.backend, "class_name", "person"))
        self.frame_id += 1
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


def detection_items(tracked: Any, class_name: str = "person") -> list[dict[str, Any]]:
    """Turn ``(track, detection)`` pairs into contract detection entries.

    The xyxy -> centre-format conversion was written out four times, once per
    platform. It is the same four lines and the same rounding every time, and a
    platform that rounded differently would show up as a fixture diff nobody
    could explain.
    """
    items = []
    for track, det in tracked:
        x1, y1, x2, y2 = det.box
        items.append(
            {
                "track_id": track.track_id,
                "class": class_name,
                "score": round(det.score, 4),
                "bbox": [
                    round((x1 + x2) / 2, 6),
                    round((y1 + y2) / 2, 6),
                    round(x2 - x1, 6),
                    round(y2 - y1, 6),
                ],
            }
        )
    return items
