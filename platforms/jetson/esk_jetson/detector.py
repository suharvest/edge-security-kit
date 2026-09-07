"""Jetson detector: NVDEC decode -> TensorRT person detection -> MQTT.

Publishes ``sensecraft.detection/1`` and ``sensecraft.status/1`` byte-for-byte
compatible with the generic reference implementation, answers ``cmd/snapshot``
with raw JPEG, and serves the latest frame over HTTP for the hub's rule canvas.
The only platform-visible differences are ``health.decode: hw``,
``health.backend: tensorrt-<version>`` and the accelerator timing in
``inference_time_ms``.

The run loop keeps a per-stage timing histogram. That is not instrumentation
for its own sake: the decision about whether any part of this pipeline needs to
leave Python is supposed to be made from the measured split, not from the
assumption that Python must be the problem. ``--stages`` prints it.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import signal
import threading
import time

import numpy as np
import paho.mqtt.client as mqtt

from esk_core import StreamSupervisor

from .backend import JetsonBackend
from .config import Config
from .letterbox import frame_norm_to_pixels
from .preview import SNAPSHOT_MAX_BYTES, encode_jpeg, start_preview_server
from .tracker import IoUTracker
from .trt_yolo import PERSON_CLASS_NAME, TRTPersonDetector, decode_person_head

LOG = logging.getLogger("esk.jetson")

DETECTION_SCHEMA = "sensecraft.detection/1"
STATUS_SCHEMA = "sensecraft.status/1"

# Stage names in pipeline order. ``decode`` is the blocking appsink pull, so on
# a source slower than the board it is mostly idle wait, not decoder work --
# reported anyway, because a decode figure that starts exceeding the frame
# period is the signal that NVDEC has fallen over to something else.
STAGES = ("decode", "letterbox", "preprocess", "infer", "postprocess", "publish")


def now_ms() -> int:
    """Epoch milliseconds. The contract forbids seconds or floats."""
    return int(time.time() * 1000)


def model_identifier(path: str) -> str:
    """``<file stem>@<sha256 prefix>`` for status.versions.model."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{stem}@{digest.hexdigest()[:12]}"


class StageTimings:
    """Per-stage wall-clock samples, summarized as p50/p95."""

    def __init__(self, maxlen: int = 2000) -> None:
        self.samples: dict[str, collections.deque[float]] = {
            name: collections.deque(maxlen=maxlen) for name in STAGES
        }
        self.pipeline: collections.deque[float] = collections.deque(maxlen=maxlen)

    def add(self, name: str, milliseconds: float) -> None:
        self.samples[name].append(milliseconds)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
        return ordered[index]

    def summary(self) -> dict:
        pipeline_p50 = self._percentile(list(self.pipeline), 0.50)
        result: dict = {"pipeline_ms": {"p50": round(pipeline_p50, 3),
                                        "p95": round(self._percentile(list(self.pipeline), 0.95), 3),
                                        "n": len(self.pipeline)}}
        for name in STAGES:
            values = list(self.samples[name])
            p50 = self._percentile(values, 0.50)
            result[name] = {
                "p50": round(p50, 3),
                "p95": round(self._percentile(values, 0.95), 3),
                # Share of the pipeline budget. ``decode`` is excluded from the
                # pipeline clock (it is the wait for the frame to exist), so its
                # share is reported against the same denominator for scale but
                # will not sum with the rest to 100%.
                "pct_of_pipeline": round(100.0 * p50 / pipeline_p50, 1) if pipeline_p50 > 0 else 0.0,
                "n": len(values),
            }
        return result


class Detector(StreamSupervisor):
    """Supervisor over N streams sharing one accelerator (see backend.py)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.session_id = str(now_ms())
        self.started_at = time.monotonic()
        self.inference_times: collections.deque[float] = collections.deque(maxlen=2000)
        self.stages = StageTimings()
        self.stop_event = threading.Event()
        self.snapshot_count = 0
        self.max_streams = cfg.max_streams

        self.model = TRTPersonDetector(
            cfg.model,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            input_size=cfg.input_size,
        )
        self.model_id = model_identifier(cfg.model)

        self.backend = JetsonBackend(cfg, self.model)

        base = f"{cfg.topic_prefix}/{cfg.device_id}"
        self.topic_base = base
        self.topic_status = f"{base}/status"
        self.topic_snapshot = f"{base}/snapshot"
        self.topic_cmd_snapshot = f"{base}/cmd/snapshot"
        self.topic_cmd_control = f"{base}/cmd/control"
        self.topic_cmd_ack = f"{base}/cmd/ack"

        self.client = self._build_client()
        self.init_streams()
        self.preview_server = None
        if cfg.preview_enabled:
            self.preview_server = start_preview_server(
                self.frame_for,
                self.stream_ids,
                cfg.preview_bind,
                cfg.preview_port,
                # /debug/decode reports the first stream's source; with several
                # cameras the per-stream decode path is in the status
                # message, which is where a consumer should read it.
                debug_source=lambda: (
                    self.workers()[0].source if self.workers() else None
                ),
            )
            LOG.info(
                "preview on http://%s:%d/preview.jpg (live /live, decode evidence /debug/decode)",
                cfg.preview_bind,
                cfg.preview_port,
            )

    # ---------------------------------------------------------------- MQTT

    def _build_client(self) -> mqtt.Client:
        client = mqtt.Client(client_id=f"esk-{self.cfg.device_id}-{self.session_id}")
        if self.cfg.mqtt_username:
            client.username_pw_set(self.cfg.mqtt_username, self.cfg.mqtt_password)
        # LWT: registered at CONNECT, so its timestamp is necessarily connect
        # time; consumers use broker receipt time as the offline instant.
        client.will_set(
            self.topic_status,
            json.dumps(
                {
                    "schema": STATUS_SCHEMA,
                    "timestamp": now_ms(),
                    "session_id": self.session_id,
                    "device_id": self.cfg.device_id,
                    "online": False,
                }
            ),
            qos=1,
            retain=True,
        )
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        return client

    def make_tracker(self) -> IoUTracker:
        """A fresh tracker per stream: track_id is per-stream by contract."""
        return IoUTracker(self.cfg.track_iou_threshold, self.cfg.track_max_lost_s)

    def _on_connect(self, client, _userdata, _flags, rc) -> None:
        if rc != 0:
            LOG.error("mqtt connect failed rc=%s", rc)
            return
        LOG.info("mqtt connected, subscribing %s and %s",
                 self.topic_cmd_snapshot, self.topic_cmd_control)
        client.subscribe(self.topic_cmd_snapshot, qos=1)
        client.subscribe(self.topic_cmd_control, qos=1)
        self.publish_status()

    def _on_message(self, _client, _userdata, msg: mqtt.MQTTMessage) -> None:
        try:
            request = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOG.warning("bad %s payload: %s", msg.topic, exc)
            return
        if msg.topic == self.topic_cmd_control:
            self.handle_control(request)
            return
        if msg.topic != self.topic_cmd_snapshot:
            return
        event_id = request.get("event_id")
        stream_id = request.get("stream_id")
        if not event_id:
            LOG.warning("cmd/snapshot without event_id, ignored")
            return
        if stream_id and stream_id not in self.stream_ids():
            return
        self.publish_snapshot(str(event_id), stream_id)

    def publish_snapshot(self, event_id: str, stream_id: str | None = None) -> bool:
        ids = self.stream_ids()
        target = str(stream_id) if stream_id else (ids[0] if ids else "")
        frame = self.frame_for(target)
        if frame is None:
            LOG.warning("snapshot %s requested before the first frame", event_id)
            return False
        payload = encode_jpeg(frame, SNAPSHOT_MAX_BYTES)
        if payload is None:
            LOG.error("snapshot %s could not be squeezed under the size cap", event_id)
            return False
        self.client.publish(f"{self.topic_snapshot}/{event_id}", payload, qos=1)
        self.snapshot_count += 1
        LOG.info("snapshot %s published, %d bytes", event_id, len(payload))
        return True

    # -------------------------------------------------------------- health

    def decode_path(self, stream_id: str | None = None) -> str:
        """The decode path of one stream, or of the first for the aggregate."""
        worker = self.worker(stream_id) if stream_id else (
            self.workers()[0] if self.workers() else None
        )
        return worker.decode if worker else self.cfg.decode_primary

    def health(self, stream_id: str | None = None) -> dict:
        """Per-stream when a stream is named, aggregate otherwise.

        The detection payload carries its own stream's rate: an aggregate there
        reports a number no single camera is running at, and the hub's
        per-stream fps column would be wrong by the number of cameras.
        """
        return {
            "fps": round(self.measured_fps(stream_id), 2),
            "decode": self.decode_path(stream_id),
            "backend": self.model.backend,
            "fallback_active": (
                any(w.decode != self.cfg.decode_primary for w in self.workers())
                if stream_id is None
                else self.decode_path(stream_id) != self.cfg.decode_primary
            ),
        }

    # publish_status / status_payload come from StreamSupervisor: the retained
    # goodbye must mirror the LWT shape (online false, no streams) on every
    # platform, and a per-platform copy is a per-platform chance to leave a
    # stopped device advertised as running forever.

    def run(self, max_frames: int = 0, max_seconds: float = 0.0, annotate: str = "") -> int:
        self.client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, self.cfg.mqtt_keepalive)
        self.client.loop_start()
        self.start_streams()

        deadline = time.monotonic() + max_seconds if max_seconds > 0 else float("inf")
        last_status = time.monotonic()
        try:
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                self.stop_event.wait(0.1)
                # A stream that hit a fatal source error -- a missing decoder
                # plugin -- is not something a supervisor can retry around.
                # Surfacing it rather than leaving a silently dead thread is
                # what keeps a misconfigured deploy from looking like a slow
                # camera.
                for worker in self.workers():
                    if worker.fatal is not None:
                        raise worker.fatal
                if time.monotonic() - last_status >= self.cfg.status_interval_s:
                    self.publish_status()
                    last_status = time.monotonic()
                if max_frames and self.published() >= max_frames:
                    break
        finally:
            self.stop_streams()
            # online=False: a clean exit means the LWT is discarded, so this
            # retained payload is the last word.
            self.publish_status(online=False)
            self.client.disconnect()
            self.client.loop_stop()
            close = getattr(self.model, "close", None)
            if callable(close):
                close()
            if self.preview_server is not None:
                self.preview_server.shutdown()
        published = self.published()
        inf = sorted(t for w in self.workers() for t in w.inference_times)
        LOG.info(
            "published %d detection messages across %d stream(s), %.2f fps, "
            "inference p50=%.2fms p95=%.2fms",
            published,
            len(self.workers()),
            self.measured_fps(),
            inf[len(inf) // 2] if inf else 0.0,
            inf[int(0.95 * (len(inf) - 1))] if inf else 0.0,
        )
        return published


def write_annotated(bundle, payload: dict, path: str) -> None:
    """Draw boxes decoded from the PUBLISHED frame_norm bbox values.

    Reconstructing from the payload rather than from the internal pixel boxes is
    what makes the image evidence: a wrong letterbox inverse shows up as
    squashed or offset rectangles rather than as a silently plausible number.
    The canvas is the active region -- padding removed -- so the drawing is
    against real decoded pixels at the source aspect ratio.
    """
    import cv2

    rgb = bundle.active_rgb()
    canvas = np.ascontiguousarray(rgb[:, :, ::-1])
    height, width = canvas.shape[:2]
    for item in payload["detections"]:
        x1, y1, x2, y2 = frame_norm_to_pixels(item["bbox"], width, height)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"id{item['track_id']} {item['class']} {item['score']:.2f}"
        cv2.putText(
            canvas, label, (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"src {payload['frame']['w']}x{payload['frame']['h']} decode={payload['health']['decode']} "
        f"frame_id={payload['frame_id']}",
        (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
    )
    cv2.imwrite(path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    LOG.info("annotated evidence frame written to %s", path)


def install_signal_handlers(detector: Detector) -> None:
    def handler(_signum, _frame):
        LOG.info("stop requested")
        detector.stop_event.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
