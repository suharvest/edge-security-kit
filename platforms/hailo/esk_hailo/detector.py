"""Hailo detector: FFmpeg decode -> Hailo-8 NPU detection -> MQTT detections.

Publishes ``sensecraft.detection/1`` and ``sensecraft.status/1`` byte-for-byte
compatible with the generic reference implementation, answers ``cmd/snapshot``
with raw JPEG, and serves the latest frame over HTTP for the hub's rule canvas.
The platform-visible differences are ``health.backend: hailort-<version>-<arch>``
and an ``inference_time_ms`` that measures the accelerator call alone.

``health.decode`` is ``sw`` here and says so. The Raspberry Pi 5 has no H.264
hardware decoder -- VideoCore VII decodes HEVC only -- so FFmpeg runs on the
Cortex-A76 cores and there is no hardware path to fall back from. That makes
``sw`` the configured primary rather than a degradation, which is why
``fallback_active`` stays false while still reporting the truth about where
decode runs. Decode is also the pipeline's cost centre on this board: the NPU
finishes long before the next frame is ready.
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

import cv2
import numpy as np
import paho.mqtt.client as mqtt

from .config import Config
from .letterbox import frame_norm_to_pixels
from .preview import SNAPSHOT_MAX_BYTES, FrameStore, encode_jpeg, start_preview_server
from .tracker import IoUTracker
from .hailo_yolo import PERSON_CLASS_NAME, HailoPersonDetector

LOG = logging.getLogger("esk.hailo")

DETECTION_SCHEMA = "sensecraft.detection/1"
STATUS_SCHEMA = "sensecraft.status/1"


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


class Detector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.session_id = str(now_ms())
        self.started_at = time.monotonic()
        self.frame_id = 0
        self.stream_state = "stopped"
        # FFmpeg on the CPU. Not a fallback on this board -- see the module
        # docstring -- so decode_primary is "sw" too and fallback_active is false.
        self.decode_path = "sw"
        self.frame_times: collections.deque[float] = collections.deque(maxlen=60)
        self.inference_times: collections.deque[float] = collections.deque(maxlen=200)
        self.pipeline_times: collections.deque[float] = collections.deque(maxlen=200)
        self.store = FrameStore()
        self.stop_event = threading.Event()
        self.snapshot_count = 0

        self.model = HailoPersonDetector(
            cfg.model,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            input_size=cfg.input_size,
            timeout_ms=cfg.infer_timeout_ms,
        )
        self.model_id = model_identifier(cfg.model)
        self.tracker = IoUTracker(cfg.track_iou_threshold, cfg.track_max_lost_s)

        base = f"{cfg.topic_prefix}/{cfg.device_id}"
        self.topic_detections = f"{base}/detections/{cfg.stream_id}"
        self.topic_status = f"{base}/status"
        self.topic_snapshot = f"{base}/snapshot"
        self.topic_cmd_snapshot = f"{base}/cmd/snapshot"

        self.client = self._build_client()
        self.preview_server = None
        if cfg.preview_enabled:
            self.preview_server = start_preview_server(
                self.store, cfg.stream_id, cfg.preview_bind, cfg.preview_port
            )
            LOG.info(
                "preview endpoint on http://%s:%d/preview.jpg (live page at /live)",
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

    def _on_connect(self, client, _userdata, _flags, rc) -> None:
        if rc != 0:
            LOG.error("mqtt connect failed rc=%s", rc)
            return
        LOG.info("mqtt connected, subscribing %s", self.topic_cmd_snapshot)
        client.subscribe(self.topic_cmd_snapshot, qos=1)
        self.publish_status()

    def _on_message(self, _client, _userdata, msg: mqtt.MQTTMessage) -> None:
        if msg.topic != self.topic_cmd_snapshot:
            return
        try:
            request = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            LOG.warning("bad cmd/snapshot payload: %s", exc)
            return
        event_id = request.get("event_id")
        stream_id = request.get("stream_id")
        if not event_id:
            LOG.warning("cmd/snapshot without event_id, ignored")
            return
        if stream_id and stream_id != self.cfg.stream_id:
            return
        self.publish_snapshot(str(event_id))

    def publish_snapshot(self, event_id: str) -> bool:
        frame = self.store.get()
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

    def measured_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return 0.0 if span <= 0 else (len(self.frame_times) - 1) / span

    def health(self) -> dict:
        return {
            "fps": round(self.measured_fps(), 2),
            "decode": self.decode_path,
            "backend": self.model.backend,
            "fallback_active": self.decode_path != self.cfg.decode_primary,
        }

    def publish_status(self, online: bool = True) -> dict:
        """Publish the retained status. ``online=False`` is the goodbye message.

        A clean shutdown sends a DISCONNECT, so the broker never delivers the
        LWT — the retained status is the only thing a consumer will ever see
        again. Publishing it with ``online: true`` therefore leaves the device
        advertised as online forever. The goodbye payload mirrors the LWT shape
        from contracts/MQTT.md (``online: false``, no ``streams`` array) so the
        hub takes the same path for a clean exit and for a yanked cable.
        """
        payload: dict = {
            "schema": STATUS_SCHEMA,
            "timestamp": now_ms(),
            "session_id": self.session_id,
            "device_id": self.cfg.device_id,
            "online": online,
        }
        if online:
            stream: dict = {
                "stream_id": self.cfg.stream_id,
                "state": self.stream_state,
                "fps": round(self.measured_fps(), 2),
                "decode": self.decode_path,
            }
            if self.cfg.preview_enabled and self.cfg.preview_advertise_host:
                base = (
                    f"http://{self.cfg.preview_advertise_host}:{self.cfg.preview_port}"
                )
                stream["preview_url"] = f"{base}/preview/{self.cfg.stream_id}.jpg"
                # The refreshing-still page served by the same server. Without it
                # the workbench "live view" button has no target and stays dead.
                stream["live_url"] = f"{base}/live/{self.cfg.stream_id}"
            payload["streams"] = [stream]
            payload["health"] = self.health()
            payload["versions"] = {"app": self.cfg.app_version, "model": self.model_id}
            payload["uptime_s"] = round(time.monotonic() - self.started_at, 1)
        self.client.publish(self.topic_status, json.dumps(payload), qos=1, retain=True)
        return payload

    # ------------------------------------------------------------ pipeline

    def open_capture(self) -> cv2.VideoCapture:
        if self.cfg.source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.cfg.rtsp_transport}"
            )
        capture = cv2.VideoCapture(self.cfg.source, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def build_detection_payload(self, frame: np.ndarray, captured_at: float) -> dict:
        canvas, transform = self.model.preprocess(frame)
        detections = self.model.detect(canvas, transform)
        self.inference_times.append(self.model.last_inference_ms)
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
        pipeline_ms = (time.monotonic() - captured_at) * 1000.0
        self.pipeline_times.append(pipeline_ms)
        return {
            "schema": DETECTION_SCHEMA,
            "timestamp": now_ms(),
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "device_id": self.cfg.device_id,
            "stream_id": self.cfg.stream_id,
            "coordinate_space": "frame_norm",
            "frame": {"w": int(width), "h": int(height)},
            # The Hailo inference call alone, which is what the contract asks
            # for: decode and the NumPy head decode are not included.
            "inference_time_ms": round(self.model.last_inference_ms, 3),
            "pipeline_ms": round(pipeline_ms, 3),
            "detections": items,
            "health": self.health(),
        }

    def run(self, max_frames: int = 0, max_seconds: float = 0.0, annotate: str = "") -> int:
        self.client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, self.cfg.mqtt_keepalive)
        self.client.loop_start()

        deadline = time.monotonic() + max_seconds if max_seconds > 0 else float("inf")
        last_status = 0.0
        backoff = 1.0
        capture: cv2.VideoCapture | None = None
        published = 0
        annotated_written = False
        try:
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()
                    self.stream_state = "reconnecting"
                    self.publish_status()
                    LOG.info("opening source %s", self.cfg.source)
                    capture = self.open_capture()
                    if not capture.isOpened():
                        LOG.warning("source unavailable, retrying in %.1fs", backoff)
                        self.stop_event.wait(backoff)
                        backoff = min(backoff * 2, 15.0)
                        continue
                    backoff = 1.0
                    self.stream_state = "running"
                    self.publish_status()
                    last_status = time.monotonic()

                ok, frame = capture.read()
                if not ok or frame is None:
                    LOG.warning("frame read failed, reopening source")
                    capture.release()
                    capture = None
                    continue
                # Clock starts here, not before the read: read() blocks until
                # the next frame exists, so timing from before it would fold
                # the idle wait for a 5 fps source into pipeline_ms and report
                # a steady ~200 ms regardless of how fast the board is. That
                # number is load-bearing -- pipeline_ms exceeding the frame
                # period is the documented signal that the tracker is being
                # starved -- so inflating it disables the diagnostic. Same
                # placement as platforms/rknn, so the two are comparable.
                captured_at = time.monotonic()

                self.store.put(frame)
                self.frame_times.append(time.monotonic())
                payload = self.build_detection_payload(frame, captured_at)
                self.client.publish(
                    self.topic_detections, json.dumps(payload), qos=0, retain=False
                )
                published += 1

                if annotate and payload["detections"] and not annotated_written:
                    write_annotated(frame, payload, annotate)
                    annotated_written = True

                if time.monotonic() - last_status >= self.cfg.status_interval_s:
                    self.publish_status()
                    last_status = time.monotonic()

                if max_frames and published >= max_frames:
                    break
        finally:
            if capture is not None:
                capture.release()
            self.stream_state = "stopped"
            # online=False: this is a clean exit, so the LWT will not be
            # delivered and this retained payload is the last word.
            self.publish_status(online=False)
            self.client.disconnect()
            self.client.loop_stop()
            self.model.close()
            if self.preview_server is not None:
                self.preview_server.shutdown()
        inf = sorted(self.inference_times)
        pipe = sorted(self.pipeline_times)

        def pct(series, q):
            return series[min(len(series) - 1, int(q * (len(series) - 1)))] if series else 0.0

        LOG.info(
            "published %d detection messages, %.2f fps, inference p50=%.2fms "
            "p95=%.2fms, pipeline p50=%.2fms p95=%.2fms",
            published,
            self.measured_fps(),
            pct(inf, 0.5), pct(inf, 0.95), pct(pipe, 0.5), pct(pipe, 0.95),
        )
        return published


def write_annotated(frame: np.ndarray, payload: dict, path: str) -> None:
    """Draw boxes decoded from the PUBLISHED frame_norm bbox values.

    Reconstructing from the payload rather than from internal pixel boxes is
    what makes the image evidence: a wrong letterbox inverse shows up as
    squashed or offset rectangles rather than as a silently plausible number.
    Drawn on the original frame at its own aspect ratio, so a 1280x720 source
    would visibly betray a square-stretch bug.
    """
    canvas = frame.copy()
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
        f"{width}x{height} frame_norm decode={payload['health']['decode']} "
        f"backend={payload['health']['backend']} frame_id={payload['frame_id']}",
        (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA,
    )
    cv2.imwrite(path, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    LOG.info("annotated evidence frame written to %s", path)


def install_signal_handlers(detector: Detector) -> None:
    def handler(_signum, _frame):
        LOG.info("stop requested")
        detector.stop_event.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
