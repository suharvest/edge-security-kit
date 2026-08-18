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

from .config import Config
from .letterbox import frame_norm_to_pixels
from .preview import SNAPSHOT_MAX_BYTES, FrameStore, encode_jpeg, start_preview_server
from .tracker import IoUTracker
from .trt_yolo import PERSON_CLASS_NAME, TRTPersonDetector, decode_person_head
from .video_source import DECODE_SW, HardwareDecodeUnavailable, open_source

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


class Detector:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.session_id = str(now_ms())
        self.started_at = time.monotonic()
        self.frame_id = 0
        self.stream_state = "stopped"
        self.decode_path = cfg.decode_primary
        self.frame_times: collections.deque[float] = collections.deque(maxlen=60)
        self.inference_times: collections.deque[float] = collections.deque(maxlen=2000)
        self.stages = StageTimings()
        self.store = FrameStore()
        self.stop_event = threading.Event()
        self.snapshot_count = 0
        self.source = None

        self.model = TRTPersonDetector(
            cfg.model,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            input_size=cfg.input_size,
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
                self.store,
                cfg.stream_id,
                cfg.preview_bind,
                cfg.preview_port,
                debug_source=lambda: self.source,
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
            # True whenever either configured-primary path is on its fallback.
            # Decode is the only one that can fall back here: there is no CPU
            # inference path, the detector refuses to start without the engine.
            "fallback_active": self.decode_path != self.cfg.decode_primary,
        }

    def publish_status(self, online: bool = True) -> dict:
        """Publish the retained status. ``online=False`` is the goodbye message.

        A clean shutdown sends a DISCONNECT, so the broker discards the LWT and
        the retained status is the last word any consumer will ever see.
        Leaving ``online: true`` there advertises a stopped device as running
        forever, so the goodbye mirrors the LWT shape exactly.
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
                base = f"http://{self.cfg.preview_advertise_host}:{self.cfg.preview_port}"
                stream["preview_url"] = f"{base}/preview/{self.cfg.stream_id}.jpg"
                stream["live_url"] = f"{base}/live/{self.cfg.stream_id}"
            payload["streams"] = [stream]
            payload["health"] = self.health()
            payload["versions"] = {"app": self.cfg.app_version, "model": self.model_id}
            payload["uptime_s"] = round(time.monotonic() - self.started_at, 1)
        self.client.publish(self.topic_status, json.dumps(payload), qos=1, retain=True)
        return payload

    # ------------------------------------------------------------ pipeline

    def open_capture(self):
        source = open_source(
            self.cfg.source,
            size=self.cfg.input_size,
            transport=self.cfg.rtsp_transport,
            codec=self.cfg.rtsp_codec,
            require_hw=self.cfg.require_hw_decode,
            latency_ms=self.cfg.rtsp_latency_ms,
            appsink_timeout_ms=self.cfg.appsink_timeout_ms,
            appsink_queue=self.cfg.appsink_queue,
        )
        self.decode_path = source.decode_path
        if self.decode_path == DECODE_SW:
            LOG.warning("running on the CPU decoder; health.decode=sw, fallback_active=true")
        return source

    def build_detection_payload(self, bundle, captured_at: float) -> dict:
        # The three model stages are driven separately rather than through
        # model.detect(), so the timing split is measured rather than guessed.
        started = time.perf_counter()
        self.model.preprocess_into(bundle.canvas)
        self.stages.add("preprocess", (time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        raw = self.model.infer()
        self.stages.add("infer", (time.perf_counter() - started) * 1000.0)
        self.inference_times.append(self.model.last_inference_ms)

        started = time.perf_counter()
        detections = decode_person_head(
            np.asarray(raw, dtype=np.float32),
            bundle.transform,
            self.cfg.conf_threshold,
            self.cfg.iou_threshold,
        )
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
        self.stages.add("postprocess", (time.perf_counter() - started) * 1000.0)

        self.frame_id += 1
        tf = bundle.transform
        return {
            "schema": DETECTION_SCHEMA,
            "timestamp": now_ms(),
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "device_id": self.cfg.device_id,
            "stream_id": self.cfg.stream_id,
            "coordinate_space": "frame_norm",
            "frame": {"w": int(tf.src_w), "h": int(tf.src_h)},
            # The TensorRT execute_async_v3 call alone, timed with CUDA events
            # on the device clock, which is what the contract asks for. Decode
            # runs on NVDEC and preprocessing on the CPU; neither is included.
            "inference_time_ms": round(self.model.last_inference_ms, 3),
            "pipeline_ms": round((time.monotonic() - captured_at) * 1000.0, 3),
            "detections": items,
            "health": self.health(),
        }

    def run(self, max_frames: int = 0, max_seconds: float = 0.0, annotate: str = "") -> int:
        self.client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, self.cfg.mqtt_keepalive)
        self.client.loop_start()

        deadline = time.monotonic() + max_seconds if max_seconds > 0 else float("inf")
        last_status = 0.0
        backoff = 1.0
        published = 0
        annotated_written = False
        try:
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                if self.source is None:
                    self.stream_state = "reconnecting"
                    self.publish_status()
                    LOG.info("opening source %s", self.cfg.source)
                    try:
                        self.source = self.open_capture()
                    except HardwareDecodeUnavailable:
                        # Configuration fault, not a transient one: retrying
                        # cannot install a GStreamer plugin.
                        raise
                    except Exception as exc:
                        LOG.warning("source unavailable (%s), retrying in %.1fs", exc, backoff)
                        self.source = None
                        self.stop_event.wait(backoff)
                        backoff = min(backoff * 2, 15.0)
                        continue
                    backoff = 1.0
                    self.stream_state = "running"
                    self.publish_status()
                    last_status = time.monotonic()

                try:
                    bundle = self.source.read()
                except Exception as exc:
                    LOG.warning("read failed (%s), reopening source", exc)
                    self.source.close()
                    self.source = None
                    continue
                if bundle is None:
                    # A pull timeout is normal on a stream that has paused.
                    continue
                # Clock starts here, not before the read: read() blocks until
                # the next frame exists, so timing from before it would fold the
                # idle wait for a 15 fps source into pipeline_ms and report a
                # steady ~66 ms regardless of how fast the board actually is.
                # That number is load-bearing -- pipeline_ms exceeding the frame
                # period is the documented signal that the tracker is being
                # starved and the hub's alerts are wrong on bad input -- so
                # inflating it would disable the diagnostic.
                captured_at = time.monotonic()
                self.stages.add("decode", bundle.decode_ms)
                self.stages.add("letterbox", bundle.letterbox_ms)

                self.store.put(bundle.active_rgb())
                self.frame_times.append(time.monotonic())
                payload = self.build_detection_payload(bundle, captured_at)

                started = time.perf_counter()
                self.client.publish(
                    self.topic_detections, json.dumps(payload), qos=0, retain=False
                )
                self.stages.add("publish", (time.perf_counter() - started) * 1000.0)
                self.stages.pipeline.append((time.monotonic() - captured_at) * 1000.0)
                published += 1

                if annotate and payload["detections"] and not annotated_written:
                    write_annotated(bundle, payload, annotate)
                    annotated_written = True

                if time.monotonic() - last_status >= self.cfg.status_interval_s:
                    self.publish_status()
                    last_status = time.monotonic()

                if max_frames and published >= max_frames:
                    break
        finally:
            if self.source is not None:
                self.source.close()
                self.source = None
            self.stream_state = "stopped"
            # online=False: a clean exit means the LWT is discarded, so this
            # retained payload is the last word.
            self.publish_status(online=False)
            self.client.disconnect()
            self.client.loop_stop()
            self.model.close()
            if self.preview_server is not None:
                self.preview_server.shutdown()
        inf = sorted(self.inference_times)
        LOG.info(
            "published %d detection messages, %.2f fps, inference p50=%.2fms p95=%.2fms",
            published,
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
