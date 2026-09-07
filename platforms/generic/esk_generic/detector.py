"""Generic CPU detector: N × RTSP -> ONNX person detection -> MQTT detections.

Publishes ``sensecraft.detection/1`` and ``sensecraft.status/1`` exactly as
specified in ``contracts/MQTT.md``, answers ``cmd/snapshot`` with raw JPEG,
answers ``cmd/control`` with ``sensecraft.ack/1``, and serves the latest frame
of every stream over HTTP for the hub's rule canvas and video wall.

``Detector`` is a supervisor. Everything genuinely shared lives here — the MQTT
client, the ONNX session, the status heartbeat, the preview server — and one
:class:`~esk_generic.streams.StreamWorker` per camera owns the rest. That split
is what lets ``add_stream`` attach a camera to a running process instead of the
operator editing a file and restarting.
"""

from __future__ import annotations

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
from .control import ControlHandler, CommandError
from .letterbox import frame_norm_to_pixels
from .preview import SNAPSHOT_MAX_BYTES, encode_jpeg, start_preview_server
from .streams import OPEN_TIMEOUT_S, StreamConfig, StreamWorker
from .tracker import IoUTracker
from .yolo import PERSON_CLASS_NAME, PersonDetector

LOG = logging.getLogger("esk.generic")

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
        self.decode_path = "sw"  # OpenCV/FFmpeg on CPU
        self.stop_event = threading.Event()
        self.snapshot_count = 0

        self.model = PersonDetector(
            cfg.model,
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
            providers=cfg.providers,
            intra_threads=cfg.intra_threads,
        )
        self.model_id = model_identifier(cfg.model)
        # One session, many streams: the generic README documents what a second
        # ORT thread pool costs on a shared box. The lock keeps two capture
        # threads out of one session at once.
        self._infer_lock = threading.Lock()

        self._workers: dict[str, StreamWorker] = {}
        self._workers_lock = threading.Lock()

        base = f"{cfg.topic_prefix}/{cfg.device_id}"
        self.topic_base = base
        self.topic_status = f"{base}/status"
        self.topic_snapshot = f"{base}/snapshot"
        self.topic_cmd_snapshot = f"{base}/cmd/snapshot"
        self.topic_cmd_control = f"{base}/cmd/control"
        self.topic_cmd_ack = f"{base}/cmd/ack"

        self.control = ControlHandler(
            cfg.device_id,
            set_threshold=self._cmd_set_threshold,
            add_stream=self._cmd_add_stream,
            remove_stream=self._cmd_remove_stream,
        )

        self.client = self._build_client()
        self.preview_server = None
        if cfg.preview_enabled:
            self.preview_server = start_preview_server(
                self._store_for, self.stream_ids, cfg.preview_bind, cfg.preview_port
            )
            LOG.info(
                "preview endpoint on http://%s:%d/preview/<stream_id>.jpg "
                "(live page at /live/<stream_id>)",
                cfg.preview_bind,
                cfg.preview_port,
            )

        for entry in cfg.stream_configs():
            self._create_worker(entry)

    # ------------------------------------------------------------- streams

    def stream_ids(self) -> list[str]:
        with self._workers_lock:
            return list(self._workers)

    def _store_for(self, stream_id: str):
        with self._workers_lock:
            worker = self._workers.get(stream_id)
        return worker.store if worker is not None else None

    def _create_worker(self, entry: dict) -> StreamWorker:
        config = StreamConfig(
            stream_id=str(entry["stream_id"]),
            source=str(entry["source"]),
            name=str(entry.get("name") or ""),
            rtsp_transport=str(entry.get("rtsp_transport") or self.cfg.rtsp_transport),
            conf_threshold=entry.get("conf_threshold"),
        )
        worker = StreamWorker(
            config=config,
            conf_threshold=float(
                config.conf_threshold
                if config.conf_threshold is not None
                else self.cfg.conf_threshold
            ),
            infer=self._infer,
            publish=self._publish_detections,
            tracker=IoUTracker(self.cfg.track_iou_threshold, self.cfg.track_max_lost_s),
            decode_primary=self.cfg.decode_primary,
        )
        with self._workers_lock:
            self._workers[worker.stream_id] = worker
        return worker

    def _infer(self, frame: np.ndarray, conf_threshold: float):
        blob, transform = self.model.preprocess(frame)
        with self._infer_lock:
            raw = self.model.infer(blob)
            inference_ms = self.model.last_inference_ms
        return self.model.postprocess(raw, transform, conf_threshold), inference_ms

    def _publish_detections(self, stream_id: str, payload: dict) -> None:
        message = {
            "schema": DETECTION_SCHEMA,
            "session_id": self.session_id,
            "device_id": self.cfg.device_id,
            **payload,
            "health": self.health(stream_id),
        }
        self.client.publish(
            f"{self.topic_base}/detections/{stream_id}",
            json.dumps(message),
            qos=0,
            retain=False,
        )

    def _persist_streams(self) -> bool:
        with self._workers_lock:
            entries = [w.config.to_dict() for w in self._workers.values()]
        return self.cfg.persist_streams(entries)

    # ------------------------------------------------------------- control

    def _cmd_set_threshold(self, stream_id: str, value: float) -> dict:
        with self._workers_lock:
            worker = self._workers.get(stream_id)
        if worker is None:
            raise CommandError(f"no such stream: {stream_id}")
        # In force from the next frame: the capture loop reads the attribute
        # each time round, so nothing has to be restarted or re-entered.
        worker.conf_threshold = value
        worker.config.conf_threshold = value
        persisted = self._persist_streams()
        self.publish_status()
        return {"stream_id": stream_id, "conf_threshold": round(value, 4),
                "persisted": persisted}

    def _cmd_add_stream(self, entry: dict) -> dict:
        stream_id = entry["stream_id"]
        with self._workers_lock:
            if stream_id in self._workers:
                raise CommandError(f"stream {stream_id} already exists")
        worker = self._create_worker(entry)
        worker.start()
        # The contract says ok:true means live, so the ack waits for the source
        # to actually open. A stream that answered "added" and then sat in
        # reconnecting forever would put a permanently grey tile on the wall
        # with nothing to explain it.
        if not worker.wait_until_open(OPEN_TIMEOUT_S):
            worker.stop()
            with self._workers_lock:
                self._workers.pop(stream_id, None)
            raise CommandError(
                worker.open_error
                or f"source did not open within {OPEN_TIMEOUT_S:g}s: {entry['source']}"
            )
        persisted = self._persist_streams()
        self.publish_status()
        return {
            "stream_id": stream_id,
            "source": entry["source"],
            "state": worker.state,
            "conf_threshold": round(worker.conf_threshold, 4),
            "persisted": persisted,
        }

    def _cmd_remove_stream(self, stream_id: str) -> dict:
        with self._workers_lock:
            worker = self._workers.pop(stream_id, None)
        if worker is None:
            raise CommandError(f"no such stream: {stream_id}")
        worker.stop()
        persisted = self._persist_streams()
        self.publish_status()
        return {"stream_id": stream_id, "removed": True, "persisted": persisted}

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

    def handle_control(self, request: dict) -> dict | None:
        """Apply one command and publish its ack. Returns the ack for tests."""
        ack = self.control.handle(request, self.session_id, now_ms())
        if ack is None:
            return None
        self.client.publish(self.topic_cmd_ack, json.dumps(ack), qos=1, retain=False)
        return ack

    def publish_snapshot(self, event_id: str, stream_id: str | None = None) -> bool:
        ids = self.stream_ids()
        target = str(stream_id) if stream_id else (ids[0] if ids else "")
        store = self._store_for(target)
        frame = store.get() if store is not None else None
        if frame is None:
            LOG.warning("snapshot %s requested before the first frame of %s",
                        event_id, target)
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

    def measured_fps(self, stream_id: str | None = None) -> float:
        with self._workers_lock:
            workers = list(self._workers.values())
        if stream_id is not None:
            worker = next((w for w in workers if w.stream_id == stream_id), None)
            return worker.measured_fps() if worker else 0.0
        return sum(w.measured_fps() for w in workers)

    def health(self, stream_id: str | None = None) -> dict:
        """Per-stream when a stream is named, aggregate otherwise.

        The detection payload carries its own stream's rate: an aggregate there
        would report a number no single camera is running at, and the hub's
        per-stream fps column would be wrong by the number of cameras.
        """
        return {
            "fps": round(self.measured_fps(stream_id), 2),
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
            with self._workers_lock:
                workers = list(self._workers.values())
            streams = []
            for worker in workers:
                entry = worker.status_entry()
                if self.cfg.preview_enabled and self.cfg.preview_advertise_host:
                    base = (
                        f"http://{self.cfg.preview_advertise_host}:"
                        f"{self.cfg.preview_port}"
                    )
                    entry["preview_url"] = f"{base}/preview/{worker.stream_id}.jpg"
                    # The refreshing-still page served by the same server. Without
                    # it the workbench "live view" button and the video wall tile
                    # have no target and stay dead.
                    entry["live_url"] = f"{base}/live/{worker.stream_id}"
                streams.append(entry)
            payload["streams"] = streams
            payload["health"] = self.health()
            payload["versions"] = {"app": self.cfg.app_version, "model": self.model_id}
            payload["uptime_s"] = round(time.monotonic() - self.started_at, 1)
        self.client.publish(self.topic_status, json.dumps(payload), qos=1, retain=True)
        return payload

    # ------------------------------------------------------------ pipeline

    def run(self, max_frames: int = 0, max_seconds: float = 0.0, annotate: str = "") -> int:
        self.client.connect(self.cfg.mqtt_host, self.cfg.mqtt_port, self.cfg.mqtt_keepalive)
        self.client.loop_start()

        with self._workers_lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.start()

        deadline = time.monotonic() + max_seconds if max_seconds > 0 else float("inf")
        last_status = time.monotonic()
        annotated_written = False
        try:
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                self.stop_event.wait(0.1)
                if time.monotonic() - last_status >= self.cfg.status_interval_s:
                    self.publish_status()
                    last_status = time.monotonic()
                if annotate and not annotated_written:
                    annotated_written = self._write_first_annotated(annotate)
                if max_frames and self.published() >= max_frames:
                    break
        finally:
            with self._workers_lock:
                workers = list(self._workers.values())
            for worker in workers:
                worker.stop()
            # online=False: this is a clean exit, so the LWT will not be
            # delivered and this retained payload is the last word.
            self.publish_status(online=False)
            self.client.disconnect()
            self.client.loop_stop()
            if self.preview_server is not None:
                self.preview_server.shutdown()
        published = self.published()
        LOG.info(
            "published %d detection messages across %d stream(s), measured %.2f fps",
            published,
            len(workers),
            self.measured_fps(),
        )
        return published

    def published(self) -> int:
        with self._workers_lock:
            return sum(w.published for w in self._workers.values())

    def _write_first_annotated(self, path: str) -> bool:
        with self._workers_lock:
            workers = list(self._workers.values())
        for worker in workers:
            frame = worker.store.get()
            if frame is None:
                continue
            detections, _ = self._infer(frame, worker.conf_threshold)
            if not detections:
                continue
            payload = {
                "frame_id": worker.frame_id,
                "detections": [
                    {
                        "track_id": 0,
                        "class": PERSON_CLASS_NAME,
                        "score": round(det.score, 4),
                        "bbox": [
                            (det.box[0] + det.box[2]) / 2,
                            (det.box[1] + det.box[3]) / 2,
                            det.box[2] - det.box[0],
                            det.box[3] - det.box[1],
                        ],
                    }
                    for det in detections
                ],
            }
            write_annotated(frame, payload, path)
            return True
        return False


def write_annotated(frame: np.ndarray, payload: dict, path: str) -> None:
    """Draw boxes decoded from the PUBLISHED frame_norm bbox values.

    Reconstructing from the payload rather than from internal pixel boxes is
    what makes the image evidence: a wrong letterbox inverse shows up as
    squashed or offset rectangles.
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
        canvas, f"{width}x{height} frame_norm frame_id={payload['frame_id']}",
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
