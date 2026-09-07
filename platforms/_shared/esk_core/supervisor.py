"""N workers in one process: the bookkeeping every platform repeats.

A platform's ``Detector`` mixes this in and supplies four things — a config, a
:class:`~esk_core.streams.StreamBackend`, a tracker factory, and a way to
publish MQTT. In exchange it gets the whole runtime-control surface: add and
remove a stream, retune one stream's threshold, route a snapshot request to the
right stream, and report every stream in the status message.

The three ``_cmd_*`` methods are the ones worth reading, because each holds one
promise the contract makes to a console:

* ``add_stream`` does not ack until the source has actually opened. Acking on
  receipt is faster and leaves a permanently grey tile on the wall with nothing
  to explain it.
* every change is written back to the config file, and the ack says whether it
  was. A threshold that reverts on the next restart is worse than one that could
  not be moved, because nothing watches for a setting quietly undoing itself.
* a stream that fails to open is removed again rather than left behind in
  ``reconnecting`` forever.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from .control import CommandError, ControlHandler
from .streams import OPEN_TIMEOUT_S, StreamConfig, StreamWorker, now_ms

LOG = logging.getLogger("esk.core.supervisor")

DETECTION_SCHEMA = "sensecraft.detection/1"
STATUS_SCHEMA = "sensecraft.status/1"


class StreamSupervisor:
    """Mixin. The host class supplies ``cfg``, ``client``, ``backend``,
    ``session_id``, ``make_tracker()`` and ``health()``."""

    # -- set up ----------------------------------------------------------
    def init_streams(self) -> None:
        self._workers: dict[str, StreamWorker] = {}
        self._workers_lock = threading.Lock()
        self.control = ControlHandler(
            self.cfg.device_id,
            set_threshold=self._cmd_set_threshold,
            add_stream=self._cmd_add_stream,
            remove_stream=self._cmd_remove_stream,
        )
        for entry in self.cfg.stream_configs():
            self._create_worker(entry)

    # -- worker registry -------------------------------------------------
    def stream_ids(self) -> list[str]:
        with self._workers_lock:
            return list(self._workers)

    def worker(self, stream_id: str) -> StreamWorker | None:
        with self._workers_lock:
            return self._workers.get(stream_id)

    def workers(self) -> list[StreamWorker]:
        with self._workers_lock:
            return list(self._workers.values())

    def frame_for(self, stream_id: str) -> Any:
        worker = self.worker(stream_id)
        return None if worker is None else worker.get_frame()

    def _create_worker(self, entry: dict[str, Any]) -> StreamWorker:
        known = {"stream_id", "source", "name", "rtsp_transport", "conf_threshold"}
        config = StreamConfig(
            stream_id=str(entry["stream_id"]),
            source=str(entry["source"]),
            name=str(entry.get("name") or ""),
            rtsp_transport=str(entry.get("rtsp_transport") or self.cfg.rtsp_transport),
            conf_threshold=entry.get("conf_threshold"),
            extra={k: v for k, v in entry.items() if k not in known},
        )
        worker = StreamWorker(
            config=config,
            conf_threshold=float(
                config.conf_threshold
                if config.conf_threshold is not None
                else self.cfg.conf_threshold
            ),
            backend=self.backend,
            publish=self._publish_detections,
            tracker=self.make_tracker(),
            decode_primary=self.cfg.decode_primary,
        )
        with self._workers_lock:
            self._workers[worker.stream_id] = worker
        return worker

    def start_streams(self) -> None:
        for worker in self.workers():
            worker.start()

    def stop_streams(self) -> None:
        for worker in self.workers():
            worker.stop()

    def published(self) -> int:
        return sum(w.published for w in self.workers())

    def measured_fps(self, stream_id: str | None = None) -> float:
        """Per-stream when named, the sum otherwise.

        The detection payload carries its own stream's rate: an aggregate there
        would report a number no single camera is running at, and the hub's
        per-stream fps column would be wrong by the number of cameras.
        """
        if stream_id is not None:
            worker = self.worker(stream_id)
            return worker.measured_fps() if worker else 0.0
        return sum(w.measured_fps() for w in self.workers())

    # -- publish ---------------------------------------------------------
    def _publish_detections(self, stream_id: str, payload: dict[str, Any]) -> None:
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

    def stream_status_entries(self) -> list[dict[str, Any]]:
        entries = []
        for worker in self.workers():
            entry = worker.status_entry()
            if self.cfg.preview_enabled and self.cfg.preview_advertise_host:
                base = f"http://{self.cfg.preview_advertise_host}:{self.cfg.preview_port}"
                entry["preview_url"] = f"{base}/preview/{worker.stream_id}.jpg"
                # The refreshing-still page served by the same server. Without
                # it the workbench "live view" button and the video wall tile
                # have no target and stay dead.
                entry["live_url"] = f"{base}/live/{worker.stream_id}"
            entries.append(entry)
        return entries

    def _persist_streams(self) -> bool:
        return self.cfg.persist_streams([w.config.to_dict() for w in self.workers()])

    # -- control ---------------------------------------------------------
    def handle_control(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Apply one command and publish its ack. Returns the ack for tests."""
        ack = self.control.handle(request, self.session_id, now_ms())
        if ack is None:
            return None
        self.client.publish(self.topic_cmd_ack, json.dumps(ack), qos=1, retain=False)
        return ack

    def _cmd_set_threshold(self, stream_id: str, value: float) -> dict[str, Any]:
        worker = self.worker(stream_id)
        if worker is None:
            raise CommandError(f"no such stream: {stream_id}")
        # In force from the next frame: the capture loop reads the attribute
        # each time round, so nothing is restarted or re-entered.
        worker.conf_threshold = value
        worker.config.conf_threshold = value
        persisted = self._persist_streams()
        self.publish_status()
        return {"stream_id": stream_id, "conf_threshold": round(value, 4),
                "persisted": persisted}

    def _cmd_add_stream(self, entry: dict[str, Any]) -> dict[str, Any]:
        stream_id = entry["stream_id"]
        if self.worker(stream_id) is not None:
            raise CommandError(f"stream {stream_id} already exists")
        if self.max_streams and len(self.stream_ids()) >= self.max_streams:
            # Refusing is kinder than accepting and degrading every existing
            # stream: the boards have a measured knee, and the operator can act
            # on a number but not on "everything got slower".
            raise CommandError(
                f"this detector is configured for at most {self.max_streams} streams"
            )
        worker = self._create_worker(entry)
        worker.start()
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
            "decode": worker.decode,
            "conf_threshold": round(worker.conf_threshold, 4),
            "persisted": persisted,
        }

    def _cmd_remove_stream(self, stream_id: str) -> dict[str, Any]:
        with self._workers_lock:
            worker = self._workers.pop(stream_id, None)
        if worker is None:
            raise CommandError(f"no such stream: {stream_id}")
        worker.stop()
        persisted = self._persist_streams()
        self.publish_status()
        return {"stream_id": stream_id, "removed": True, "persisted": persisted}

    # -- status ----------------------------------------------------------
    def status_payload(self, online: bool = True) -> dict[str, Any]:
        """The retained status. ``online=False`` is the goodbye message.

        A clean shutdown sends a DISCONNECT, so the broker never delivers the
        LWT — the retained status is the only thing a consumer will ever see
        again. Publishing it with ``online: true`` leaves the device advertised
        as online forever. The goodbye mirrors the LWT shape from
        contracts/MQTT.md (``online: false``, no ``streams``) so the hub takes
        one path for a clean exit and for a yanked cable.
        """
        payload: dict[str, Any] = {
            "schema": STATUS_SCHEMA,
            "timestamp": now_ms(),
            "session_id": self.session_id,
            "device_id": self.cfg.device_id,
            "online": online,
        }
        if online:
            payload["streams"] = self.stream_status_entries()
            payload["health"] = self.health()
            payload["versions"] = {"app": self.cfg.app_version, "model": self.model_id}
            payload["uptime_s"] = round(time.monotonic() - self.started_at, 1)
        return payload

    def publish_status(self, online: bool = True) -> dict[str, Any]:
        payload = self.status_payload(online)
        self.client.publish(self.topic_status, json.dumps(payload), qos=1, retain=True)
        return payload
