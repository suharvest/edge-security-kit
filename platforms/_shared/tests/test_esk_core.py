"""The shared capture loop's contract, with a fake backend.

No camera, no accelerator: what is tested is the part that is the same on every
board and that a one-camera run cannot show — the per-stream state that must not
be shared, the two read outcomes that must not be confused, and the config
write-back that decides whether a console change survives a restart.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from esk_core import (  # noqa: E402
    CommandError, ControlHandler, FatalSourceError, SourceLost, StreamConfig,
    StreamWorker, config_streams, detection_items,
)


class Det:
    def __init__(self, box, score=0.9):
        self.box = box
        self.score = score


class Track:
    def __init__(self, track_id):
        self.track_id = track_id


class FakeTracker:
    """Hands out ids from 1, per instance. Mirrors IoUTracker's contract."""

    def __init__(self):
        self.next_id = 1

    def update(self, detections, _now):
        out = []
        for det in detections:
            out.append((Track(self.next_id), det))
            self.next_id += 1
        return out


class FakeBackend:
    class_name = "person"

    def __init__(self):
        self.seen = []
        self.opens = 0
        self.closes = 0
        self.script = []          # values read() yields in order
        self.open_raises = None
        self.decode = "hw"

    def open(self, config):
        self.opens += 1
        if self.open_raises:
            raise self.open_raises
        return {"config": config}

    def read(self, source):
        if not self.script:
            time.sleep(0.01)
            return None
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self, source):
        self.closes += 1

    def decode_path(self, source):
        return self.decode

    def frame_of(self, bundle):
        return bundle

    def detect(self, bundle, conf_threshold):
        self.seen.append(conf_threshold)
        return [Det([0.1, 0.1, 0.2, 0.3])], 1.5, 1280, 720


def worker(backend, stream_id="cam-0", threshold=0.35, published=None):
    return StreamWorker(
        config=StreamConfig(stream_id=stream_id, source="rtsp://x"),
        conf_threshold=threshold,
        backend=backend,
        publish=((lambda _sid, payload: published.append(payload))
                 if published is not None else (lambda *_: None)),
        tracker=FakeTracker(),
        decode_primary="hw",
    )


# -- per-stream state --------------------------------------------------------

def test_each_stream_infers_at_its_own_threshold():
    backend = FakeBackend()
    a, b = worker(backend, "cam-0", 0.35), worker(backend, "cam-1", 0.7)
    a.build_payload({}, 0.0)
    b.build_payload({}, 0.0)
    assert backend.seen == [0.35, 0.7]


def test_a_threshold_change_takes_effect_on_the_next_frame():
    backend = FakeBackend()
    w = worker(backend, threshold=0.35)
    w.build_payload({}, 0.0)
    w.conf_threshold = 0.6
    w.build_payload({}, 0.0)
    assert backend.seen == [0.35, 0.6]


def test_track_ids_and_frame_ids_are_per_stream():
    """A shared tracker would give cam-1's first person id 2, not 1."""
    backend = FakeBackend()
    a, b = worker(backend, "cam-0"), worker(backend, "cam-1")
    for _ in range(3):
        a.build_payload({}, 0.0)
    pa = a.build_payload({}, 0.0)
    pb = b.build_payload({}, 0.0)
    assert pa["frame_id"] == 4 and pb["frame_id"] == 1
    assert pa["detections"][0]["track_id"] == 4
    assert pb["detections"][0]["track_id"] == 1


def test_payload_reports_the_original_frame_size_not_the_model_input():
    payload = worker(FakeBackend()).build_payload({}, 0.0)
    assert payload["frame"] == {"w": 1280, "h": 720}
    assert payload["coordinate_space"] == "frame_norm"


def test_detection_items_convert_xyxy_to_centre_format():
    items = detection_items([(Track(3), Det([0.2, 0.1, 0.4, 0.5], 0.87654))])
    assert items[0]["bbox"] == [0.3, 0.3, 0.2, 0.4]
    assert items[0]["score"] == 0.8765
    assert items[0]["track_id"] == 3


def test_status_entry_reports_the_running_threshold_and_decode():
    w = worker(FakeBackend())
    w.conf_threshold = 0.62
    w.decode = "sw"
    entry = w.status_entry()
    assert entry["conf_threshold"] == 0.62
    assert entry["decode"] == "sw"
    assert entry["fallback_active"] is True


# -- the two read outcomes ---------------------------------------------------

def test_a_none_read_keeps_the_source_open():
    """A pull timeout on a paused 5 fps stream must not tear down the pipeline."""
    backend = FakeBackend()
    backend.script = [None, None, {"f": 1}]
    published = []
    w = worker(backend, published=published)
    w.start()
    deadline = time.monotonic() + 3
    while not published and time.monotonic() < deadline:
        time.sleep(0.02)
    w.stop()
    assert published, "the frame after two Nones was never published"
    assert backend.opens == 1, "the source was reopened over a pull timeout"


def test_source_lost_reopens():
    backend = FakeBackend()
    backend.script = [SourceLost("bus error"), {"f": 1}]
    published = []
    w = worker(backend, published=published)
    w.start()
    deadline = time.monotonic() + 3
    while not published and time.monotonic() < deadline:
        time.sleep(0.02)
    w.stop()
    assert backend.opens >= 2 and backend.closes >= 1


def test_a_fatal_source_error_is_not_retried():
    """No amount of waiting installs a decoder plugin."""
    backend = FakeBackend()
    backend.open_raises = FatalSourceError("libgstrockchipmpp.so is not installed")
    w = worker(backend)
    w.start()
    assert w.wait_until_open(2.0) is False
    w.stop()
    assert backend.opens == 1
    assert "libgstrockchipmpp" in w.open_error


def test_an_ordinary_open_failure_is_retried():
    backend = FakeBackend()
    backend.open_raises = RuntimeError("connection refused")
    w = worker(backend)
    w.start()
    assert w.wait_until_open(1.5) is False
    w.stop()
    assert backend.opens >= 2, "a camera that is rebooting deserves the backoff loop"


def test_the_pipeline_clock_starts_after_the_read():
    """Timing from before read() folds the idle wait into pipeline_ms.

    A 5 fps source would then report a steady ~200 ms regardless of how fast the
    board is, which disables pipeline_ms as the starved-tracker signal.
    """
    backend = FakeBackend()
    slow = {"f": 1}
    real_read = backend.read

    def read(source):
        time.sleep(0.15)
        return real_read(source)

    backend.read = read
    backend.script = [slow]
    published = []
    w = worker(backend, published=published)
    w.start()
    deadline = time.monotonic() + 3
    while not published and time.monotonic() < deadline:
        time.sleep(0.02)
    w.stop()
    assert published[0]["pipeline_ms"] < 100, published[0]["pipeline_ms"]


# -- config ------------------------------------------------------------------

class Cfg:
    def __init__(self, path=None, streams=None):
        self.path = path
        self.streams = streams or []
        self.stream_id = "cam-0"
        self.source = "rtsp://a"
        self.rtsp_transport = "tcp"


def test_a_single_stream_config_still_describes_one_stream():
    assert config_streams.stream_configs(Cfg()) == [
        {"stream_id": "cam-0", "source": "rtsp://a", "rtsp_transport": "tcp"}
    ]


def test_a_streams_list_wins_over_the_shorthand():
    cfg = Cfg(streams=[{"stream_id": "cam-1", "source": "rtsp://b"}])
    assert [s["stream_id"] for s in config_streams.stream_configs(cfg)] == ["cam-1"]


def test_persist_drops_the_contradicting_shorthand(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(
        {"device_id": "rk3588-01", "stream_id": "cam-0", "source": "rtsp://a",
         "conf_threshold": 0.35}))
    assert config_streams.persist_streams(Cfg(path=str(path)), [
        {"stream_id": "cam-0", "source": "rtsp://a", "conf_threshold": 0.6},
        {"stream_id": "cam-1", "source": "rtsp://b", "name": "North gate"},
    ]) is True
    written = yaml.safe_load(path.read_text())
    assert [s["stream_id"] for s in written["streams"]] == ["cam-0", "cam-1"]
    # Leaving the shorthand behind makes the file describe cam-0 twice.
    assert "source" not in written and "stream_id" not in written
    assert written["conf_threshold"] == 0.35   # the process default is untouched
    assert not (tmp_path / "config.yaml.tmp").exists()


def test_a_flag_configured_process_persists_nothing_and_says_so():
    assert config_streams.persist_streams(Cfg(), [{"stream_id": "a", "source": "b"}]) is False


# -- control -----------------------------------------------------------------

def test_a_redelivered_request_id_applies_once_and_acks_twice():
    calls = []
    handler = ControlHandler(
        "rk3588-01",
        set_threshold=lambda s, v: {"stream_id": s, "conf_threshold": v},
        add_stream=lambda e: calls.append(e) or {"stream_id": e["stream_id"]},
        remove_stream=lambda s: {"stream_id": s},
    )
    cmd = {"schema": "sensecraft.command/1", "timestamp": 1, "request_id": "hub-1",
           "device_id": "rk3588-01", "command": "add_stream",
           "params": {"stream_id": "cam-2", "source": "rtsp://x"}}
    first = handler.handle(cmd, "s1", 1)
    second = handler.handle(cmd, "s1", 2)
    assert len(calls) == 1, "QoS 1 redelivery attached the camera twice"
    assert first["ok"] is True and second["ok"] is True


def test_a_refusal_carries_a_reason():
    handler = ControlHandler(
        "rk3588-01",
        set_threshold=lambda s, v: (_ for _ in ()).throw(CommandError("no such stream: " + s)),
        add_stream=lambda e: {}, remove_stream=lambda s: {},
    )
    ack = handler.handle(
        {"schema": "sensecraft.command/1", "timestamp": 1, "request_id": "r",
         "device_id": "rk3588-01", "command": "set_conf_threshold",
         "params": {"stream_id": "cam-9", "conf_threshold": 0.4}}, "s1", 1)
    assert ack["ok"] is False and ack["error"] == "no such stream: cam-9"


def test_open_means_producing_frames_not_pipeline_built():
    """Measured on an RK3588: the GStreamer pipeline for an unreachable RTSP URL
    constructs and reports decode=hw before the connection is attempted, so
    signalling "open" there made add_stream ack ok:true -- state "running" -- for
    a camera that does not exist. The contract says ok:true means live."""
    backend = FakeBackend()
    backend.script = [SourceLost("Failed to connect")]
    w = worker(backend)
    w.start()
    assert w.wait_until_open(1.5) is False
    w.stop()
