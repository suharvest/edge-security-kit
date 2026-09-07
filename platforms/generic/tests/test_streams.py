"""Per-stream state that must not be shared, and the config write-back.

The two bugs this file exists to prevent are both invisible in a single-stream
run: a tracker shared across cameras hands one track_id to two different
people, and a threshold stored on the model session moves every camera when one
slider moves.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from esk_generic.config import Config
from esk_generic.streams import StreamConfig, StreamWorker
from esk_generic.tracker import IoUTracker


class Det:
    def __init__(self, box, score=0.9):
        self.box = box
        self.score = score


def worker(stream_id: str, threshold: float, seen: list) -> StreamWorker:
    def infer(frame, conf_threshold):
        seen.append((stream_id, conf_threshold))
        return [Det([0.1, 0.1, 0.2, 0.3])], 1.5

    return StreamWorker(
        config=StreamConfig(stream_id=stream_id, source="rtsp://x"),
        conf_threshold=threshold,
        infer=infer,
        publish=lambda _sid, _payload: None,
        tracker=IoUTracker(0.2, 0.75),
    )


def frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_each_stream_infers_at_its_own_threshold():
    seen: list = []
    a, b = worker("cam-0", 0.35, seen), worker("cam-1", 0.7, seen)
    a.build_payload(frame(), 0.0)
    b.build_payload(frame(), 0.0)
    assert seen == [("cam-0", 0.35), ("cam-1", 0.7)]


def test_a_threshold_change_takes_effect_on_the_next_frame():
    seen: list = []
    w = worker("cam-0", 0.35, seen)
    w.build_payload(frame(), 0.0)
    w.conf_threshold = 0.6
    w.build_payload(frame(), 0.0)
    assert [t for _, t in seen] == [0.35, 0.6]


def test_track_ids_and_frame_ids_are_per_stream():
    seen: list = []
    a, b = worker("cam-0", 0.35, seen), worker("cam-1", 0.35, seen)
    for _ in range(3):
        a.build_payload(frame(), 0.0)
    payload_a = a.build_payload(frame(), 0.0)
    payload_b = b.build_payload(frame(), 0.0)
    assert payload_a["frame_id"] == 4
    assert payload_b["frame_id"] == 1
    # Both streams start their own numbering at 1 -- the contract's per-stream
    # guarantee. A shared tracker would have given cam-1's first person id 2.
    assert payload_a["detections"][0]["track_id"] == 1
    assert payload_b["detections"][0]["track_id"] == 1


def test_status_entry_reports_the_running_threshold():
    """The console's slider renders this, not the config file's value."""
    w = worker("cam-0", 0.35, [])
    w.conf_threshold = 0.62
    entry = w.status_entry()
    assert entry["stream_id"] == "cam-0"
    assert entry["conf_threshold"] == 0.62
    assert entry["state"] == "stopped"


# -- config ------------------------------------------------------------------

def test_a_single_stream_config_still_describes_one_stream(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(
        {"device_id": "generic-01", "stream_id": "cam-0", "source": "rtsp://a"}))
    cfg = Config.load(str(path))
    assert cfg.stream_configs() == [
        {"stream_id": "cam-0", "source": "rtsp://a", "rtsp_transport": "tcp"}
    ]


def test_a_streams_list_wins_over_the_shorthand(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "device_id": "generic-01",
        "stream_id": "cam-0",
        "source": "rtsp://ignored",
        "streams": [
            {"stream_id": "cam-0", "source": "rtsp://a"},
            {"stream_id": "cam-1", "source": "rtsp://b", "conf_threshold": 0.5},
        ],
    }))
    cfg = Config.load(str(path))
    ids = [s["stream_id"] for s in cfg.stream_configs()]
    assert ids == ["cam-0", "cam-1"]
    assert cfg.stream_configs()[1]["conf_threshold"] == 0.5


def test_persist_rewrites_the_file_and_drops_the_contradicting_shorthand(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(
        {"device_id": "generic-01", "stream_id": "cam-0", "source": "rtsp://a",
         "conf_threshold": 0.35}))
    cfg = Config.load(str(path))

    assert cfg.persist_streams([
        {"stream_id": "cam-0", "source": "rtsp://a", "conf_threshold": 0.6},
        {"stream_id": "cam-1", "source": "rtsp://b", "name": "North gate"},
    ]) is True

    written = yaml.safe_load(path.read_text())
    assert [s["stream_id"] for s in written["streams"]] == ["cam-0", "cam-1"]
    # Leaving the shorthand behind would make the file describe cam-0 twice,
    # with two different sources after the next edit.
    assert "source" not in written and "stream_id" not in written
    assert written["conf_threshold"] == 0.35  # the process default is untouched
    assert Config.load(str(path)).stream_configs()[1]["name"] == "North gate"
    assert not (tmp_path / "config.yaml.tmp").exists()


def test_a_flag_configured_process_persists_nothing_and_says_so():
    """No file to write means the ack must report persisted: false, not fail."""
    cfg = Config()
    assert cfg.path is None
    assert cfg.persist_streams([{"stream_id": "cam-0", "source": "rtsp://a"}]) is False


def test_unknown_config_keys_are_still_refused(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"device_id": "x", "nonsense": 1}))
    with pytest.raises(ValueError, match="nonsense"):
        Config.load(str(path))
