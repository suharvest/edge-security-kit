"""This platform is wired to the shared runtime-control base.

The shared behaviour is covered once in ``platforms/_shared/tests``. What can
only break per platform is the wiring: a detector that never subscribes to
cmd/control, a backend missing a method the capture loop calls, or a config that
cannot describe a second camera. Each of those turns the console's four controls
into a timeout on this board and nothing else would notice.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))

import esk_core  # noqa: E402


PKG = {"rknn": "esk_rknn", "hailo": "esk_hailo", "jetson": "esk_jetson",
       "generic": "esk_generic"}[Path(__file__).resolve().parents[1].name]


def _mod(name: str):
    import importlib
    return importlib.import_module(f"{PKG}.{name}")


def test_the_detector_subscribes_to_cmd_control():
    src = inspect.getsource(_mod("detector"))
    assert "topic_cmd_control" in src
    assert "client.subscribe(self.topic_cmd_control" in src
    assert "self.handle_control(request)" in src


def test_the_detector_is_a_stream_supervisor():
    assert issubclass(_mod("detector").Detector, esk_core.StreamSupervisor)
    assert hasattr(_mod("detector").Detector, "make_tracker")


def test_the_backend_satisfies_what_the_capture_loop_calls():
    backend = next(
        obj for _n, obj in vars(_mod("backend")).items()
        if inspect.isclass(obj) and _n.endswith("Backend")
    )
    for name in ("open", "read", "close", "detect", "frame_of", "decode_path"):
        assert callable(getattr(backend, name, None)), name


def test_the_model_takes_a_per_call_confidence_threshold():
    """Without it one slider moves every camera on the board."""
    module = _mod("detector")
    for name in dir(module):
        obj = getattr(module, name)
        if inspect.isclass(obj) and hasattr(obj, "detect") and name.endswith("Detector") \
                and name != "Detector":
            sig = inspect.signature(obj.detect)
            assert "conf_threshold" in sig.parameters, name


def test_the_config_describes_several_streams_and_remembers_its_path():
    cfg = _mod("config").Config()
    assert hasattr(cfg, "streams") and hasattr(cfg, "path")
    assert hasattr(cfg, "max_streams")
    assert cfg.stream_configs()[0]["stream_id"] == cfg.stream_id
    assert cfg.persist_streams([]) is False  # no path -> persisted: false, not a crash
