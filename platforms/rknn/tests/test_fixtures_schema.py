"""Conformance: every payload kind this platform publishes passes the schema.

MQTT.md "Conformance": each platform keeps a representative fixture per message
kind and validates it in its host-only tests. The fixtures in ``fixtures/`` were
captured verbatim off the broker during a real RK3588 run -- not hand-written,
and not copied from the generic platform.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT.parents[1] / "contracts" / "mqtt-detection.schema.json"
FIXTURES = sorted((ROOT / "fixtures").glob("*.json"))


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_fixtures_exist():
    assert FIXTURES, "no captured fixtures under platforms/rknn/fixtures/"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_validates(path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    payload = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(payload, load_schema())


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_detection_fixture_invariants(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "sensecraft.detection/1":
        return
    assert payload["coordinate_space"] == "frame_norm"
    assert isinstance(payload["timestamp"], int)
    for det in payload["detections"]:
        assert det["track_id"] >= 1, "0 means untracked; the hub ignores those"
        cx, cy, w, h = det["bbox"]
        assert 0.0 <= cx - w / 2 and cx + w / 2 <= 1.0
        assert 0.0 <= cy - h / 2 and cy + h / 2 <= 1.0


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_rk_platform_reports_hardware_decode(path: Path):
    """The whole point of this platform: decode must be reported, and be hw.

    A silent fall back to CPU ffmpeg is the Rockchip failure mode, so a captured
    fixture that says ``sw`` means the run being certified was not the run this
    platform is supposed to deliver.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    health = payload.get("health")
    if not health:
        return
    assert health["decode"] == "hw"
    assert health["fallback_active"] is False
    assert health["backend"].startswith("rknn-lite2-")


def test_detection_fixture_frame_is_the_original_not_the_letterbox():
    """frame.w/h must be the source size, never the 640x640 model input.

    Publishing the canvas size would make every normalized coordinate wrong in
    a way that still validates against the schema.
    """
    for path in FIXTURES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "sensecraft.detection/1":
            continue
        assert (payload["frame"]["w"], payload["frame"]["h"]) != (640, 640)
        assert payload["frame"]["w"] > payload["frame"]["h"], "fixture must be non-square"


#: platform fixture -> its promoted copy under contracts/fixtures/
PROMOTED = {
    "detection.json": "detection-rknn-720p.json",
    "status.json": "status-rknn.json",
    "status-lwt.json": "status-rknn-lwt.json",
}


@pytest.mark.parametrize("local,promoted", sorted(PROMOTED.items()))
def test_promoted_copy_matches_this_platform(local: str, promoted: str):
    """contracts/check_fixtures.sh gates on the promoted copy, so it must not
    drift from the payload this platform actually captured."""
    mine = (ROOT / "fixtures" / local).read_text(encoding="utf-8")
    theirs = (SCHEMA_PATH.parent / "fixtures" / promoted).read_text(encoding="utf-8")
    assert json.loads(mine) == json.loads(theirs)
