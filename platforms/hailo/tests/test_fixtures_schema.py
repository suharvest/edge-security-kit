"""Conformance: every payload kind this platform publishes passes the schema.

MQTT.md "Conformance": each platform keeps a representative fixture per message
kind, captured verbatim off the broker during a real run, and validates it in
its host-only tests.

This platform has no captured fixtures yet. The Hailo-8 in the reference board
is held exclusively by an unrelated service (see README, "Not yet verified on
hardware"), and HailoRT gives one process at a time a VDevice, so no run has
produced payloads to capture. The module skips rather than passing vacuously:
the tests activate the moment ``fixtures/`` is populated, and until then the
platform is explicitly not conformance-certified. Hand-written fixtures would
defeat the point of the clause -- the value is that the bytes came off a wire.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT.parents[1] / "contracts" / "mqtt-detection.schema.json"
FIXTURES = sorted((ROOT / "fixtures").glob("*.json"))

pytestmark = pytest.mark.skipif(
    not FIXTURES,
    reason="no payloads captured from the board yet; see README 'Not yet verified'",
)


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_validates(path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(json.loads(path.read_text(encoding="utf-8")), load_schema())


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
def test_health_reports_this_platform_honestly(path: Path):
    """decode must say sw, and say it without claiming a fallback.

    The Pi 5 has no H.264 hardware decoder, so ``sw`` is the truth and the
    configured primary at once. A fixture reporting ``hw`` would mean the field
    was written to look good rather than measured; a fixture reporting
    ``fallback_active: true`` would mean the platform is advertising a
    degradation it cannot actually be in.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    health = payload.get("health")
    if not health:
        return
    assert health["decode"] == "sw"
    assert health["fallback_active"] is False
    assert health["backend"].startswith("hailort-")


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_detection_fixture_frame_is_the_original_not_the_letterbox(path: Path):
    """frame.w/h must be the source size, never the 640x640 model input.

    Publishing the canvas size would make every normalized coordinate wrong in
    a way that still validates against the schema.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "sensecraft.detection/1":
        return
    assert (payload["frame"]["w"], payload["frame"]["h"]) != (640, 640)
    assert payload["frame"]["w"] > payload["frame"]["h"], "fixture must be non-square"
