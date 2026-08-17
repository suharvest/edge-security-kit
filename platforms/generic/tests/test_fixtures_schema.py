"""Conformance: every payload kind this platform publishes passes the schema.

MQTT.md "Conformance": each platform keeps a representative fixture per message
kind and validates it in its host-only tests. The fixtures in ``fixtures/`` are
captured verbatim off the broker, not hand-written.
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
    assert FIXTURES, "no captured fixtures under platforms/generic/fixtures/"


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


#: platform fixture -> its promoted copy under contracts/fixtures/
PROMOTED = {
    "detection.json": "detection-generic-720p.json",
    "status.json": "status-generic.json",
    "status-lwt.json": "status-generic-lwt.json",
}


@pytest.mark.parametrize("local,promoted", sorted(PROMOTED.items()))
def test_promoted_copy_matches_this_platform(local: str, promoted: str):
    """contracts/check_fixtures.sh gates on the promoted copy, so it must not
    drift from the payload this platform actually captured."""
    mine = (ROOT / "fixtures" / local).read_text(encoding="utf-8")
    theirs = (SCHEMA_PATH.parent / "fixtures" / promoted).read_text(encoding="utf-8")
    assert json.loads(mine) == json.loads(theirs)
