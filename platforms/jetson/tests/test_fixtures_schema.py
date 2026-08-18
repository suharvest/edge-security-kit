"""Conformance: every payload kind this platform publishes passes the schema.

MQTT.md "Conformance": each platform keeps a representative fixture per message
kind and validates it in its host-only tests. The fixtures in ``fixtures/`` were
captured verbatim off the broker during a real Orin NX run against the truth
video -- not hand-written, and not copied from another platform.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT.parents[1] / "contracts" / "mqtt-detection.schema.json"
FIXTURES = sorted((ROOT / "fixtures").glob("*.json"))


@pytest.fixture(scope="module")
def validator():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    return jsonschema.Draft202012Validator(schema)


def test_fixtures_exist():
    assert {p.name for p in FIXTURES} >= {
        "detection.json",
        "status.json",
        "status-lwt.json",
    }


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fixture_matches_schema(validator, path):
    errors = sorted(validator.iter_errors(json.loads(path.read_text())), key=str)
    assert not errors, "\n".join(f"{list(e.path)}: {e.message}" for e in errors)


def test_detection_reports_this_platform():
    payload = json.loads((ROOT / "fixtures" / "detection.json").read_text())
    assert payload["coordinate_space"] == "frame_norm"
    # The source is non-square on purpose: a square one passes even when the
    # letterbox inverse is missing entirely.
    assert payload["frame"] == {"w": 1280, "h": 720}
    assert payload["health"]["decode"] == "hw"
    assert payload["health"]["backend"].startswith("tensorrt-")
    assert payload["health"]["fallback_active"] is False
    # HUB_SPEC 2.1: the hub keeps rule state only for track_id >= 1.
    assert all(d["track_id"] >= 1 for d in payload["detections"])


def test_status_advertises_preview_and_live():
    payload = json.loads((ROOT / "fixtures" / "status.json").read_text())
    stream = payload["streams"][0]
    assert stream["decode"] == "hw"
    assert stream["preview_url"].endswith("/preview/cam-0.jpg")
    assert stream["live_url"].endswith("/live/cam-0")


def test_lwt_has_no_streams():
    """The goodbye and the LWT must be the same shape: one consumer code path."""
    payload = json.loads((ROOT / "fixtures" / "status-lwt.json").read_text())
    assert payload["online"] is False
    assert "streams" not in payload
