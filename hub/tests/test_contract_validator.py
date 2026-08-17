"""contracts/validate_payload.py must agree with the JSON Schema, with no deps."""

from __future__ import annotations

import json

import pytest
from conftest import detection, device_event, status

import validate_payload
from edge_hub.validation import ContractValidator


@pytest.fixture(scope="module")
def jsonschema_validator() -> ContractValidator:
    return ContractValidator()


@pytest.mark.parametrize(
    "payload",
    [
        detection(0.5, 0.5),
        status(),
        status(online=False),
        device_event(),
        device_event(event_type="line_cross"),
        device_event(event_type="loitering"),
    ],
)
def test_good_payloads_pass_both_validators(payload, jsonschema_validator):
    assert validate_payload.validate(payload) == payload["schema"]
    assert jsonschema_validator.validate(json.dumps(payload)) is not None


def _break(payload: dict, **changes) -> dict:
    out = dict(payload)
    for key, value in changes.items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


@pytest.mark.parametrize(
    "payload",
    [
        _break(detection(0.5, 0.5), coordinate_space=None),
        _break(detection(0.5, 0.5), coordinate_space="letterbox_norm"),
        _break(detection(0.5, 0.5), timestamp=1.5),
        _break(detection(0.5, 0.5), timestamp=-1),
        _break(detection(0.5, 0.5), session_id=None),
        _break(detection(0.5, 0.5), frame=({"w": 1920, "h": 1080, "extra": 1})),
        detection(1.5, 0.5),
        _break(status(), online="yes"),
        _break(device_event(event_type="line_cross"), direction=None),
        _break(device_event(event_type="loitering"), dwell_s=None),
        _break(device_event(), event_type="fall"),
        _break(device_event(), bbox=[0.5, 0.5, 0.1]),
        {"schema": "sensecraft.other/1"},
    ],
)
def test_bad_payloads_fail_both_validators(payload, jsonschema_validator):
    with pytest.raises(ValueError):
        validate_payload.validate(payload)
    assert jsonschema_validator.validate(json.dumps(payload)) is None


def test_lwt_payload_must_not_carry_streams():
    # contracts/MQTT.md: the LWT payload has online false and no streams array.
    bad = status(online=False)
    bad["streams"] = [{"stream_id": "cam-0", "state": "running"}]
    with pytest.raises(ValueError, match="LWT"):
        validate_payload.validate(bad)


def test_cli_reports_the_matched_kind(tmp_path, capsys):
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(detection(0.5, 0.5)))
    import sys

    argv = sys.argv
    sys.argv = ["validate_payload.py", str(path)]
    try:
        validate_payload.main()
    finally:
        sys.argv = argv
    assert "sensecraft.detection/1" in capsys.readouterr().out
