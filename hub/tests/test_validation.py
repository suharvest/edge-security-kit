"""Contract validation and ingest dispatch (HUB_SPEC §1, §10)."""

from __future__ import annotations

import json

from conftest import DEVICE, STREAM, detection, device_event, status

from edge_hub.mqtt_ingest import MqttIngest
from edge_hub.validation import ContractValidator

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32


def test_valid_payloads_of_all_three_kinds_pass():
    validator = ContractValidator()
    assert validator.validate(json.dumps(detection(0.5, 0.5))) is not None
    assert validator.validate(json.dumps(status())) is not None
    assert validator.validate(json.dumps(device_event())) is not None
    assert validator.failure_count == 0


def test_missing_coordinate_space_is_rejected_and_counted():
    validator = ContractValidator()
    payload = detection(0.5, 0.5)
    del payload["coordinate_space"]
    assert validator.validate(json.dumps(payload)) is None
    assert validator.failure_count == 1


def test_coordinates_outside_the_unit_range_are_rejected():
    validator = ContractValidator()
    payload = detection(1.4, 0.5)
    assert validator.validate(json.dumps(payload)) is None
    assert validator.failure_count == 1


def test_letterboxed_coordinate_space_value_is_rejected():
    validator = ContractValidator()
    payload = detection(0.5, 0.5)
    payload["coordinate_space"] = "letterbox_norm"
    assert validator.validate(json.dumps(payload)) is None


def test_float_timestamp_is_rejected():
    validator = ContractValidator()
    payload = detection(0.5, 0.5)
    payload["timestamp"] = 1_755_400_000.5
    assert validator.validate(json.dumps(payload)) is None


def test_line_cross_event_without_direction_is_rejected():
    validator = ContractValidator()
    payload = device_event(event_type="line_cross")
    del payload["direction"]
    assert validator.validate(json.dumps(payload)) is None


def test_wrong_kind_for_the_topic_is_rejected():
    validator = ContractValidator()
    assert validator.validate(json.dumps(status()), expected="sensecraft.detection/1") is None
    assert validator.failures["wrong_topic_kind"] == 1


def test_malformed_json_and_unknown_schema_are_counted_separately():
    validator = ContractValidator()
    assert validator.validate(b"{not json") is None
    assert validator.validate(json.dumps({"schema": "other/1"})) is None
    assert validator.failures["json_parse"] == 1
    assert validator.failures["unknown_schema"] == 1


def test_snapshot_magic_bytes_and_size_cap():
    validator = ContractValidator()
    assert validator.snapshot_ok(JPEG) is True
    assert validator.snapshot_ok(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32) is False
    assert validator.snapshot_ok(b"\xff\xd8\xff" + b"\x00" * (200 * 1024)) is False
    assert validator.snapshot_ok(b"\xff") is False
    assert validator.failures["snapshot_not_jpeg"] == 1
    assert validator.failures["snapshot_too_large"] == 1
    assert validator.failures["snapshot_too_short"] == 1


def test_stats_shape_is_health_ready():
    validator = ContractValidator()
    validator.validate(b"nope")
    stats = validator.stats()
    assert stats["schema_failures"] == 1
    assert stats["schema_failures_by_reason"]["json_parse"] == 1


# -- ingest dispatch ----------------------------------------------------
def make_ingest(seen: dict) -> MqttIngest:
    async def on_detections(payload):
        seen.setdefault("detections", []).append(payload)

    async def on_status(payload):
        seen.setdefault("status", []).append(payload)

    async def on_event(payload):
        seen.setdefault("events", []).append(payload)

    async def on_snapshot(device_id, event_id, payload):
        seen.setdefault("snapshots", []).append((device_id, event_id, payload))

    return MqttIngest(
        host="localhost",
        on_detections=on_detections,
        on_status=on_status,
        on_event=on_event,
        on_snapshot=on_snapshot,
    )


def test_subscription_topics_match_the_contract():
    ingest = MqttIngest(host="localhost")
    assert ingest.subscriptions == [
        "sensecraft/security/+/detections/+",
        "sensecraft/security/+/events/+",
        "sensecraft/security/+/status",
        "sensecraft/security/+/snapshot/+",
        # The control ack is an uplink like the other four: the hub subscribes
        # to it, the detector publishes it. cmd/control itself is not here --
        # the hub is that topic's publisher.
        "sensecraft/security/+/cmd/ack",
    ]


async def test_dispatch_routes_each_topic_family():
    seen: dict = {}
    ingest = make_ingest(seen)
    base = f"sensecraft/security/{DEVICE}"
    await ingest.dispatch(f"{base}/detections/{STREAM}", json.dumps(detection(0.5, 0.5)).encode())
    await ingest.dispatch(f"{base}/status", json.dumps(status()).encode())
    await ingest.dispatch(
        f"sensecraft/security/recamera-01/events/cam0", json.dumps(device_event()).encode()
    )
    await ingest.dispatch(f"{base}/snapshot/evt-1", JPEG)
    assert len(seen["detections"]) == 1
    assert len(seen["status"]) == 1
    assert len(seen["events"]) == 1
    assert seen["snapshots"] == [(DEVICE, "evt-1", JPEG)]
    assert ingest.validator.failure_count == 0


async def test_dispatch_rejects_a_device_id_that_contradicts_the_topic():
    seen: dict = {}
    ingest = make_ingest(seen)
    await ingest.dispatch(
        "sensecraft/security/other/detections/cam-0",
        json.dumps(detection(0.5, 0.5)).encode(),
    )
    assert "detections" not in seen
    assert ingest.validator.failures["device_id_topic_mismatch"] == 1


async def test_dispatch_ignores_foreign_prefixes_and_bad_snapshots():
    seen: dict = {}
    ingest = make_ingest(seen)
    await ingest.dispatch("other/root/detections/cam", b"{}")
    await ingest.dispatch(f"sensecraft/security/{DEVICE}/snapshot/x", b"notjpeg")
    assert seen == {}
    assert ingest.validator.failures["unknown_topic"] == 1
    assert ingest.validator.failures["snapshot_not_jpeg"] == 1
