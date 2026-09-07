"""The detector's cmd/control decision table (contracts/MQTT.md).

No broker, no camera, no model: ControlHandler takes three callbacks, so what
is tested here is the part that has to be right — which commands are honoured,
what the ack says, and what happens twice when the broker delivers twice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "contracts"))
from validate_payload import validate  # noqa: E402

from esk_generic.control import CommandError, ControlHandler  # noqa: E402

DEVICE = "generic-01"


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.fail: str | None = None

    def set_threshold(self, stream_id: str, value: float) -> dict:
        self.calls.append(("set", stream_id, value))
        if self.fail:
            raise CommandError(self.fail)
        return {"stream_id": stream_id, "conf_threshold": value, "persisted": True}

    def add(self, entry: dict) -> dict:
        self.calls.append(("add", entry["stream_id"], entry["source"]))
        if self.fail:
            raise CommandError(self.fail)
        return {"stream_id": entry["stream_id"], "state": "running", "persisted": True}

    def remove(self, stream_id: str) -> dict:
        self.calls.append(("remove", stream_id))
        if self.fail:
            raise CommandError(self.fail)
        return {"stream_id": stream_id, "removed": True, "persisted": True}


@pytest.fixture
def handler():
    rec = Recorder()
    h = ControlHandler(DEVICE, rec.set_threshold, rec.add, rec.remove)
    h.recorder = rec
    return h


def command(command: str, params: dict, request_id: str = "hub-1", device_id=DEVICE) -> dict:
    return {
        "schema": "sensecraft.command/1",
        "timestamp": 1_757_203_411_000,
        "request_id": request_id,
        "device_id": device_id,
        "command": command,
        "params": params,
    }


def test_ack_is_a_conformant_payload(handler):
    ack = handler.handle(
        command("set_conf_threshold", {"stream_id": "cam-0", "conf_threshold": 0.42}),
        "sess-1", 1_757_203_411_087,
    )
    assert validate(ack) == "sensecraft.ack/1"
    assert ack["ok"] is True
    assert ack["applied"]["conf_threshold"] == 0.42


def test_a_failure_ack_is_conformant_too(handler):
    """A refusal that a consumer cannot parse is as bad as no answer at all."""
    handler.recorder.fail = "no such stream: cam-9"
    ack = handler.handle(
        command("set_conf_threshold", {"stream_id": "cam-9", "conf_threshold": 0.4}),
        "sess-1", 1_757_203_411_087,
    )
    assert validate(ack) == "sensecraft.ack/1"
    assert ack["ok"] is False
    assert ack["error"] == "no such stream: cam-9"


def test_a_redelivered_request_id_applies_once_and_acks_twice(handler):
    """QoS 1 redelivery is normal; adding the camera twice is not."""
    payload = command("add_stream", {"stream_id": "cam-2", "source": "rtsp://x"})
    first = handler.handle(payload, "sess-1", 1)
    second = handler.handle(payload, "sess-1", 2)
    assert [c[0] for c in handler.recorder.calls] == ["add"]
    assert second["ok"] == first["ok"]
    assert second["request_id"] == first["request_id"]
    assert second["timestamp"] == 2  # re-acked now, not replayed with a stale time


def test_a_command_for_another_device_is_ignored_entirely(handler):
    assert handler.handle(
        command("remove_stream", {"stream_id": "cam-0"}, device_id="rk3588-02"),
        "sess-1", 1,
    ) is None
    assert handler.recorder.calls == []


def test_a_non_command_payload_is_ignored(handler):
    assert handler.handle({"schema": "sensecraft.status/1"}, "sess-1", 1) is None


def test_a_command_without_request_id_is_ignored(handler):
    """There is nothing to correlate an ack to, so answering would be noise."""
    payload = command("remove_stream", {"stream_id": "cam-0"})
    del payload["request_id"]
    assert handler.handle(payload, "sess-1", 1) is None


@pytest.mark.parametrize("payload,reason", [
    (command("reboot", {"stream_id": "cam-0"}), "unknown command"),
    (command("set_conf_threshold", {"conf_threshold": 0.4}), "stream_id is required"),
    (command("set_conf_threshold", {"stream_id": "cam-0"}), "must be a number"),
    (command("set_conf_threshold", {"stream_id": "cam-0", "conf_threshold": 1.4}),
     "between 0 and 1"),
    (command("set_conf_threshold", {"stream_id": "cam-0", "conf_threshold": True}),
     "must be a number"),
    (command("add_stream", {"stream_id": "cam-2"}), "source is required"),
    (command("add_stream", {"stream_id": "cam-2", "source": "rtsp://x",
                            "rtsp_transport": "sctp"}), "tcp or udp"),
])
def test_a_bad_command_is_refused_with_a_reason_not_dropped(handler, payload, reason):
    ack = handler.handle(payload, "sess-1", 1)
    assert ack is not None and ack["ok"] is False
    assert reason in ack["error"]
    assert validate(ack) == "sensecraft.ack/1"
    assert handler.recorder.calls == []


def test_an_unexpected_fault_still_produces_an_ack(handler):
    """Silence would be read as 'device unreachable', which is a lie."""
    def explode(_stream_id, _value):
        raise RuntimeError("model session is gone")

    handler.set_threshold = explode
    ack = handler.handle(
        command("set_conf_threshold", {"stream_id": "cam-0", "conf_threshold": 0.4}),
        "sess-1", 1,
    )
    assert ack["ok"] is False
    assert "RuntimeError" in ack["error"]
