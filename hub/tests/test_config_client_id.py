"""The MQTT client id must be unique per hub instance (HUB_SPEC §8).

A shared id is not a cosmetic problem. MQTT ids are exclusive per broker: the
second connection presenting an id already in use takes over the session and
the first client is dropped. Two hubs pointed at one broker then flap in a ~1 s
reconnect loop, and because detections are QoS 0 the disconnected side loses
messages with nothing logged. Observed in acceptance: ``line_cross`` (needs two
consecutive frames) failed about half the time while ``zone_enter`` (needs any
one frame) looked healthy.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from edge_hub import config as config_module
from edge_hub.mqtt_ingest import MqttIngest

HUB_ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"^edge-security-hub-[a-z0-9-]+-[0-9a-f]{4}$")


def test_default_client_id_is_unique_per_call():
    ids = {config_module.default_client_id() for _ in range(50)}
    assert len(ids) == 50
    for value in ids:
        assert PATTERN.match(value), value


def test_two_hub_processes_get_different_default_client_ids():
    """The case that matters: two `edge-security-hub` processes, one broker."""
    code = "from edge_hub.config import resolve; print(resolve(None)['mqtt_client_id'])"
    got = [
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=HUB_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert PATTERN.match(got[0]) and PATTERN.match(got[1]), got
    assert got[0] != got[1], got


def test_resolve_is_stable_within_one_process():
    """Otherwise every config PUT would report a spurious restart_required."""
    first = config_module.resolve(None)
    second = config_module.resolve(None)
    assert first["mqtt_client_id"] == second["mqtt_client_id"]
    assert config_module.restart_required(first, second) == []


def test_explicit_client_id_wins():
    assert config_module.resolve({"mqtt_client_id": "fixed"})["mqtt_client_id"] == "fixed"
    from_env = config_module.resolve(None, env={"MQTT_CLIENT_ID": "from-env"})
    assert from_env["mqtt_client_id"] == "from-env"


def test_ingest_generates_its_own_id_when_unset():
    a = MqttIngest(host="localhost")
    b = MqttIngest(host="localhost")
    assert a.client_id != b.client_id
    assert PATTERN.match(a.client_id) and PATTERN.match(b.client_id)
    assert MqttIngest(host="localhost", client_id="pinned").client_id == "pinned"


def test_health_reports_the_client_id(hub):
    hub.build_ingest()
    assert hub.health()["mqtt_client_id"] == hub.ingest.client_id
    assert PATTERN.match(hub.health()["mqtt_client_id"])
