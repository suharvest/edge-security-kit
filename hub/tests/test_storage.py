"""SQLite persistence, the §3 state machine and the §4 config guarantees."""

from __future__ import annotations

import json
import os

import pytest

from edge_hub.storage import Storage, atomic_write_json

NOW = 1_755_400_000_000


def add_alert(storage: Storage, n: int = 1, **overrides) -> dict:
    fields = {
        "event_id": f"dev-cam-s1-{n}",
        "ts_ms": NOW + n,
        "received_ms": NOW + n,
        "device_id": "dev",
        "stream_id": "cam",
        "event_type": "zone_enter",
        "rule_name": "bay",
        "track_id": n,
        "score": 0.9,
        "bbox": [0.5, 0.5, 0.1, 0.2],
        "snapshot_state": "none",
        "meta": {},
    }
    fields.update(overrides)
    return storage.insert_alert(**fields)


# -- schema -------------------------------------------------------------
def test_wal_mode_is_enabled(storage):
    mode = storage.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_all_spec_tables_exist(storage):
    names = {
        row[0]
        for row in storage.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"alerts", "devices", "rules", "config_versions", "auth"} <= names


def test_event_id_is_unique(storage):
    add_alert(storage, 1)
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        add_alert(storage, 1)


def test_event_type_check_constraint(storage):
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        add_alert(storage, 2, event_type="fall")


# -- state machine ------------------------------------------------------
def test_new_to_acked_and_dismissed(storage):
    first = add_alert(storage, 1)
    result, alert = storage.transition_alert(first["id"], "acked", "admin", NOW)
    assert (result, alert["state"], alert["acted_by"], alert["acted_at"]) == (
        "ok", "acked", "admin", NOW,
    )
    second = add_alert(storage, 2)
    result, alert = storage.transition_alert(second["id"], "dismissed", "admin", NOW)
    assert (result, alert["state"]) == ("ok", "dismissed")


def test_acked_and_dismissed_are_mutually_reversible(storage):
    alert = add_alert(storage, 1)
    assert storage.transition_alert(alert["id"], "acked", "admin", NOW)[0] == "ok"
    assert storage.transition_alert(alert["id"], "dismissed", "admin", NOW)[0] == "ok"
    assert storage.transition_alert(alert["id"], "acked", "admin", NOW)[0] == "ok"


@pytest.mark.parametrize("target", ["acked", "dismissed"])
def test_repeating_the_current_state_is_a_conflict(storage, target):
    alert = add_alert(storage, 1)
    storage.transition_alert(alert["id"], target, "admin", NOW)
    result, current = storage.transition_alert(alert["id"], target, "admin", NOW)
    assert result == "conflict"
    assert current["state"] == target


def test_there_is_no_transition_back_to_new(storage):
    alert = add_alert(storage, 1)
    storage.transition_alert(alert["id"], "acked", "admin", NOW)
    result, _ = storage.transition_alert(alert["id"], "new", "admin", NOW)
    assert result == "conflict"


def test_missing_alert_is_not_found(storage):
    assert storage.transition_alert(999, "acked", "admin", NOW)[0] == "not_found"


def test_dismissed_statistics_group_by_rule(storage):
    add_alert(storage, 1, rule_name="bay")
    second = add_alert(storage, 2, rule_name="bay")
    add_alert(storage, 3, rule_name="gate")
    storage.transition_alert(second["id"], "dismissed", "admin", NOW)
    stats = {row["rule_name"]: row for row in storage.dismissed_by_rule()}
    assert stats["bay"]["dismissed"] == 1 and stats["bay"]["total"] == 2
    assert stats["gate"]["dismissed"] == 0


# -- queries ------------------------------------------------------------
def test_default_order_is_newest_first(storage):
    for n in range(1, 4):
        add_alert(storage, n)
    assert [a["id"] for a in storage.query_alerts()] == [3, 2, 1]


def test_after_id_returns_ascending_ids_greater_than_the_cursor(storage):
    for n in range(1, 6):
        add_alert(storage, n)
    rows = storage.query_alerts(after_id=2)
    assert [a["id"] for a in rows] == [3, 4, 5]
    # A cursor at the head returns nothing, which is what a caught-up WS client
    # sees after reconnecting (HUB_SPEC §5).
    assert storage.query_alerts(after_id=5) == []


def test_after_id_composes_with_filters(storage):
    for n in range(1, 6):
        add_alert(storage, n, device_id="a" if n % 2 else "b")
    rows = storage.query_alerts(after_id=1, device_id="a")
    assert [a["id"] for a in rows] == [3, 5]


def test_filters_and_paging(storage):
    for n in range(1, 6):
        add_alert(storage, n, event_type="line_cross" if n > 3 else "zone_enter",
                  direction="forward" if n > 3 else None)
    assert len(storage.query_alerts(event_type="line_cross")) == 2
    assert [a["id"] for a in storage.query_alerts(limit=2)] == [5, 4]
    assert [a["id"] for a in storage.query_alerts(limit=2, offset=2)] == [3, 2]
    assert len(storage.query_alerts(date_from=NOW + 4)) == 2
    assert len(storage.query_alerts(date_to=NOW + 2)) == 2


# -- rules / config -----------------------------------------------------
def test_rev_increments_and_versions_accumulate(storage):
    first = storage.put_rules("dev", "cam", {"zones": [], "cooldown": 30}, NOW)
    assert first["rev"] == 1
    second = storage.put_rules("dev", "cam", {"zones": [], "cooldown": 15}, NOW + 1)
    assert second["rev"] == 2
    assert storage.get_rules("dev", "cam")["body"]["cooldown"] == 15
    revs = [v["rev"] for v in storage.config_versions("rules:dev/cam")]
    assert revs == [2, 1]


def test_rev_is_per_stream(storage):
    storage.put_rules("dev", "cam-0", {"zones": []}, NOW)
    storage.put_rules("dev", "cam-0", {"zones": []}, NOW)
    assert storage.put_rules("dev", "cam-1", {"zones": []}, NOW)["rev"] == 1
    assert storage.get_rules("dev", "cam-0")["rev"] == 2


def test_config_versions_are_capped_at_fifty(storage):
    for n in range(55):
        storage.put_rules("dev", "cam", {"zones": [], "n": n}, NOW + n)
    versions = storage.config_versions("rules:dev/cam")
    assert len(versions) == 50
    assert versions[0]["rev"] == 55
    assert versions[-1]["rev"] == 6


def test_put_rules_mirrors_config_json_atomically(storage):
    storage.put_rules("dev", "cam", {"zones": [], "cooldown": 7}, NOW)
    assert storage.config_path.is_file()
    mirror = json.loads(storage.config_path.read_text())
    assert mirror["rules"]["dev"]["cam"]["body"]["cooldown"] == 7
    assert mirror["rules"]["dev"]["cam"]["rev"] == 1
    # No temporary file is left behind by the tmp + fsync + rename sequence.
    assert not (storage.config_path.with_name(storage.config_path.name + ".tmp")).exists()


def test_atomic_write_replaces_the_previous_content_in_one_step(tmp_path):
    target = tmp_path / "config.json"
    atomic_write_json(target, {"a": 1})
    inode = os.stat(target).st_ino
    atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text()) == {"a": 2}
    # rename() gives the file a new inode; a reader holding the old handle keeps
    # reading valid JSON rather than a half-written file.
    assert os.stat(target).st_ino != inode


def test_hub_config_rev_increments(storage):
    assert storage.get_hub_config() is None
    assert storage.put_hub_config({"retention_days": 7}, NOW)["rev"] == 1
    assert storage.put_hub_config({"retention_days": 14}, NOW)["rev"] == 2
    current = storage.get_hub_config()
    assert (current["rev"], current["body"]["retention_days"]) == (2, 14)
    mirror = json.loads(storage.config_path.read_text())
    assert mirror["hub"]["retention_days"] == 14 and mirror["hub_rev"] == 2


def test_device_config_restore_bumps_every_stream(storage):
    storage.put_rules("dev", "cam-0", {"zones": []}, NOW)
    results = storage.put_device_rules(
        "dev", {"cam-0": {"zones": [], "cooldown": 5}, "cam-1": {"zones": []}}, NOW
    )
    assert {r["stream_id"]: r["rev"] for r in results} == {"cam-0": 2, "cam-1": 1}


def test_rules_survive_a_reopen(storage, tmp_path):
    storage.put_rules("dev", "cam", {"zones": [], "cooldown": 11}, NOW)
    alert = add_alert(storage, 1)
    storage.transition_alert(alert["id"], "acked", "admin", NOW)
    storage.close()
    reopened = Storage(tmp_path / "data")
    try:
        assert reopened.get_rules("dev", "cam")["body"]["cooldown"] == 11
        assert reopened.get_alert(alert["id"])["state"] == "acked"
    finally:
        reopened.close()


# -- devices / retention ------------------------------------------------
def test_device_upsert_and_listing(storage):
    storage.upsert_device("dev", True, NOW, "hub", {"streams": []})
    storage.upsert_device("dev", False, NOW + 1, "single_box", {"streams": [1]})
    devices = storage.list_devices()
    assert len(devices) == 1
    assert devices[0]["online"] is False
    assert devices[0]["mode"] == "single_box"
    assert storage.device_mode("dev") == "single_box"
    assert storage.device_mode("unknown") == "hub"


def test_retention_removes_old_alerts_and_their_snapshots(storage):
    old = add_alert(storage, 1, received_ms=NOW - 40 * 86_400_000)
    recent = add_alert(storage, 2, received_ms=NOW)
    target = storage.snapshot_target(old["event_id"], NOW)
    target.write_bytes(b"\xff\xd8\xff")
    storage.set_snapshot(old["id"], str(target))
    removed = storage.purge_older_than(NOW - 30 * 86_400_000)
    assert removed == 1
    assert storage.get_alert(old["id"]) is None
    assert storage.get_alert(recent["id"]) is not None
    assert not target.exists()
