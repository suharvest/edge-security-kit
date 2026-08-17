"""Cooldown, event_id, snapshot state machine (HUB_SPEC §3, §3.1)."""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import DEVICE, STREAM, device_event, rules_body

from edge_hub.rules.engine import Candidate

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


def candidate(clock, track_id: int = 1, rule_name: str = "bay",
              event_type: str = "zone_enter", stream_id: str = STREAM) -> Candidate:
    return Candidate(
        device_id=DEVICE,
        stream_id=stream_id,
        session_id="s1",
        event_type=event_type,
        rule_name=rule_name,
        rule_id=rule_name,
        track_id=track_id,
        bbox=[0.5, 0.5, 0.1, 0.2],
        ts_ms=clock.wall_ms(),
        received_ms=clock.wall_ms(),
        received_mono=clock.mono_ms(),
        score=0.9,
        cls="person",
    )


# -- cooldown -----------------------------------------------------------
def test_four_tuple_cooldown_suppresses_the_same_track(clock, alerts):
    body = rules_body(cooldown=30)
    assert alerts.allow(candidate(clock), body) is True
    clock.advance(29_000)
    assert alerts.allow(candidate(clock), body) is False
    clock.advance(1_100)
    assert alerts.allow(candidate(clock), body) is True


def test_zone_entry_does_not_swallow_the_loitering_escalation(clock, alerts):
    """zone_enter and loitering share a rule_name (the zone).

    Upstream keyed the cooldown per event type (`track.fired_events[etype]`).
    Without event_type in the key, entering a zone suppresses that zone's
    loitering alert for the whole cooldown window — losing the more serious
    signal. Found while capturing conformance fixtures: loitering never fired.
    """
    body = rules_body(cooldown=30)
    assert alerts.allow(candidate(clock, rule_name="bay",
                                  event_type="zone_enter"), body) is True
    clock.advance(6_000)
    assert alerts.allow(candidate(clock, rule_name="bay",
                                  event_type="loitering"), body) is True
    # Each type still holds its own budget against a repeat.
    assert alerts.allow(candidate(clock, rule_name="bay",
                                  event_type="loitering"), body) is False


def test_cooldown_is_per_rule_and_per_stream(clock, alerts):
    body = rules_body(cooldown=30)
    assert alerts.allow(candidate(clock, rule_name="bay"), body) is True
    # A different rule on the same track has its own budget.
    assert alerts.allow(candidate(clock, rule_name="gate"), body) is True
    # ...as does the same rule on a different stream.
    assert alerts.allow(candidate(clock, rule_name="bay", stream_id="cam-1"), body) is True


def test_stream_rate_limit_stops_a_track_id_change_from_re_firing(clock, alerts):
    """The upstream defect: cooldown hung on the track, so a new tracker ID
    re-alerted immediately (multi_camera_manager.py:464-482)."""
    body = rules_body(cooldown=30, stream_rate_limit_s=30)
    assert alerts.allow(candidate(clock, track_id=1), body) is True
    clock.advance(500)
    # Same person, tracker re-assigned an ID: the four-tuple key differs but the
    # stream-level limit still holds.
    assert alerts.allow(candidate(clock, track_id=2), body) is False
    clock.advance(30_000)
    assert alerts.allow(candidate(clock, track_id=3), body) is True


def test_without_the_stream_limit_distinct_tracks_both_fire(clock, alerts):
    body = rules_body(cooldown=30, stream_rate_limit_s=0)
    assert alerts.allow(candidate(clock, track_id=1), body) is True
    assert alerts.allow(candidate(clock, track_id=2), body) is True


def test_stream_rate_limit_defaults_to_off(clock, alerts):
    """Adjudicated: the default must not collapse two people into one alert.

    The body omits ``stream_rate_limit_s`` entirely, so this pins the default
    rather than an explicit 0 (which the sibling test above already covers).
    """
    body = rules_body(cooldown=30)
    del body["stream_rate_limit_s"]
    assert alerts.allow(candidate(clock, track_id=1), body) is True
    assert alerts.allow(candidate(clock, track_id=2), body) is True
    # The four-tuple cooldown still holds for the same track.
    assert alerts.allow(candidate(clock, track_id=1), body) is False


async def test_handle_candidates_counts_suppressions(clock, alerts):
    body = rules_body(cooldown=30, stream_rate_limit_s=30)
    stored = await alerts.handle_candidates(
        [candidate(clock, track_id=1), candidate(clock, track_id=2)], lambda d, s: body
    )
    assert len(stored) == 1
    assert alerts.suppressed == 1


# -- event ids ----------------------------------------------------------
def test_event_ids_are_unique_and_sequential(clock, alerts):
    ids = [alerts.next_event_id(DEVICE, STREAM, "s1") for _ in range(3)]
    assert ids == [f"{DEVICE}-{STREAM}-s1-{n}" for n in (1, 2, 3)]
    # A restart (new session) cannot collide with the previous generation.
    assert alerts.next_event_id(DEVICE, STREAM, "s2") == f"{DEVICE}-{STREAM}-s2-1"


# -- snapshot state machine ---------------------------------------------
async def test_hub_mode_inserts_pending_and_requests_a_snapshot(clock, alerts):
    alert = await alerts.fire(candidate(clock), mode="hub")
    assert alert["snapshot_state"] == "pending"
    topics = [t for t, _, _ in alerts.published]
    assert f"sensecraft/security/{DEVICE}/cmd/snapshot" in topics
    assert f"sensecraft/security/{DEVICE}/events/{STREAM}" in topics
    cmd = next(json.loads(p) for t, p, _ in alerts.published if t.endswith("cmd/snapshot"))
    assert cmd == {"stream_id": STREAM, "event_id": alert["event_id"]}
    assert [m["type"] for m in alerts.pushed] == ["alert.new"]


async def test_republished_event_matches_the_contract(clock, alerts):
    await alerts.fire(candidate(clock, event_type="line_cross"), mode="hub")
    payload = next(
        json.loads(p) for t, p, _ in alerts.published if "/events/" in t
    )
    assert payload["schema"] == "sensecraft.event/1"
    assert payload["event_type"] == "line_cross"
    assert payload["device_id"] == DEVICE


async def test_snapshot_received_updates_and_pushes(clock, alerts):
    alert = await alerts.fire(candidate(clock), mode="hub")
    updated = await alerts.attach_snapshot(alert["event_id"], JPEG)
    assert updated["snapshot_state"] == "received"
    assert updated["snapshot_url"] == f"/api/alerts/{alert['id']}/snapshot.jpg"
    assert [m["type"] for m in alerts.pushed] == ["alert.new", "alert.update"]
    stored = alerts.storage.alert_snapshot_path(alert["id"])
    assert stored is not None and open(stored, "rb").read() == JPEG


async def test_snapshot_timeout_then_late_arrival_backfills(clock, alerts):
    alert = await alerts.fire(candidate(clock), mode="hub")
    timed_out = await alerts.expire_snapshot(alert["id"])
    assert timed_out["snapshot_state"] == "timeout"
    assert [m["type"] for m in alerts.pushed] == ["alert.new", "alert.update"]
    # HUB_SPEC §3: a snapshot arriving after the timeout is still associated.
    late = await alerts.attach_snapshot(alert["event_id"], JPEG)
    assert late["snapshot_state"] == "received"
    assert [m["type"] for m in alerts.pushed][-1] == "alert.update"


async def test_expire_is_a_noop_once_received(clock, alerts):
    alert = await alerts.fire(candidate(clock), mode="hub")
    await alerts.attach_snapshot(alert["event_id"], JPEG)
    after = await alerts.expire_snapshot(alert["id"])
    assert after["snapshot_state"] == "received"


async def test_single_box_event_starts_at_none(clock, alerts):
    alert = await alerts.ingest_device_event(device_event())
    assert alert["snapshot_state"] == "none"
    # No cmd/snapshot round trip in single-box mode.
    assert not [t for t, _, _ in alerts.published if t.endswith("cmd/snapshot")]
    received = await alerts.attach_snapshot(alert["event_id"], JPEG)
    assert received["snapshot_state"] == "received"


async def test_duplicate_device_event_id_is_ignored(clock, alerts):
    first = await alerts.ingest_device_event(device_event(event_id="dup-1"))
    assert first is not None
    assert await alerts.ingest_device_event(device_event(event_id="dup-1")) is None


async def test_snapshot_arriving_before_its_event_is_parked_and_associated(clock, alerts):
    event = device_event(event_id="early-1")
    assert await alerts.attach_snapshot("early-1", JPEG) is None
    alert = await alerts.ingest_device_event(event)
    assert alert["snapshot_state"] == "received"


async def test_parked_snapshot_is_swept_after_the_timeout(clock, alerts):
    await alerts.attach_snapshot("orphan-1", JPEG)
    clock.advance(61_000)
    assert alerts.sweep_parked() == 1
    alert = await alerts.ingest_device_event(device_event(event_id="orphan-1"))
    assert alert["snapshot_state"] == "none"


async def test_restart_rearms_pending_snapshots(clock, alerts):
    alert = await alerts.fire(candidate(clock), mode="hub")
    await alerts.shutdown()
    assert alerts.storage.get_alert(alert["id"])["snapshot_state"] == "pending"
    # Simulate the process coming back up with a short timer.
    alerts.snapshot_timeout_s = 0.01
    assert await alerts.rearm_pending_snapshots() == 1
    await asyncio.sleep(0.05)
    assert alerts.storage.get_alert(alert["id"])["snapshot_state"] == "timeout"
    assert [m["type"] for m in alerts.pushed][-1] == "alert.update"


async def test_rearm_skips_rows_that_already_have_evidence(clock, alerts):
    alert = await alerts.fire(candidate(clock), mode="hub")
    await alerts.attach_snapshot(alert["event_id"], JPEG)
    assert await alerts.rearm_pending_snapshots() == 0


# -- disposition --------------------------------------------------------
@pytest.mark.parametrize("target", ["acked", "dismissed"])
async def test_transition_pushes_an_update(clock, alerts, target):
    alert = await alerts.fire(candidate(clock), mode="hub")
    result, updated = await alerts.transition(alert["id"], target, "admin")
    assert result == "ok"
    assert updated["state"] == target
    assert updated["acted_by"] == "admin"
    assert [m["type"] for m in alerts.pushed][-1] == "alert.update"
