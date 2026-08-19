"""The status topic must not reset a generation the frames already established.

HUB_SPEC §2.1 resets all per-track state for a device when its ``session_id``
changes. Two channels carry that id — the detections topic (every frame) and the
status topic (every 30 s) — and the status one can be a full heartbeat interval
late: its first message of a session is dropped whenever the device publishes it
before the hub subscribed, the broker loses it across a reconnect, or the
contract validator rejects it (the reCamera Pro app announced
``state: "starting"``, which is not in the ``stream_status`` enum, so its connect
status never reached the registry).

When it finally lands, the engine is mid-scene on that same session. Resetting
there wipes ``zone_entered_at`` for a track that never left the zone, and the
loitering escalation it was owed is published as a second ``zone_enter``.
Measured on the reCamera Pro truth-clip runs: one such alert per detector
session, at 30.21–30.41 s after session start, in 7 of 7 sessions
(alerts 293, 346, 408, 437, 452, 480, 510).
"""

from __future__ import annotations

from conftest import DEVICE, STREAM, detection, rules_body, status

OUTSIDE = (0.2, 0.55)
INSIDE = (0.75, 0.55)

OLD_SESSION = "sess-0"
LIVE_SESSION = "sess-1"


def install_rules(hub, **overrides) -> None:
    overrides.setdefault("lines", [])
    overrides.setdefault("cooldown", 3)
    hub.storage.put_rules(DEVICE, STREAM, rules_body(**overrides), hub.clock.wall_ms())


async def walk_into_the_zone(hub, session_id: str = LIVE_SESSION) -> None:
    await hub.on_detections(detection(*OUTSIDE, frame_id=1, session_id=session_id))
    await hub.on_detections(detection(*INSIDE, frame_id=2, session_id=session_id))


def event_types(hub) -> list[str]:
    """Stored alerts oldest first — query_alerts() answers newest first."""
    return [row["event_type"] for row in sorted(hub.storage.query_alerts(),
                                                key=lambda r: r["id"])]


async def test_a_late_first_status_does_not_wipe_the_running_generation(hub, clock):
    install_rules(hub)
    # The registry's last word on this device is the previous session: the
    # current one's connect status never arrived.
    await hub.on_status(status(session_id=OLD_SESSION))

    await walk_into_the_zone(hub)
    assert event_types(hub) == ["zone_enter"]

    # 30 s in, the first heartbeat of the running session reaches the registry.
    # The track has not moved and its dwell is already past dwell_seconds.
    clock.advance(30_000)
    await hub.on_status(status(session_id=LIVE_SESSION))
    await hub.on_detections(detection(*INSIDE, frame_id=3, session_id=LIVE_SESSION))

    assert event_types(hub) == ["zone_enter", "loitering"]
    escalation = max(hub.storage.query_alerts(), key=lambda r: r["id"])
    assert escalation["dwell_s"] == 30.0


async def test_a_late_first_status_keeps_the_cooldown_state(hub, clock):
    """The generation's cooldowns belong to it too, and were dropped with it."""
    install_rules(hub, cooldown=30)
    await hub.on_status(status(session_id=OLD_SESSION))
    await walk_into_the_zone(hub)
    assert event_types(hub) == ["zone_enter"]

    clock.advance(1_000)
    await hub.on_status(status(session_id=LIVE_SESSION))
    # Leaving and re-entering inside the cooldown window must stay suppressed.
    await hub.on_detections(detection(*OUTSIDE, frame_id=3, session_id=LIVE_SESSION))
    await hub.on_detections(detection(*INSIDE, frame_id=4, session_id=LIVE_SESSION))
    assert event_types(hub) == ["zone_enter"]


async def test_a_status_announcing_an_unseen_session_still_resets(hub, clock):
    """The clause the guard must not disable (HUB_SPEC §2.1).

    A heartbeat that names a session the detections path has never evaluated is
    a real generation change: the detector restarted, its track ids restarted
    with it, and the state held for the old generation is meaningless.
    """
    install_rules(hub)
    await hub.on_status(status(session_id=OLD_SESSION))
    await walk_into_the_zone(hub, session_id=OLD_SESSION)
    assert event_types(hub) == ["zone_enter"]
    assert hub.engine.adopted_session(DEVICE) == OLD_SESSION

    clock.advance(30_000)
    await hub.on_status(status(session_id=LIVE_SESSION))
    assert hub.engine.adopted_session(DEVICE) is None
    assert hub.engine.track_count(DEVICE, STREAM) == 0

    # Track 1 of the new generation is a different subject: standing where the
    # old one stood is an entry, not a continuation.
    await hub.on_detections(detection(*INSIDE, frame_id=1, session_id=LIVE_SESSION))
    assert event_types(hub) == ["zone_enter", "zone_enter"]


async def test_adopted_session_follows_the_detections_path(hub):
    install_rules(hub)
    assert hub.engine.adopted_session(DEVICE) is None
    await hub.on_detections(detection(*OUTSIDE, frame_id=1, session_id=LIVE_SESSION))
    assert hub.engine.adopted_session(DEVICE) == LIVE_SESSION
    await hub.on_detections(detection(*OUTSIDE, frame_id=1, session_id="sess-2"))
    assert hub.engine.adopted_session(DEVICE) == "sess-2"
