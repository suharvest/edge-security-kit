"""Alert lifecycle: cooldown, event_id, snapshot state machine, WS fan-out.

HUB_SPEC §3 and §3.1. Two behaviours here are direct fixes of upstream defects:

* cooldown is keyed ``(device_id, stream_id, rule_name, track_id)`` *and* rate
  limited per ``(device_id, stream_id, rule_name)``. Upstream hung the cooldown
  on the track object, so a tracker ID change re-fired immediately
  (multi_camera_manager.py:464-482).
* the snapshot round trip is asynchronous: the alert is stored and pushed first,
  the evidence is attached later, and a late snapshot still lands after the
  60 s timeout.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .clock import Clock
from .rules.engine import DEFAULT_COOLDOWN_S, Candidate
from .storage import Storage

log = logging.getLogger("edge_hub.alerts")

Publisher = Callable[[str, bytes, int], Awaitable[None]]
Broadcaster = Callable[[dict[str, Any]], Any]

#: Marker the hub stamps on events it republishes on a device's behalf.
ORIGIN_HUB = "hub"


@dataclass
class _Parked:
    """A snapshot that arrived before its event (single-box ordering, §3.1)."""

    payload: bytes
    received_mono: float


class AlertManager:
    def __init__(
        self,
        storage: Storage,
        clock: Clock | None = None,
        publisher: Publisher | None = None,
        broadcaster: Broadcaster | None = None,
        topic_prefix: str = "sensecraft/security",
        snapshot_timeout_s: float = 60.0,
        default_cooldown_s: float = DEFAULT_COOLDOWN_S,
    ) -> None:
        self.storage = storage
        self.clock = clock or Clock()
        self.publisher = publisher
        self.broadcaster = broadcaster
        self.topic_prefix = topic_prefix
        self.snapshot_timeout_s = snapshot_timeout_s
        self.default_cooldown_s = default_cooldown_s
        #: (device, stream, rule_name, track_id) -> last fire, hub monotonic ms
        self._track_cooldown: dict[tuple[str, str, str, int], float] = {}
        #: (device, stream, rule_name) -> last fire, hub monotonic ms
        self._stream_cooldown: dict[tuple[str, str, str], float] = {}
        #: (device, stream, session) -> event sequence number
        self._seq: dict[tuple[str, str, str], int] = {}
        self._parked_snapshots: dict[str, _Parked] = {}
        self._timers: dict[int, asyncio.Task[None]] = {}
        self.suppressed = 0

    # -- helpers ---------------------------------------------------------
    def next_event_id(self, device_id: str, stream_id: str, session_id: str) -> str:
        key = (device_id, stream_id, session_id)
        last = self._seq.get(key)
        if last is None:
            # First event of this generation *in this process*, but the generation
            # can be older than the process: a hub restart leaves an already
            # connected device on its existing session_id, so restarting the
            # counter at 1 regenerates event_ids that are already in the table.
            # event_id is UNIQUE, so every insert for that device would fail for
            # as long as it stays connected. Seed from the stored high-water mark.
            last = self.storage.max_event_seq(device_id, stream_id, session_id)
        seq = last + 1
        self._seq[key] = seq
        return f"{device_id}-{stream_id}-{session_id}-{seq}"

    def reset_session(self, device_id: str) -> None:
        """Drop cooldown/sequence state for a device on a generation change."""
        for key in [k for k in self._track_cooldown if k[0] == device_id]:
            del self._track_cooldown[key]
        for key in [k for k in self._stream_cooldown if k[0] == device_id]:
            del self._stream_cooldown[key]

    def _cooldowns(self, body: dict[str, Any] | None) -> tuple[float, float]:
        cooldown = float((body or {}).get("cooldown", self.default_cooldown_s))
        # Default 0 = no stream-level limit. A non-zero value collapses two
        # different people tripping the same rule within the window into one
        # alert, so it is opt-in; the per-track cooldown always applies.
        rate_limit = float((body or {}).get("stream_rate_limit_s", 0.0))
        return cooldown * 1000.0, rate_limit * 1000.0

    def allow(self, candidate: Candidate, body: dict[str, Any] | None) -> bool:
        """Cooldown + stream rate limit gate (HUB_SPEC §2 `_emit` row)."""
        cooldown_ms, rate_ms = self._cooldowns(body)
        now = candidate.received_mono
        # event_type is part of the key: zone_enter and loitering share a
        # rule_name (the zone), and the upstream cooldown was per event type
        # (`track.fired_events[etype]`). Dropping it here would let the entry
        # alert swallow the loitering escalation for the same track — losing the
        # more serious of the two.
        tkey = (
            candidate.device_id,
            candidate.stream_id,
            candidate.rule_name,
            candidate.event_type,
            candidate.track_id,
        )
        skey = (
            candidate.device_id,
            candidate.stream_id,
            candidate.rule_name,
            candidate.event_type,
        )
        last_track = self._track_cooldown.get(tkey)
        if last_track is not None and now - last_track < cooldown_ms:
            return False
        last_stream = self._stream_cooldown.get(skey)
        if rate_ms > 0 and last_stream is not None and now - last_stream < rate_ms:
            return False
        self._track_cooldown[tkey] = now
        self._stream_cooldown[skey] = now
        return True

    # -- firing ----------------------------------------------------------
    async def handle_candidates(
        self,
        candidates: list[Candidate],
        rules_provider: Callable[[str, str], dict[str, Any] | None],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for candidate in candidates:
            body = rules_provider(candidate.device_id, candidate.stream_id)
            if not self.allow(candidate, body):
                self.suppressed += 1
                continue
            alert = await self.fire(candidate, mode="hub")
            out.append(alert)
        return out

    async def fire(
        self,
        candidate: Candidate,
        mode: str = "hub",
        event_id: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store, push, republish, and (hub mode) request the snapshot."""
        event_id = event_id or self.next_event_id(
            candidate.device_id, candidate.stream_id, candidate.session_id
        )
        meta = dict(candidate.meta)
        if extra_meta:
            meta.update(extra_meta)
        meta.setdefault("mode", mode)
        # §3: hub mode arms a cmd/snapshot round trip -> pending.
        # single-box mode has no round trip -> none until a snapshot shows up.
        snapshot_state = "pending" if mode == "hub" else "none"
        alert = self.storage.insert_alert(
            event_id=event_id,
            ts_ms=candidate.ts_ms,
            received_ms=candidate.received_ms,
            device_id=candidate.device_id,
            stream_id=candidate.stream_id,
            event_type=candidate.event_type,
            rule_name=candidate.rule_name,
            track_id=candidate.track_id,
            score=candidate.score,
            bbox=candidate.bbox,
            direction=candidate.direction,
            dwell_s=candidate.dwell_s,
            snapshot_state=snapshot_state,
            meta=meta,
        )
        await self._broadcast({"type": "alert.new", "alert": alert})
        if mode == "hub":
            await self._republish_event(candidate, event_id)
            await self._request_snapshot(candidate.device_id, candidate.stream_id, event_id)
        # A snapshot that beat its event (single-box) is waiting here.
        parked = self._parked_snapshots.pop(event_id, None)
        if parked is not None:
            await self.attach_snapshot(event_id, parked.payload)
        elif mode == "hub":
            self.arm_snapshot_timeout(alert["id"])
        return self.storage.get_alert(alert["id"]) or alert

    async def ingest_device_event(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Single-box mode: trust the device's own event (HUB_SPEC §1 rule row)."""
        event_id = str(payload["event_id"])
        if self.storage.get_alert_by_event_id(event_id):
            return None
        candidate = Candidate(
            device_id=str(payload["device_id"]),
            stream_id=str(payload["stream_id"]),
            session_id=str(payload["session_id"]),
            event_type=str(payload["event_type"]),
            rule_name=str(payload["rule_name"]),
            rule_id=str(payload["rule_name"]),
            track_id=int(payload["track_id"]),
            bbox=[float(v) for v in payload["bbox"]],
            ts_ms=int(payload["timestamp"]),
            received_ms=self.clock.wall_ms(),
            received_mono=self.clock.mono_ms(),
            score=payload.get("score"),
            cls=payload.get("class"),
            direction=payload.get("direction"),
            dwell_s=payload.get("dwell_s"),
            meta={"source": "device"},
        )
        return await self.fire(candidate, mode="single_box", event_id=event_id)

    # -- snapshots -------------------------------------------------------
    async def _request_snapshot(self, device_id: str, stream_id: str, event_id: str) -> None:
        if self.publisher is None:
            return
        topic = f"{self.topic_prefix}/{device_id}/cmd/snapshot"
        body = json.dumps({"stream_id": stream_id, "event_id": event_id}).encode()
        await self.publisher(topic, body, 1)

    async def _republish_event(self, candidate: Candidate, event_id: str) -> None:
        """Hub-judged events go back on events/<stream_id> (MQTT.md Modes)."""
        if self.publisher is None:
            return
        payload: dict[str, Any] = {
            "schema": "sensecraft.event/1",
            # `origin` is an additive field (the schema allows extra properties).
            # The hub subscribes to events/+ for single-box devices, so without a
            # marker it would ingest its own republished event and conclude that
            # the publishing device runs rules locally.
            "origin": ORIGIN_HUB,
            "timestamp": candidate.ts_ms,
            "session_id": candidate.session_id,
            "event_id": event_id,
            "device_id": candidate.device_id,
            "stream_id": candidate.stream_id,
            "event_type": candidate.event_type,
            "rule_name": candidate.rule_name,
            "track_id": candidate.track_id,
            "bbox": candidate.bbox,
        }
        if candidate.direction is not None:
            payload["direction"] = candidate.direction
        if candidate.dwell_s is not None:
            payload["dwell_s"] = candidate.dwell_s
        if candidate.score is not None:
            payload["score"] = candidate.score
        if candidate.cls is not None:
            payload["class"] = candidate.cls
        topic = (
            f"{self.topic_prefix}/{candidate.device_id}/events/{candidate.stream_id}"
        )
        await self.publisher(topic, json.dumps(payload).encode(), 1)

    def arm_snapshot_timeout(self, alert_id: int, delay_s: float | None = None) -> None:
        """Start the §3 60 s timer; a no-op without a running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        delay = self.snapshot_timeout_s if delay_s is None else delay_s
        old = self._timers.pop(alert_id, None)
        if old is not None:
            old.cancel()
        self._timers[alert_id] = loop.create_task(self._snapshot_timeout(alert_id, delay))

    async def _snapshot_timeout(self, alert_id: int, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        await self.expire_snapshot(alert_id)

    async def expire_snapshot(self, alert_id: int) -> dict[str, Any] | None:
        """Move a still-pending alert to ``timeout`` and push the update."""
        self._timers.pop(alert_id, None)
        alert = self.storage.get_alert(alert_id)
        if alert is None or alert["snapshot_state"] != "pending":
            return alert
        updated = self.storage.set_snapshot_state(alert_id, "timeout")
        if updated:
            await self._broadcast({"type": "alert.update", "alert": updated})
        return updated

    async def rearm_pending_snapshots(self) -> int:
        """HUB_SPEC §3: on restart, re-arm the timer for every pending row."""
        rows = self.storage.pending_snapshot_alerts()
        for row in rows:
            self.arm_snapshot_timeout(row["id"])
        return len(rows)

    async def attach_snapshot(self, event_id: str, payload: bytes) -> dict[str, Any] | None:
        """Associate a JPEG with its alert; valid from pending/timeout/none.

        A snapshot with no alert yet is parked for ``snapshot_timeout_s`` so the
        single-box ordering (snapshot first, event second) still associates.
        """
        alert = self.storage.get_alert_by_event_id(event_id)
        if alert is None:
            self._parked_snapshots[event_id] = _Parked(
                payload=payload, received_mono=self.clock.mono_ms()
            )
            self.sweep_parked()
            return None
        timer = self._timers.pop(alert["id"], None)
        if timer is not None:
            timer.cancel()
        target = self.storage.snapshot_target(event_id, self.clock.wall_ms())
        target.write_bytes(payload)
        updated = self.storage.set_snapshot(alert["id"], str(target), "received")
        if updated:
            await self._broadcast({"type": "alert.update", "alert": updated})
        return updated

    def sweep_parked(self) -> int:
        cutoff = self.clock.mono_ms() - self.snapshot_timeout_s * 1000.0
        stale = [k for k, v in self._parked_snapshots.items() if v.received_mono < cutoff]
        for key in stale:
            del self._parked_snapshots[key]
        return len(stale)

    # -- disposition -----------------------------------------------------
    async def transition(self, alert_id: int, target: str, actor: str) -> tuple[str, dict[str, Any] | None]:
        result, alert = self.storage.transition_alert(
            alert_id, target, actor, self.clock.wall_ms()
        )
        if result == "ok" and alert is not None:
            await self._broadcast({"type": "alert.update", "alert": alert})
        return result, alert

    # -- fan-out ---------------------------------------------------------
    async def _broadcast(self, message: dict[str, Any]) -> None:
        if self.broadcaster is None:
            return
        result = self.broadcaster(message)
        if asyncio.iscoroutine(result):
            await result

    async def shutdown(self) -> None:
        for task in list(self._timers.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._timers.clear()
