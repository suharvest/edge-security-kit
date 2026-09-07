"""Device / stream liveness from the retained status topic + LWT (HUB_SPEC §1).

The registry is the only writer of the ``devices`` table. Per contracts/MQTT.md
the LWT payload carries the *connect* timestamp, so the offline instant recorded
here is the hub's own receipt time, never ``payload["timestamp"]``.

It also keeps the most recent detections message per stream in memory — the
backing store for ``GET /api/live/{device_id}/{stream_id}``. Detections are never
written to SQLite (HUB_SPEC §11).
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from .clock import Clock
from .storage import Storage

Broadcaster = Callable[[dict[str, Any]], Any]


class DeviceRegistry:
    def __init__(
        self,
        storage: Storage,
        clock: Clock | None = None,
        broadcaster: Broadcaster | None = None,
        on_session_change: Callable[[str, str], Awaitable[None] | None] | None = None,
    ) -> None:
        self.storage = storage
        self.clock = clock or Clock()
        self.broadcaster = broadcaster
        self.on_session_change = on_session_change
        self._devices: dict[str, dict[str, Any]] = {}
        #: (device_id, stream_id) -> last detections payload + receipt time
        self._live: dict[tuple[str, str], dict[str, Any]] = {}
        for row in storage.list_devices():
            self._devices[row["device_id"]] = row

    # -- status ----------------------------------------------------------
    async def on_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        device_id = str(payload["device_id"])
        received_ms = self.clock.wall_ms()
        online = bool(payload.get("online", True))
        previous = self._devices.get(device_id, {})
        mode = str(payload.get("mode") or previous.get("mode") or "hub")
        if mode not in ("hub", "single_box"):
            mode = "hub"

        streams = payload.get("streams")
        if streams is None:
            # LWT / goodbye payload: no streams array. Keep the last known stream
            # list so the UI still knows which streams the device had, but stop
            # reporting them as running at their last fps -- the device is gone.
            streams = [{**s, "state": "stopped", "fps": 0.0}
                       for s in (previous.get("streams") or [])]

        entry = {
            "device_id": device_id,
            "online": online,
            "last_seen_ms": received_ms,
            "mode": mode,
            "session_id": payload.get("session_id"),
            "device_ts_ms": payload.get("timestamp"),
            "versions": payload.get("versions") or previous.get("versions") or {},
            "uptime_s": payload.get("uptime_s"),
            "streams": [dict(s) for s in streams],
            # Carried through from health so /api/devices can answer "which
            # runtime is this actually running" without a subscriber having to
            # tap MQTT. Falls back to the previous value like versions does: a
            # goodbye or LWT payload has no health block, and blanking the
            # field on the way down would lose it exactly when someone is
            # looking at why the device went away.
            "backend": (payload.get("health") or {}).get("backend")
            or previous.get("backend"),
            "fallback_active": bool((payload.get("health") or {}).get("fallback_active"))
            or any(bool(s.get("fallback_active")) for s in streams),
        }
        changed = self._materially_changed(previous, entry)
        self._devices[device_id] = entry
        self.storage.upsert_device(
            device_id=device_id,
            online=online,
            last_seen_ms=received_ms,
            mode=mode,
            info={
                k: v
                for k, v in entry.items()
                if k not in ("device_id", "online", "last_seen_ms", "mode")
            },
        )
        session_id = payload.get("session_id")
        if (
            session_id
            and previous.get("session_id")
            and previous["session_id"] != session_id
            and self.on_session_change is not None
        ):
            result = self.on_session_change(device_id, str(session_id))
            if result is not None and hasattr(result, "__await__"):
                await result
        if changed:
            await self._broadcast(entry)
        return entry

    @staticmethod
    def _materially_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
        """§5: push device.status on up/down, decode change, fallback flip."""
        if not before:
            return True
        if before.get("online") != after.get("online"):
            return True
        if before.get("fallback_active") != after.get("fallback_active"):
            return True
        # A runtime swapped underneath a device -- a rebuilt image, a driver
        # upgrade -- changes what its numbers mean, so it is worth a push
        # rather than something a reader notices on the next poll.
        if before.get("backend") != after.get("backend"):
            return True
        def digest(entry: dict[str, Any]) -> list[tuple[Any, ...]]:
            return [
                (s.get("stream_id"), s.get("state"), s.get("decode"))
                for s in entry.get("streams") or []
            ]
        return digest(before) != digest(after)

    def note_single_box(self, device_id: str) -> None:
        """A device publishing its own events is in single-box mode."""
        entry = self._devices.setdefault(
            device_id,
            {"device_id": device_id, "online": True, "last_seen_ms": self.clock.wall_ms()},
        )
        if entry.get("mode") != "single_box":
            entry["mode"] = "single_box"
            self.storage.upsert_device(
                device_id=device_id,
                online=bool(entry.get("online", True)),
                last_seen_ms=int(entry.get("last_seen_ms") or self.clock.wall_ms()),
                mode="single_box",
                info={
                    k: v
                    for k, v in entry.items()
                    if k not in ("device_id", "online", "last_seen_ms", "mode")
                },
            )

    def mode(self, device_id: str) -> str:
        entry = self._devices.get(device_id)
        if entry and entry.get("mode"):
            return str(entry["mode"])
        return self.storage.device_mode(device_id)

    # -- detections cache -------------------------------------------------
    def note_detections(self, payload: dict[str, Any]) -> None:
        key = (str(payload["device_id"]), str(payload["stream_id"]))
        self._live[key] = {"received_ms": self.clock.wall_ms(), "payload": payload}

    def live(self, device_id: str, stream_id: str) -> dict[str, Any] | None:
        return self._live.get((device_id, stream_id))

    def live_all(self) -> list[dict[str, Any]]:
        """Every stream's last detections payload, newest state, one call.

        The video wall draws overlay boxes on up to nine tiles at a few hertz.
        Nine separate ``GET /live/{d}/{s}`` round trips per tick is nine times
        the request overhead for data that all comes out of the same in-memory
        dict, and the tiles then disagree with each other by a request latency.
        One call keeps the wall's boxes from the same instant.
        """
        return [
            {"device_id": device_id, "stream_id": stream_id, **entry}
            for (device_id, stream_id), entry in sorted(self._live.items())
        ]

    # -- views -----------------------------------------------------------
    def list_devices(self) -> list[dict[str, Any]]:
        return [self._devices[k] for k in sorted(self._devices)]

    async def _broadcast(self, entry: dict[str, Any]) -> None:
        if self.broadcaster is None:
            return
        result = self.broadcaster({"type": "device.status", "device": entry})
        if result is not None and hasattr(result, "__await__"):
            await result
