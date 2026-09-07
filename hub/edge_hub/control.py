"""Runtime control downlink: publish one command, wait for its ack.

``contracts/MQTT.md`` "Control downlink" defines the pair. This module is the
hub's half of it, and it exists to hold one property that a fire-and-forget
publish cannot: **the REST caller learns whether the change is live.**

The console has a confidence slider and an "add camera" dialog. Both are
operations an operator performs while looking at the scene, so both need an
answer — the slider must not settle at a value the detector never took, and the
camera list must not sprout a row for a stream that failed to open. So every
command carries a ``request_id``, the hub parks a future under it, and the
detector's ack resolves it. Silence resolves nothing: the wait times out and the
REST layer answers 504 rather than 200.

Nothing here retries. A redelivered command is the broker's business (QoS 1) and
a detector must apply a repeated ``request_id`` once; a hub-side retry would
turn one operator click into two ``add_stream`` calls with no way for the
detector to tell them apart.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Awaitable, Callable

log = logging.getLogger("edge_hub.control")

COMMAND_SCHEMA = "sensecraft.command/1"
ACK_SCHEMA = "sensecraft.ack/1"
#: Long enough for a detector to actually open an RTSP source before answering,
#: short enough that an operator does not think the console has hung. The
#: detector is expected to ack the failure itself well inside this.
DEFAULT_TIMEOUT_S = 8.0

Publisher = Callable[[str, bytes, int], Awaitable[None]]


class ControlTimeout(Exception):
    """No ack arrived inside the window. Says nothing about the device state."""


class ControlRejected(Exception):
    """The detector answered ``ok: false``. ``error`` is operator-facing."""

    def __init__(self, message: str, applied: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.applied = applied or {}


class ControlPlane:
    def __init__(
        self,
        publisher: Publisher,
        topic_prefix: str = "sensecraft/security",
        clock: Any = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.publisher = publisher
        self.topic_prefix = topic_prefix.rstrip("/")
        self.clock = clock
        self.timeout_s = timeout_s
        #: request_id -> future resolved by :meth:`on_ack`
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.sent = 0
        self.acked = 0
        self.timeouts = 0
        #: acks for a request nobody is waiting on — a late answer to a request
        #: that already timed out, or a duplicate. Counted rather than logged at
        #: warning level, because both are normal on a lossy link.
        self.orphan_acks = 0

    def topic(self, device_id: str) -> str:
        return f"{self.topic_prefix}/{device_id}/cmd/control"

    def _now_ms(self) -> int:
        if self.clock is not None:
            return int(self.clock.wall_ms())
        import time

        return int(time.time() * 1000)

    async def send(
        self,
        device_id: str,
        command: str,
        params: dict[str, Any],
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Publish one command and return the detector's ``applied`` block.

        Raises :class:`ControlTimeout` on silence and :class:`ControlRejected`
        when the detector answers ``ok: false``.
        """
        request_id = f"hub-{uuid.uuid4().hex[:12]}"
        payload = {
            "schema": COMMAND_SCHEMA,
            "timestamp": self._now_ms(),
            "request_id": request_id,
            "device_id": device_id,
            "command": command,
            "params": params,
        }
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        try:
            await self.publisher(
                self.topic(device_id), json.dumps(payload).encode("utf-8"), 1
            )
            self.sent += 1
            ack = await asyncio.wait_for(future, timeout_s or self.timeout_s)
        except asyncio.TimeoutError:
            self.timeouts += 1
            raise ControlTimeout(
                f"{device_id} did not answer {command} within "
                f"{timeout_s or self.timeout_s:g}s"
            ) from None
        finally:
            self._pending.pop(request_id, None)
        self.acked += 1
        if not ack.get("ok"):
            raise ControlRejected(
                str(ack.get("error") or "device rejected the command"),
                ack.get("applied") if isinstance(ack.get("applied"), dict) else None,
            )
        applied = ack.get("applied")
        return applied if isinstance(applied, dict) else {}

    async def on_ack(self, payload: dict[str, Any]) -> None:
        """Resolve the parked request. Called from the MQTT ingest path."""
        request_id = str(payload.get("request_id") or "")
        future = self._pending.get(request_id)
        if future is None or future.done():
            self.orphan_acks += 1
            log.debug("ack for an unknown or settled request_id %r", request_id)
            return
        future.set_result(payload)

    def stats(self) -> dict[str, int]:
        return {
            "control_sent": self.sent,
            "control_acked": self.acked,
            "control_timeouts": self.timeouts,
            "control_orphan_acks": self.orphan_acks,
        }
