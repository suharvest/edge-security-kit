"""Clock abstraction.

HUB_SPEC §2.1 clock clause: dwell, cooldown and line-chain expiry all run on the
hub's own monotonic clock at receive time. The device `timestamp` is display /
evidence only. Wall time is used solely for stored `ts_ms` / `received_ms`
columns and for retention math.

Tests inject :class:`FakeClock` so timing behaviour is deterministic.
"""

from __future__ import annotations

import time


class Clock:
    """Real clock: monotonic milliseconds plus wall-clock epoch milliseconds."""

    def mono_ms(self) -> float:
        return time.monotonic() * 1000.0

    def wall_ms(self) -> int:
        return int(time.time() * 1000)


class FakeClock(Clock):
    """Manually advanced clock for tests."""

    def __init__(self, mono_ms: float = 0.0, wall_ms: int = 1_755_400_000_000) -> None:
        self._mono = float(mono_ms)
        self._wall = int(wall_ms)

    def mono_ms(self) -> float:
        return self._mono

    def wall_ms(self) -> int:
        return self._wall

    def advance(self, ms: float) -> None:
        """Advance both clocks by ``ms`` milliseconds."""
        self._mono += float(ms)
        self._wall += int(ms)
