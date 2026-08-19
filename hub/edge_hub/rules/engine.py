"""Rule engine: detections -> candidate alerts (HUB_SPEC §2, §2.1).

The engine is synchronous and side-effect free apart from its own in-memory
state, so every boundary clause in §2.1 is unit-testable without a broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..clock import Clock
from .line import evaluate_lines
from .state import DeviceState
from .zone import evaluate_zones

DEFAULT_TRACK_EXPIRY_S = 5.0
DEFAULT_LINE_CHAIN_GAP_MS = 1000.0
DEFAULT_COOLDOWN_S = 30.0

#: reason strings recorded in :attr:`RuleEngine.stats` when a message is dropped
DROP_OUT_OF_ORDER = "frame_out_of_order"
DROP_NO_RULES = "no_rules"


@dataclass
class Candidate:
    """A rule firing, before cooldown / de-duplication."""

    device_id: str
    stream_id: str
    session_id: str
    event_type: str
    rule_name: str
    rule_id: str
    track_id: int
    bbox: list[float]
    ts_ms: int
    received_ms: int
    received_mono: float
    score: float | None = None
    cls: str | None = None
    direction: str | None = None
    dwell_s: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class RuleEngine:
    """Evaluates zone / loitering / line rules for hub-mode devices."""

    def __init__(
        self,
        rules_provider: Callable[[str, str], dict[str, Any] | None],
        clock: Clock | None = None,
    ) -> None:
        self._rules_provider = rules_provider
        self._clock = clock or Clock()
        self._devices: dict[str, DeviceState] = {}
        self.stats: dict[str, int] = {DROP_OUT_OF_ORDER: 0, DROP_NO_RULES: 0}

    # -- state -----------------------------------------------------------
    def device(self, device_id: str) -> DeviceState:
        st = self._devices.get(device_id)
        if st is None:
            st = DeviceState(device_id=device_id)
            self._devices[device_id] = st
        return st

    def reset_device(self, device_id: str) -> None:
        self._devices.pop(device_id, None)

    def adopted_session(self, device_id: str) -> str | None:
        """The session_id whose frames this engine currently holds state for.

        ``None`` until the device's first detection message. Callers that reset
        the generation from a slower channel (the status topic) compare against
        it, so a late announcement of a session the frames already established
        does not wipe that session's live state.
        """
        dev = self._devices.get(device_id)
        return None if dev is None else dev.session_id

    def track_count(self, device_id: str, stream_id: str) -> int:
        dev = self._devices.get(device_id)
        if dev is None or stream_id not in dev.streams:
            return 0
        return len(dev.streams[stream_id].tracks)

    # -- evaluation ------------------------------------------------------
    def on_detections(self, payload: dict[str, Any]) -> list[Candidate]:
        """Evaluate one validated ``sensecraft.detection/1`` payload."""
        mono_ms = self._clock.mono_ms()
        received_ms = self._clock.wall_ms()
        device_id = str(payload["device_id"])
        stream_id = str(payload["stream_id"])
        session_id = str(payload["session_id"])
        frame_id = int(payload["frame_id"])

        dev = self.device(device_id)
        # §2.1 generation reset: a new session_id drops all per-track state for
        # the whole device (every stream), including line-chain origins.
        if dev.session_id != session_id:
            dev.reset_generation(session_id)

        stream = dev.stream(stream_id)
        # §2.1 out-of-order guard: frame_id <= last is dropped outright.
        if stream.last_frame_id is not None and frame_id <= stream.last_frame_id:
            self.stats[DROP_OUT_OF_ORDER] += 1
            return []
        stream.last_frame_id = frame_id

        body = self._rules_provider(device_id, stream_id)
        if not body:
            self.stats[DROP_NO_RULES] += 1
            return []

        zones = body.get("zones") or []
        lines = body.get("lines") or []
        features = body.get("features") or {}
        gap_ms = float(body.get("line_chain_gap_ms", DEFAULT_LINE_CHAIN_GAP_MS))
        expiry_ms = float(body.get("track_expiry_s", DEFAULT_TRACK_EXPIRY_S)) * 1000.0

        candidates: list[Candidate] = []
        for det in payload.get("detections") or []:
            track_id = int(det.get("track_id", 0))
            # §2.1 untracked clause: track_id 0 counts toward statistics only.
            if track_id < 1:
                continue
            bbox = [float(v) for v in det["bbox"]]
            centroid = (bbox[0], bbox[1])
            track = stream.track(track_id, mono_ms)

            fired = evaluate_zones(track, centroid, zones, features, mono_ms)
            fired += evaluate_lines(
                track, centroid, lines, features, mono_ms, gap_ms
            )
            for hit in fired:
                candidates.append(
                    Candidate(
                        device_id=device_id,
                        stream_id=stream_id,
                        session_id=session_id,
                        event_type=hit["event_type"],
                        rule_name=hit["rule_name"],
                        rule_id=hit["rule_id"],
                        track_id=track_id,
                        bbox=bbox,
                        ts_ms=int(payload["timestamp"]),
                        received_ms=received_ms,
                        received_mono=mono_ms,
                        score=det.get("score"),
                        cls=det.get("class"),
                        direction=hit.get("direction"),
                        dwell_s=hit.get("dwell_s"),
                        meta={"frame_id": frame_id},
                    )
                )

        stream.expire_tracks(mono_ms, expiry_ms)
        return candidates
