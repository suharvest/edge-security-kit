"""Per-track rule state, indexed device / stream / track (HUB_SPEC §2 last row).

Replaces the upstream per-``Track`` attributes (`zone_entered_at`,
`fired_events`, `history`) which lived inside the video pipeline. Cooldown
bookkeeping moved out to :mod:`edge_hub.alert_manager`; what stays here is the
state the geometry needs: zone entry instants and the previous centroid of the
line chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass
class TrackState:
    track_id: int
    last_seen_mono: float
    #: zone key -> hub monotonic ms at which the track entered that zone
    zone_entered_at: dict[str, float] = field(default_factory=dict)
    #: previous centroid, and the hub monotonic ms at which it was received
    last_point: Point | None = None
    last_point_mono: float | None = None

    def reset_line_chain(self, point: Point, mono_ms: float) -> None:
        self.last_point = point
        self.last_point_mono = mono_ms


@dataclass
class StreamState:
    stream_id: str
    #: highest frame_id accepted for this (device, stream, session)
    last_frame_id: int | None = None
    tracks: dict[int, TrackState] = field(default_factory=dict)

    def track(self, track_id: int, mono_ms: float) -> TrackState:
        st = self.tracks.get(track_id)
        if st is None:
            st = TrackState(track_id=track_id, last_seen_mono=mono_ms)
            self.tracks[track_id] = st
        else:
            st.last_seen_mono = mono_ms
        return st

    def expire_tracks(self, mono_ms: float, expiry_ms: float) -> None:
        stale = [
            tid
            for tid, st in self.tracks.items()
            if mono_ms - st.last_seen_mono > expiry_ms
        ]
        for tid in stale:
            del self.tracks[tid]


@dataclass
class DeviceState:
    device_id: str
    session_id: str | None = None
    streams: dict[str, StreamState] = field(default_factory=dict)

    def stream(self, stream_id: str) -> StreamState:
        st = self.streams.get(stream_id)
        if st is None:
            st = StreamState(stream_id=stream_id)
            self.streams[stream_id] = st
        return st

    def reset_generation(self, session_id: str) -> None:
        """HUB_SPEC §2.1 generation clause.

        A new ``session_id`` drops every stream's track state and line-chain
        origin for this device, and forgets the frame_id watermark (monotonicity
        only holds within one session).
        """
        self.session_id = session_id
        self.streams.clear()
