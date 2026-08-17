"""Tracker contract: IDs start at 1, never 0, and survive small motion."""

from __future__ import annotations

from dataclasses import dataclass

from esk_generic.tracker import IoUTracker


@dataclass
class Det:
    box: list[float]
    score: float = 0.9


def test_first_track_id_is_one():
    tracker = IoUTracker()
    tracked = tracker.update([Det([0.1, 0.1, 0.2, 0.4])], 0.0)
    assert [t.track_id for t, _ in tracked] == [1]


def test_id_is_stable_across_small_motion():
    tracker = IoUTracker()
    tracker.update([Det([0.10, 0.10, 0.20, 0.40])], 0.0)
    tracked = tracker.update([Det([0.11, 0.10, 0.21, 0.40])], 0.1)
    assert [t.track_id for t, _ in tracked] == [1]


def test_expired_track_gets_a_new_higher_id():
    tracker = IoUTracker(max_lost_sec=0.5)
    tracker.update([Det([0.1, 0.1, 0.2, 0.4])], 0.0)
    tracker.update([], 1.0)  # expired
    tracked = tracker.update([Det([0.1, 0.1, 0.2, 0.4])], 1.1)
    assert [t.track_id for t, _ in tracked] == [2]


def test_ids_are_monotonic_and_never_zero():
    tracker = IoUTracker()
    seen = []
    for step in range(5):
        offset = step * 0.15
        tracked = tracker.update(
            [Det([0.05 + offset, 0.1, 0.10 + offset, 0.4]), Det([0.6, 0.5, 0.7, 0.9])],
            step * 0.1,
        )
        seen += [t.track_id for t, _ in tracked]
    assert 0 not in seen
    assert min(seen) == 1
    assert max(seen) == tracker.next_id - 1
