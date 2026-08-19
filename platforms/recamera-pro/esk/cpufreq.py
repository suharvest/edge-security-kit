"""CPU frequency governor control and sampling -- a MEASUREMENT tool.

Why this module exists
----------------------

The zero-copy frame path (``059e689``) halved this app's per-frame CPU work and
``inference_time_ms`` *rose*, 30.6 -> 41.8 ms p50. The board's governor is
``interactive`` over 594 MHz - 1.608 GHz on a single policy covering all four
cores, and with the CPU work halved the load estimator settles at the bottom
step. The host side of an RKNN call -- input marshalling, the nine-output
dequant -- is CPU-bound, so a downclocked core stretches the call even though
the work shrank.

"Some of that 41.8 ms is DVFS" is a hypothesis until the clock is pinned. This
module pins it, so a baseline can be taken with the frequency held at the top
step and the two numbers can be subtracted.

**It is not a product setting.** ``performance`` on this policy holds all four
cores at 1.608 GHz whether or not anything is running, which is a power cost the
app has no business imposing on a camera that idles most of the day. The
governor is therefore restored by :class:`GovernorLock` on exit -- normal exit,
exception and SIGTERM alike -- and the default configuration locks nothing.

Writing ``scaling_governor`` needs root. Apps started by appmgr are root; the
unprivileged fleet account is not, so :func:`set_governor` reports the ``EACCES``
rather than raising, and a headless run simply measures whatever the board was
already doing.
"""

from __future__ import annotations

import threading
import time

POLICY_DIR = "/sys/devices/system/cpu/cpufreq/policy0"
CPU0_DIR = "/sys/devices/system/cpu/cpu0/cpufreq"


def _dirs():
    """Both spellings of the same policy; this kernel exposes both."""
    return (POLICY_DIR, CPU0_DIR)


def _read(name: str):
    for base in _dirs():
        try:
            with open(f"{base}/{name}") as fh:
                return fh.read().strip()
        except OSError:
            continue
    return None


def _read_int(name: str):
    raw = _read(name)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _write(name: str, value: str):
    """Write to the first policy path that accepts it.

    Returns ``(ok, detail)``. A failure is reported, never raised: locking the
    clock is an optional measurement aid, and a run that cannot lock it should
    still produce numbers -- clearly labelled as unlocked.
    """
    last = "no cpufreq policy directory found"
    for base in _dirs():
        try:
            with open(f"{base}/{name}", "w") as fh:
                fh.write(value)
            return True, f"{base}/{name}={value}"
        except OSError as exc:
            last = f"{base}/{name}: {exc}"
    return False, last


def snapshot() -> dict:
    """Everything about the current DVFS state, for the evidence record."""
    available = _read("scaling_available_frequencies") or ""
    return {
        "governor": _read("scaling_governor"),
        "available_governors": (_read("scaling_available_governors") or "").split(),
        "available_khz": [int(v) for v in available.split() if v.isdigit()],
        "cur_khz": _read_int("scaling_cur_freq"),
        "min_khz": _read_int("scaling_min_freq"),
        "max_khz": _read_int("scaling_max_freq"),
        "cpuinfo_cur_khz": _read_int("cpuinfo_cur_freq"),
    }


def cur_khz():
    """One ``scaling_cur_freq`` reading, world-readable on this kernel."""
    return _read_int("scaling_cur_freq")


def set_governor(spec: str) -> dict:
    """Switch the governor, and pin the clock to one step.

    ``spec`` is a governor name, optionally with a frequency: ``performance``
    pins to the top step, ``performance@594000`` pins to that step instead.

    Setting the governor alone is not enough to guarantee a pinned clock.
    ``performance`` runs at ``scaling_max_freq``, so raising ``min`` to meet
    ``max`` is what actually removes the remaining freedom -- and the explicit
    ``@khz`` form is what makes the *slow* baseline reproducible. The 41.8 ms
    inference figure was recorded with the board sitting at 594 MHz under
    ``interactive``; "how much of that was the clock" is only answerable by
    holding the clock at 594 MHz deliberately and comparing against the same
    board held at 1.608 GHz. Waiting for ``interactive`` to choose 594 MHz again
    is not an experiment, because the load that made it choose that is exactly
    what the comparison changes.
    """
    name, _, khz = spec.partition("@")
    before = snapshot()
    ok, detail = _write("scaling_governor", name)
    target = None
    pin = None
    if ok and name == "performance":
        target = int(khz) if khz.isdigit() else before.get("max_khz")
    if target:
        # max first when lowering, min first when raising: the kernel rejects a
        # min above the current max and a max below the current min, so the
        # order has to follow the direction of travel or one of the two writes
        # silently does nothing.
        cur_min = before.get("min_khz") or 0
        order = ("scaling_min_freq", "scaling_max_freq") if target >= cur_min else (
            "scaling_max_freq", "scaling_min_freq"
        )
        pin = {}
        for knob in order:
            knob_ok, knob_detail = _write(knob, str(target))
            pin[knob] = {"ok": knob_ok, "detail": knob_detail}
    time.sleep(0.3)
    after = snapshot()
    return {
        "requested": spec,
        "governor": name,
        "target_khz": target,
        "ok": ok,
        "detail": detail,
        "pin": pin,
        # The check that matters: did the board actually land where it was told?
        "pinned": bool(target) and after.get("cur_khz") == target,
        "before": before,
        "after": after,
    }


class GovernorLock:
    """Hold a governor for the life of a run and put the board back afterwards.

    The restore is the point. ``performance`` costs power on a device that spends
    most of its life idle, so a measurement run must not be able to leave it
    behind -- not on a clean exit, not on an exception, and not on the SIGTERM
    appmgr sends when it switches apps. ``restore()`` is idempotent so all three
    paths can call it.
    """

    def __init__(self, governor: str):
        self.governor = governor
        self.applied = None
        self._prev = None
        self._restored = False

    def apply(self) -> dict:
        self._prev = snapshot()
        self.applied = set_governor(self.governor)
        return self.applied

    def restore(self) -> dict:
        if self._restored or self._prev is None:
            return {"restored": self._restored, "reason": "nothing to restore"}
        self._restored = True
        prev = self._prev
        # Widen the window before handing the policy back, and widen it in the
        # order the kernel accepts (max up first, then min down) -- restoring
        # `min` while `max` is still pinned low is rejected, which would leave
        # the board clamped at the measurement frequency under a governor that
        # looks correct.
        detail = {}
        for knob, key in (
            ("scaling_max_freq", "max_khz"),
            ("scaling_min_freq", "min_khz"),
        ):
            if prev.get(key) is not None:
                ok, raw = _write(knob, str(prev[key]))
                detail[knob] = {"ok": ok, "detail": raw}
        gov_ok, gov_detail = _write("scaling_governor", prev.get("governor") or "")
        time.sleep(0.3)
        after = snapshot()
        return {
            "restored": True,
            "governor_ok": gov_ok,
            "governor_detail": gov_detail,
            "freq": detail,
            "before_restore": self.applied.get("after") if self.applied else None,
            "after": after,
            # Restoration is only claimed when the board matches what was found.
            "matches_original": (
                after.get("governor") == prev.get("governor")
                and after.get("min_khz") == prev.get("min_khz")
                and after.get("max_khz") == prev.get("max_khz")
            ),
        }

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, *exc):
        self.restore()
        return False


class FreqSampler:
    """Background ``scaling_cur_freq`` sampler.

    A single reading taken next to a latency number proves nothing -- the whole
    question is whether the clock *stayed* where it was put. This keeps a
    histogram of every sample over the run, which is what distinguishes "pinned
    at 1.608 GHz" from "averaged 1.4 GHz while bouncing".
    """

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = float(interval_s)
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self) -> "FreqSampler":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="cpufreq-sampler"
            )
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            khz = cur_khz()
            if khz is None:
                continue
            with self._lock:
                self._counts[khz] = self._counts.get(khz, 0) + 1

    def stop(self) -> None:
        self._stop.set()

    def report(self) -> dict:
        with self._lock:
            counts = dict(self._counts)
        total = sum(counts.values())
        if not total:
            return {"samples": 0, "histogram_khz": {}, "mean_khz": None}
        mean = sum(k * n for k, n in counts.items()) / total
        return {
            "samples": total,
            "histogram_khz": {str(k): counts[k] for k in sorted(counts)},
            "mean_khz": round(mean, 1),
            "min_khz": min(counts),
            "max_khz": max(counts),
            "pinned_pct": round(100.0 * counts[max(counts)] / total, 2),
        }
