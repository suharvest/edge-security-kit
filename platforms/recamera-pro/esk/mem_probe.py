"""Locate a growing RSS from inside the process, over HTTP.

The detector's RSS climbs while it runs (see this platform's README). ``ps`` can
say how fast but never what, and the app is started by appmgr as root with no
readable stdout, so the usual `tracemalloc` print-to-console loop is not
available. This module makes the answer fetchable: ``/debug/memory`` returns,
for the live process,

* ``rss_kb`` / ``vms_kb`` from ``/proc/self/status`` -- the number ``ps`` shows,
  so the HTTP answer and the external sampler are comparable;
* ``gc``: object counts by type, and the delta against the previous call. A
  Python-level leak shows up as one type climbing monotonically across calls;
* ``tracemalloc``: the top allocation sites by size, and their delta against the
  previous call. Only present when tracing was started (config ``mem_probe``);
* ``malloc``: ``sys._debugmallocstats``-free summary from ``/proc/self/smaps_rollup``
  when available, which is what separates "Python is holding objects" from
  "the allocator/an extension is holding pages Python no longer owns".

Reading it costs a `gc.get_objects()` walk (tens of ms at this heap size) and is
therefore an explicit debug endpoint, not something the loop calls.

Interpreting the three layers, in order
---------------------------------------
1. ``gc.type_counts`` grows      -> a Python object is being retained; the type
   names the owner and ``tracemalloc`` names the line.
2. ``gc`` flat, ``tracemalloc`` grows -> Python allocations that are not
   GC-tracked (bytes, str, numpy payloads) are being retained.
3. both flat, ``rss_kb`` grows   -> the growth is outside CPython's allocator:
   a native extension (librknnrt, librga, the SDK) or allocator fragmentation.
   This is the case that no Python-level fix can reach.
"""

from __future__ import annotations

import gc
import sys

_prev_types: dict = {}
_prev_trace: dict = {}
_tracing = False
_allocator_tuned: dict = {"applied": False}

#: glibc mallopt parameter numbers (malloc.h). Negative by design.
M_TRIM_THRESHOLD = -1
M_MMAP_THRESHOLD = -3


def tune_allocator(mmap_threshold: int = 128 * 1024,
                   trim_threshold: int = 256 * 1024) -> dict:
    """Pin glibc's mmap threshold. **Tested against the RSS growth; did not fix it.**

    Kept as an opt-in switch (``malloc_tune``), not a default, with the result
    recorded here so the hypothesis is not re-litigated from scratch.

    The hypothesis. Over 7 minutes at 6 fps, ``rss_kb`` rose 234 916 -> 276 368 kB
    while ``gc.tracked_objects`` moved 48 299 -> 48 739 and every entry in
    ``gc.type_counts`` stayed flat. Python was holding nothing; the pages were
    ``Private_Dirty`` anonymous memory.

    Why that happens here. Each frame allocates several megabyte-scale blocks --
    the RGA full-resolution RGB, the letterbox canvas, the frame store's copy --
    and frees them a few milliseconds later. glibc 2.38 serves blocks above
    ``M_MMAP_THRESHOLD`` (128 kB initially) with ``mmap`` and returns them on
    free, but it also *auto-tunes that threshold upward* -- up to 32 MB -- every
    time an mmap'd block is freed, on the theory that a program repeatedly
    allocating that size will want it back. After a few dozen frames every one
    of those buffers is being carved out of the heap instead, and a heap block
    freed below a live one cannot be returned to the kernel. RSS then only ever
    goes up, at a rate set by fragmentation rather than by any leak.

    Pinning the threshold disables that auto-tuning (an explicit ``mallopt``
    call is documented as doing exactly that), so those buffers should keep
    going through ``mmap``/``munmap``.

    The measurement, on this board, glibc 2.38, ``rc: [1, 1]`` (both calls
    accepted): RSS still climbed -- 107 276 -> 175 404 kB in 136 s, ~30 MB/min,
    *faster* than the 12 MB/min without it -- and the extra page faults pushed
    the inference stage from 44.3 to 63.2 ms p50 and CPU from 27 % to 43 %. So
    the growth is not glibc heap fragmentation, and this is not the fix. Use
    ``/debug/memory``'s ``malloc_info`` and ``vma`` blocks, which were added to
    replace this guess with a measurement.

    Failure is not fatal: on a non-glibc libc the symbol is absent and the app
    runs exactly as before, which the returned dict records.
    """
    global _allocator_tuned
    result = {"applied": False, "libc": None,
              "mmap_threshold": mmap_threshold, "trim_threshold": trim_threshold}
    try:
        import ctypes
        import platform

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        result["libc"] = "-".join(str(p) for p in platform.libc_ver() if p)
        ok_mmap = libc.mallopt(M_MMAP_THRESHOLD, int(mmap_threshold))
        ok_trim = libc.mallopt(M_TRIM_THRESHOLD, int(trim_threshold))
        result["applied"] = bool(ok_mmap) and bool(ok_trim)
        result["rc"] = [int(ok_mmap), int(ok_trim)]
    except Exception as exc:
        result["error"] = repr(exc)
    _allocator_tuned = result
    return result


def start_tracing(frames: int = 5) -> bool:
    """Turn on tracemalloc. Call once at startup; costs ~2x allocation time."""
    global _tracing
    try:
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start(frames)
        _tracing = True
    except Exception:
        _tracing = False
    return _tracing


def _proc_status() -> dict:
    out = {}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                if key in ("VmRSS", "VmSize", "VmData", "RssAnon", "RssFile"):
                    out[key] = int(value.split()[0])
    except OSError:
        pass
    return out


def _smaps_rollup() -> dict:
    out = {}
    try:
        with open("/proc/self/smaps_rollup") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                if key in ("Rss", "Pss", "Private_Dirty", "Anonymous"):
                    out[key + "_kb"] = int(value.split()[0])
    except OSError:
        pass
    return out


def _type_counts(top: int) -> tuple:
    counts: dict = {}
    for obj in gc.get_objects():
        name = type(obj).__name__
        counts[name] = counts.get(name, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    delta = {
        name: count - _prev_types.get(name, 0)
        for name, count in ordered
        if count - _prev_types.get(name, 0) != 0
    }
    return dict(ordered), delta, counts


def _vma_summary(top: int = 8) -> dict:
    """Mappings, by count and size -- the layer above the allocator.

    Separates the two ways anonymous RSS can grow: the heap getting bigger
    (``[heap]`` grows, mapping count flat) versus mappings accumulating (count
    climbs, each one small-ish). A native library that imports a dma-buf or
    mmaps a scratch buffer per frame and never unmaps shows up only here --
    ``gc``, ``tracemalloc`` and ``malloc_info`` are all blind to it.
    """
    regions = []
    total_anon = heap = 0
    count = 0
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                count += 1
                parts = line.split()
                span = parts[0]
                lo, _, hi = span.partition("-")
                size = int(hi, 16) - int(lo, 16)
                name = parts[5] if len(parts) > 5 else ""
                if name == "[heap]":
                    heap += size
                if not name or name.startswith("["):
                    total_anon += size
                    regions.append((size, name or "anon", span))
    except OSError:
        return {}
    regions.sort(reverse=True)
    return {
        "mapping_count": count,
        "anon_bytes": total_anon,
        "heap_bytes": heap,
        "largest_anon": [
            {"size_kb": s // 1024, "name": n, "range": r} for s, n, r in regions[:top]
        ],
    }


def _malloc_info() -> dict:
    """glibc's own accounting: how much the allocator holds vs has returned.

    ``malloc_info`` writes an XML summary of every arena. The two numbers that
    matter are the total ``system`` bytes the allocator has taken from the
    kernel and the ``rest``/free totals it is sitting on. If ``system`` tracks
    RSS growth the memory is inside malloc (a leak or fragmentation); if it
    stays flat while RSS climbs, the growth is somewhere malloc never saw.

    Opt-in (``/debug/memory?deep=1``) because it is the one part of this module
    that leaves Python. Every ``restype``/``argtypes`` below is load-bearing:
    ctypes defaults an unprototyped return to C ``int``, so on aarch64 the
    ``FILE *`` from ``open_memstream`` comes back truncated to 32 bits, and
    handing that to ``malloc_info`` segfaults the interpreter -- which is
    exactly what it did on this board before the prototypes were added, killing
    the detector mid-run.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.open_memstream.restype = ctypes.c_void_p
        libc.open_memstream.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)
        ]
        libc.malloc_info.restype = ctypes.c_int
        libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
        libc.fclose.restype = ctypes.c_int
        libc.fclose.argtypes = [ctypes.c_void_p]
        libc.free.restype = None
        libc.free.argtypes = [ctypes.c_void_p]

        buf = ctypes.c_void_p()
        size = ctypes.c_size_t()
        stream = libc.open_memstream(ctypes.byref(buf), ctypes.byref(size))
        if not stream:
            return {"available": False, "error": "open_memstream returned NULL"}
        rc = libc.malloc_info(0, stream)
        libc.fclose(stream)          # flush: buf/size are only valid after this
        if rc != 0 or not buf.value:
            libc.free(buf)
            return {"available": False, "error": "malloc_info rc=%d" % rc}
        xml = ctypes.string_at(buf.value, size.value).decode("utf-8", "replace")
        libc.free(buf)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    out = {"available": True, "arenas": xml.count("<heap ")}
    # The trailing <total .../> elements are the whole-process rollup.
    for kind in ("fast", "rest", "mmap", "system", "max_system"):
        marker = '<total type="%s"' % kind
        idx = xml.rfind(marker)
        if idx < 0:
            continue
        seg = xml[idx:xml.find("/>", idx)]
        for token in seg.split():
            if token.startswith('size="'):
                out[kind + "_bytes"] = int(token[6:].rstrip('"'))
    return out


def _numpy_bytes() -> dict:
    """Total bytes held by live numpy arrays, and how many there are.

    The frame store, the letterbox canvas and every model input are ndarrays; if
    RSS growth is Python-visible at all, this is where it would land, and
    ``gc.type_counts`` alone would not show it (one extra array of 2.7 MB is one
    extra object).
    """
    try:
        import numpy as np
    except Exception:
        return {}
    total = 0
    count = 0
    for obj in gc.get_objects():
        if isinstance(obj, np.ndarray):
            count += 1
            if obj.base is None:          # count owners only, not views
                total += obj.nbytes
    return {"arrays": count, "owned_bytes": total}


def _tracemalloc_top(top: int) -> dict:
    if not _tracing:
        return {"tracing": False}
    import tracemalloc

    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics("lineno")[:top]
    rows = []
    for stat in stats:
        key = str(stat.traceback[0])
        rows.append(
            {
                "site": key,
                "size_kb": stat.size // 1024,
                "count": stat.count,
                "delta_kb": (stat.size - _prev_trace.get(key, 0)) // 1024,
            }
        )
        _prev_trace[key] = stat.size
    current, peak = tracemalloc.get_traced_memory()
    return {
        "tracing": True,
        "traced_current_kb": current // 1024,
        "traced_peak_kb": peak // 1024,
        "top": rows,
    }


def report(top: int = 25, deep: bool = False) -> dict:
    """One snapshot; deltas are against the previous call to this function.

    ``deep`` additionally calls into libc (``malloc_info``); see that function
    for why it is not on by default.
    """
    global _prev_types
    status = _proc_status()
    counts, delta, full = _type_counts(top)
    _prev_types = full
    return {
        "rss_kb": status.get("VmRSS"),
        "vms_kb": status.get("VmSize"),
        "data_kb": status.get("VmData"),
        "rss_anon_kb": status.get("RssAnon"),
        "rss_file_kb": status.get("RssFile"),
        "smaps_rollup": _smaps_rollup(),
        "gc": {
            "tracked_objects": len(gc.get_objects()),
            "garbage": len(gc.garbage),
            "counts": gc.get_count(),
            "type_counts": counts,
            "type_delta": delta,
        },
        "numpy": _numpy_bytes(),
        "vma": _vma_summary(),
        "malloc_info": _malloc_info() if deep else {"available": False,
                                                    "hint": "pass ?deep=1"},
        "allocator": _allocator_tuned,
        "tracemalloc": _tracemalloc_top(top),
        "python": sys.version.split()[0],
    }
