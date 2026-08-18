"""Kernel-side evidence for which frame path is actually live.

``health.decode`` is a two-value field the app fills in itself, so on its own it
proves nothing -- an app that hard-codes ``"hw"`` reports ``"hw"``. This module
exists so ``/debug/decode`` can serve facts the app cannot fake: what the kernel
says this process has open, which shared objects are mapped into it, and which
of the platform's hardware paths exist on the filesystem at all.

Everything here reads ``/proc/self`` and ``/dev``; nothing is measured, nothing
is cached across calls except the ffmpeg decoder list (one subprocess). If a
field cannot be read the key is present with ``null`` rather than omitted, so a
missing capability and an unreadable one stay distinguishable.

What counts as evidence for each path
-------------------------------------
``zero-copy (official broker)``
    ``/run/recamera/frame.sock`` in ``open_unix_sockets``; ``librecamera_ext.so``
    and (when the RGA convert is latched on) ``librga.so`` in ``mapped_libs``;
    ``/dev/rga`` and one or more ``dmabuf`` entries in ``open_devices``. No
    ``ffmpeg`` child process.
``ffmpeg software decode``
    an ``ffmpeg`` entry in ``child_processes`` and a pipe fd; no
    ``/dev/mpp_service``, no ``dmabuf``.
``MPP hardware decode``
    ``/dev/mpp_service`` in ``open_devices`` AND ``librockchip_mpp.so`` in
    ``mapped_libs``. Presence of the device node alone is not evidence: it is
    world-visible on this firmware whether or not anything opened it.
"""

from __future__ import annotations

import os
import re
import subprocess

#: Official extension-API endpoints (kit/adapters/registry.py owns the same
#: paths; duplicated here so the probe works even when the kit is not imported).
OFFICIAL_SOCKETS = {
    "frame": "/run/recamera/frame.sock",
    "result_in": "/run/recamera/result-in.sock",
    "audio": "/run/recamera/audio.sock",
    "probe": "/run/recamera/probe.sock",
}

#: Rockchip hardware nodes and the userspace libraries that drive them.
HW_NODES = ["/dev/mpp_service", "/dev/rga", "/dev/dri/card0", "/dev/dma_heap"]
HW_LIBS = {
    "mpp": "/oem/usr/lib/librockchip_mpp.so",
    "rga": "/oem/usr/lib/librga.so",
    "recamera_ext": "/oem/usr/lib/librecamera_ext.so",
}

_LIB_RE = re.compile(r"(librockchip_mpp|librga|librecamera_ext|librknn\w*)\.so[.\d]*$")


def _access(path: str) -> dict:
    """Existence vs readability, kept apart.

    ``/run/recamera`` is mode 0750 root:root on this firmware, so an app running
    as ``admin`` gets ``exists: false`` for every socket under it while an app
    started by appmgr (root) sees them. Reporting ``dir_listable`` alongside
    keeps that difference legible instead of looking like a missing feature.
    """
    parent = os.path.dirname(path)
    return {
        "path": path,
        "exists": os.path.exists(path),
        "dir_listable": os.access(parent, os.R_OK | os.X_OK),
    }


def capability_probe() -> dict:
    """Which frame/decode paths this firmware offers, from the filesystem."""
    return {
        "official_sockets": {k: _access(v) for k, v in OFFICIAL_SOCKETS.items()},
        "hw_nodes": {p: os.path.exists(p) for p in HW_NODES},
        "hw_libs": {k: os.path.exists(v) for k, v in HW_LIBS.items()},
        "euid": os.geteuid(),
    }


def ffmpeg_decoders() -> dict:
    """The H.264/H.265 decoders this build of ffmpeg actually carries.

    Answers "is rkmpp compiled in" without anyone having to trust a comment.
    """
    out = {"available": False, "h264": [], "hevc": []}
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "quiet", "-decoders"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20,
        )
    except Exception:
        return out
    out["available"] = True
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("V"):
            continue
        name = parts[1]
        if "h264" in name:
            out["h264"].append(name)
        elif "hevc" in name or "h265" in name:
            out["hevc"].append(name)
    return out


def _proc_self() -> dict:
    """What the kernel says this process holds: fds, threads, mapped libraries."""
    devices, unix_socks, pipes, mapped = [], [], 0, []
    fd_dir = "/proc/self/fd"
    try:
        for name in os.listdir(fd_dir):
            try:
                target = os.readlink(os.path.join(fd_dir, name))
            except OSError:
                continue
            if target.startswith("/dev/") or "dmabuf" in target:
                devices.append(target)
            elif target.startswith("socket:") or target.startswith("/run/"):
                unix_socks.append(target)
            elif target.startswith("pipe:"):
                pipes += 1
    except OSError:
        devices = unix_socks = None

    # /proc/self/net/unix maps this process's socket inodes back to their paths;
    # readlink on a socket fd only yields `socket:[inode]`, which names nothing.
    sock_paths = []
    if unix_socks:
        inodes = {t[8:-1] for t in unix_socks if t.startswith("socket:[")}
        try:
            with open("/proc/self/net/unix", "r") as fh:
                for line in fh:
                    cols = line.split()
                    if len(cols) >= 8 and cols[6] in inodes:
                        sock_paths.append(cols[7])
        except OSError:
            sock_paths = None

    try:
        seen = set()
        with open("/proc/self/maps", "r") as fh:
            for line in fh:
                path = line.rstrip("\n").split(" ", 5)[-1].strip()
                if path.startswith("/") and _LIB_RE.search(path) and path not in seen:
                    seen.add(path)
                    mapped.append(path)
    except OSError:
        mapped = None

    threads = []
    try:
        for tid in os.listdir("/proc/self/task"):
            try:
                with open(f"/proc/self/task/{tid}/comm") as fh:
                    threads.append(fh.read().strip())
            except OSError:
                continue
    except OSError:
        threads = None

    return {
        "open_devices": sorted(set(devices)) if devices is not None else None,
        "open_unix_socket_paths": sorted(set(sock_paths)) if sock_paths else sock_paths,
        "open_pipe_fds": pipes,
        "mapped_libs": mapped,
        "threads": sorted(threads) if threads else threads,
    }


def _children() -> list:
    """Direct child processes, by name -- an ffmpeg decoder cannot hide from this."""
    me = os.getpid()
    found = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as fh:
                    stat = fh.read()
                # comm is parenthesised and may contain spaces; ppid is the
                # second field after the closing paren.
                comm = stat[stat.index("(") + 1: stat.rindex(")")]
                rest = stat[stat.rindex(")") + 2:].split()
                if int(rest[1]) == me:
                    found.append({"pid": int(entry), "comm": comm})
            except (OSError, ValueError, IndexError):
                continue
    except OSError:
        return []
    return found


def classify(source) -> dict:
    """Name the live path from the source object plus the kernel's view.

    ``verdict`` is derived, never asserted: it is the strongest claim the
    evidence in the same payload supports, so a reader can check it against the
    fields below without trusting this function.
    """
    proc = _proc_self()
    devices = proc.get("open_devices") or []
    socks = proc.get("open_unix_socket_paths") or []
    libs = proc.get("mapped_libs") or []
    children = _children()

    frame_sock_open = any("frame.sock" in s for s in socks)
    dmabuf_open = any("dmabuf" in d for d in devices)
    mpp_open = any(d.startswith("/dev/mpp") for d in devices)
    mpp_mapped = any("librockchip_mpp" in p for p in libs)
    ffmpeg_child = [c for c in children if "ffmpeg" in c["comm"]]

    if frame_sock_open or dmabuf_open:
        verdict = "zero-copy-isp"
        decode = "hw"
        detail = ("frames arrive as ISP dma-buf over the official frame broker; "
                  "no H.264/H.265 decode happens in this process")
    elif mpp_open and mpp_mapped:
        verdict = "mpp-hw-decode"
        decode = "hw"
        detail = "H.264 decoded by the MPP hardware decoder via librockchip_mpp"
    elif ffmpeg_child:
        verdict = "ffmpeg-sw-decode"
        decode = "sw"
        detail = "H.264/H.265 decoded on the CPU by an ffmpeg child process"
    else:
        verdict = "unknown"
        decode = getattr(source, "decode_path", None)
        detail = "no decoder evidence found; the source may not have started yet"

    return {
        "verdict": verdict,
        "decode": decode,
        "detail": detail,
        "evidence": {
            "frame_sock_open": frame_sock_open,
            "dmabuf_fd_open": dmabuf_open,
            "mpp_device_open": mpp_open,
            "mpp_lib_mapped": mpp_mapped,
            "ffmpeg_children": ffmpeg_child,
        },
        "proc_self": proc,
        "children": children,
    }


def full_report(source) -> dict:
    """Everything ``/debug/decode`` serves, in one call."""
    report = {
        "backend_class": type(source).__name__ if source is not None else None,
        "claimed_decode": getattr(source, "decode_path", None),
        "decoder_cmd": getattr(source, "decoder_cmd", None),
        "decoder_report": getattr(source, "decoder_report", None),
        "source_size": getattr(source, "source_size", None),
        "capabilities": capability_probe(),
        "ffmpeg_decoders": ffmpeg_decoders(),
    }
    report.update(classify(source))
    return report
