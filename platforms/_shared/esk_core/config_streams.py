"""Stream-list reading and atomic write-back, shared by every platform Config.

Two forms describe the same thing. Existing config files name one camera with
top-level ``stream_id``/``source``; a multi-camera file carries a ``streams``
list. :func:`stream_configs` folds the two into one so nothing downstream
branches on which was written, and no deployment needs editing to keep working.

:func:`persist_streams` is what makes a console change survive a restart. It is
atomic — a power cut during the write must not leave a detector with a truncated
config file and no way to start — and it drops the single-stream shorthand once
a list exists, so the file never describes the same camera twice.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

LOG = logging.getLogger("esk.core.config")


def stream_configs(cfg: Any) -> list[dict[str, Any]]:
    """The stream list, whichever form the file used."""
    streams = getattr(cfg, "streams", None)
    if streams:
        return [dict(entry) for entry in streams]
    return [{"stream_id": cfg.stream_id, "source": cfg.source,
             "rtsp_transport": getattr(cfg, "rtsp_transport", "tcp")}]


def persist_streams(cfg: Any, streams: list[dict[str, Any]]) -> bool:
    """Write the stream list back to the file ``cfg`` was loaded from.

    Returns whether it was persisted. ``False`` is a normal answer, not a
    failure: a detector started from flags has nowhere to write, and the ack
    reports that as ``applied.persisted: false`` rather than pretending.
    """
    path = getattr(cfg, "path", None)
    if not path:
        return False
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml ships with every runtime here
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read()
        data = (json.loads(raw) if path.endswith(".json") else yaml.safe_load(raw)) or {}
        data["streams"] = streams
        # The shorthand keys would now contradict the list; drop them so the
        # file has exactly one description of what this detector watches.
        data.pop("source", None)
        data.pop("stream_id", None)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            if path.endswith(".json"):
                json.dump(data, handle, indent=2, ensure_ascii=False)
            else:
                yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return True
    except (OSError, ValueError) as exc:  # noqa: BLE001
        LOG.warning("could not persist config to %s: %s", path, exc)
        return False
