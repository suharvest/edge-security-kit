"""The detector's half of ``cmd/control`` (contracts/MQTT.md "Control downlink").

Shared by every platform: the decision table is about the contract, not about
the accelerator, so a second copy of it is a second place for the two rules
below to be got wrong.

Kept apart from the MQTT plumbing and from the capture loop, because the part
worth reading is the decision table, not the transport: which commands this
runtime can honour, what "applied" means for each, and what a refusal has to
say. The supervisor passes in three callbacks and this module never touches
sockets, threads or OpenCV — so the table is testable without a broker, a
camera or a model.

Two rules from the contract are enforced here rather than left to the caller:

* **``ok: true`` means live.** ``add_stream`` does not ack until the source has
  actually opened; a threshold change does not ack until the value is the one
  the next frame will use.
* **A repeated ``request_id`` applies once.** QoS 1 redelivery is normal, and a
  command applied twice would attach the same camera twice.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

LOG = logging.getLogger("esk.core.control")

ACK_SCHEMA = "sensecraft.ack/1"
COMMANDS = ("set_conf_threshold", "add_stream", "remove_stream")
#: How many settled request_ids to remember. A redelivery arrives within
#: seconds; this only has to outlive the broker's retry, not the process.
SEEN_LIMIT = 256


class CommandError(Exception):
    """A refusal with an operator-facing reason (acked as ``ok: false``)."""


class ControlHandler:
    def __init__(
        self,
        device_id: str,
        set_threshold: Callable[[str, float], dict[str, Any]],
        add_stream: Callable[[dict[str, Any]], dict[str, Any]],
        remove_stream: Callable[[str], dict[str, Any]],
    ) -> None:
        self.device_id = device_id
        self.set_threshold = set_threshold
        self.add_stream = add_stream
        self.remove_stream = remove_stream
        #: request_id -> the ack already sent for it
        self._seen: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    def handle(self, request: dict[str, Any], session_id: str, now_ms: int) -> dict[str, Any] | None:
        """Turn one command payload into the ack to publish, or None to ignore.

        ``None`` means the message was not addressed to this detector or was not
        a command at all. Everything else — including every failure — produces
        an ack, because a caller reads silence as "device unreachable" and would
        otherwise be told the wrong thing about a command that was received and
        refused.
        """
        if request.get("schema") != "sensecraft.command/1":
            return None
        if str(request.get("device_id") or "") != self.device_id:
            return None
        request_id = str(request.get("request_id") or "")
        command = str(request.get("command") or "")
        if not request_id:
            LOG.warning("cmd/control without request_id, ignored: %r", command)
            return None

        previous = self._seen.get(request_id)
        if previous is not None:
            # Redelivery: ack again with the same verdict, apply nothing.
            LOG.info("cmd/control %s redelivered, re-acking without re-applying",
                     request_id)
            return dict(previous, timestamp=now_ms)

        ack: dict[str, Any] = {
            "schema": ACK_SCHEMA,
            "timestamp": now_ms,
            "session_id": session_id,
            "device_id": self.device_id,
            "request_id": request_id,
            "command": command if command in COMMANDS else "set_conf_threshold",
        }
        params = request.get("params")
        try:
            if command not in COMMANDS:
                raise CommandError(f"unknown command {command!r}")
            if not isinstance(params, dict):
                raise CommandError("params must be an object")
            applied = self._apply(command, params)
        except CommandError as exc:
            ack["ok"] = False
            ack["error"] = str(exc)
            LOG.warning("cmd/control %s refused: %s", command, exc)
        except Exception as exc:  # noqa: BLE001 - a fault must still be acked
            ack["ok"] = False
            ack["error"] = f"{type(exc).__name__}: {exc}"
            LOG.exception("cmd/control %s failed", command)
        else:
            ack["ok"] = True
            ack["applied"] = applied
            LOG.info("cmd/control %s applied: %s", command, applied)
        self._remember(request_id, ack)
        return ack

    def _apply(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        stream_id = str(params.get("stream_id") or "")
        if not stream_id:
            raise CommandError("params.stream_id is required")
        if command == "set_conf_threshold":
            raw = params.get("conf_threshold")
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise CommandError("params.conf_threshold must be a number")
            value = float(raw)
            if not 0.0 <= value <= 1.0:
                raise CommandError("params.conf_threshold must be between 0 and 1")
            return self.set_threshold(stream_id, value)
        if command == "add_stream":
            source = str(params.get("source") or "")
            if not source:
                raise CommandError("params.source is required")
            transport = str(params.get("rtsp_transport") or "tcp")
            if transport not in ("tcp", "udp"):
                raise CommandError("params.rtsp_transport must be tcp or udp")
            return self.add_stream(
                {
                    "stream_id": stream_id,
                    "source": source,
                    "name": str(params.get("name") or ""),
                    "rtsp_transport": transport,
                }
            )
        return self.remove_stream(stream_id)

    def _remember(self, request_id: str, ack: dict[str, Any]) -> None:
        self._seen[request_id] = ack
        self._order.append(request_id)
        while len(self._order) > SEEN_LIMIT:
            self._seen.pop(self._order.pop(0), None)
