"""Contract validation (HUB_SPEC §1, §10).

The three JSON message kinds are validated against
``contracts/mqtt-detection.schema.json``. A failure increments a counter that
``GET /api/health`` exposes and the message is discarded — never raised
(HUB_SPEC §1 ingest row).

Snapshots are binary JPEG and out of the schema's scope; they get the magic-byte
and size check from contracts/MQTT.md instead.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

JPEG_SOI = b"\xff\xd8\xff"
MAX_SNAPSHOT_BYTES = 200 * 1024

SCHEMA_KINDS = {
    "sensecraft.detection/1": "detection_message",
    "sensecraft.status/1": "status_message",
    "sensecraft.event/1": "event_message",
    "sensecraft.command/1": "command_message",
    "sensecraft.ack/1": "ack_message",
}


def schema_path() -> Path:
    """Locate ``contracts/mqtt-detection.schema.json``.

    ``HUB_CONTRACT_SCHEMA`` wins (the container copies contracts/ to a fixed
    path); otherwise walk up from this file to the repo checkout.
    """
    override = os.environ.get("HUB_CONTRACT_SCHEMA")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "contracts" / "mqtt-detection.schema.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("contracts/mqtt-detection.schema.json not found")


class ContractValidator:
    """Validates payloads and counts rejections by reason."""

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self.schema = schema if schema is not None else json.loads(
            schema_path().read_text(encoding="utf-8")
        )
        self._whole = Draft202012Validator(self.schema)
        # Per-kind validators give a precise error instead of the oneOf blob.
        self._per_kind = {
            kind: Draft202012Validator(
                {
                    "$schema": self.schema["$schema"],
                    "$ref": f"#/$defs/{ref}",
                    "$defs": self.schema["$defs"],
                }
            )
            for kind, ref in SCHEMA_KINDS.items()
        }
        self.failures: Counter[str] = Counter()
        self.accepted: Counter[str] = Counter()
        self.last_error: str | None = None

    @property
    def failure_count(self) -> int:
        return sum(self.failures.values())

    def validate(self, raw: bytes | str, expected: str | None = None) -> dict[str, Any] | None:
        """Return the parsed payload, or ``None`` after counting the failure."""
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            self.failures["json_parse"] += 1
            return None
        if not isinstance(payload, dict):
            self.failures["not_object"] += 1
            return None
        kind = payload.get("schema")
        if kind not in SCHEMA_KINDS:
            self.failures["unknown_schema"] += 1
            return None
        if expected is not None and kind != expected:
            self.failures["wrong_topic_kind"] += 1
            return None
        error = next(iter(self._per_kind[kind].iter_errors(payload)), None)
        if error is not None:
            self.failures[kind] += 1
            self.last_error = f"{'/'.join(str(p) for p in error.absolute_path)}: {error.message}"
            return None
        self.accepted[kind] += 1
        return payload

    def snapshot_ok(self, payload: bytes, max_bytes: int = MAX_SNAPSHOT_BYTES) -> bool:
        """JPEG magic bytes + size cap (contracts/MQTT.md Snapshots)."""
        if not isinstance(payload, (bytes, bytearray)) or len(payload) < 4:
            self.failures["snapshot_too_short"] += 1
            return False
        if not bytes(payload[:3]) == JPEG_SOI:
            self.failures["snapshot_not_jpeg"] += 1
            return False
        if len(payload) > max_bytes:
            self.failures["snapshot_too_large"] += 1
            return False
        self.accepted["snapshot"] += 1
        return True

    def stats(self) -> dict[str, Any]:
        return {
            "schema_failures": self.failure_count,
            "schema_failures_by_reason": dict(self.failures),
            "accepted": dict(self.accepted),
        }
