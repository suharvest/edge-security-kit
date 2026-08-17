"""MQTT ingest (HUB_SPEC §1, contracts/MQTT.md Topics).

Subscribes to the four inbound topic families, validates, and hands the payload
to the callbacks the application wires in. Validation failures are counted and
dropped, never raised. The publish side (``cmd/snapshot`` downlink and the
``events`` republish) reuses the same client through :meth:`publish`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from .config import default_client_id
from .validation import ContractValidator

log = logging.getLogger("edge_hub.mqtt")

DETECTIONS = "detections"
EVENTS = "events"
STATUS = "status"
SNAPSHOT = "snapshot"


class MqttIngest:
    def __init__(
        self,
        host: str,
        port: int = 1883,
        username: str | None = None,
        password: str | None = None,
        client_id: str | None = None,
        topic_prefix: str = "sensecraft/security",
        validator: ContractValidator | None = None,
        on_detections: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        on_status: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        on_event: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        on_snapshot: Callable[[str, str, bytes], Awaitable[None]] | None = None,
        max_snapshot_bytes: int = 200 * 1024,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        # A shared client id makes two hubs disconnect each other in a loop
        # and silently lose QoS 0 detections (see config.default_client_id).
        self.client_id = client_id or default_client_id()
        self.topic_prefix = topic_prefix.rstrip("/")
        self.validator = validator or ContractValidator()
        self.on_detections = on_detections
        self.on_status = on_status
        self.on_event = on_event
        self.on_snapshot = on_snapshot
        self.max_snapshot_bytes = max_snapshot_bytes
        self.connected = False
        self.messages = 0
        #: handler faults contained per message (see :meth:`_run`)
        self.handler_errors = 0
        self._client: Any = None
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    @property
    def subscriptions(self) -> list[str]:
        p = self.topic_prefix
        return [
            f"{p}/+/{DETECTIONS}/+",
            f"{p}/+/{EVENTS}/+",
            f"{p}/+/{STATUS}",
            f"{p}/+/{SNAPSHOT}/+",
        ]

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _run(self) -> None:
        import aiomqtt

        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with aiomqtt.Client(
                    hostname=self.host,
                    port=self.port,
                    username=self.username,
                    password=self.password,
                    identifier=self.client_id,
                ) as client:
                    self._client = client
                    self.connected = True
                    backoff = 1.0
                    for topic in self.subscriptions:
                        await client.subscribe(topic, qos=1)
                    log.info("mqtt connected to %s:%s as client_id=%s",
                             self.host, self.port, self.client_id)
                    async for message in client.messages:
                        # A handler fault is a message-level problem, not a
                        # transport one. Letting it escape drops the broker
                        # connection and starts the reconnect backoff, so a
                        # single failing payload takes the entire ingest path
                        # down for as long as the fault persists.
                        try:
                            await self.dispatch(str(message.topic), message.payload)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:  # noqa: BLE001
                            self.handler_errors += 1
                            log.exception("handler failed for %s: %s",
                                          message.topic, exc)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any broker error
                log.warning("mqtt connection lost (client_id=%s): %s",
                            self.client_id, exc)
            finally:
                self.connected = False
                self._client = None
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)

    async def publish(self, topic: str, payload: bytes, qos: int = 1) -> None:
        client = self._client
        if client is None:
            log.debug("publish dropped, no broker connection: %s", topic)
            return
        await client.publish(topic, payload, qos=qos)

    # -- dispatch --------------------------------------------------------
    def parse_topic(self, topic: str) -> tuple[str, str, str | None] | None:
        """``<prefix>/<device>/<kind>[/<stream|event_id>]`` -> parts."""
        if not topic.startswith(self.topic_prefix + "/"):
            return None
        rest = topic[len(self.topic_prefix) + 1 :].split("/")
        if len(rest) == 2:
            return rest[0], rest[1], None
        if len(rest) >= 3:
            return rest[0], rest[1], "/".join(rest[2:])
        return None

    async def dispatch(self, topic: str, payload: bytes) -> None:
        self.messages += 1
        parts = self.parse_topic(topic)
        if parts is None:
            self.validator.failures["unknown_topic"] += 1
            return
        device_id, kind, tail = parts

        if kind == SNAPSHOT:
            if tail is None:
                self.validator.failures["snapshot_no_event_id"] += 1
                return
            if not self.validator.snapshot_ok(payload, self.max_snapshot_bytes):
                return
            if self.on_snapshot is not None:
                await self.on_snapshot(device_id, tail, bytes(payload))
            return

        expected = {
            DETECTIONS: "sensecraft.detection/1",
            EVENTS: "sensecraft.event/1",
            STATUS: "sensecraft.status/1",
        }.get(kind)
        if expected is None:
            self.validator.failures["unknown_topic"] += 1
            return
        message = self.validator.validate(payload, expected=expected)
        if message is None:
            log.debug("rejected %s: %s", topic, self.validator.last_error)
            return
        # contracts/MQTT.md: the payload's own device_id is authoritative, but a
        # mismatch with the topic segment is a misconfigured publisher.
        if str(message.get("device_id")) != device_id:
            self.validator.failures["device_id_topic_mismatch"] += 1
            return
        if kind == DETECTIONS and self.on_detections is not None:
            await self.on_detections(message)
        elif kind == EVENTS and self.on_event is not None:
            await self.on_event(message)
        elif kind == STATUS and self.on_status is not None:
            await self.on_status(message)
