# MQTT contract: sensecraft.detection/1

Every detector runtime publishes the same core JSON shapes defined by
[`mqtt-detection.schema.json`](mqtt-detection.schema.json). Platform-specific
fields may be added (`additionalProperties: true` everywhere), but the required
fields, units and meanings must not change.

Three message kinds share one schema file, discriminated by the `schema` field:

| `schema` | Published by | Purpose |
|---|---|---|
| `sensecraft.detection/1` | detector | per-frame detection results |
| `sensecraft.status/1` | detector | heartbeat / health, plus LWT offline |
| `sensecraft.event/1` | rule engine (device-local or hub) | rule events: zone_enter / loitering / line_cross |

## Topics

```
sensecraft/security/<device_id>/detections/<stream_id>   QoS0, no retain
sensecraft/security/<device_id>/events/<stream_id>       QoS1, no retain
sensecraft/security/<device_id>/status                   QoS1, retain, + LWT
sensecraft/security/<device_id>/snapshot/<event_id>      QoS1, no retain
sensecraft/security/<device_id>/cmd/snapshot             QoS1, no retain (downlink)
```

- `<device_id>` and `<stream_id>` are stable configured IDs. Both are also
  required inside every payload; consumers must not depend on topic parsing.
- `cmd/snapshot` is the one downlink: in hub mode the rule engine runs off-device,
  so the hub requests evidence by publishing `{"stream_id": ..., "event_id": ...}`
  there; the device answers on the snapshot topic. Devices that run rules locally
  (single-box mode, reCamera) publish snapshots unprompted.

## Field semantics

- `timestamp`: Unix epoch milliseconds. Never monotonic seconds or a float.
  Used for display and evidence only. All rule timing (dwell, cooldown,
  line-chain expiry) and cross-device ordering happen on the hub's own clock at
  receive time, so device NTP sync is recommended but is not a correctness
  requirement.
- `session_id`: generated once per detector process start (boot epoch ms or
  uuid recommended). Required in all three message kinds. A changed value
  signals that `frame_id`, `track_id` and event sequence counters have reset;
  the hub drops all per-track rule state for the device when it changes.
- `coordinate_space`: always the literal `frame_norm`. The sender must reverse
  any letterbox/pad transform and normalize against the original frame
  (`frame.w` × `frame.h`) before publishing. Letterboxed coordinates must never
  leave the device. There is no key-presence branching on the consumer side:
  a payload without the field fails validation.
- `bbox`: normalized `[center_x, center_y, width, height]`, same convention as
  the fall-detection contract.
- `inference_time_ms`: accelerator inference call only where the API exposes
  it. If a backend can measure only a larger region, document that fact and
  publish the larger measurement separately as `pipeline_ms`.
- `health.decode`: `hw` or `sw`. Every platform has a silent-CPU-fallback
  failure mode (GStreamer NVDEC open failure on Jetson, missing MPP plugins on
  Rockchip); the active path must be reported so the hub can surface it.
  `health.fallback_active` is true whenever decode or inference runs on a
  fallback path.
- `track_id`: stable per-stream ID, `>= 1`; `0` means the detection is not
  associated with a track. Untracked detections never participate in rule
  evaluation — they count only toward statistics. A platform whose detections
  feed hub-side rules must therefore run a tracker; a device without tracking
  (reCamera-class) must run single-box mode, judge rules locally and publish
  only events.
- `event_id` (event messages): globally unique within the publishing rule
  engine (`<device_id>-<stream_id>-<session_id>-<seq>` recommended, so a
  restart cannot reuse an ID). It names the snapshot topic segment; the hub
  stores it as an association column, not as the alert primary key.
- `direction` (line_cross): lines are directed segments `start -> end`. Let
  `side(p) = sign((end-start) × (p-start))` (2-D cross product). A crossing is
  `forward` when the track centroid moves from `side > 0` to `side < 0`
  (left-to-right relative to the arrow), `backward` for the reverse.

## Status and LWT

The status message is published retained on connect, then every 30 s (runtime
configurable). The LWT payload — registered at CONNECT with `online: false` and
no `streams` array — is delivered retained by the broker when the connection
drops without a DISCONNECT. The LWT `timestamp` is necessarily the connect
time; consumers use broker receipt time as the offline instant. A stream entry
may carry optional `preview_url` (device-local single-frame JPEG endpoint) and
`live_url` (device-local live view); the hub passes both through to its UI and
never proxies either.

## Snapshots

Snapshot payloads are raw JPEG bytes (not JSON), ≤ 200 KB. The publisher
downscales/re-encodes as needed to stay under the cap. Encoding to base64 is
not used on the wire; the cap applies to the binary payload. The JSON Schema
does not apply to this topic — the hub validates the JPEG magic bytes and the
size cap instead.

## Modes

**Single-box** (detector and rule engine on the same host): the device-local
rule engine publishes `events/<stream_id>` and unprompted snapshots. A hub, if
present, subscribes to events instead of judging rules for that device.

**Hub mode**: devices publish only detections + status. The hub judges rules,
requests snapshots via `cmd/snapshot`, and republishes its resulting events to
the same `events/<stream_id>` topic on behalf of the device, so third-party
integrations see one uniform topic regardless of where rules ran. A device is
in exactly one mode at a time, set by device configuration.

## QoS / retain rationale

Detections are high-rate and disposable: QoS0, no retain. Events and snapshots
are the product: QoS1. Status is retained so a late-joining hub learns the
fleet state without waiting a heartbeat interval. QoS0 loss and reordering are
handled hub-side (frame_id ordering guard, line-chain expiry — see
HUB_SPEC §2); publishers only guarantee per-stream monotonic `frame_id` within
a session.

## Conformance

Each platform must keep a representative payload fixture for every message
kind it publishes and validate it against the schema in CI or its host-only
tests. A benchmark is not complete until its MQTT fixtures pass. For
dependency-free checks:

```bash
python3 contracts/validate_payload.py path/to/payload.json
```

## Relation to the fall-detection contract v1

This contract reuses the fall-detection contract's conventions (epoch-ms
timestamps, payload-embedded `stream_id`, normalized center-format bbox,
accelerator-only `inference_time_ms`, fixture-based conformance) but is a
separate, incompatible schema: the fall contract's required surface is
fall-specific (`persons[].state`, `pose17`, `features`) and cannot be satisfied
by a detection-only runtime without publishing dead fields, so the two stay
parallel rather than merged.
