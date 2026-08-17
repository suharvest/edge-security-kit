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
- `events/<stream_id>` carries traffic in **both** directions relative to the
  hub: a single-box device publishes its own verdicts there, and a hub
  republishes its verdicts on the same topic on behalf of hub-mode devices. The
  `origin` field separates the two (see below); a consumer that both publishes
  and subscribes here must filter on it.

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
- `origin` (event messages): `device` or `hub`, defaulting to `device` when
  absent so existing single-box publishers stay conformant. A hub that
  republishes its own verdicts must stamp `hub`, must ignore inbound events
  carrying `hub`, and may only set a device's mode from an `origin: device`
  event. This is not a cosmetic field: the hub subscribes to the same
  `events/+` family it publishes to, so an unmarked echo makes it classify a
  hub-mode device as single-box and stop judging that device's rules
  altogether — observed in integration, where a run produced two zone alerts
  and then silence.
- `direction` (line_cross): lines are directed segments `start -> end`. Let
  `side(p) = sign((end-start) × (p-start))` (2-D cross product), which is `+1`,
  `-1` or `0`. A crossing is `forward` when the track centroid moves from
  `side > 0` to `side < 0` (left-to-right relative to the arrow), `backward`
  for the reverse.

  **`side == 0` is specified, not a degenerate case.** A rule engine MUST keep,
  per track and per line, the last side it observed that was non-zero
  (`last_nonzero_side`), and decide against that value rather than against the
  immediately preceding frame:

  - a frame whose centroid gives `side == 0` MUST NOT update
    `last_nonzero_side` and MUST NOT report a crossing — the track counts as
    still being on the side it last had;
  - when the side becomes non-zero and carries the opposite sign to
    `last_nonzero_side`, that is a crossing; the direction is read off
    `last_nonzero_side -> current side`, and the finite-segment test runs
    between the centroid that produced `last_nonzero_side` and the current one;
  - a track with no `last_nonzero_side` yet — its first frame, or a first
    observation that was on the line — only seeds the value. Leaving the line
    towards one side is not a crossing.

  The reason is quantization, not floating-point pedantry. Detector centroids
  arrive on a coarse grid: the RK3588 preprocessor scales a 1280-wide frame to
  the 640-wide model input, so `cx` snaps to multiples of 1/640 and `x = 0.5`
  is exactly 320/640. A line drawn down the middle of the frame — the most
  natural thing a user does on the rule canvas — then reads `side == 0` on
  every frame of the traverse. Decided frame to frame, that line never fires,
  and the symptom is one rule silently never triggering while every other rule
  on the same stream works. Carrying the last non-zero side resolves the
  crossing, and still refuses to fire for a track that merely sits on the line
  and jitters, because its side never takes the opposite sign.

  A break in the frame chain (`line_chain_gap_ms`, HUB_SPEC §2.1) discards
  `last_nonzero_side` along with the rest of the chain: the track is re-seeded
  from wherever it is next observed.

## Status and LWT

The status message is published retained on connect, then every 30 s (runtime
configurable). The LWT payload — registered at CONNECT with `online: false` and
no `streams` array — is delivered retained by the broker when the connection
drops without a DISCONNECT. The LWT `timestamp` is necessarily the connect
time; consumers use broker receipt time as the offline instant. A stream entry
may carry optional `preview_url` (device-local single-frame JPEG endpoint) and
`live_url` (device-local live view); the hub passes both through to its UI and
never proxies either.

**A clean exit must publish `online: false` itself.** The LWT covers exactly one
case — the connection dying without a DISCONNECT. A publisher that shuts down
normally (SIGTERM, container stop, `docker compose stop`) sends a DISCONNECT, the
broker discards the will, and the last retained message on the status topic stays
whatever the publisher put there — its `online: true` heartbeat. That retained
payload is then the device's permanent last word: every consumer that connects
afterwards, including a hub restarted hours later, reads a device that is not
running as online. The observable result is a zombie device on the fleet page
with no LWT ever arriving to correct it.

So the shutdown path is normative, not an implementation nicety: before
disconnecting, publish the status topic retained with `online: false` and no
`streams` array — the same shape as the registered LWT, so consumers need one
code path for "device gone" regardless of how it went away. The reference
implementation is `platforms/generic/esk_generic/detector.py`
(`publish_status(online=False)` in the shutdown `finally`).

## Snapshots

Snapshot payloads are raw JPEG bytes (not JSON), ≤ 200 KB. The publisher
downscales/re-encodes as needed to stay under the cap. Encoding to base64 is
not used on the wire; the cap applies to the binary payload. The JSON Schema
does not apply to this topic — the hub validates the JPEG magic bytes and the
size cap instead.

## Modes

**Single-box** (detector and rule engine on the same host): the device-local
rule engine publishes `events/<stream_id>` with `origin: device` (or the field
omitted) plus unprompted snapshots. A hub, if present, subscribes to events
instead of judging rules for that device.

**Hub mode**: devices publish only detections + status. The hub judges rules,
requests snapshots via `cmd/snapshot`, and republishes its resulting events to
the same `events/<stream_id>` topic on behalf of the device — stamped
`origin: hub` — so third-party integrations see one uniform topic regardless of
where rules ran. A device is in exactly one mode at a time, set by device
configuration.

Mode inference must be echo-safe. A consumer deriving mode from observed
traffic follows two rules: only an `origin: device` event may mark a device as
single-box, and an event whose `event_id` is already stored is a duplicate
rather than fresh evidence of a device-side engine.

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

A copy of each platform's captured fixtures is promoted into
`contracts/fixtures/` and gated by `sh contracts/check_fixtures.sh`, so the
contract is backed by real payloads from more than one implementation — today
`platforms/generic` (ONNX, x86) and `platforms/rknn` (RK3588). Each platform's
`test_fixtures_schema.py` asserts its promoted copy still matches what it
captured.

## Relation to the fall-detection contract v1

This contract reuses the fall-detection contract's conventions (epoch-ms
timestamps, payload-embedded `stream_id`, normalized center-format bbox,
accelerator-only `inference_time_ms`, fixture-based conformance) but is a
separate, incompatible schema: the fall contract's required surface is
fall-specific (`persons[].state`, `pose17`, `features`) and cannot be satisfied
by a detection-only runtime without publishing dead fields, so the two stay
parallel rather than merged.
