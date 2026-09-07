# Generic CPU detector (ONNX Runtime)

RTSP → person detection → `sensecraft.detection/1` over MQTT. Pure CPU, no
accelerator SDK, so it runs anywhere Python does and serves as the reference
implementation of the contract in [`contracts/MQTT.md`](../../contracts/MQTT.md):
every accelerated platform must produce the same payloads this one produces.

It is a hub-mode detector — it publishes detections and status, never events.
Rules are judged by the hub, which requires tracked targets (`track_id >= 1`,
HUB_SPEC §2.1), so a greedy IoU tracker runs here even though detection alone
would not need it.

## What it publishes

| Topic | Payload | QoS / retain |
|---|---|---|
| `sensecraft/security/<device_id>/detections/<stream_id>` | `sensecraft.detection/1` | 0, no retain |
| `sensecraft/security/<device_id>/status` | `sensecraft.status/1` | 1, retained, LWT registered at CONNECT |
| `sensecraft/security/<device_id>/snapshot/<event_id>` | raw JPEG ≤ 200 KB | 1, no retain |
| `sensecraft/security/<device_id>/cmd/ack` | `sensecraft.ack/1` | 1, no retain |

Subscribes to `sensecraft/security/<device_id>/cmd/snapshot` and answers
`{"stream_id": ..., "event_id": ...}` with the latest decoded frame as JPEG.
`session_id` is the process start epoch-ms, so a restart tells the hub to drop
its per-track state.

## Several cameras in one process

One process carries N streams. Each has its own capture loop, tracker, frame
counter, preview endpoint and confidence threshold; they share the ONNX session
under a lock, and the status message lists them all.

Two things are per-stream because sharing them is a correctness bug, not an
efficiency one. `track_id` is per-stream by contract, so one tracker across two
cameras would hand the same id to two different people. The confidence
threshold is what an operator retunes per camera, so a single value on the model
session would move every camera when one slider moved.

Inference is shared on purpose: a second ORT session doubles the memory and the
thread pool, and the thread-budget section below is what that costs.

## Runtime control (`cmd/control`)

Subscribes to `sensecraft/security/<device_id>/cmd/control` and answers every
command on `cmd/ack` — see contracts/MQTT.md "Control downlink" for the payload
shapes. Three commands are implemented:

| Command | Effect here |
|---|---|
| `set_conf_threshold` | the stream's next frame uses the new value; no restart, no reconnect |
| `add_stream` | starts a capture loop, **waits for the source to open**, then acks |
| `remove_stream` | stops that loop and drops it from the status message |

Two properties are worth knowing before building on it:

- **The ack is not sent until the change is live.** `add_stream` waits up to 6 s
  for the source to open and acks `ok: false` with the reason if it does not.
  Acking on receipt would be faster and would put a permanently grey tile on the
  hub's video wall with nothing to explain it.
- **The change is written back to the config file.** `applied.persisted` says
  whether it was: a threshold moved from the console and lost on the next
  restart is worse than one that could not be moved at all, because nobody
  watches for a setting quietly reverting. A detector started from flags rather
  than a config file has nowhere to write and reports `persisted: false`.

The write is atomic (`.tmp` + `fsync` + `rename`) and drops the single-stream
`source`/`stream_id` shorthand once a `streams` list exists, so the file never
describes the same camera twice.

## Run

```bash
uv sync
cp config.example.yaml config.yaml   # edit device_id / source / mqtt_host
curl -L -o models/yolov8n.onnx \
  https://raw.githubusercontent.com/Zhang-zu-hao/Industrial-security-demo/master/models/yolov8n.onnx
uv run python -m esk_generic --config config.yaml
```

Useful flags: `--source`, `--mqtt-host`, `--seconds N` / `--frames N` for bounded
test runs, `--annotate out.jpg` to dump one frame with the boxes redrawn from
the published `frame_norm` values.

A local test rig (RTSP on 8555 + mosquitto on 1884) lives in
[`tools/rtsp-fixture/`](../../tools/rtsp-fixture/).

## Coordinates

The model input is square, the source usually is not. The frame is letterboxed
(aspect preserved, grey padding) and the pad is reversed before publishing, so
`bbox` is always normalized against the original frame — `coordinate_space` is
the literal `frame_norm` and nothing else ever leaves the process.

Skipping the pad reversal is the failure this is guarding against, and it is not
subtle: on a 1280×720 frame with a 640×640 input, the pad is 140 px top and
bottom, so a box published without the reversal comes out **261 px too short**
vertically while its x coordinates look fine. `tests/test_letterbox.py` pins the
round trip; `evidence-annotated.jpg` shows the boxes redrawn from published
values on a real 720p frame.

## Preview endpoint

`preview_enabled` starts one HTTP server (default `:8099`) covering every stream
the process carries. Routes are resolved per request, so a camera attached by
`add_stream` serves frames immediately — no port to allocate, nothing to
restart.

| Route | Serves | Advertised as |
|---|---|---|
| `/preview.jpg`, `/preview/<stream_id>.jpg` | the latest frame as JPEG | `status.streams[].preview_url` |
| `/live`, `/live/<stream_id>`, `/` | an HTML page reloading that JPEG once a second | `status.streams[].live_url` |

Both carry `Access-Control-Allow-Origin: *`. The JPEG route is what the hub's
single-frame proxy fetches for the rule-canvas backdrop (HUB_SPEC §4); the HTML
route is what the workbench "live view" button opens. The live page is a
refreshing still, not a video stream — no second encode path, and it renders in
any browser.

Both URLs are advertised only when `preview_advertise_host` is set: the device
cannot guess which address the browser can reach, so an unset value omits the
fields rather than publishing wrong ones. Note that a `preview_advertise_host` of
`127.0.0.1` is correct for the device itself and useless to a remote browser —
that is precisely the case the hub proxy covers, and the reason `live_url` should
point at an address other machines can resolve when one exists.

## Health reporting

`health.decode` is `sw`: decoding is OpenCV/FFmpeg on the CPU, which is this
platform's *primary* path, so `fallback_active` stays false. Set
`decode_primary: hw` only if you have replaced the decoder — the detector then
reports `fallback_active: true` whenever it is actually running on `sw`.

`inference_time_ms` is the `onnxruntime` `session.run` call alone.
`pipeline_ms` covers capture → publish for the same frame.

## Measured on a 20-core aarch64 workstation (aarch64, ONNX Runtime 1.28 CPU, 1280×720)

| | |
|---|---|
| Live RTSP, 15 fps source | 14.92 fps sustained (source-limited) |
| Same file without pacing | 26.82 fps (compute-limited) |
| `inference_time_ms` | 30–37 ms |
| `pipeline_ms` | 58–72 ms |

## Thread budget: set `intra_threads`

`intra_threads: 0` hands ONNX Runtime its default, which is one intra-op thread
per core — **per session**. That is fine for exactly one detector on an otherwise
idle host and wrong everywhere else. Two detectors on that 20-core box with
the cap unset:

| | unset (`0`) | `intra_threads: 6` |
|---|---|---|
| CPU per detector | ~900 % | ~250 % |
| `inference_time_ms` p50 | 138 ms | 50 ms |
| throughput | 6.2 fps | source rate (15 fps) |

`config.example.yaml` ships `4`. Budget cores across the detectors you actually
run rather than letting each one claim the whole machine.

### The failure disguises itself as a rule bug

This is the diagnostic clue worth keeping: CPU oversubscription does not surface
as "slow". It surfaces as **wrong alerts**, and the first hypothesis it invites is
a bug in the hub's rule engine.

The chain, as observed:

1. Inference falls behind the RTSP source, so decoded frames queue up.
2. The reader drains the backlog in bursts — frames arrive in clumps, and the gap
   between two consecutive *published* frames stretches. Measured: 1.66 s.
3. That gap is longer than `track_max_lost_s` (default 0.75 s), so the tracker
   retires the track and issues a **new `track_id`** for the same person.
4. Rule state on the hub is keyed by `(device_id, stream_id, track_id)`
   (HUB_SPEC §2), so the new ID is a person who has never been in the zone:
   `zone_enter` fires again. Cooldown does not help — it is keyed by track too.

The visible symptom is duplicate `zone_enter` alerts for one person standing
still, which reads as a broken zone rule. Before touching rule code, check the
detector's `inference_time_ms` and the frame interval on the detections topic: if
`pipeline_ms` exceeds the source frame period, the tracker is being starved and
the rules are behaving correctly on bad input. Raising `track_max_lost_s` hides
the symptom while leaving the pipeline unable to keep up; capping threads fixes
the cause.

## Tests

```bash
uv run pytest          # letterbox round trip, tracker IDs, fixture conformance
```

`fixtures/*.json` are captured verbatim off the broker during a real run, not
hand-written, and are validated against `contracts/mqtt-detection.schema.json`
as required by the contract's Conformance section.
