# Conformance fixtures

Every fixture here is a payload **captured from a real run**, not hand-written
JSON. That is the point: a field rename or a type change in an implementation
shows up as a fixture diff, and `check_fixtures.sh` fails in CI rather than the
drift reaching a device.

Run the gate:

```bash
sh contracts/check_fixtures.sh
```

| Fixture | Captured from | What it pins |
|---|---|---|
| `detection-generic-720p.json` | `platforms/generic` on spark, live 1280×720 RTSP via `mosquitto_sub` | The full detection contract on a **non-square** source: `coordinate_space: frame_norm`, `frame.w/h` = 1280×720, a `track_id >= 1`, and a bbox whose letterbox pad has been reversed. A square source would have passed a broken inverse — see `platforms/generic/README` for the 261 px error this shape catches. |
| `status-generic.json` | same run, retained `status` topic | Heartbeat shape: `online: true`, per-stream `state`/`fps`/`decode`, `preview_url` (the rule-canvas backdrop), `versions`, `uptime_s`. |
| `status-generic-lwt.json` | same run, after `SIGKILL` (no DISCONNECT) | The LWT payload the broker delivers: `online: false` and **no** `streams` array. Its `timestamp` is the CONNECT time, so consumers must use receipt time as the offline instant. |
| `detection-rknn-720p.json` | `platforms/rknn` on a Radxa Rock 5B (RK3588), live 1280×720 RTSP | The same contract from a second, independent implementation: RGA/MPP preprocessing, `decode: "hw"`, `backend: "rknn-lite2-*"`. Its `bbox` cx values land on the 1/640 grid the RK preprocessor quantizes to — the grid that puts a mid-frame line on `side == 0` and forced the `direction` rule in `MQTT.md`. |
| `status-rknn.json` | same run, retained `status` topic | Heartbeat from the RK platform: `preview_url`/`live_url` and a `versions.model` naming the `.rknn` artifact. |
| `status-rknn-lwt.json` | same run, connection dropped without a DISCONNECT | The RK platform's LWT — the same shape as the generic one, which is the point: consumers need one "device gone" code path regardless of platform. |
| `event-zone-enter-hub.json` | `hub/tools/capture_fixtures.py`, republished off `events/cam-0` | A hub-judged event: `origin: "hub"`. This marker is what stops the hub ingesting its own echo and reclassifying a hub-mode device as single-box. |
| `event-line-cross-hub.json` | same | `line_cross` carries `direction` (schema requires it conditionally). |
| `event-loitering-hub.json` | same | `loitering` carries `dwell_s`. Capturing this one surfaced a defect: with `event_type` missing from the cooldown key, the zone's `zone_enter` swallowed its own `loitering` escalation and the event never fired. |
| `event-zone-enter-device.json` | shape a single-box publisher (reCamera-class) sends | `origin: "device"` — the only origin permitted to mark a device as single-box. |

Two independent detector implementations now back the same contract. The
platform copies stay under `platforms/<name>/fixtures/`; the files here are
promoted copies, and each platform's `test_fixtures_schema.py` asserts the two
are identical, so a drift shows up in the platform test as well as in the gate.

## Regenerating

The detector fixtures come off a live device; do not edit them by hand. The
hub-side event fixtures are reproducible on any host:

```bash
cd hub && uv run python tools/capture_fixtures.py
```

A diff in that output is a contract change and needs review, not a re-commit.
