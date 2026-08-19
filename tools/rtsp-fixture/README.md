# RTSP + MQTT test fixture

Self-contained loopback rig for detector verification: one mediamtx serving
RTSP/TCP on **8555** and one mosquitto on **1884**. Both ports and both
container names (`edge-sec-mediamtx`, `edge-sec-mosquitto`) are chosen so this
rig can run alongside the fall-detection fixture (`fall-e2e-mediamtx`, 8554)
and any production broker on 1883 without touching them.

```bash
docker compose up -d
chmod +x publish.sh
./publish.sh /path/to/source.mp4                # 1280x720 @ 15 fps
./publish.sh /path/to/source.mp4 edge-sec-1080p 127.0.0.1 1920x1080
```

Client URLs:

```
rtsp://HOST:8555/edge-sec-720p
rtsp://HOST:8555/edge-sec-1080p
mqtt://HOST:1884
```

The 720p profile is the default because the detector's letterbox inverse is
only exercised by a non-square source. A square fixture passes even when the
inverse is missing, so do not verify coordinates against one.

Confirm the stream is actually readable — an open TCP port is not the same as
a successful DESCRIBE:

```bash
ffprobe -v error -rtsp_transport tcp \
  -show_entries stream=codec_name,profile,width,height,r_frame_rate \
  -of compact rtsp://127.0.0.1:8555/edge-sec-720p
```

Watch the wire while the detector runs:

```bash
mosquitto_sub -h 127.0.0.1 -p 1884 -v -t 'sensecraft/security/#'
```

Request an evidence snapshot (the detector answers on
`sensecraft/security/<device_id>/snapshot/<event_id>` with raw JPEG):

```bash
mosquitto_pub -h 127.0.0.1 -p 1884 \
  -t sensecraft/security/generic-01/cmd/snapshot \
  -m '{"stream_id":"cam-0","event_id":"manual-1"}'
```

Teardown touches only this rig's two services:

```bash
docker compose stop mediamtx mosquitto && docker compose rm -f mediamtx mosquitto
```

---

## Ground-truth acceptance harness

The files below are the harness the hub/detector acceptance run was executed
with, promoted here verbatim apart from having their absolute paths turned into
overridable variables. They exist because "the alert looked right" is not a
result: every assertion here is against a video whose person trajectory is
*designed*, so the expected crossing and entry instants are known before the
system is started.

| File | Role |
|---|---|
| `find_person.py` | Scan a real clip for the clearest standing person, cut the patch (`person_patch.png` + a detection probe report). The subject is real pixels, not a drawn rectangle — a synthetic blob is not something a person detector is guaranteed to find. |
| `build_truth.py` | Compose `truth.mp4` (1280×720, 31.0 s) by translating that patch along a trajectory defined **in seconds**, and export `truth.json`: per-frame centroid plus the line/zone geometry and dwell threshold. Seconds rather than frames means the same wall-clock schedule at any frame rate. |
| `publish_truth.sh` | Publish `truth.mp4` on `rtsp://…:8555/edge-sec-truth` at its own 5 fps (`publish.sh` forces 15 and would duplicate frames past what the CPU detector can consume). |
| `start_stack.sh` | Bring up hub (`:18080`, own data dir) + two detectors, one on the truth stream, one on a real ADL clip. Kills only processes matched on its own config/data paths. |
| `verify_e2e.py` | The assertion pass: log in, PUT rules for both streams, tap the detections topic, align the observed centroid series against `truth.json`, derive the ground-truth event instants from the very frames the hub judged, then match every new alert to its instant **per rule** and assert. Also pulls snapshots and `/api/devices`. |
| `diagnose.py` | Post-mortem on a tap: inter-frame gaps, fps, track lifetimes. Run this first when alerts look wrong. |
| `restart_fixture.sh` | Republish truth at 5 fps and restart both detectors with a bounded ONNX thread pool. |
| `regress_restart.sh` | Regression for the `event_id` collision: restart the hub while the detectors keep running (so `session_id` is unchanged) and confirm alerting resumes (HUB_SPEC §3). |
| `restart_hub_verify.sh` | Restart-persistence check: alerts, dispositions **and the session cookie** survive a hub restart (HUB_SPEC §10, §7). |
| `verify_ws_catchup.py` | Drops the WS for a configurable gap while the stack keeps firing, then reconnects and asserts `GET /alerts?after_id=` returns exactly the missed rows (FRONTEND_SPEC §7). The browser run cannot show this path: a socket that never drops never takes it. |
| `test_shutdown.sh` | A detector that exits cleanly must leave a retained `online: false` status and show offline on the hub (contracts/MQTT.md, Status and LWT). Uses its own device_id and preview port, so the two running detectors are untouched. |
| `truth.json` | The trajectory table from the run in the report. Regenerate with `build_truth.py` rather than editing. |

### Reproducing the run

```bash
export ESK_FIXTURE_DIR=$HOME/edge-security-fixture   # scratch dir for videos + logs
mkdir -p "$ESK_FIXTURE_DIR"
cd tools/rtsp-fixture
docker compose up -d                                 # mediamtx :8555 + mosquitto :1884

# 1. a real person patch, then the designed clip + its truth table
cp /path/to/real_clip.mp4 "$ESK_FIXTURE_DIR/source_720p.mp4"
python3 find_person.py
python3 build_truth.py 5                             # -> truth.mp4 + truth.json

# 2. publish both streams
./publish.sh "$ESK_FIXTURE_DIR/source_720p.mp4"      # edge-sec-720p, the ADL stream
./publish_truth.sh "$ESK_FIXTURE_DIR/truth.mp4"      # edge-sec-truth, 5 fps

# 3. hub + two detectors
./start_stack.sh

# 4. assert
python3 verify_e2e.py 130                            # observation window, seconds
```

Overridable variables (all have defaults matching the layout above):
`ESK_ROOT`, `ESK_FIXTURE_DIR`, `ESK_GENERIC_DIR`, `ESK_HUB_DIR`, `ESK_HUB_DATA`,
`ESK_MODEL`, `ESK_HUB`, `ESK_MQTT_HOST`, `ESK_MQTT_PORT`, `ESK_HUB_USER`,
`ESK_HUB_PASSWORD`.

`verify_e2e.py` and `find_person.py` / `build_truth.py` import
`esk_generic.yolo`, so run them with the generic detector's environment on the
path (`cd platforms/generic && uv run python …/verify_e2e.py`) or point
`ESK_GENERIC_DIR` at a checkout whose deps are installed.

### How the harness matches alerts to truth instants

The clip loops (~31 s) inside a much longer observation window, so the truth
series is regenerated per loop from the tapped detections — a 130 s window holds
about four crossings, four zone entries and four escalations, not one of each.

Two rules keep the accounting honest, and both exist because breaking them
produced failures that looked like platform regressions:

**Alerts pair with truth instants by smallest |Δt|, globally** — not in id order,
and not "for each alert, its nearest unused instant". Both of those cascade: one
unpaired alert makes every later alert take the *next* loop's instant, so a
single anomaly is reported as N failures each off by exactly one clip period.
That is what the "leading `zone_enter` artefact" in the RK3588, RK3576 and Jetson
platform READMEs actually was — the artefact was one alert, the five failures
were the matcher. Those runs blamed the observation phase and re-ran; the
matcher was never the suspect. Nothing is suppressed by the fix: an alert with no
instant within `TOL_S`, or an instant with no alert, is still a failure.

**A truth instant is only asserted if the run watched long enough to see it
answered** (`assertable_until`). The tap close is one edge; the last detection
the stream published is the other, and it is often tighter. A `loitering` instant
is entry + `dwell_seconds`, and the hub only escalates while it keeps seeing the
track inside the zone — so an entry near the end of the window never escalates
once the stream stops, and demanding its alert fails on something that cannot
exist. A `zone_enter` instant is itself an observed detection, so it never hits
this edge.

`test_verify_e2e.py` (`python3 test_verify_e2e.py`, no broker or device needed)
pins both against a recorded RK3576 run, in both directions: the recorded run
must report exactly one complaint, and corrupted variants of it — every alert 5 s
late, an alert deleted, a wrong `track_id` — must still report.

### Publish rate is part of the fixture

The truth clip is published at 5 fps, not 15, and that is a correctness
requirement rather than a convenience. The CPU detector on the reference host
sustains ~6.2 fps when its ONNX thread pool is unbounded. Publish faster than
the consumer and the OpenCV RTSP reader accumulates, then fast-forwards: whole
trajectory segments are skipped, the frame gap exceeds `track_max_lost_s`, the
tracker issues a fresh `track_id` for the same person, and the hub — correctly,
on the input it was given — fires `zone_enter` again. The alerts look wrong and
the rule engine looks guilty. See `platforms/generic/README.md`
("The failure disguises itself as a rule bug") and cap `intra_threads`.
