# reCamera Pro (RV1126B) detector

Hub-mode detector for reCamera Pro, packaged as a reCamera app rather than a
container: the device runs Buildroot with no Docker, and its NPU is reached
through the on-device app manager.

Verified on the board on 2026-08-18: RV1126B NPU inference, the same
`tools/rtsp-fixture/verify_e2e.py` assertions the RK3588 and Orin NX platforms
run, **exit 0, no failures**. Broker, hub and detector all ran on the camera
itself.

## The camera path does not decode anything

The first version of this platform pointed the detector at the device's own
rkipc/go2rtc RTSP stream and decoded H.264 with ffmpeg on the CPU. That is an
encode/decode round trip the camera never needed: the sensor's frames are
already in system memory as ISP NV12 before anything encodes them, and this
firmware publishes them on `/run/recamera/frame.sock` (the extension API's
frame proxy, dma-buf + `SCM_RIGHTS`). The kit's capability registry selects
`OfficialFrameSource` for that socket automatically — the app only had to stop
overriding the source.

So the question "how do we get MPP hardware decode on this board" was the wrong
one, and its answer is anyway "you cannot": `ffmpeg -decoders` on this image
lists `h264`, `h264_v4l2m2m`, `hevc`, `hevc_v4l2m2m` and no `*_rkmpp`; every
`/dev/video*` node is an `rkisp`/`rkcif` capture path, not a stateful m2m
decoder; and there is no `mppvideodec` GStreamer plugin. `librockchip_mpp.so`
and `/dev/mpp_service` exist, so a ctypes MPP decoder is *possible*, but it
would only ever serve the replay fixture below — the live camera has no
bitstream to decode.

Two consequences for `health.decode`:

* **camera path → `hw`.** No CPU decode happens; RGA does NV12→RGB and the
  letterbox resize. `decode` is no longer a constant in the source — it is
  resolved from the live source object (`_resolve_decode_path`), so it cannot
  disagree with reality.
* **replay path (`source_file`) → `sw`,** truthfully: that fixture really is an
  H.264 file decoded by ffmpeg on the CPU.

### Kernel-side evidence, not a self-report

`health.decode` is a field the app writes, so `GET /debug/decode` (preview port,
default 8099) serves what the app does *not* author — `/proc/self`:

```json
"verdict": "zero-copy-isp",
"evidence": {"dmabuf_fd_open": true, "mpp_device_open": false,
             "mpp_lib_mapped": false, "ffmpeg_children": []},
"proc_self": {
  "open_devices": ["/dev/dma_heap/system", "/dev/null", "/dev/rga",
                   "/dev/rknpu", "/dmabuf:"],
  "mapped_libs": ["/oem/usr/lib/librga.so",
                  "/oem/usr/lib/librecamera_ext.so.1.0.0",
                  "/oem/usr/lib/librknnrt.so"],
  "open_pipe_fds": 0},
"capabilities": {"official_sockets": {"frame": {"exists": true}}, "euid": 0}
```

A dma-buf fd and `librecamera_ext` mapped, **zero** ffmpeg children and zero
pipe fds: nothing in this process could be decoding a bitstream. The same
endpoint on the replay path shows the mirror image — an `ffmpeg` child and no
dma-buf.

`/run/recamera` is `drwxr-x--- root root`, so the sockets are invisible to the
unprivileged fleet account and the capability probe reports `exists: false` for
all of them. That is why this had to be checked from inside an appmgr-started
(root) app, and why `capability_probe()` reports `dir_listable` next to
`exists`: "absent" and "not permitted to look" are different answers.

## The letterbox was the expensive stage, not the decode

`pipeline_ms` is measured *after* the frame read returns, so decode time was
never in it — the 45 ms between `inference_time_ms` and `pipeline_ms` in the old
table was Pillow resizing 1280×720 → 640×360 and padding it into a 640×640
canvas, once per frame, in Python. `model_frame = "hw"` hands that to RGA:
`Frame.model_data` arrives already letterboxed while `Frame.data` stays
full-resolution for the preview, the snapshot reply and the evidence frame.

The RGA geometry is checked against `esk.letterbox.fit_geometry` on every frame
(`_model_canvas`) and only used when the scaled size and both pad offsets match
exactly; a mismatch falls back to the Python letterbox rather than publishing
shifted boxes. Over the runs below: 992 frames, 992 hardware, 0 fallbacks, 0
mismatches.

Stage p50, from `GET /debug/decode` → `app.stage_ms_p50` (camera path, 6 fps):

| stage | before (`cpu` letterbox) | after (`hw` letterbox) |
|---|---:|---:|
| letterbox | ~46 ms (by difference) | **0.25 ms** |
| frame store copy | 2.2 ms | 2.2 ms |
| `detect()` (NPU call + NumPy head) | 44 ms | 44.3 ms |
| tracker | 0.15 ms | 0.15 ms |
| MQTT publish | 0.9 ms | 0.9 ms |
| kit `emit()` | 0.45 ms | 0.45 ms |
| **`pipeline_ms` p50 / p95** | **94.2 / 98.1 ms** | **47.4 / 50.7 ms** |
| **CPU** | 51.5 % of one core | **27.0 %** |
| **CPU-time per frame** | 85.8 ms | **44.9 ms** |

`inference_time_ms` *rose*, 30.6 → 41.8 ms p50, and that is not noise: the
governor is `interactive` over 594 MHz–1.608 GHz, and with the per-frame CPU
work nearly halved the board settles at `scaling_cur_freq = 594000`. The
host-side part of the RKNN call (input marshalling, output dequant) is CPU-bound
and runs proportionally slower there. Per-frame *work* fell by 48 %; per-call
*latency* on a downclocked core rose. Both numbers are above; neither alone
describes the change.

fps is unchanged at 6.0 because the frame broker, not the pipeline, sets it —
`pipeline_ms` 47 ms would support ~21 fps.

## Measured on a reCamera Pro (RV1126B, librknnrt 2.3.2, 1280×720 H.264 @ 5 fps replay)

The table below is the **replay fixture** (ffmpeg software decode of
`truth.mp4`), which is what the cross-platform assertions run against. It is not
the camera path; for that, see the stage table above.

| | value |
|---|---|
| `inference_time_ms` p50 / p95 (in-pipeline) | **29.9 / 32.3 ms** |
| `pipeline_ms` p50 / p95 (capture → published) | **74.7 / 90.4 ms** |
| detector CPU | 38–40 % of one core |
| detector RSS | 247 MB after 5 min, 408 MB after 20 min — **grows, see below** |
| sustained fps | 5.0 (source-limited), 651 msgs / 130.0 s = 5.01/s |
| decode | `sw` (ffmpeg), the only userspace path on this firmware |
| capture → alert p50 / p95 | 111.7 / 299.6 ms |

Against the other two platforms, same clip, same assertions:

| | Orin NX (TensorRT fp16) | RK3588 (RKNN int8) | **reCamera Pro (RKNN int8)** |
|---|---:|---:|---:|
| `inference_time_ms` p50 | 4.13 ms | 41.9 ms | **29.9 ms** |
| `inference_time_ms` p95 | 4.14 ms | 56.5 ms | **32.3 ms** |
| `pipeline_ms` p50 | 7.24 ms | 44.3 ms | **74.7 ms** |
| detector CPU | 8.5–12.5 % of one core | 21 % | **38–40 %** |
| RSS | 309 MB | 206 MB | **247 MB rising** |
| decode | NVDEC (`hw`) | MPP (`hw`) | ffmpeg (`sw`) |

The inference figure beating RK3588's int8 is not a faster NPU. Both rows are
in-pipeline `inference_time_ms`, but the two boards run different models:
RK3588's 41.9 ms is YOLOv8n at 640×640 against a **live** MPP pipeline whose
weight traffic is re-paid per frame, while this row is the same 640×640 graph
on a board doing nothing else. The honest reading is that they are the same
order of magnitude, not that RV1126B is 30 % faster than RK3588. No back-to-back
benchmark loop was run here, so unlike the RK3588 README there is no second,
smaller number to quote — every figure above came out of the live pipeline.

`pipeline_ms` is 45 ms wider than inference because letterbox, the
NumPy head decode and JPEG work are all on this CPU; RK3588 offloads decode and
scaling to MPP/RGA, and Orin to NVDEC.

## RSS growth: where it is, and where it is not

**RSS grows about 12–13 MB/min** and the rate is the same on both frame paths —
13 MB/min at 5 fps on the replay fixture, 12.3 MB/min at 6 fps on the zero-copy
camera path (182 244 → 234 916 kB over 256 s). On a 2 GB board that is a few
hours to trouble. `GET /debug/memory` (and `?deep=1` for the libc arena
summary) is the probe; here is what four layers of it say, over a 4-minute
window on the camera path:

| layer | A | B | reading |
|---|---:|---:|---|
| `rss_kb` | 202 424 | 259 400 | +57 MB |
| `rss_anon_kb` | 180 632 | 237 608 | all of it anonymous |
| `gc.tracked_objects` | 48 041 | 48 527 | flat |
| `gc.type_counts` | — | — | flat; largest delta `dict` +224 |
| `vma.mapping_count` | 318 | 318 | **flat** |
| `vma.heap_bytes` | 156 393 472 | 214 044 672 | **+56 MB — the whole delta** |
| `malloc_info` `rest` (free held) | 104 116 983 | 91 298 042 | **fell 13 MB** |
| `malloc_info` `mmap` | 5 767 168 | 5 767 168 | flat |

Read together these exclude three of the four candidates and name the fourth:

* **not a Python object leak** — object counts and per-type counts are flat
  while RSS climbs 14 MB/min;
* **not mapping accumulation** — a native library importing a dma-buf or mmapping
  a scratch buffer per frame and never unmapping would move `mapping_count`;
  it does not move at all;
* **not glibc fragmentation** — the allocator's free pool *shrinks* while the
  heap grows, so these are in-use blocks, not freed-but-unreturned ones. Tested
  directly as well: pinning `M_MMAP_THRESHOLD` at 128 kB
  (`malloc_tune`/`ESK_MALLOC_TUNE`, `mallopt` returning 1) made it **worse**
  (30 MB/min) and cost 19 ms per frame in page faults. That switch is kept, off
  by default, with the result recorded in `esk/mem_probe.tune_allocator` so the
  hypothesis is not re-tested from scratch;
* **it is `malloc`'d memory a C extension holds and never frees.** The
  process's native components are `librknnrt`, `librga` and `librecamera_ext`,
  and the growth rate is unchanged between the two frame paths — the replay path
  loads neither `librga` nor `librecamera_ext`. The only native component common
  to both is the RKNN runtime, reached once per frame through
  `kit.runtime.engine.infer` → `rknn_lite.inference()`. Per-frame arithmetic
  agrees: 12.3 MB/min at 6 fps is 34 kB per inference, 13 MB/min at 5 fps is
  43 kB per inference — the same order, scaling with inference count rather
  than with frame size or fps alone.

Not yet proven, and what would prove it: a standalone loop calling
`rknn_lite.inference()` on a constant array with no frame source, no MQTT and no
preview server. That needs `/dev/rknpu`, which is root-only, so it has to be
packaged as an appmgr app to run at all — the reason it is not in this round.
Note that the shipped `fall-detection` app uses the same `kit.runtime.engine`
wrapper and had been running 12 h at 263 MB when this work started, so if the
RKNN runtime is the source it is model- or output-shape-dependent (this app's
`rknn_model_zoo` head returns nine output tensors per frame) rather than
universal.

**Do not run this unattended until that is settled.** A supervisor restart on an
RSS ceiling is the available mitigation today.

## End to end, against the truth video

`tools/rtsp-fixture/verify_e2e.py`, 130 s observation, hub and broker on the
camera. Truth line at x=0.5 (no `ESK_LINE_X` nudge — this platform decodes at
full resolution in software, so published `cx` is not quantized to the 1/640
grid that put the RK3588 centroid exactly on the line).

| assertion | tolerance | measured |
|---|---|---|
| `line_cross` instants (×8) | ±0.5 s | 0.036 – 0.225 s |
| `line_cross` direction | exact | 8/8 correct |
| `line-fwd` fired on a backward crossing | none | **NONE** (4 alerts, all forward) |
| `line-bwd` fired on a forward crossing | none | **NONE** (4 alerts, all backward) |
| `zone_enter` instants (×4) | ±0.5 s | 0.009 – 0.010 s |
| `loitering` dwell (configured 10 s) | ≤1 s | 10.00 – 10.01 s |
| `loitering` first-fire instant | ±0.5 s | 0.010 – 0.019 s |
| snapshots | real JPEG | 8/8, `image/jpeg`, 44.4 – 46.3 KB |
| trajectory alignment vs `truth.json` | — | mean \|cx\| err 0.00231, p95 0.00447, max 0.02401 |
| stream continuity | — | 651 msgs / 130.0 s, gap p50 199 ms max 442 ms, one track id |

`fixtures/` holds the payloads captured off the broker during that run, all
three passing `contracts/validate_payload.py`.

## The NPU is actually running

`health.backend` is the process describing itself, which proves nothing. The
independent parts:

* the vendor runtime reports what it opened, not what the app claims —
  `RKNN Driver Information, version: 0.9.8` and
  `RKNN Model Information, ... target platform: rv1126b` come out of
  `librknnrt` after the driver ioctl succeeds;
* the negative control: the same package, same model, launched as the
  unprivileged fleet account instead of through appmgr, fails at
  `init_runtime` with
  `failed to open rknpu module, need to insmod rknpu dirver!` →
  `RKNN_ERR_FAIL`. `/dev/rknpu` is `crw------- root root`, so a run that
  reaches inference at all has opened it.

`/sys/class/devfreq/22000000.npu/load` is world-readable but reads
`100@800000000Hz` with the detector running **and** with no app running at all,
so on this kernel it is not a busy/idle discriminator. Do not cite it.

## Running it on the board

The device has no Docker, no root shell for the fleet account, and no MQTT
broker or hub of its own. All three parts run on the camera; a laptop only
opens the hub in a browser.

### Broker and hub

Both go in their own venv — **never into `/userdata/rknnenv`**, which is the app
runtime.

```sh
python3 -m venv /userdata/esk-hubenv
/userdata/esk-hubenv/bin/pip install amqtt "aiohttp>=3.9" aiomqtt bcrypt \
    jsonschema "paho-mqtt<2" sqlean.py
```

Three firmware facts this has to work around:

* **No MQTT broker.** The image ships `mosquitto_pub`/`_sub` but no
  `mosquitto`. `amqtt` is pure Python and installs from PyPI.
* **No `sqlite3` in CPython.** The image has no `_sqlite3` extension and no
  libsqlite3 at all, so `import sqlite3` fails and the hub cannot start.
  `sqlean.py` is a DB-API 2.0 module with SQLite statically linked and an
  aarch64 wheel; a `sitecustomize.py` in the venv aliases it into
  `sys.modules["sqlite3"]` before anything imports the stdlib name. Scoped to
  the venv.
* **amqtt 0.12.0 leaks retained messages across subscribers.** On CONNECT it
  replays retained messages for every filter in the broker-wide registry, not
  the connecting client's own, so a client subscribed to `.../detections/+`
  is handed the retained `.../status` message and any consumer that assumes the
  topic filter held will misparse it. One-line fix in `Broker.client_connected`:
  skip filters this session does not hold.

Then, with the broker on 1884 and the hub on 18080:

```sh
MQTT_HOST=127.0.0.1 MQTT_PORT=1884 HUB_HTTP_PORT=18080 \
  HUB_ADMIN_PASSWORD=... HUB_WEB_DIR=<web/dist> \
  /userdata/esk-hubenv/bin/python -m edge_hub --data-dir /userdata/esk-hubdata
```

### The detector

`/dev/rknpu` is root-only and the fleet account is uid 1000, so the app has to
go through the device's app manager, which is **single-active**: activating it
stops whatever app is currently running. Record the current `active_app` first
and put it back afterwards.

Packages must be signed — `POST /api/appMgr/install` refuses an unsigned one
with *"package is unsigned and signature verification is required"* — so build
the tar with `manifest.json` at the **top level** (not inside a directory) and
sign it with the release key before pushing:

```sh
tar -czf intrusion-detection.tar.gz manifest.json app.py esk models
python3 <recamera_pro>/market/packaging/sign.py --dist . --pkg intrusion-detection.tar.gz
# POST /api/appMgr/install {"path": "/userdata/.../intrusion-detection.tar.gz"}
# POST /api/appMgr/config  {"id": "intrusion-detection", "config": {...}}
# POST /api/appMgr/switch  {"id": "intrusion-detection"}
```

Everything the app needs is in the manifest's `config_schema`: `mqtt_host`,
`mqtt_port`, `device_id`, `stream_id`, `source_file` (replay instead of the
sensor), `annotate_path`, `preview_port`. appmgr owns the environment, so the
`ESK_*` variables only reach headless runs.

The app writes stdout to `<app_dir>/logs/app.log`, which is root-owned and
unreadable from the fleet account; appmgr exposes no log endpoint either. A
failed start shows only `last_exit.code` in `/api/appMgr/list`. Diagnosing the
first three failures needed a temporary package whose `app.py` `dup2`s its own
stdout/stderr to a world-readable path.

## What the board changed in this code

Three defects that only a real run could surface, all fixed here:

* `esk/file_source.py` probed the clip size with `ffprobe`. The image ships
  `ffmpeg` **without** `ffprobe`, so replay — the whole reason this module
  exists — could never have worked on the one platform it was written for. It
  now falls back to parsing `ffmpeg -i`'s stream line.
* Pillow cannot write JPEG on this firmware: the bundled Pillow 9.4.0 was built
  against libjpeg 9 and the image ships libjpeg 8, so every save dies with
  `Wrong JPEG library version: library is 80, caller expects 90` →
  `OSError: encoder error -2`. `PIL.features` cheerfully reports "JPEG support
  ok", so the check is an actual encode, once per process; the fallback shells
  out to `ffmpeg`, which links its own encoder. This affects the preview
  endpoint, the snapshot reply and the annotated evidence frame — i.e. every
  image the hub can ask for.
* `tools/rtsp-fixture/verify_e2e.py` now scopes alerts to the observation
  window and fetches them at tap close. On a fast host the harness's brute-force
  trajectory alignment takes seconds; here it takes ~5 minutes, during which the
  hub keeps firing, and those extra alerts were being zipped against truth
  crossings — reporting a clean run as eight wrong directions with −25 s errors.

## Coordinates

The model input is square, the source is not, so the frame is scaled with its
aspect preserved and padded, and the pad is reversed before publishing;
`coordinate_space` is the literal `frame_norm`, normalized against the original
1280×720 frame.

`fixtures/evidence-annotated.jpg` is one frame with the boxes **redrawn from the
published `frame_norm` values**, not from the detector's internal pixel boxes —
a wrong inverse shows up as a squashed or offset rectangle rather than as a
plausible number. The clip composites a 297×662 person patch at y=58, so the
drawing can be checked against arithmetic instead of taste:

| | truth patch | from the published bbox | delta |
|---|---:|---:|---:|
| box top | 58 px | 55.7 px | −2.3 px |
| box bottom | 720 px | 718.9 px | −1.1 px |
| box height | 662 px | 663.2 px | +1.2 px |

The padded axis here is the vertical one (720 → 360 inside a 640-tall canvas),
which is exactly where an unreversed pad would show: it would put the box 140 px
out, not 2 px. The residual is the detector boxing the person rather than the
patch rectangle. Width is 274 px against the patch's 297 for the same reason —
the body does not fill the patch's width.

## Layout

`esk/letterbox.py` and `esk/tracker.py` are byte-identical copies of their
`platforms/rknn` counterparts. `esk/zoo_head.py` reuses the same nine-output
decoder and adopts the kit-owned RKNN handle instead of loading a second copy.
`esk/preview.py` is ported off OpenCV to Pillow because this image carries no
cv2. `esk/file_source.py` replays a local file through the kit's frame source,
which is what makes the same truth-video assertions possible here as on RK3588.

The kit framework itself lives outside this repository, in the reCamera Pro
application tree; `manifest.json` is what the device's app manager consumes.

## Model

Binaries are not committed. `models/SHA256SUMS` pins what was built; regenerate
with the RK3588 conversion chain in `../rknn/tools/`, passing the `rv1126b`
platform and the same 400-image calibration list. `yolov8n` zoo ONNX converted
to RV1126B int8, `librknnc 2.3.2` matching the board's `librknnrt 2.3.2`; every
Conv/Split/Add/Concat is placed on the NPU, only the input and the nine output
operators sit on the CPU.

No accuracy sweep has been run on this platform. The RK3588 COCO numbers do not
transfer: same ONNX and the same calibration list, but a different quantizer
target.
