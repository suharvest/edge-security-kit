# RKNPU2 detector (RK3588 / RK3576, RKNN NPU + Rockchip MPP)

RTSP → person detection → `sensecraft.detection/1` over MQTT, with inference on
the Rockchip NPU and decode on the Rockchip MPP hardware decoder. It publishes
the same payloads as [`platforms/generic`](../generic), so the hub cannot tell
the platforms apart except through `health`.

**Two SoCs, one implementation.** RK3588 and RK3576 are both RKNPU2 behind the
same `librknnrt`, and both negotiate the same MPP/RGA output geometry, so there
is no chip branch anywhere in this package — not in the decoder, the letterbox,
the tracker or the publisher. The only per-chip input is which `.rknn` the
config points at, and a model built for the wrong SoC is refused at
`init_runtime` rather than run:

```
E RKNN: This rknn model is for RK3588, but current platform is RK3576
```

Both are verified on hardware; the measurements are in
[Measured on a Radxa Rock 5T](#measured-on-a-radxa-rock-5t-rk3588-librknnrt-232-1280720-h264--5-fps)
and [Measured on a LubanCat-3](#measured-on-a-lubancat-3-rk3576-librknnrt-232-1280720-h264--5-fps).

Like the generic platform it is a hub-mode detector: it publishes detections and
status, never events. The hub judges rules, which requires tracked targets
(`track_id >= 1`, HUB_SPEC §2.1), so a greedy IoU tracker runs here too.

| Topic | Payload | QoS / retain |
|---|---|---|
| `sensecraft/security/<device_id>/detections/<stream_id>` | `sensecraft.detection/1` | 0, no retain |
| `sensecraft/security/<device_id>/status` | `sensecraft.status/1` | 1, retained, LWT at CONNECT |
| `sensecraft/security/<device_id>/snapshot/<event_id>` | raw JPEG ≤ 200 KB | 1, no retain |

## Run

The board supplies `rknn-toolkit-lite2` and the GStreamer bindings; everything
else comes from the venv. The venv **must** be created with
`--system-site-packages`, because those two packages have to be the OS copies:
the RKNN user library has to match the kernel NPU driver, and the GStreamer
bindings have to match the system GStreamer.

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python \
  "opencv-python-headless>=4.9" "paho-mqtt>=1.6,<2.0" "pyyaml>=6.0"

cp config.example.yaml config/config.yaml           # RK3588
cp config.rk3576.example.yaml config/config.yaml   # RK3576
# then edit device_id / source / mqtt_host
./.venv/bin/python -m esk_rknn --config config/config.yaml
```

`--validate` loads the model and exits, `--seconds N` / `--frames N` bound a test
run, `--annotate out.jpg` dumps one frame with the boxes redrawn from the
*published* `frame_norm` values, and `--allow-sw-decode` permits the CPU
fallback (see below). Note that a relative `model:` path resolves against the
**config file's** directory, not the working directory.

## The model

`tools/prepare_model.sh` converts the same `yolov8n.onnx` the generic platform
uses (md5 `eda19d19a8aa73d54411a6fc1d091c7f`), so a coordinate difference
between the two platforms can never be blamed on different weights.

```bash
tools/prepare_model.sh --onnx /path/to/yolov8n.onnx --builder-host <x86_64-build-host>
tools/prepare_model.sh --onnx /path/to/yolov8n.onnx --platform rk3576 \
  --builder-host <x86_64-build-host>
```

`--platform` is the only thing that differs between the two SoCs' builds — same
ONNX, same calibration list, same toolkit version — so a difference measured
between the boards cannot be attributed to a different model.

Conversion is x86_64-only — there is no aarch64 RKNN Toolkit 2 wheel — so the
script delegates to an x86_64 Fleet host and pulls the artifact back.

**The toolkit version must match the runtime, and the filename lies about it.**
On the board `/usr/lib/librknnrt.so` is a symlink to `librknnrt.so.2.3.0`, while the
version compiled into that library is **2.3.2**:

```
$ strings /usr/lib/librknnrt.so | grep 'librknnrt version'
librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
```

So `health.backend` is derived by reading that string out of the binary rather
than from the SONAME, and the model is converted with `rknn-toolkit2==2.3.2`.
The board confirms the pairing at load time:

```
RKNN Runtime Information, librknnrt version: 2.3.2
RKNN Model Information, version: 6, toolkit version: 2.3.2, target platform: rk3588
```

The RK3576 board carries the same runtime, so it converts with the same 2.3.2:

```
RKNN Runtime Information, librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
RKNN Driver Information, version: 0.9.8
RKNN Model Information, version: 6, toolkit version: 2.3.2, target: RKNPU f2, target platform: rk3576
```

`W Query dynamic range failed ... RKNN_ERR_MODEL_INVALID` on every start is
harmless noise from a static-shape model, not a defect.

## Three models, and which one to ship

Two things were conflated in the original fp16 build: the ONNX export and the
precision. `airockchip/rknn_model_zoo` ships a *different* YOLOv8n ONNX with
DFL and the output concat cut out of the graph, and RKNPU2 has int8 kernels for
everything that is left. Testing int8 against the stock export alone would have
credited the graph change to quantization, so three models were measured:

| | ONNX | precision | file |
|---|---|---|---|
| **A** | stock ultralytics | fp16 | `yolov8n_fp16.rk3588.rknn` |
| **B** | model zoo | fp16 | `yolov8n_zoo_fp16.rk3588.rknn` |
| **C** | model zoo | int8, 400-image PTQ | `yolov8n_zoo_int8.rk3588.rknn` |

```bash
tools/prepare_model.sh --onnx yolov8n_zoo.onnx --graph zoo --dtype int8 \
  --calib-dir /path/on/builder/images --builder-host <x86_64-build-host>
```

`esk_rknn.rknn_yolo` decodes both head layouts and picks between them from the
number of tensors the runtime returns, so the model is swappable by config
alone. The zoo head's DFL runs in NumPy here, on the anchors that survive the
person-score threshold — single digits at 0.35, which is why moving that
arithmetic out of the graph costs nothing.

### Accuracy: COCO val2017, person class, 500 images

Detection at `conf 0.001 / iou 0.65` so the full PR curve is traced; the
detector ships at 0.35/0.45. The evaluation half (odd `image_id`) is disjoint
from the calibration half (even). GT instances: 736 small, 679 medium, 490
large.

| | AP@.5 | AP@.5:.95 | AP small | AP med | AP large |
|---|---:|---:|---:|---:|---:|
| A stock fp16 | 0.7836 | 0.5374 | 0.2787 | 0.6353 | 0.7736 |
| B zoo fp16 | 0.7838 | 0.5375 | 0.2786 | 0.6362 | 0.7727 |
| C zoo int8 | 0.7784 | 0.5322 | 0.2726 | 0.6299 | 0.7708 |

**A and B agree to the fourth decimal.** Same weights, same answers, different
graph decomposition — which is also the check that the second decoder is right.

**int8 costs half an AP point**: −0.52 on AP@.5:.95, −0.61 on small (−2.2%
relative), −0.28 on large. Small targets lose the most, as expected, and the
margin is small enough that a coarser sweep would not resolve it, which is why
detection runs at 0.001 rather than at the deployed threshold.

`tools/eval_coco_person.py` runs this: `subset` on a workstation, `infer` on
the board, `score` back on the workstation with pycocotools.

### Speed, measured twice on purpose

| | A stock fp16 | B zoo fp16 | C zoo int8 |
|---|---:|---:|---:|
| **in-pipeline** `inference_time_ms` p50 / p95 | 72.3 / 125.5 | 66.7 / 121.1 | **36.2–44.5 / 55.8–56.2** |
| **in-pipeline** `pipeline_ms` p50 | 73.1 | 67.5 | 38.1–46.3 |
| back-to-back over 500 stills, p50 / p95 | 48.9 / 55.3 | 44.4 / 62.7 | 18.8 / 22.8 |
| NPU Core0 @ 1.0 GHz | 26 % | 23 % | **8–11 %** |
| detector CPU (one core) | ~14 % | ~18 % | 16.4–17.9 % |
| RSS after 20 s | 292 MB | 224 MB | 214 MB |
| sustained fps | 5.0 (source-limited) | 5.0 | 5.0 |

The int8 column is the controlled 2026-08-18 re-measurement (four 120 s
windows, see below); A and B are the earlier single runs and are kept for the
shape of the comparison, not as same-session numbers.

The two latency rows disagree by more than 2× and the top one is the real
number. A benchmark loop amortizes weight traffic that a 5 fps stream re-pays
on every frame (see the `core_mask` note below); quoting 18.8 ms as this
board's per-frame latency overstates it by 2–2.5×.

### The controlled re-measurement, 2026-08-18

Model C, four consecutive 120 s samples, each preceded by 30 s of warmup, same
clip at 5 fps, `conf 0.35` / `iou 0.45`, 640 input, `pipeline_ms` and
`inference_time_ms` read off the published payload so the figures line up with
the Jetson board's. 600 messages per window in every case, 5.01 fps published.

| window | board state | infer p50 | infer p95 | pipeline p50 | pipeline p95 | CPU | RSS |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | as-found, detector 2 days old | 36.91 | 56.12 | 39.00 | 59.86 | 17.1 % | 1850 MB |
| 2 | as-found, detector restarted | 36.22 | 56.08 | 38.08 | 59.80 | 16.4 % | 214 MB |
| 3 | quiesced | 39.62 | 55.80 | 41.27 | 59.60 | 16.9 % | 300 MB |
| 4 | quiesced, repeat | 44.47 | 56.16 | 46.30 | 59.94 | 17.9 % | 338 MB |

`quiesced` means the two co-tenant containers that measurably used CPU — an
unrelated RTSP fixture pair at 3.18 % and 5.36 % of one core — were stopped for
the run and restarted afterwards. The other six containers on the board,
including a voice stack, all measured at or below 1.00 % of one core and were
left alone.

**Quiescing did not help.** p50 moved 8 ms in the wrong direction while p95
stayed inside 0.4 ms across all four windows. Over the same windows the NPU held
1.0 GHz, the DDR controller held 534 MHz, `npu-thermal` sat at 62 °C and NPU
Core0 read 8–11 % with cores 1–2 idle, so there is no thermal or DVFS story
behind the p50 movement either — it is this board's window-to-window spread.
**Report RK3588 p50 as a range and lean on p95, which is the reproducible
number.** Anyone comparing a single RK3588 p50 against another board's is
comparing against noise of ±8 ms.

RSS grows with process age: 214 MB after a minute, 338 MB after ten, 1850 MB
after two days. Latency did not degrade with it — the two-day-old process turned
in the fastest p50 of the four windows — but a long-lived deployment should
expect the working set to keep climbing.

### Single-stream ceiling: 31.4 inferences/s

One process, flat out, preprocess + infer + head decode, 60 s, model C, with the
pipeline detector stopped:

| loop | inferences/s | infer p50 | infer p95 |
|---|---:|---:|---:|
| full: preprocess + infer + head decode | **31.37** | 34.24 ms | 37.26 ms |
| pure `rknn.inference()`, no head decode | 34.78 | 26.84 ms | 37.34 ms |

Both loops are measured with the same `last_inference_ms` timer around the same
`rknn.inference()` call, so the 7.4 ms difference between them is not decode
work being counted — it is the NumPy head decode running *between* inferences
and slowing the inference itself, which is the same weight-traffic effect the
5 fps stream pays. This is why the loop you pick decides the answer: against an
in-pipeline p50 of 36.2–44.5 ms, the full loop is optimistic by 6–30 % and the
pure-inference loop by 35–66 %.

The Orin NX ceiling, measured with the identical full-loop recipe, is 167
inferences/s. That ratio — 5.3× — is the honest single-stream comparison; the
ratio of the two boards' pure-inference loops would flatter RK3588.

Reading the in-pipeline row: the graph change is worth **8 %**, quantization a
further **37 %**, and together they take inference from 72 ms to 42 ms and the
NPU from 26 % busy to 8 %. p95 improves more than p50 (125 → 57 ms), which is
the number that decides whether a detector keeps up with a stream.

CPU goes slightly *up* on the zoo models: the DFL that left the graph landed in
NumPy. A fair trade at 5 fps, worth re-checking at 25.

### End to end, against the truth video

`tools/rtsp-fixture/verify_e2e.py`, fixture clip, 130 s, each model in turn on
the live MPP pipeline:

| | line_cross err | direction | zone_enter err | loitering dwell err | result |
|---|---|---|---|---|---|
| A | 0.086–0.261 s | 8/8 correct | (see note) | 0.064–0.103 s | crossings pass |
| B | 0.078–0.251 s | 8/8 correct | 0.053–0.054 s | pass | 1 tail artefact |
| C | 0.085–0.248 s | 8/8 correct | 0.056–0.057 s | 0.080–0.233 s | **exit 0, all pass** |

No forward-line alert fired on a backward crossing, and none the other way, for
any model. The crossing instants move by under 20 ms between fp16 and int8 —
inside the ±0.5 s tolerance and inside the 200 ms frame period. int8 loses
nothing the rules can see.

The A and B rows each carry one matching artefact rather than a detection
failure: an alert at the edge of the observation window whose truth instant
lies outside it. It appears when a run starts against a hub that has no rules
yet, and does not recur once the loop is established — which is why C's run is
clean.

### What int8 actually changes, visually

`tools/compare_annotate.py` overlays both models on identical pixels.

* `fixtures/int8-vs-fp16-surveillance.jpg` — two people in a corridor. The
  boxes agree at IoU ≥ 0.7, corners move ~1 px, scores 0.848 → 0.839 and
  0.733 → 0.696.
* `fixtures/int8-vs-fp16-cctv.jpg` — same, scores 0.891 → 0.884 and
  0.856 → 0.868. One down, one up.
* `fixtures/int8-vs-fp16-coco-small.jpg` — the interesting one. int8 emits an
  *extra* box at 0.365 on a dark object on a dining table. Not a recovered
  distant person: a false positive fp16 kept below threshold. The quantized
  score head is noisier near the threshold in both directions, so the failure
  mode to watch for is extra low-confidence boxes, not silence.

### Verdict

**Ship C.** 37 % less inference latency in the real pipeline, p95 more than
halved, NPU load 26 % → 8 %, for half an AP point overall and 0.6 on small
targets, with the end-to-end assertions passing unchanged. Mixed quantization
was not needed: no layer here loses enough to justify keeping it in float.

Two caveats stay attached:

* The small-target result rests on COCO's 736 small instances. The surveillance
  footage used for calibration turned out to contain almost none — 4 small
  boxes against 533 large across 300 frames — so nothing here tests a person at
  30 m on a real camera. Before deploying to a wide-angle overhead site, re-run
  the AP sweep on footage from it.
* A 20-image generic calibration set scores the same as the 400-image
  surveillance one, on COCO and on a surveillance holdout alike. The set in
  `calibration/` is kept because it is the right shape for the deployment, not
  because it was measured to help. See `calibration/README.md`.

`config.example.yaml` ships C. `config.rk3576.example.yaml` ships the RK3576
build of C for the same reasons, re-measured on that board rather than assumed —
see below.

## Coordinates

The model input is square, the source is not, so the frame is scaled with its
aspect preserved and padded, and the pad is reversed before publishing —
`coordinate_space` is the literal `frame_norm`, normalized against the original
frame.

This platform has a failure mode the CPU one does not: **the scaling happens in
two different places.** On the hardware path `mppvideodec` is told the
aspect-fitted output size and RGA scales during decode; on the software fallback
OpenCV does it. If the two disagree by one pixel of pad, the same scene produces
different published coordinates depending on which decoder started. Both
therefore call `letterbox.fit_geometry`, and the inverse is built from the
integer geometry actually applied. `tests/test_letterbox.py` pins that they
agree, and `fixtures/rk3588-evidence-annotated.jpg` shows boxes redrawn from
published values on a real 1280×720 frame.

**Re-verified per SoC, not inherited.** RGA is a different block on RK3576 and
nothing guarantees it rounds a scale the same way, so the inverse was measured
again there rather than assumed from the RK3588 pass. It negotiates the
identical 640×360 output, and `fixtures/rk3576-evidence-annotated.jpg` (subject
right of the line) and `-left.jpg` (subject left of it) show the boxes on that
board. The number behind the pictures is in the RK3576 section below: against
`truth.json`, which describes the subject in *original-frame* normalized units,
the published box height is off by a mean of 0.0035 — 2.5 px of 720. An
unreversed pad would put it 261 px out.

The scale is kept per-axis rather than as one shared ratio: after integer
rounding a 1281-wide source does not scale identically in both axes, and one
shared scale biases every box by up to half a source pixel.

## Hardware decode is verified, not assumed

`health.decode` exists because every platform has a silent-CPU-fallback failure
mode, and on Rockchip it is the *normal* one. If `mppvideodec` is not visible to
GStreamer — missing plugin, a stale plugin registry in `~/.cache/gstreamer-1.0`,
or `PYTHONPATH` replaced rather than appended so PyGObject disappeared with it —
the obvious implementation quietly opens the stream with ffmpeg on the CPU and
everything keeps working, slower, forever.

Two things prevent that:

* the pipeline is built element by element, and `ElementFactory.make` returning
  `None` is raised rather than swallowed, so "MPP is not installed" cannot be
  mistaken for "MPP is installed and slow";
* `require_hw_decode: true` (the default) **refuses to run** on the CPU decoder
  rather than merely reporting it. Set it false and the detector starts anyway
  with `decode: sw` and `fallback_active: true` on every message.

`GET :8099/debug/decode` reports the element factory GStreamer actually
instantiated and the caps it negotiated, read off the live pipeline:

```json
{"decode": "hw", "backend_class": "GStreamerMPP", "decoder_factory": "mppvideodec",
 "negotiated_caps": "video/x-raw, format=(string)RGB, width=(int)640, height=(int)360, ...",
 "source_size": [1280, 720]}
```

That is still the process describing itself. The kernel is the independent
witness — with the detector running, the process holds the hardware decoder and
scaler device nodes and hosts the MPP driver's worker threads:

```
$ ls -l /proc/$PID/fd | grep -oE '/dev/[a-z0-9/_]+' | sort | uniq -c
      4 /dev/dma_heap/cma
      1 /dev/dri/card1
      1 /dev/mpp_service      <- Rockchip hardware video decoder
      1 /dev/rga              <- 2D hardware scaler
$ ps -L -p $PID -o comm= | grep -i mpp
mpp_dec_parser
mpp_dec_hal
```

## Preview endpoint

`preview_enabled` starts an HTTP server (default `:8099`):

| Route | Serves | Advertised as |
|---|---|---|
| `/preview.jpg`, `/preview/<stream_id>.jpg` | latest frame as JPEG | `status.streams[].preview_url` |
| `/live`, `/live/<stream_id>`, `/` | page reloading that JPEG once a second | `status.streams[].live_url` |
| `/debug/decode` | live decoder identity (above) | — |

Both URLs are advertised only when `preview_advertise_host` is set: the device
cannot guess which address a browser can reach, and a wrong URL is worse than an
absent one.

The served frame is the letterbox canvas with the padding cut back off. On the
MPP path the full-resolution frame never exists in system memory — RGA scales
during decode — so a full-resolution preview would have to be invented. What is
served is genuinely decoded pixels at the source aspect ratio, just smaller
(1280×720 arrives as 640×360). The rule canvas works in normalized coordinates
and is unaffected.

## Measured on a Radxa Rock 5T (RK3588, librknnrt 2.3.2, 1280×720 H.264 @ 5 fps)

Model **A** only, and the original acceptance run rather than the one in the
comparison above; a repeat measured 72.3 / 125.5 ms, so treat the difference
between 77 and 72 as this board's run-to-run spread, not as a change. The
2026-08-18 int8 re-measurement puts a number on that spread: four consecutive
120 s windows moved p50 by 8 ms with p95 fixed to within 0.4 ms.

| | RK3588 (this) | 20-core aarch64 workstation, CPU (generic) |
|---|---|---|
| `inference_time_ms` p50 / p95 | 77 / 116 ms | 30–37 ms |
| back-to-back inference (no stream) | 46 ms p50 | — |
| `pipeline_ms` p50 | 78 ms | 58–72 ms |
| sustained fps | 5.0 (source-limited) | 14.92 (source-limited) |
| decode | `hw` (MPP), ~0 % CPU | `sw` (ffmpeg) |
| detector CPU / RSS | 21 % of one core / 262 MB | ~250 % / — |
| NPU load | Core0 27 % @ 1.0 GHz, cores 1–2 idle | — |

**The NPU is slower per inference than a 20-core server CPU here, and that is
the real result rather than a misconfiguration.** RKNPU2 is optimized for int8;
an fp16 YOLOv8n does not play to it. What the board wins is everything else: one
fifth of a core against two and a half cores, and decode that costs no CPU at
all, so the RK3588 has almost the whole SoC left over while the workstation
does not.

Two measurements worth keeping:

* **Isolated inference is ~46 ms; in the live 5 fps pipeline the same call takes
  77–116 ms.** It is not CPU contention (board load average 0.60) and not DVFS
  (the NPU sits at its maximum 1.0 GHz either way). Back-to-back inference
  amortizes weight traffic that a sparse 5 fps stream pays on every frame. So a
  benchmark loop overstates this board's real per-frame latency, and capacity
  numbers derived from one are optimistic. How much it overstates depends on
  what the loop contains: on the int8 model the 2026-08-18 re-measurement put a
  full preprocess + infer + decode loop 6–30 % fast and a pure-inference loop
  35–66 % fast against the same pipeline. Quote the full loop, or quote the
  pipeline.
* **`core_mask` does not help.** `NPU_CORE_0_1_2` moves p50 by under a
  millisecond and makes p95 worse (104 ms vs 51 ms); `/sys/kernel/debug/rknpu/load`
  shows Core0 busy and cores 1–2 idle regardless. Multi-core is for running
  several models concurrently, not for splitting one small one — `npu_core_mask`
  is left `null`.

## Measured on a LubanCat-3 (RK3576, librknnrt 2.3.2, 1280×720 H.264 @ 5 fps)

EmbedFire LubanCat-3, `lubancat-rk3576-debian12-gnome-20250721`, kernel
`6.1.99-rk3576`, `librockchip-mpp1 1.5.0-1`, `librga2 2.2.0-1`, RKNPU driver
`0.9.8`, NPU pinned at 950 MHz by the `performance` governor. Broker, hub and
detector all on the board; RTSP served from the board too, so network jitter
cannot be read as detector latency.

### The NPU is reachable, and not through a `/dev/rknpu` node

Worth stating because the obvious check fails: **there is no NPU device node
under `/dev/` on this image.** `ls /dev | grep -i npu` returns nothing, and it
would be easy to conclude the driver is absent. It is not — RKNPU registers as a
DRM device, and the runtime reaches it through `/dev/dri/card1`:

```
$ cat /sys/class/drm/card1/device/uevent | grep DRIVER
DRIVER=RKNPU
$ readlink -f /sys/class/drm/renderD129/device
/sys/devices/platform/27700000.npu
```

`init_runtime` is the test that settles it, and it also settles the model
question in one step, because a model for the other SoC is rejected there:

```
RKNN Runtime Information, librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
RKNN Driver Information, version: 0.9.8
RKNN Model Information, ... target: RKNPU f2, target platform: rk3576
INIT_RUNTIME_RC = 0
```

### int8 against fp16, in the live pipeline

70 s per model on the running stream, same board, same source:

| | zoo fp16 | zoo int8 |
|---|---:|---:|
| `inference_time_ms` p50 / p95 | 48.60 / 55.27 | **23.88 / 27.87** |
| detector CPU (one core) | 16.3 – 19.3 % | 8.6 – 14.0 % |
| RSS | 155 MB | 200 MB |
| NPU Core0 @ 950 MHz | 9 – 10 % | 5 – 6 % |
| NPU Core1 | 0 % | 0 % |
| sustained fps | 5.00 (source-limited) | 5.00 (source-limited) |

int8 halves inference latency here, as it does on RK3588, so **C is what
`config.rk3576.example.yaml` ships**. Two things in that table are not what the
RK3588 measurement would predict and are reported as measured rather than
explained: int8's RSS is *higher* than fp16's on this board, and its CPU is
lower — the opposite of RK3588, where the zoo/NumPy DFL pushed CPU up. Neither
was chased down; at 5 fps neither changes a decision.

No accuracy sweep was run on this board. The COCO and surveillance numbers above
are RK3588's, and they are properties of the quantized graph and its calibration
set, both of which are byte-identical here — but that is an argument, not a
measurement, and small-target AP on RK3576 is **unverified**.

### Against the four platforms

| | Orin NX 16GB | Rock 5T (RK3588) | **LubanCat-3 (RK3576)** | reCamera Pro (RV1126B) |
|---|---:|---:|---:|---:|
| precision | TensorRT FP16 | RKNN int8 | **RKNN int8** | RKNN int8 |
| inference p50, in pipeline | 4.13 ms | 41.9 ms | **23.9 ms** | 29.9 ms |
| full pipeline p50 | 7.24 ms | 44.3 ms | **26.2 ms** | 74.7 ms |
| detector CPU | 8.5 – 12.5 % | 21 % | **8.6 – 14.0 %** | 38 – 40 % |
| decode | NVDEC | MPP | **MPP** | ffmpeg (software) |

**RK3576 runs this model at roughly half of RK3588's per-frame latency.** Both
boards load `librknnrt 2.3.2`, both were given a model converted from the same
ONNX with the same toolkit and the same 400-image calibration list, and the only
argument that differed was `--platform`, so the model is ruled out. What is not
ruled out is the boards: the two numbers come from different systems measured on
different days, and no attempt was made to hold DVFS, thermals or kernel version
equal. Read it as "RK3576 is not the slower part it is priced as", not as a
per-TOPS ranking of the two NPUs.

### End to end, against the truth video

`tools/rtsp-fixture/verify_e2e.py`, 130 s, int8, hub and broker on the board:

```
== FAILURES
  none
```

| assertion | tolerance | measured |
|---|---|---|
| `line_cross` forward instants (×4) | ±0.5 s | 0.177 – 0.189 s |
| `line_cross` backward instants (×4) | ±0.5 s | 0.011 – 0.013 s |
| direction correctness | 8/8 | 8/8 |
| forward-only line fired on a backward crossing | never | **NONE** |
| backward-only line fired on a forward crossing | never | **NONE** |
| `zone_enter` instants (×4) | ±0.5 s | 0.002 – 0.004 s |
| `loitering` dwell error | ≤1 s | 0.006 – 0.204 s |
| snapshots | real JPEG | 8/8, `image/jpeg`, 17.5 – 17.7 KB |
| trajectory alignment vs `truth.json` | — | mean \|cx\| err 0.00204, p95 0.00322, max 0.00463 |
| stream continuity | — | 651 msgs / 130.0 s = 5.01/s, gap p50 202 ms max 217 ms, one track id |

Capture-to-alert p50 **39.4 ms**, p95 **224.3 ms**.

The truth line is nudged to x=0.503 (`ESK_LINE_X`), the same offset RK3588 and
Jetson used. RK3576 needs it for the same reason RK3588 does and not by
inheritance: MPP/RGA emits 640 px wide here too, so a published `cx` can only
land on a multiple of 1/640 and 0.5 is exactly on that grid.

The first run of these assertions reported five `zone_enter` failures, each off
by 31.0 s — one truth-video loop period exactly. That is the artefact already
documented for RK3588 and Jetson: the run began mid-loop with the subject
already inside the zone, the hub emitted a leading `zone_enter` the moment its
rules were installed, and the in-order matcher then paired every later alert
with the previous loop's instant. The table above is the immediately following
run on the established loop, unchanged in every other respect.

### Letterbox, checked on the vertical axis

The x axis is covered by the trajectory alignment above, which is against
original-frame normalized coordinates. `truth.json` also fixes the subject's
vertical geometry — `cy_const` 0.54028, patch height 662/720 = 0.91944 — so the
published boxes can be scored against it directly. Over 301 consecutive
detections:

| | truth | mean published | mean abs err | max abs err |
|---|---:|---:|---:|---:|
| `bbox` cy | 0.54028 | 0.53675 | 0.00353 (2.5 px) | 0.00959 (6.9 px) |
| `bbox` h | 0.91944 | 0.91649 | 0.00354 (2.5 px) | 0.01534 (11 px) |

A letterbox pad that was never reversed would put `h` at ~0.36 rather than 0.92.
`bbox` w is legitimately narrower than the patch (0.214 against 0.232): the
patch is a rectangular crop containing a person, and the detector boxes the
person.

### Hardware decode, from the kernel side

`GET :8099/debug/decode` reports `mppvideodec` and RGB 640×360, and the kernel
agrees — with the detector running, that process is the only holder of
`/dev/mpp_service` on the board:

```
$ lsof /dev/mpp_service
COMMAND     PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
python  2034513  cat   25u   CHR  241,0      0t0   91 /dev/mpp_service

$ ls -l /proc/2034513/fd | grep -oE '/dev/[a-z0-9/_]+' | sort | uniq -c
      4 /dev/dma_heap/system
      1 /dev/dri/card1        <- RKNPU (there is no /dev/rknpu on this image)
      1 /dev/mpp_service      <- Rockchip hardware video decoder
      1 /dev/rga              <- 2D hardware scaler

$ ps -L -p 2034513 -o comm= | grep -i mpp
mpp_dec_hal
mpp_dec_parser
```

One difference from the RK3588 board, which changes nothing but is real: the DMA
buffers come from `/dev/dma_heap/system` rather than `/dev/dma_heap/cma`.
`health.decode` reports `hw` with `fallback_active: false` throughout, and the
captured `fixtures/rk3576-detection.json` carries that.

## Tests

```bash
uv run pytest     # letterbox/geometry, tracker IDs, head decode, fixture conformance
```

All host-only: no NPU, no board, no network. `fixtures/rk3588-*.json` and
`fixtures/rk3576-*.json` were captured verbatim off the broker during a real run
on each board — one set per SoC, neither copied from the other — and validated against
`contracts/mqtt-detection.schema.json` as the contract's Conformance section
requires. `tests/test_fixtures_schema.py` additionally asserts the captured
fixtures report `decode: hw` — a fixture that says `sw` means the run being
certified was not the run this platform exists to deliver.
