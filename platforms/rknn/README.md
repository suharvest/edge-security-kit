# RK3588 detector (RKNN NPU + Rockchip MPP)

RTSP → person detection → `sensecraft.detection/1` over MQTT, with inference on
the RK3588 NPU and decode on the Rockchip MPP hardware decoder. It publishes the
same payloads as [`platforms/generic`](../generic), so the hub cannot tell the
platforms apart except through `health`.

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

cp config.example.yaml config/config.yaml     # edit device_id / source / mqtt_host
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
tools/prepare_model.sh --onnx /path/to/yolov8n.onnx --builder-host wsl2-local
```

Conversion is x86_64-only — there is no aarch64 RKNN Toolkit 2 wheel — so the
script delegates to an x86_64 Fleet host and pulls the artifact back.

**The toolkit version must match the runtime, and the filename lies about it.**
On radxa `/usr/lib/librknnrt.so` is a symlink to `librknnrt.so.2.3.0`, while the
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
  --calib-dir /path/on/builder/images --builder-host wsl2-local
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
| **in-pipeline** `inference_time_ms` p50 / p95 | 72.3 / 125.5 | 66.7 / 121.1 | **41.9 / 56.5** |
| **in-pipeline** `pipeline_ms` p50 | 73.1 | 67.5 | 44.3 |
| back-to-back over 500 stills, p50 / p95 | 48.9 / 55.3 | 44.4 / 62.7 | 18.8 / 22.8 |
| NPU Core0 @ 1.0 GHz | 26 % | 23 % | **8 %** |
| detector CPU (one core) | ~14 % | ~18 % | ~17 % |
| RSS after 20 s | 292 MB | 224 MB | 206 MB |
| sustained fps | 5.0 (source-limited) | 5.0 | 5.0 |

The two latency rows disagree by more than 2× and the top one is the real
number. A benchmark loop amortizes weight traffic that a 5 fps stream re-pays
on every frame (see the `core_mask` note below); quoting 18.8 ms as this
board's per-frame latency overstates it by 2.2×.

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

`config.example.yaml` still points at A. Switching the default is a deliberate
act, not a side effect of this measurement.

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
agree, and `fixture/evidence-annotated.jpg` shows boxes redrawn from published
values on a real 1280×720 frame.

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

## Measured on radxa (RK3588, librknnrt 2.3.2, 1280×720 H.264 @ 5 fps)

Model **A** only, and the original acceptance run rather than the one in the
comparison above; a repeat measured 72.3 / 125.5 ms, so treat the difference
between 77 and 72 as this board's run-to-run spread, not as a change.

| | RK3588 (this) | spark CPU (generic) |
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
all, so the RK3588 has almost the whole SoC left over while spark does not.

Two measurements worth keeping:

* **Isolated inference is ~46 ms; in the live 5 fps pipeline the same call takes
  77–116 ms.** It is not CPU contention (board load average 0.60) and not DVFS
  (the NPU sits at its maximum 1.0 GHz either way). Back-to-back inference
  amortizes weight traffic that a sparse 5 fps stream pays on every frame. So a
  benchmark loop overstates this board's real per-frame latency by roughly 2×,
  and capacity numbers derived from one are optimistic.
* **`core_mask` does not help.** `NPU_CORE_0_1_2` moves p50 by under a
  millisecond and makes p95 worse (104 ms vs 51 ms); `/sys/kernel/debug/rknpu/load`
  shows Core0 busy and cores 1–2 idle regardless. Multi-core is for running
  several models concurrently, not for splitting one small one — `npu_core_mask`
  is left `null`.

## Tests

```bash
uv run pytest     # letterbox/geometry, tracker IDs, head decode, fixture conformance
```

All host-only: no NPU, no board, no network. `fixtures/*.json` were captured
verbatim off the broker during a real RK3588 run and are validated against
`contracts/mqtt-detection.schema.json` as the contract's Conformance section
requires. `tests/test_fixtures_schema.py` additionally asserts the captured
fixtures report `decode: hw` — a fixture that says `sw` means the run being
certified was not the run this platform exists to deliver.
