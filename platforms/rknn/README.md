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

The model is fp16, not int8. fp16 needs no calibration set and is numerically
close to the ONNX reference, which is what makes the two platforms comparable;
int8 would be roughly twice as fast but needs a calibration set representative
of the deployed scene, and getting that wrong degrades the score head in a way
that shows up as tracker flicker rather than as an obvious error.

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
