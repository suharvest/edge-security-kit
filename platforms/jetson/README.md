# Jetson Orin detector (NVDEC + TensorRT)

Person detection on a Jetson Orin, publishing `sensecraft.detection/1` to the
hub. NVDEC decodes the RTSP stream, TensorRT runs YOLOv8n on the GPU, and the
rest — letterbox, head decode, tracking, MQTT, preview — is NumPy and stdlib
Python shared with the other platforms in this repository.

Measured on an Orin NX 16 GB, JetPack 6.1 (L4T R36.4.3), TensorRT 10.3.0,
CUDA 12.6, against the 1280×720 H.264 truth clip at 5 fps.

| | value |
|---|---|
| `inference_time_ms` p50 / p95 (in-pipeline, CUDA events) | **4.16 / 4.19 ms** |
| `pipeline_ms` p50 / p95 (published payload field) | **6.99 / 7.03 ms** |
| `pipeline_ms` p50 / p95 (`--stages`, capture → published) | 7.26 / 7.34 ms |
| detector CPU | 5.5 % of one core |
| RSS | 310 MB |
| single-stream ceiling, one process flat out | **167 inferences/s** |
| GPU inference ceiling, all contexts | **236 inferences/s** |
| sustained fps | 5.0 (source-limited) |
| decode | `hw` (NVDEC), verified |
| engine build time on device | 307 s |

Re-measured 2026-08-18: 30 s warmup, then a 120 s sample of 601 published
payloads, `conf 0.35` / `iou 0.45`, 640 input, against the same clip. The
detector's own summary agrees with the payload sample to the second decimal
(`published 1187 detection messages, 5.00 fps, inference p50=4.16ms
p95=4.19ms`), which is the check that the collector is reading what the
detector thinks it produced.

Two rows changed meaning rather than value. `pipeline_ms` is now quoted from
the payload field, because that is the field the RK3588 board also publishes
and the two boards are only comparable on one definition; the `--stages`
number, which additionally covers the publish call, is kept alongside it.
Detector CPU dropped from a previously quoted 8.5–12.5 % because that range
came from `ps -o pcpu`, which averages over the whole process lifetime and so
carries engine load and TensorRT warmup into a steady-state figure. 5.5 % is
`utime+stime` deltas from `/proc/<pid>/stat` across the 120 s window, sampled
every 5 s, and it never left 5.19–5.60 %.

**Nothing had to be stopped for this.** `docker stats` during the run showed
only this project's own RTSP fixture — `esk-jetson-rtsp-pub` 0.06 %,
`esk-jetson-rtsp-server` 0.10 % — and the board's load average was 0.00 before
the detector started, so the as-found and quiesced conditions the top-level
README describes are the same state on this board. `tegrastats` at 1 s
intervals read `GR3D_FREQ 0%` in 215 of 239 samples and never above 24 %,
which is what a 5 fps source and a 4 ms inference should look like: a 2 %
duty cycle that a 1 s sampler mostly misses.

The single-stream ceiling is one process running flat out through
preprocess + infer + head decode: **166.99 inferences/s over 60 s, 10020
inferences, per-inference p50 3.82 / p95 3.83 ms**. Against the in-pipeline
4.16 ms that loop is optimistic by 8 % — small enough that on this board a
benchmark loop is a fair proxy, which is not true on RK3588 (see
`platforms/rknn/README.md`).

## Container image

`Dockerfile` builds `edge-security-detector-jetson`, 682 MB, arm64. It is a
plain `ubuntu:22.04` base — L4T r36.x userspace *is* Ubuntu 22.04, so the image
needs no CUDA, no TensorRT and no L4T multimedia stack of its own, which is the
difference between this and a multi-GB `l4t-cuda` build.

Everything version-locked to the board arrives at runtime instead:
`runtime: nvidia` injects libcuda, the `/usr/lib/aarch64-linux-gnu/nvidia` tree,
the two L4T GStreamer plugins (`libgstnvvideo4linux2.so`, `libgstnvvidconv.so`)
and the decoder device nodes; libnvinfer and the `tensorrt` Python package are
bind-mounted at the private paths `/host-trt` and `/host-python`. They must not
be mounted over the container's own `/usr/lib/aarch64-linux-gnu` or
`/usr/lib/python3.10/dist-packages` — that hides the image's own libraries and
numpy fails with `libblas.so.3: cannot open shared object file`.

The engine is not in the image (see below); the ONNX is, at
`/app/models/yolov8n.onnx`, so a deploy step can copy it out and build the
engine on the target.

```bash
docker build -f platforms/jetson/Dockerfile \
  -t <registry>/edge-security-detector-jetson:0.1.0 .
```

Verified containerised on an Orin NX 16GB: `decode: hw` through
`nvv4l2decoder`, `/dev/v4l2-nvdec` and `/dev/nvmap` open in the container
process, published `inference_time_ms` 4.13 — within 0.03 ms of the bare-metal
run, so the container costs nothing measurable on the inference path.

## Run

```bash
# 1. venv. --system-site-packages because tensorrt and PyGObject must come from
#    the board; numpy/opencv are installed INTO the venv because the L4T system
#    numpy on this image is broken (missing libblas.so.3), which takes the
#    system cv2 down with it.
uv venv --system-site-packages --python /usr/bin/python3.10 .venv
uv pip install --python .venv/bin/python \
  "numpy>=2,<3" "opencv-python-headless>=4.9" "paho-mqtt>=1.6,<2.0" \
  "pyyaml>=6.0" "cuda-python>=12.2,<13"

# 2. engine, ON THIS DEVICE (see "The engine is not a file you can copy")
./tools/build_engine.sh models/yolov8n.onnx models/yolov8n_fp16.orin.engine

# 3. config, then run
cp config.example.yaml config.yaml
.venv/bin/python -m esk_jetson --config config.yaml --validate   # loads the engine, exits
.venv/bin/python -m esk_jetson --config config.yaml
```

Useful flags: `--seconds N` / `--frames N` to bound a run, `--annotate FILE` to
write one evidence JPEG drawn from the published coordinates, `--stages FILE`
to dump the per-stage timing breakdown on exit.

## The model

Stock ultralytics YOLOv8n, `models/yolov8n.onnx`, md5
`eda19d19a8aa73d54411a6fc1d091c7f`, single output `(1, 84, 8400)` with DFL
already applied inside the graph. Same file the `platforms/generic` reference
uses, so `status.versions.model` is comparable across platforms.

**Do not use the `airockchip/rknn_model_zoo` export here.** That graph has DFL
and the final concat cut out because RKNPU2 has no int8 kernel for the
transposed softmax. TensorRT has no such limitation, so the nine-output head
would only move arithmetic out of the GPU and into NumPy.

### The engine is not a file you can copy

A serialized TensorRT engine is valid only for the exact TensorRT version and
GPU architecture that built it. Copying one from another machine — or from the
same machine after a JetPack upgrade — makes `deserialize_cuda_engine` return
`None` with no other diagnostic. `models/*.engine` is gitignored for that
reason and `tools/build_engine.sh` is meant to run on the target.

The build takes about five minutes and is a one-off:

```
Engine generation completed in 302.465 seconds.
Created engine with size: 9.18205 MiB
Engine written: models/yolov8n_fp16.orin.engine
Build seconds:  307
507e44814947a01441c087fcc2451a52f399cfe9df0b4d344217dbeb70fe1408
```

`build_engine.sh` handles both ultralytics export flavours. A dynamic export
needs an explicit optimization profile; a static one rejects it outright with
`Static model does not take explicit shapes`. The script tries with a profile
and retries without on exactly that error, so the operator does not have to
know which export they were handed.

## Where the frame time actually goes

`--stages` over a 120 s run, 1187 frames (2026-08-18):

| stage | p50 | p95 | share of `pipeline_ms` |
|---|---:|---:|---:|
| decode (blocking appsink pull) | 191.0 ms | 196.4 ms | — |
| letterbox (OpenCV resize + pad) | 0.75 ms | 0.79 ms | 10.3 % |
| preprocess (HWC→NCHW, /255, into pinned) | 1.79 ms | 1.83 ms | 24.7 % |
| infer (TensorRT, host wall) | 4.63 ms | 4.66 ms | 63.7 % |
| postprocess (NumPy decode + NMS + tracker) | 0.43 ms | 0.45 ms | 5.9 % |
| publish (json.dumps + MQTT) | 0.25 ms | 0.30 ms | 3.5 % |
| **pipeline total** | **7.26 ms** | **7.34 ms** | 100 % |

`decode` is excluded from the pipeline clock and is not decoder work: it is the
blocking wait for the next frame to exist on a 5 fps source, which is why it
sits at ~191 ms against a 200 ms frame period. It is still reported, because a
decode figure that starts *exceeding* the frame period is the signal that NVDEC
has stopped keeping up.

`infer` (4.62 ms host wall) is larger than the published `inference_time_ms`
(4.16 ms) because the latter is a CUDA event pair around `execute_async_v3`
only, as the contract asks; the difference is the H2D/D2H copies and the host
sync.

### Does any of this need to leave Python?

No, and the breakdown is why.

Postprocess is **6.0 %** of the frame budget — below the 10 % that would make
it worth rewriting. Preprocess (24.6 %) and letterbox (11.7 %) are each above
it, and together they are 2.62 ms, 36 % of `pipeline_ms`. Read as a percentage
that looks like the thing to fuse into a CUDA kernel. Read in absolute terms it
is 2.62 ms out of a 200 ms frame period — a 1.3 % duty cycle — and the number
it feeds into is not latency anybody experiences: measured capture-to-alert is
**117 ms p50**, of which the detector contributes 7 ms and the hub's rule
evaluation and MQTT round trip the other 110.

Extrapolating to the case that would justify the work: eight 1080p streams at
15 fps is 120 frames/s of preprocessing. At 1080p the letterbox costs about
2.25× the 720p figure (pixel count), so ≈ 3.7 ms of CPU per frame, ≈ 0.49 of
one core out of eight. Still not the constraint. **The GPU ceiling of 236
inferences/s is reached before the CPU preprocessing is.**

So: no C or CUDA. If that ever changes, the segment to move is
letterbox+preprocess *as one fused kernel* — they are the same pass over the
same pixels, and splitting them would pay the traversal twice. Expected gain is
bounded by 2.6 ms/frame of CPU, which would raise the single-process ceiling
from the measured 167 inferences/s toward the 236 the GPU allows. That only
matters for a design where one process serves many streams, which is not the
design recommended below.

## Multi-stream: three options, and where each stops making sense

Measured inputs, all on this Orin NX 16 GB:

* **GPU ceiling ≈ 236 inferences/s.** Worker processes running flat out
  (preprocess + TensorRT + head decode, no RTSP, no MQTT): 1 → 166.7/s,
  2 → 231.2/s, 4 → 237.0/s, 8 → 235.8/s. Per-inference p50 rises 3.84 → 6.43 →
  14.55 → 31.60 ms as the queue deepens; aggregate throughput is flat from two
  workers on, so the GPU is the shared resource and it saturates at ~236.
* **208 MB of unified memory per additional process** holding its own engine,
  context, stream and Python runtime. System used RAM: 3977 MB baseline →
  4172 / 4394 / 4600 / 4822 MB at 1/2/3/4 processes.
* **No interference at realistic rates.** Four processes each inferring at 5 fps
  measured p50 4.02 / 4.07 / 4.04 / 4.02 ms — indistinguishable from one
  process alone.

### 1. Multi-process, one TRT context each

Run N copies of this detector with distinct `device_id` / `stream_id` /
`preview_port`. No new code at all.

* **Memory:** 208 MB × N. Four streams = 0.83 GB, eight = 1.7 GB, against
  15.6 GB total and ~4 GB already in use on this board. Fits with room to spare;
  memory is not the limit.
* **GPU:** eight 1080p streams at 15 fps is 120 inferences/s against the 236
  ceiling — 51 % utilization, so per-frame latency stays near the unloaded
  3.8 ms rather than climbing into the queueing regime.
* **CPU:** measured 5.5 % of one core per stream at 5 fps 720p (steady state,
  `/proc` deltas). Scaling linearly to 15 fps 1080p gives roughly 0.35–0.45 of a
  core per stream, ~3 of the 8 cores at eight streams. **Extrapolated, not
  measured — needs verifying before anyone sells an eight-stream box on it.**
  The earlier 8.5–12.5 % figure here was a `ps -o pcpu` lifetime average and
  carried engine warmup into the per-stream cost.
* **GIL:** sidestepped entirely, which is the whole point.
* **Not verified:** NVDEC's own capacity for 8×1080p15 concurrent sessions.
  Orin NX's decoder is specified well above that in aggregate pixel rate, but
  this was measured with one stream. **需核实.**

### 2. Single process, dynamic-batch engine

Build the engine with a dynamic batch profile, collect frames from N streams
into one `enqueueV3`.

* **Throughput:** trtexec reports 1.21 ms of the 3.81 ms per-inference GPU time
  as enqueue overhead, so batching should amortize a real fraction of it. A
  2× aggregate improvement is plausible. **Not measured — no dynamic-batch
  engine was built. 需核实.**
* **Latency cost, and this is the part that decides it:** a batch of N frames
  drawn one per stream cannot dispatch until the slowest stream delivers. At
  15 fps that is up to a 67 ms wait, ~33 ms on average, added to every alert.
  Measured capture-to-alert today is 117 ms p50, so batch=8 would push it toward
  150–185 ms. Inside the ±0.5 s assertion tolerance, but it spends a third of
  the latency budget to buy GPU headroom that option 1 says is not needed at
  eight streams.
* **Implementation:** a frame collector with a partial-batch timeout, per-slot
  letterbox transforms carried alongside the batch, batched head decode, and a
  new failure mode when one stream stalls and the others wait on it. Order of
  magnitude ~300–400 lines plus a rebuilt engine.

### 3. DeepStream (nvinfer + nvtracker, NVMM zero-copy)

* **Disk:** DeepStream 7.x for JetPack 6.1 is roughly 1.5–2 GB installed before
  build space. This device has 2.9 GB free, so it does not fit today without
  clearing something first.
* **Version coupling:** the DeepStream release is pinned to the JetPack
  release, and `pyds` is pinned to the DeepStream release. A JetPack upgrade
  forces a DeepStream upgrade and a `pyds` rebuild — the detector in this
  directory needs only a re-run of `build_engine.sh`.
* **What it genuinely adds:** the decoded frame never leaves NVMM, so the H2D
  copy, the letterbox and the normalization all disappear from the CPU;
  `nvstreammux` batches across streams without the hand-written collector of
  option 2; `nvtracker` moves tracking to the GPU; tiled composition comes free.
* **What that is worth here:** it buys back the 2.62 ms/frame of CPU
  preprocessing — ≈ 0.5 of a core at eight 1080p15 streams, ~6 % of this SoC.
  In exchange the entire data plane becomes a GStreamer pipeline with probe
  callbacks, and the tracker, MQTT publisher, preview server and snapshot
  handling all have to be re-hosted inside it.

### Recommendation

| streams (1080p @ 15 fps) | option |
|---|---|
| 1 – 8 | **multi-process** (option 1). No new code; 208 MB and ~0.5 core each; GPU at ~51 %. |
| 8 – 16 | **dynamic batch** (option 2), once someone has measured the batched throughput and confirmed the extra 30–70 ms of alert latency is acceptable. |
| 16+, or 4K, or per-stream tiled display | **DeepStream** (option 3). Past 16 streams the CPU preprocessing does become the constraint and NVMM zero-copy is the only way around it. |

For the eight-stream case the user asked about, option 1 is the answer and it
requires writing nothing. The costs that decide it — 208 MB and roughly half a
core per stream, against a 236 inference/s GPU ceiling — are measured on this
board, except the per-stream CPU at 15 fps 1080p, which is extrapolated from
5 fps 720p and should be confirmed on the target resolution.

## Coordinates

`coordinate_space` is always `frame_norm`: normalized against the **original**
frame, never the padded 640×640 canvas. `esk_jetson/letterbox.py` is the same
module `platforms/rknn` uses, and the forward and inverse geometry both come
out of `fit_geometry()` so they cannot drift.

This is also why the decoder is *not* asked to scale. `nvvidconv` could
letterbox on the VIC, and it would be free, but then the forward transform
would live in GStreamer caps while the inverse lived in Python. The frame is
converted to BGRx at source resolution and the letterbox is done in OpenCV,
which costs 0.84 ms and keeps one source of truth.

The acceptance fixture is 1280×720 on purpose: a square source passes even when
the inverse is missing entirely.

`evidence/annotated-*.jpg` (written by `--annotate`) draws boxes reconstructed
from the **published** `frame_norm` values rather than from the internal pixel
boxes, so a broken inverse shows up as an offset or squashed rectangle instead
of a plausible number.

## Hardware decode is verified, not assumed

`health.decode` exists because every platform has a silent-CPU-fallback failure
mode. On Jetson it is `nvv4l2decoder`: it ships in the L4T multimedia plugin
set, not in stock GStreamer, so a container from a plain base image or a stale
registry in `~/.cache/gstreamer-1.0` sends the obvious implementation to ffmpeg
on the CPU, on a box bought for its GPU.

`require_hw_decode: true` (the default) refuses to start rather than degrade.
The evidence that it worked is not the self-reported field:

* **`/dev/v4l2-nvdec` is open in the detector process.** `ls -l /proc/<pid>/fd`
  shows it alongside `/dev/nvmap`. This is the NVDEC hardware device node; an
  ffmpeg decoder never opens it.
* **The negotiated appsink caps carry NVMM provenance:**
  `nvbuf-memory-type=(string)nvbuf-mem-surface-array, gpu-id=(int)0`. Only the
  NVIDIA converter fed by NVIDIA decoder surfaces produces those fields.
* **`NvMMLiteOpen : Block : BlockType = 261`** on stdout at startup — the L4T
  multimedia API opening the decoder block.
* `GET /debug/decode` on the preview port reports the GStreamer factory that
  was actually instantiated, read off the running pipeline on each request.

CPU cost is *not* good evidence at this frame rate, and the measurement is here
so nobody reaches for it: decoding this 5 fps 720p clip costs 4.2 % of one core
on the CPU path against 3.0 % on NVDEC. The gap is real but small, because a
5 fps clip is a light decode. NVDEC's value shows up at higher rates and stream
counts, not on this fixture.

## Preview endpoint

`GET /preview/<stream_id>.jpg` (also `/preview.jpg`) serves one current frame;
`GET /live/<stream_id>` is a page that reloads it on a timer. Both are
advertised in `status.streams[]` only when `preview_advertise_host` is set —
the device cannot guess which address a browser can reach, and a wrong URL is
worse than an absent one. The hub proxies the still at
`/api/devices/<id>/streams/<sid>/preview.jpg`.

The frame served is the *active region* of the letterbox canvas: real decoded
pixels at the source aspect ratio, 640×360 for a 1280×720 source. NVDEC does
hand back full resolution, unlike the Rockchip path, but copying 1280×720×4
into the frame store on every frame to serve an endpoint polled once a second
is not worth it, and matching `platforms/rknn`'s behaviour keeps the hub's rule
canvas identical across platforms.

## Measured against the other platforms

Same truth clip, same assertions, same hub.

The Orin NX and RK3588 int8 columns are the controlled 2026-08-18 re-measurement
described at the top of this file — same clip, same thresholds, same 30 s
warmup and 120 s sample, payload fields on both sides. The RK3588 fp16 and
workstation columns are earlier runs kept for shape, not part of that pairing.

| | **Orin NX (this)** | RK3588 int8 | RK3588 fp16 | workstation CPU (generic) |
|---|---:|---:|---:|---:|
| `inference_time_ms` p50 | **4.16 ms** | 36.2–44.5 ms | 72.3 ms | 30–37 ms |
| `inference_time_ms` p95 | **4.19 ms** | 55.8–56.2 ms | 125.5 ms | — |
| `pipeline_ms` p50 | **6.99 ms** | 38.1–46.3 ms | 73.1 ms | 58–72 ms |
| single-stream ceiling (full loop) | **167 inf/s** | 31.4 inf/s | — | — |
| detector CPU | 5.5 % of one core | 16.4–17.9 % | ~14 % | ~250 % |
| RSS | 310 MB | 214 MB at start | 292 MB | — |
| decode | `hw` (NVDEC) | `hw` (MPP) | `hw` (MPP) | `sw` (ffmpeg) |
| accelerator headroom | GR3D 0 % in 215/239 samples, 236 inf/s ceiling | NPU Core0 8–11 % busy | NPU 26 % busy | — |

Orin NX runs inference **9–11× faster than RK3588 int8** and **7–9× faster than
a 20-core x86 CPU**, at a fraction of the CPU cost of the latter. The range on
the RK3588 side is that board's window-to-window spread across four 120 s
samples, not measurement slop in the comparison: its p95 held to ±0.2 ms while
its p50 moved 8 ms.

### The benchmark number and the real number

Unlike RK3588 — where the same loop overstates per-frame speed by 35–66 % once
the head decode is left out of it — the two agree closely here:

| | value |
|---|---:|
| `trtexec` loop, 300 iterations, no data transfers | 3.81 ms p50, 262 qps |
| flat-out loop, preprocess + infer + decode, 60 s | 3.82 ms p50, 167 inf/s |
| in-pipeline `inference_time_ms` (CUDA events) | 4.16 ms p50 |
| in-pipeline `infer` stage (host wall, incl. copies) | 4.62 ms p50 |

The benchmark understates the real per-frame call by 8 %, and the full host
stage by 21 %. Quote 4.2 ms, not 3.8, and never quote 262 qps as a stream
capacity — the measured multi-context ceiling is **236 inferences/s**, and at
that load per-inference latency is 31.6 ms, not 3.8. That the trtexec loop and
the full flat-out loop land within 0.01 ms of each other is the thing worth
carrying to another platform: on this board the head decode is cheap enough to
hide behind the GPU, which is exactly what does not hold on RK3588.

### Against the `industrial_security_jetson` claim

That solution's headline is "30+ FPS on Orin NX" with TensorRT GPU inference.
This platform's measured numbers on the same class of hardware:

* single stream, no source limit: **167 inferences/s** measured flat out
  through preprocess + TensorRT + head decode, against a multi-context GPU
  ceiling of **236 inferences/s**;
* per-frame `pipeline_ms` **7.24 ms**, i.e. a 138 fps single-stream budget;
* on the 5 fps fixture the detector idles at a 3.6 % duty cycle.

So the 30+ FPS claim is met with a **4–5× margin** on a single stream, and
exceeded on aggregate throughput while serving four concurrent streams
(237 inferences/s = 59 fps each). The CPU/ONNX path this repository shipped
before was 30–37 ms per inference and ~2.5 cores; that was the regression
against `industrial_security_jetson`, and it is closed.

## End to end, against the truth video

`tools/rtsp-fixture/verify_e2e.py`, 130 s, RTSP served on the board itself
(`tools/rtsp-fixture/start_jetson_fixture.sh`) so network jitter cannot be
mistaken for detector latency. Hub and broker on a separate host.

```
== FAILURES
  none
```

| assertion | tolerance | measured |
|---|---|---|
| `line_cross` forward instants (×4) | ±0.5 s | 0.267 – 0.268 s |
| `line_cross` backward instants (×4) | ±0.5 s | 0.109 – 0.110 s |
| direction correctness | 8/8 | 8/8 |
| forward-only line fired on a backward crossing | never | **NONE** |
| backward-only line fired on a forward crossing | never | **NONE** |
| `zone_enter` instants (×4) | ±0.5 s | 0.090 s each |
| `loitering` dwell error | ≤1 s | 0.090 – 0.290 s |
| snapshots | real JPEG | 8/8, `image/jpeg`, 17.3–17.6 KB |
| trajectory alignment vs `truth.json` | — | mean \|cx\| err 0.00115, max 0.00298 |
| stream continuity | — | 650 msgs / 129.8 s = 5.01/s, gap p50 200 ms max 208 ms, one track id |

Capture-to-alert latency p50 **117.4 ms**, p95 **297.0 ms**, of which the
detector is 7 ms.

The truth line is nudged to x=0.503 (`ESK_LINE_X`), the same offset the RK
acceptance run used, so the two results compare directly. Jetson coordinates
are not quantized to the decoder's output grid the way RK3588's int8 path is,
so this platform probably does not need the nudge — untested, and left matching
RK for comparability.

An earlier run of the same assertions reported five `zone_enter` failures. They
were one artefact, not five: the run began with the subject already inside the
zone, so the hub emitted an extra leading `zone_enter` the moment its rules were
installed, and the harness's in-order matcher then paired every later alert with
the previous loop's truth instant. The same artefact appears in the RK
acceptance log (`rk-A`) and clears once the run starts on a clean phase.

## Tests

```bash
uv run pytest     # letterbox/geometry, tracker IDs, head decode, fixture conformance
```

All host-only: no GPU, no Jetson, no network — 33 tests. `fixtures/*.json` were
captured verbatim off the broker during a real Orin NX run and are validated
against `contracts/mqtt-detection.schema.json` as the contract's Conformance
section requires. The capture deliberately skips the first frames of a session:
the first TensorRT call carries lazy CUDA module load (38.6 ms against a warm
4.1 ms) and the first status is published while the stream is still
`reconnecting`, and shipping either as the platform's contract fixture would
document a number the platform does not run at.

`tests/test_fixtures_schema.py` additionally asserts the captured fixtures
report `decode: hw` — a fixture that says `sw` means the run being certified was
not the run this platform exists to deliver.
