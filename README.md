# edge-security-kit

Intrusion, line-crossing and loitering detection for edge devices, with the
detector and the rule engine split across a published MQTT contract.

Detectors run on the hardware that suits the site — a Jetson Orin, an RK3588
board, a plain CPU — and report over one payload schema. The hub judges the
rules, keeps the alert history and serves the workbench. One hub can carry
detectors of different platforms at once; a single board can also carry both.

## What it detects

Person detection (COCO class 0) plus three rules:

| Rule | Fires when |
|---|---|
| `zone_enter` | a tracked person's centroid enters a drawn polygon |
| `loitering` | that person stays inside it past `dwell_seconds` |
| `line_cross` | a track crosses a drawn line, optionally only in one direction |

Rules are drawn in the browser on a live frame and stored in normalized
coordinates, so they survive a change of resolution.

## Measured

Same 1280×720 H.264 source, same truth video, same assertions.

| | Jetson Orin NX 16GB | Radxa Rock 5T (RK3588) | LubanCat-3 (RK3576) | reCamera Pro (RV1126B) |
|---|---|---|---|---|
| Accelerator | TensorRT 10.3, FP16 | RKNN 2.3.2, int8 | RKNN 2.3.2, int8 | RKNN 2.3.2, int8 |
| Inference p50, in pipeline | 4.16 ms | 36.2–44.5 ms | 23.9 ms | 29.9 ms |
| Inference p95, in pipeline | 4.19 ms | 55.8–56.2 ms | 27.9 ms | — |
| Full pipeline p50 | 6.99 ms | 38.1–46.3 ms | 26.2 ms | 74.7 ms |
| Detector CPU | 5.5% of one core | 16.4–17.9% of one core | 8.6–14.0% of one core | 38–40% of one core |
| Detector RSS | 310 MB | 214 MB at start | 200 MB | — |
| Accelerator busy | GR3D 0% in 215 of 239 1 s samples | NPU Core0 8–11%, cores 1–2 idle | NPU Core0 5–6%, Core1 idle | — |
| Board state while measured | idle, load avg 0.00 | 8 containers, 2 above 1% CPU | idle, no containers | — |
| Decode | NVDEC, confirmed in-kernel | Rockchip MPP, confirmed in-kernel | Rockchip MPP, confirmed in-kernel | ffmpeg, software (no userspace MPP) |
| Single-stream ceiling | 167 inferences/s | 31.4 inferences/s | not measured | not measured |
| Sustained fps | 5.0 (source-limited) | 5.0 (source-limited) | 5.0 (source-limited) | 5.0 (source-limited) |

The Orin NX and RK3588 columns were re-measured on 2026-08-18 under the method
below. The RK3576 and reCamera Pro columns are earlier runs on their own boards
and have not been repeated, so read them as indicative rather than as part of
the controlled comparison.

### How these were measured

Both re-measured boards ran the same 1280×720 H.264 truth clip at 5 fps,
republished on the board itself with `-c copy` so no re-encode sits between the
fixture and the decoder, at `conf 0.35` / `iou 0.45` and a 640 input. Each run
is a 30 s warmup followed by a 120 s sample of the published
`sensecraft.detection/1` payloads — `inference_time_ms` and `pipeline_ms` are
read off the contract fields, so the two platforms are compared on identical
definitions. Detector CPU is `utime+stime` deltas from `/proc/<pid>/stat` over
5 s intervals, as a percentage of one core; RSS is `/proc/<pid>/statm`.

**Two numbers per board, not one.** Every board was measured twice: `as-found`,
with whatever else the device was running left alone, and `quiesced`, after
stopping the co-tenants that measurably used CPU. Deployments rarely get a
board to themselves, so the as-found figure is the one a site will see and the
quiesced figure is the ceiling. On these two boards the gap turned out to be
nil, and the per-board READMEs give the pairs:

* **Orin NX** had no co-tenant to stop. `docker stats` showed only this
  project's own RTSP fixture, at 0.16 % of one core; load average was 0.00.
  as-found and quiesced are the same state.
* **RK3588** carried eight containers, of which six measured at or under 1 % of
  one core (`openvoicestream` 1.00 %, `ovs-llm` 0.40 %, `ovs-agent` 0.07 %,
  `wyoming-slv` 0.00 %) and were left running. Stopping the two that did use
  CPU — an unrelated RTSP fixture pair at 3.18 % and 5.36 % — moved inference
  p50 the wrong way, from 36.2 ms to 39.6 and 44.5 ms across the next two
  windows, while p95 stayed inside 55.8–56.2 ms. That 8 ms spread is this
  board's window-to-window variation, not contention: the NPU held 1.0 GHz, the
  DDR controller held 534 MHz and `npu-thermal` sat at 62 °C throughout. **Quote
  RK3588 p50 as a range across runs; its p95 is the reproducible number.**

**A benchmark loop is not a latency figure.** Running the model flat out
amortizes weight traffic that a 5 fps stream re-pays on every frame, so a loop
always reads faster than the pipeline it is supposed to describe — but by how
much is platform-specific, and the gap is not a constant you can correct for:

| | flat-out loop p50 | in-pipeline p50 | loop is optimistic by |
|---|---:|---:|---|
| Orin NX, fp16 | 3.82 ms | 4.16 ms | 8 % |
| RK3588, int8, full loop (preprocess + infer + decode) | 34.24 ms | 36.2–44.5 ms | 6–30 % |
| RK3588, int8, pure `rknn.inference()` loop | 26.84 ms | 36.2–44.5 ms | 35–66 % |

The narrower the loop, the more it flatters the board: on RK3588, dropping the
head decode out of the loop takes 7 ms off the apparent per-inference cost that
the real pipeline still pays. The single-stream ceilings in the table above are
the *full* loop on both boards — one process, flat out, preprocess + infer +
head decode — so 167 and 31.4 inferences/s are measured by the same recipe.

RK3576 and RK3588 run one implementation — `platforms/rknn/` has no chip branch,
only a different `.rknn` — and RK3576 turns in roughly half to two-thirds of
RK3588's per-frame latency on the same model built from the same ONNX and
calibration set.

The load explanation for that gap was tested and does not hold. RK3588 was
measured with eight containers on the board and RK3576 with none, so the
suspicion was contention; stopping the RK3588 co-tenants that measurably used
CPU moved its inference p50 the wrong way, from 36.2 ms to 39.6 and 44.5 ms,
and never toward RK3576's 23.9 ms. NPU occupancy says the same thing from the
other side: Core0 reads 8–11 % on RK3588 against 5–6 % on RK3576, and a
container competing for CPU cannot inflate NPU busy time.

What remains uncontrolled is the boards themselves: different kernels,
different days, no attempt to hold thermals or DVFS equal, and RK3576's figure
is one 70 s window with no repeat while RK3588's p50 is known to move 8 ms
between windows. So it is a result about these two systems, not a per-TOPS
ranking of the two NPUs. Board detail:
`lubancat-rk3576-debian12-gnome-20250721`, kernel `6.1.99-rk3576`, RKNPU driver
0.9.8.

The hub costs 3.7% of one core and 52.8 MB RSS on the RK3588 board while that
board also runs a detector, so the aggregation host does not need to be a
server.

Multi-stream on Orin NX, measured: the GPU saturates at 236 inferences/s and
each extra detector process costs 208 MB. One process per stream needs no new
code up to 8 streams. Beyond that, dynamic batching, then DeepStream. The CPU
cost at 8×1080p@15fps and NVDEC's session ceiling are extrapolated — **needs
verifying**.

int8 on RK3588 costs 0.52 AP@.5:.95 against fp16 on 500 COCO person images
(0.61 on small targets) and returns 37% of the inference latency. Its failure
mode is extra low-confidence boxes rather than missed people, so raise the
threshold before reaching for fp16.

## Layout

```
contracts/     MQTT payload schema, semantics, conformance fixtures + gate
hub/           rule engine, alert lifecycle, SQLite, REST/WS API, cookie sessions
web/           workbench, rule editor, device page (Preact + htm, no build step)
platforms/
  jetson/      NVDEC + TensorRT, all Python
  rknn/        MPP + RGA + RKNN, int8 and fp16, RK3588 and RK3576
  generic/     CPU / ONNX Runtime, the reference implementation and test rig
  recamera-pro/ RV1126B, RKNN int8 on the camera itself, broker and hub included
tools/
  rtsp-fixture/ mediamtx, the truth-video generator and the acceptance harness
```

## The contract

Every detector publishes `sensecraft.detection/1`. Three requirements are worth
knowing before writing a new one:

- Coordinates are `frame_norm` — the letterbox pad must be reversed before
  normalizing. A square test source will pass a broken inverse; on a 720p frame
  an unreversed box is 261 px short while its x coordinates look correct.
- `track_id` starts at 1. A detector without tracking cannot feed hub-side
  rules, because dwell and crossing are per-track state.
- `session_id` changes on every process start, so the hub can tell a restarted
  device from a continuing one and reset its rule state.

`contracts/check_fixtures.sh` validates captured payloads from two independent
platforms against the schema. Adding a platform means adding its fixture.

## Verification

`tools/rtsp-fixture/` builds a video whose trajectory is known per frame and
asserts against it: crossing instants within 0.5 s and in the right direction,
a forward-only line silent on a backward crossing, dwell within 1 s, snapshots
that are real frames. Both shipped detectors pass it on their own hardware.

This is how the rule engine is checked rather than watched. It is also what
caught a line drawn down the middle of the frame never firing: the RK
preprocessor quantizes cx to a 1/640 grid and 0.5 lands exactly on it, so the
centroid sat on the line and no sign flip occurred.

## Status

Jetson, RK3588, RK3576 and the hub are verified on hardware; Jetson and RK3588
including containerised deployment. RK3576 is verified as of 2026-08-19 on an
EmbedFire LubanCat-3: NPU inference at 23.9 ms p50, MPP decode confirmed from
the kernel side, the truth-video assertions passing with no failures, and
broker, hub and detector all on the board. Its caveat is that no accuracy sweep
was run there — the COCO numbers are RK3588's, and the graph and calibration set
are identical, but that is an argument rather than a measurement. The RKNPU2
container image now carries both SoCs' models; it has not been rebuilt since.
reCamera Pro is verified too, as of 2026-08-18: RV1126B NPU
inference at 29.9 ms p50, the same truth-video assertions passing with no
failures, and broker, hub and detector all running on the camera. Two caveats
stay attached — its RSS grows ~13 MB/min, which is unexplained, and it carries
no accuracy sweep.

Not built: Hailo. Not measured: anything above two concurrent streams.
