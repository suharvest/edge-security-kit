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

| | Jetson Orin NX 16GB | Radxa Rock 5T (RK3588) | reCamera Pro (RV1126B) |
|---|---|---|---|
| Accelerator | TensorRT 10.3, FP16 | RKNN 2.3.2, int8 | RKNN 2.3.2, int8 |
| Inference p50, in pipeline | 4.13 ms | 41.9 ms | 29.9 ms |
| Full pipeline p50 | 7.24 ms | 44.3 ms | 74.7 ms |
| Detector CPU | 8.5–12.5% of one core | 21% of one core | 38–40% of one core |
| Decode | NVDEC, confirmed in-kernel | Rockchip MPP, confirmed in-kernel | ffmpeg, software (no userspace MPP) |
| Single-stream ceiling | 167 inferences/s | not measured | not measured |

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
  rknn/        MPP + RGA + RKNN, int8 and fp16
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

Jetson, RK3588 and the hub are verified on hardware, including containerised
deployment. reCamera Pro is verified too, as of 2026-08-18: RV1126B NPU
inference at 29.9 ms p50, the same truth-video assertions passing with no
failures, and broker, hub and detector all running on the camera. Two caveats
stay attached — its RSS grows ~13 MB/min, which is unexplained, and it carries
no accuracy sweep.

Not built: Hailo. Not measured: anything above two concurrent streams.
