# Hailo-8 / 8L detector (Raspberry Pi 5)

RTSP → FFmpeg decode → YOLOv8n on the Hailo NPU → `sensecraft.detection/1` on
MQTT. Same contract, same tracker, same `frame_norm` publishing path as
`platforms/generic` and `platforms/rknn`; the platform-specific parts are the
HEF build and the head decode.

**Status: compiled and unit-tested, not yet run on the NPU.** Read
"Not yet verified on hardware" before quoting anything from here as a result.

## What is platform-specific

| Concern | This platform |
|---|---|
| Accelerator | Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS) over PCIe, `/dev/hailo0` |
| Model | `models/yolov8n.hef`, built from the repo's shared `yolov8n.onnx` — see `models/README.md` |
| Preprocess | letterbox to 640×640, uint8 NHWC RGB; the `/255` is folded into the HEF input layer |
| Head decode | six-output split head decoded in NumPy (`esk_hailo/hailo_yolo.py`) |
| `health.decode` | `sw`, and `fallback_active: false` — see below |
| `health.backend` | `hailort-<version>-<arch>`, e.g. `hailort-4.21.0-hailo8` |

### Decode is `sw`, and that is the honest answer

The Raspberry Pi 5 removed the H.264 hardware decoder the Pi 4 had; VideoCore
VII decodes HEVC only. An H.264 RTSP stream is therefore decoded by FFmpeg on
the Cortex-A76 cores, and there is no hardware path being silently missed.

So this platform reports `decode: sw` with `fallback_active: false`, and
`config.example.yaml` has no `require_hw_decode` switch — unlike the Rockchip
platform, where a silent drop from MPP to CPU is the characteristic failure and
the detector refuses to start on the fallback. Reporting `hw` here to make the
fleet page look uniform would be a lie about where the cycles go, and the field
exists precisely so an operator can see that.

### The head is cut at the six convolutions

The HEF ends at the detection head's convolutions; DFL, the box assembly and
NMS run in NumPy on the host. That is the same split `platforms/rknn` uses for
the model-zoo export, and `esk_hailo/hailo_yolo.py` is a port of that decoder —
same DFL mean, same anchor centres, same greedy NMS — so a coordinate bug
cannot hide on one platform while the other looks fine.

Two Hailo-specific details:

* The class branch carries an on-chip sigmoid (`change_output_activation`), so
  the values thresholded are probabilities. `HailoPersonDetector` checks that
  range on its first inference and raises if the HEF was built without it,
  because logits published as `score` would still look plausible and would put
  every threshold in the stack quietly off.
* Branches are paired **by tensor shape**, never by output name or order.
  HailoRT keys outputs by graph layer name (`yolov8n/conv41` …) and those names
  change on every re-translation, so a name-dependent decoder would break
  silently at the next recompile.

## Install on the board

HailoRT's 4.x ioctl protocol is not forward compatible: the kernel driver, the
userspace library and the Python bindings must be the same major.minor. The
reference board runs `hailort 4.21.0` and `hailort-pcie-driver 4.21.0`, both
`apt-mark hold`ed, so the bindings are installed from the matching 4.21.0
wheel rather than resolved from an index. (The PyPI project named `hailort` is
an unrelated stub.)

Raspberry Pi OS ships Python 3.13 and Hailo publishes no cp313 wheel for
4.21.0, so the venv is pinned to 3.11 — `uv` fetches the interpreter, which is
lighter than the Docker route and matters on a board with 2.6 GB free:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  numpy 'opencv-python-headless>=4.9' 'paho-mqtt>=1.6,<2.0' 'pyyaml>=6.0' \
  ./hailort-4.21.0-cp311-cp311-linux_aarch64.whl
```

The Pi 5 also needs `force_desc_page_size=4096` on the `hailo_pci` module —
the kernel's 16 KB pages exceed the Hailo-8 PCIe descriptor limit and HEF
configuration fails while `fw-control identify` still succeeds:

```bash
echo "options hailo_pci force_desc_page_size=4096" | sudo tee /etc/modprobe.d/hailo.conf
```

Check the HEF loads and its shapes are what the decoder expects, without
starting the pipeline:

```bash
.venv/bin/python -m esk_hailo --config config.yaml --validate
```

## Run

```bash
.venv/bin/python -m esk_hailo --config config.yaml
.venv/bin/python -m esk_hailo --config config.yaml --seconds 60 --annotate evidence.jpg
```

`--annotate` draws boxes reconstructed from the **published** `frame_norm`
values, not from internal pixel boxes, so a wrong letterbox inverse shows up as
squashed or offset rectangles instead of as a plausible number.

## Tests

Host-only, no NPU and no HEF needed:

```bash
uv run pytest        # 25 passed, 4 skipped
```

`test_decode_head.py` plants one-hot DFL distributions into a synthetic head,
so the expected box is analytic and the test does not re-implement the softmax
it is checking. It pins branch pairing, stride derivation, the DFL sign
convention, output-order independence and the letterbox inverse on a 16:9
source. `test_fixtures_schema.py` skips while `fixtures/` is empty — see below.

## Correctness evidence so far

The quantized graph was run in the Dataflow Compiler's emulator against real
1280×720 frames and decoded by the committed `esk_hailo` code. This exercises
the head decode, the letterbox inverse and the `frame_norm` conversion with the
exact weights baked into the HEF. **It is not a hardware run and says nothing
about latency.**

![emulated detection on the truth clip](evidence-emulated-truth-720p.jpg)

| Frame | Truth `cx` | Published `cx` | Published `w × h` |
|---|---|---|---|
| `truth.mp4` t = 3.0 s (frame 15) | 0.50039 | 0.500933 | 0.2151 × 0.9212 |
| `truth.mp4` t = 16.0 s (frame 80) | 0.20039 | 0.201799 | 0.2168 × 0.9214 |
| ADL clip t = 12.0 s | — | 0.35053 | 0.1774 × 0.6425 |

The truth clip translates a 297 × 662 px person patch along a schedule defined
in seconds, so the expected centroid is known before anything runs. The
published centres land 0.7 px and 1.8 px from it, and the box height (0.921 of
720 px = 663 px) matches the patch to a pixel. On a 1280×720 source the
letterbox pads 140 px top and bottom; an inverse that ignored the pad or
divided x and y by different factors could not produce those numbers.

## Not yet verified on hardware

Every claim above about latency, throughput, CPU, NPU occupancy and end-to-end
rule behaviour is **absent, not pending publication**. No inference has run on
the NPU.

The reason is not a code or toolchain problem. On the reference board the
Hailo-8 is held exclusively by an unrelated long-running service; HailoRT hands
a `VDevice` to one process at a time unless every participant goes through the
multi-process service, which the incumbent does not:

```
$ hailortcli run models/yolov8n.hef -t 3
[HailoRT] [error] CHECK failed - Failed to create vdevice. there are not
enough free devices. requested: 1, found: 0
[HailoRT CLI] [error] CHECK_SUCCESS failed with status=
HAILO_OUT_OF_PHYSICAL_DEVICES(74) - Failed creating vdevice
```

`hailortcli fw-control identify` still succeeds against an occupied device — it
opens a control handle, not a `VDevice` — so it is not a usable check for
"the NPU is free". The device-level check is the `run` above, or any
`VDevice()` construction.

What is still owed before this platform belongs in the top-level `Measured`
table:

- captured MQTT fixtures in `fixtures/` (detection, status, status-LWT) and the
  promotion into `contracts/fixtures/`;
- the ground-truth harness pass (`tools/rtsp-fixture/verify_e2e.py`) against
  `truth.json`: line-cross instants and direction, the forward-only line not
  firing on the backward traverse, zone entry, loitering dwell, real snapshots;
- in-pipeline inference p50/p95, pipeline p50, sustained fps, detector CPU and
  RSS, and NPU occupancy from `hailortcli monitor`;
- an annotated evidence frame produced by `--annotate` on the board rather than
  by the emulator.

Freeing the device means stopping the process that holds it, which was out of
scope for the run that produced this platform.
