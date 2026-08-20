# Model weights

Neither the ONNX input nor the compiled HEF is tracked in git.

## Source ONNX

The same artifact every other platform in this repo runs, so the platforms stay
comparable:

```bash
curl -L -o yolov8n.onnx \
  https://raw.githubusercontent.com/Zhang-zu-hao/Industrial-security-demo/master/models/yolov8n.onnx
```

`yolov8n.onnx`, 12851087 bytes, md5 `eda19d19a8aa73d54411a6fc1d091c7f`.

## Compiling the HEF

`build_hef.py` is the compile, run inside the Hailo AI Software Suite container
on an **x86_64** host — the Dataflow Compiler has no ARM build, so this step
cannot happen on the Pi.

```bash
docker run --rm -v "$PWD":/work hailo_ai_sw_suite_2025-04:1 python /work/build_hef.py
```

It expects `/work/yolov8n.onnx` and `/work/calib-frames.tgz` (paths overridable
via `ESK_HAILO_ONNX`, `ESK_HAILO_CALIB`, `ESK_HAILO_WORK`) and writes
`yolov8n.hef` plus the quantized HAR next to them.

What it does, and why:

| Step | Choice |
|---|---|
| Graph cut | End nodes are the six detection-head convolutions (`/model.22/cv2.N/cv2.N.2/Conv`, `/model.22/cv3.N/cv3.N.2/Conv`). DFL, the class sigmoid concat and the final box assembly stay off-chip and are decoded in NumPy by `esk_hailo.hailo_yolo` — the same arithmetic `platforms/rknn` runs on its model-zoo export. |
| Input normalization | `normalization([0,0,0],[255,255,255])` folded into the input layer, so the HEF takes uint8 NHWC RGB and the `/255` never costs a CPU pass on the Pi. |
| Class activation | `change_output_activation(convNN, sigmoid)` on the three class branches. Two reasons: the published `score` is then a probability without a host-side sigmoid, and the optimizer can pin the output range to exactly `[0,1]` instead of spending the int8 range on logits nobody reads. `HailoPersonDetector` asserts that range on its first inference, so a HEF built without these lines fails loudly rather than publishing logits as scores. |
| Calibration set | The 81 frames of `calib-frames.tgz` — the same ADL clip frames the RKNN int8 build was calibrated on, letterboxed to 640x640 exactly as the runtime does. |

The model script this produces is committed as `yolov8n.alls` for reference.

## The artifact this platform was built against

| Field | Value |
|---|---|
| File | `yolov8n.hef`, 4430780 bytes |
| sha256 | `dbabd55f42c9309358398985e055a5c0b45ddbb7a1baae5ccc03e0496e501475` |
| md5 | `1d9aa53d55f27c8e1fabc0e29281f3ce` |
| Dataflow Compiler | 3.31.0 (`hailo_ai_sw_suite_2025-04:1`) |
| HailoRT the suite pairs with | 4.21.0 |
| Target | HAILO8, single context |
| Utilization | control 75%, compute 44.1%, memory 27.7% |

The HailoRT version is not incidental. HailoRT's 4.x ioctl protocol is not
forward compatible, so the userspace library, the Python bindings and the
`hailo_pci` kernel driver on the board must all be the same major.minor. The
board this was built for reports `hailort 4.21.0` and
`hailort-pcie-driver 4.21.0`, both held (`apt-mark showhold`), which is why
3.31.0 is the compiler version rather than the newest available.

`status.versions.model` reports `yolov8n@dbabd55f42c9`.

## Calibration ran at optimization level 0

The compile log says so explicitly:

```
[warning] Reducing optimization level to 0 (the accuracy won't be optimized and
compression won't be used) because there's less data than the recommended
amount (1024), and there's no available GPU
[info] Using dataset with 64 entries for calibration
[info] Finetune encoding skipped
[info] Bias Correction skipped
[info] Adaround skipped
```

That is plain post-training quantization with no bias correction — the same
class of quantization the RKNN int8 build used, which is why the two are worth
comparing at all. Raising it means both a calibration set past 1024 frames and
a CUDA GPU visible to the suite container; neither was available on the build
host. Anyone re-running this with a GPU should expect the accuracy to move and
should re-measure rather than inherit these numbers.
