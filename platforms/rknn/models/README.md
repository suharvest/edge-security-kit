# Models

Three models live here, built by `../tools/prepare_model.sh`. See
`../README.md` for the accuracy and latency comparison between them.

| file | ONNX | md5 of the ONNX | precision |
|---|---|---|---|
| `yolov8n_fp16.rk3588.rknn` | stock ultralytics, same file the generic platform uses | `eda19d19a8aa73d54411a6fc1d091c7f` | fp16 |
| `yolov8n_zoo_fp16.rk3588.rknn` | `airockchip/rknn_model_zoo` | `2a48b1cef26b6722547807a883079f51` | fp16 |
| `yolov8n_zoo_int8.rk3588.rknn` | `airockchip/rknn_model_zoo` | `2a48b1cef26b6722547807a883079f51` | int8, 400-image PTQ |

The stock export is kept because it is byte-identical to the generic platform's
weights: a coordinate difference between the two platforms can then never be
blamed on the model. The zoo export is a different graph (DFL and the final
concat removed) and is the one that quantizes — every op left in it has an
RKNPU2 int8 kernel.

The model-zoo ONNX comes from `examples/yolov8/model/download_model.sh`:

```bash
curl -sSL -o yolov8n_zoo.onnx \
  https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8n.onnx
# sha256 0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5
```

int8 additionally needs `--calib-dir`; the set and the reasoning are in
`../calibration/`.

The `.rknn` is not committed. `SHA256SUMS` is, so a rebuild is verifiable:

```bash
../tools/prepare_model.sh --onnx /path/to/yolov8n.onnx --builder-host wsl2-local
sha256sum -c SHA256SUMS
```

## Toolkit version must match the runtime, and the filename lies

radxa's `/usr/lib/librknnrt.so` is a symlink to `librknnrt.so.2.3.0`, but the
version string compiled into that library is **2.3.2**:

```
$ strings /usr/lib/librknnrt.so | grep 'librknnrt version'
librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
```

Convert with `rknn-toolkit2==2.3.2`. Trusting the SONAME and converting with
2.3.0 produces a model the runtime may still load, since `load_rknn` is
tolerant; the failure appears later at `init_runtime` or as wrong output.

Conversion is x86_64-only — there is no aarch64 RKNN Toolkit 2 wheel — so
`prepare_model.sh` delegates to a Fleet x86_64 host. The board only ever needs
`rknn-toolkit-lite2`, which it already has from the OS image.
