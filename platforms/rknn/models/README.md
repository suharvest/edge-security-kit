# Models

`yolov8n_fp16.rk3588.rknn` is built by `../tools/prepare_model.sh` from the same
`yolov8n.onnx` the generic platform uses (md5
`eda19d19a8aa73d54411a6fc1d091c7f`), so both platforms detect with identical
weights and a coordinate difference cannot be blamed on the model.

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
