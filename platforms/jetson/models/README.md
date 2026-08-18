# Model weights and engines

Nothing here is tracked in git.

**ONNX** (~12 MB), fetch before the first build:

```bash
curl -L -o yolov8n.onnx \
  https://raw.githubusercontent.com/Zhang-zu-hao/Industrial-security-demo/master/models/yolov8n.onnx
```

Verified artifact: `yolov8n.onnx`, 12851087 bytes, md5
`eda19d19a8aa73d54411a6fc1d091c7f`, sha256 prefix `fb38b3933007`. The same file
`platforms/generic` uses, so `status.versions.model` stays comparable.

Do not substitute the `airockchip/rknn_model_zoo` export: it has DFL cut out of
the graph for RKNPU2's benefit and only costs work on TensorRT.

**Engine**, built on the target device:

```bash
../tools/build_engine.sh yolov8n.onnx yolov8n_fp16.orin.engine
```

A serialized TensorRT engine is valid only for the exact TensorRT version and
GPU that built it, so it is a per-device build product rather than a release
artifact. The Orin NX 16 GB / JetPack 6.1 / TensorRT 10.3.0 build used for the
measurements in `../README.md`:

```
sha256  507e44814947a01441c087fcc2451a52f399cfe9df0b4d344217dbeb70fe1408
size    9.18 MiB
build   307 s
```
