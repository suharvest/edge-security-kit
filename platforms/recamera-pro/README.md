# reCamera Pro (RV1126B) detector

Hub-mode detector for reCamera Pro, packaged as a reCamera app rather than a
container: the device runs Buildroot with no Docker, and its NPU is reached
through the on-device app manager.

## Status

**The model is converted and the app is written; it has not run on the board.**
Two things block an on-device run, both of them environmental rather than
code:

- `/dev/rknpu` is `root`-only and the fleet account is uid 1000. The supported
  root path is the device's app manager, which is single-active by design:
  activating this app stops whatever app is currently running.
- The device sits on a different overlay network from the hub used for
  testing, with no route between them.

Nothing here is estimated. There are no performance numbers, no captured
payloads and no accuracy figures for this platform, and there will not be any
until it runs on the board.

## What is verified

- `yolov8n` zoo ONNX converted to RV1126B int8, `librknnc 2.3.2` matching the
  board's `librknnrt 2.3.2`. Every Conv/Split/Add/Concat is placed on the NPU;
  only the input and the nine output operators sit on the CPU.
- The letterbox inverse round-trips exactly on host, and the tracker's first id
  is 1, as the contract requires.

## Layout

`esk/letterbox.py` and `esk/tracker.py` are byte-identical copies of their
`platforms/rknn` counterparts. `esk/zoo_head.py` reuses the same nine-output
decoder and adopts the kit-owned RKNN handle instead of loading a second copy.
`esk/preview.py` is ported off OpenCV to Pillow because this image carries no
cv2. `esk/file_source.py` replays a local file through the kit's frame source,
which is what makes the same truth-video assertions possible here as on RK3588.

The kit framework itself lives outside this repository, in the reCamera Pro
application tree; `manifest.json` is what the device's app manager consumes.

## Model

Binaries are not committed. `models/SHA256SUMS` pins what was built; regenerate
with the RK3588 conversion chain in `../rknn/tools/`, passing the `rv1126b`
platform and the same 400-image calibration list.
