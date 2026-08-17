# Model weights

Not tracked in git (~12 MB). Fetch before the first run:

```bash
curl -L -o yolov8n.onnx \
  https://raw.githubusercontent.com/Zhang-zu-hao/Industrial-security-demo/master/models/yolov8n.onnx
```

Verified artifact: `yolov8n.onnx`, 12851087 bytes, md5
`eda19d19a8aa73d54411a6fc1d091c7f`, sha256 prefix `fb38b3933007` (the prefix is
what `status.versions.model` reports as `yolov8n@fb38b3933007`).

The detector reads a single-output YOLOv8/YOLO26 head — `(1, 4 + nc, anchors)`
or its transpose — and keeps COCO class 0 only. Swapping in another export of
the same head shape needs no code change; `status.versions.model` changes with
the file hash so the hub can tell the fleet apart.
