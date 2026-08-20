"""ONNX -> HEF for edge-security-kit's yolov8n, Hailo-8.

Cuts the graph at the six detection-head convolutions (DFL, the class sigmoid
and the final concat stay off-chip, decoded in numpy by esk_hailo.hailo_yolo),
folds the /255 normalization into the input layer so the HEF takes uint8 RGB,
and appends a sigmoid to the three class convs so the published score is a
probability and quantization sees a bounded range.
"""
import glob
import os
import sys
import tarfile

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

WORK = os.environ.get("ESK_HAILO_WORK", "/work/work")
ONNX = os.environ.get("ESK_HAILO_ONNX", "/work/yolov8n.onnx")
CALIB_TGZ = os.environ.get("ESK_HAILO_CALIB", "/work/calib-frames.tgz")
INPUT = 640
PAD = 114

END = [
    "/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv",
    "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv",
    "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv",
]

ALLS = """
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
change_output_activation(conv42, sigmoid)
change_output_activation(conv53, sigmoid)
change_output_activation(conv63, sigmoid)
"""


def letterbox(img, dst=INPUT):
    w, h = img.size
    scale = min(dst / w, dst / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = img.convert("RGB").resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (dst, dst), (PAD, PAD, PAD))
    canvas.paste(resized, ((dst - nw) // 2, (dst - nh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def build_calib():
    out = os.path.join(WORK, "calib")
    if not os.path.isdir(out):
        os.makedirs(out, exist_ok=True)
        with tarfile.open(CALIB_TGZ) as tar:
            tar.extractall(out)
    files = sorted(glob.glob(os.path.join(out, "**", "*.jpg"), recursive=True))
    if not files:
        raise SystemExit("no calibration frames found")
    arr = np.stack([letterbox(Image.open(f)) for f in files])
    print(f"calib set: {arr.shape} dtype={arr.dtype} from {len(files)} frames")
    return arr


def main():
    os.makedirs(WORK, exist_ok=True)
    calib = build_calib()

    runner = ClientRunner(hw_arch="hailo8")
    runner.translate_onnx_model(
        ONNX, "yolov8n",
        start_node_names=["images"], end_node_names=END,
        net_input_shapes={"images": [1, 3, INPUT, INPUT]},
    )
    alls_path = os.path.join(WORK, "yolov8n.alls")
    with open(alls_path, "w") as fh:
        fh.write(ALLS)
    runner.load_model_script(alls_path)
    runner.optimize(calib)
    runner.save_har(os.path.join(WORK, "yolov8n_quantized.har"))
    hef = runner.compile()
    out = os.path.join(WORK, "yolov8n.hef")
    with open(out, "wb") as fh:
        fh.write(hef)
    print("HEF written:", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    sys.exit(main())
