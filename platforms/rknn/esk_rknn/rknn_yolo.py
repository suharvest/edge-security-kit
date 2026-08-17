"""RKNN Lite person detector (YOLOv8n head, COCO class 0).

The model was converted with ``mean_values=0`` / ``std_values=255``, so the NPU
does the normalization and this module hands it raw uint8 NHWC RGB. That is not
a micro-optimization: converting a 640x640x3 frame to float32 on the CPU costs
more than the inference call itself on this board.

Decoding is deliberately the same NumPy postprocess the generic platform uses.
One decoder for both platforms means a coordinate bug cannot hide on one of
them, and the head is small enough that the decode is not the bottleneck.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from .letterbox import LetterboxTransform, xyxy_to_frame_norm

PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"


@dataclass
class Detection:
    """One post-NMS detection. ``box`` is xyxy in frame_norm units."""

    box: list[float]
    score: float
    label: str = PERSON_CLASS_NAME


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS on xyxy boxes. Returns kept indices, highest score first."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def decode_person_head(
    raw: np.ndarray,
    tf: LetterboxTransform,
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    """Decode a (1, 4+nc, N) or (1, N, 4+nc) YOLOv8 head into frame_norm boxes."""
    pred = np.squeeze(raw, axis=0) if raw.ndim == 3 else raw
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T  # (channels, anchors) -> (anchors, channels)
    if pred.shape[1] < 5:
        return []

    scores = pred[:, 4 + PERSON_CLASS_ID]
    mask = scores >= conf_threshold
    if not np.any(mask):
        return []
    kept, kept_scores = pred[mask, :4], scores[mask]

    cx, cy, w, h = kept[:, 0], kept[:, 1], kept[:, 2], kept[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    detections: list[Detection] = []
    for i in nms(boxes, kept_scores, iou_threshold):
        bbox = xyxy_to_frame_norm(tuple(boxes[i]), tf)
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue  # fully clipped by the frame edge
        detections.append(
            Detection(
                box=[
                    float(bbox[0] - bbox[2] / 2),
                    float(bbox[1] - bbox[3] / 2),
                    float(bbox[0] + bbox[2] / 2),
                    float(bbox[1] + bbox[3] / 2),
                ],
                score=float(kept_scores[i]),
            )
        )
    return detections


class RKNNPersonDetector:
    """YOLOv8n .rknn restricted to COCO class 0, on the RK3588 NPU."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        core_mask: int | None = None,
        input_size: int = 640,
    ) -> None:
        from rknnlite.api import RKNNLite

        self.input_size = int(input_size)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.last_inference_ms = 0.0

        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model_path)
        if ret:
            raise RuntimeError(f"load_rknn({model_path}) returned {ret}")
        # init_runtime is where a model built for the wrong SoC or against a
        # newer librknnrt actually fails -- load_rknn succeeds regardless.
        ret = self.rknn.init_runtime(**({"core_mask": core_mask} if core_mask is not None else {}))
        if ret:
            raise RuntimeError(f"init_runtime({model_path}) returned {ret}")
        self.runtime_version = _librknnrt_version()

    @property
    def backend(self) -> str:
        return f"rknn-lite2-{self.runtime_version}"

    def infer(self, rgb_canvas: np.ndarray) -> np.ndarray:
        """Run one inference on a uint8 HWC RGB letterbox canvas."""
        if rgb_canvas.dtype != np.uint8:
            raise ValueError("RKNN input must stay uint8; the model normalizes")
        batched = np.ascontiguousarray(rgb_canvas[None])
        started = time.perf_counter()
        outputs = self.rknn.inference(inputs=[batched])
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        if outputs is None:
            # RKNN reports some input-signature errors only on stderr and hands
            # back None rather than raising.
            raise RuntimeError("RKNN inference returned None")
        return np.asarray(outputs[0])

    def detect(self, rgb_canvas: np.ndarray, tf: LetterboxTransform) -> list[Detection]:
        raw = self.infer(rgb_canvas)
        return decode_person_head(raw, tf, self.conf_threshold, self.iou_threshold)

    def close(self) -> None:
        try:
            self.rknn.release()
        except Exception:  # pragma: no cover - teardown only
            pass


_LIBRKNNRT_CANDIDATES = (
    "/usr/lib/librknnrt.so",
    "/usr/lib/aarch64-linux-gnu/librknnrt.so",
    "/usr/local/lib/librknnrt.so",
)


def _librknnrt_version() -> str:
    """Read the version string out of the librknnrt binary itself.

    ``rknn_get_sdk_version`` needs a live context, so it cannot be queried
    before a model is loaded, and the SONAME is not usable as a substitute: on
    this board ``/usr/lib/librknnrt.so`` points at ``librknnrt.so.2.3.0`` while
    the library reports **2.3.2**. Since the toolkit used for conversion has to
    match the reported version, the reported one is what belongs in
    ``health.backend`` -- reading it from the file is the only way to get it
    without guessing from a filename that is wrong.
    """
    import re

    pattern = re.compile(rb"librknnrt version:\s*([0-9][0-9A-Za-z._-]*)")
    seen: set[str] = set()
    for name in _LIBRKNNRT_CANDIDATES:
        real = os.path.realpath(name)
        if real in seen or not os.path.exists(real):
            continue
        seen.add(real)
        try:
            with open(real, "rb") as handle:
                match = pattern.search(handle.read())
        except OSError:
            continue
        if match:
            return match.group(1).decode("ascii", "replace")
    return "unknown"
