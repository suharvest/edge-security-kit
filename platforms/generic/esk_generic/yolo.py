"""ONNX Runtime person detector (YOLOv8/YOLO26 single-output head)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from .letterbox import LetterboxTransform, letterbox, xyxy_to_frame_norm

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


class PersonDetector:
    """YOLOv8n-class ONNX model restricted to COCO class 0."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        providers: list[str] | None = None,
        intra_threads: int = 0,
    ) -> None:
        opts = ort.SessionOptions()
        if intra_threads > 0:
            opts.intra_op_num_threads = intra_threads
        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # NCHW; dynamic axes come back as strings -> fall back to 640.
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.providers = self.session.get_providers()
        self.last_inference_ms = 0.0

    @property
    def backend(self) -> str:
        provider = self.providers[0].replace("ExecutionProvider", "").lower()
        return f"onnxruntime-{ort.__version__}-{provider}"

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        padded, tf = letterbox(frame, self.input_w, self.input_h)
        rgb = padded[:, :, ::-1]
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        blob /= 255.0
        return blob, tf

    def infer(self, blob: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        return outputs[0]

    def postprocess(
        self,
        raw: np.ndarray,
        tf: LetterboxTransform,
        conf_threshold: float | None = None,
    ) -> list[Detection]:
        """Decode a (1, 4+nc, N) or (1, N, 4+nc) head into frame_norm boxes.

        ``conf_threshold`` overrides the session default for this call. One
        model session is shared by every stream on the process, but the
        threshold is a per-stream setting an operator retunes at runtime, so it
        cannot live on the session -- setting it there would move every camera
        when one slider moved.
        """
        pred = np.squeeze(raw, axis=0) if raw.ndim == 3 else raw
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # (channels, anchors) -> (anchors, channels)
        if pred.shape[1] < 5:
            return []

        threshold = self.conf_threshold if conf_threshold is None else conf_threshold
        scores = pred[:, 4 + PERSON_CLASS_ID]
        mask = scores >= threshold
        if not np.any(mask):
            return []
        kept, kept_scores = pred[mask, :4], scores[mask]

        cx, cy, w, h = kept[:, 0], kept[:, 1], kept[:, 2], kept[:, 3]
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        detections: list[Detection] = []
        for i in nms(boxes, kept_scores, self.iou_threshold):
            bbox = xyxy_to_frame_norm(tuple(boxes[i]), tf)
            if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                continue  # fully clipped by the frame edge
            # Cast to Python floats: numpy scalars would leak into the payload
            # and json.dumps has no encoder for np.float32.
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

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        blob, tf = self.preprocess(frame)
        return self.postprocess(self.infer(blob), tf)
