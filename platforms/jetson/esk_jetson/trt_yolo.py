"""TensorRT person detector (YOLOv8n detect head) on Jetson Orin.

Everything here is Python. The TensorRT Python bindings ship with JetPack and
``cuda-python`` supplies ``cudaMalloc``/``cudaMemcpyAsync``, so the runtime is
``execute_async_v3`` on a stream with pinned host staging on both ends -- no
native extension, no ctypes shim, nothing to cross-compile.

The decode is the same NumPy postprocess as ``platforms/generic``: the stock
ultralytics export has one ``(1, 84, 8400)`` output with DFL already applied
inside the graph, so decoding is a threshold, a centre/corner conversion and
NMS, all vectorized. Keeping it identical to the ONNX reference is deliberate:
``coordinate_space: frame_norm`` is a contract promise, and a platform with its
own private decoder is a platform where a coordinate bug can hide.

Note on the model: the RKNPU2-optimized export in ``airockchip/rknn_model_zoo``
must NOT be used here. It has DFL and the final concat cut out of the graph
because RKNPU2 has no int8 kernel for the transposed softmax; TensorRT has no
such limitation and the nine-output head would only add work.
"""

from __future__ import annotations

import ctypes
import logging
import os
from dataclasses import dataclass

import numpy as np

from .letterbox import LetterboxTransform, xyxy_to_frame_norm

LOG = logging.getLogger("esk.jetson.trt")

PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"

# Anchors kept before NMS, highest score first. At the deployed threshold this
# never engages; it bounds the quadratic NMS for an AP sweep that thresholds at
# 0.001. COCO scores the top 100 detections per image, so the cut cannot move
# the metric.
MAX_PRE_NMS = 1000


def _import_cudart():
    """cuda-python moved the runtime module between 12.x releases."""
    try:
        from cuda.bindings import runtime as cudart  # cuda-python >= 12.8
    except ImportError:
        from cuda import cudart  # cuda-python 12.0 - 12.6
    return cudart


@dataclass
class Detection:
    """One post-NMS detection. ``box`` is xyxy in frame_norm units."""

    box: list[float]
    score: float
    label: str = PERSON_CLASS_NAME


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy NMS on xyxy boxes. Returns kept indices, highest score first.

    Shared verbatim with ``platforms/generic`` and ``platforms/rknn``.
    """
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
    """Decode a (1, 4+nc, N) or (1, N, 4+nc) YOLOv8 head into frame_norm boxes.

    Same body as ``esk_generic.yolo.PersonDetector.postprocess`` and
    ``esk_rknn.rknn_yolo.decode_person_head``. Vectorized throughout: the
    per-anchor work is a masked NumPy gather, and the only Python-level loop
    runs over the handful of boxes NMS keeps.
    """
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
    if kept_scores.size > MAX_PRE_NMS:
        top = np.argpartition(kept_scores, -MAX_PRE_NMS)[-MAX_PRE_NMS:]
        kept, kept_scores = kept[top], kept_scores[top]

    cx, cy, w, h = kept[:, 0], kept[:, 1], kept[:, 2], kept[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    detections: list[Detection] = []
    for i in nms(boxes, kept_scores, iou_threshold):
        bbox = xyxy_to_frame_norm(tuple(boxes[i]), tf)
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue  # fully clipped by the frame edge
        # Builtin floats: json.dumps has no encoder for np.float32.
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


class _CudaError(RuntimeError):
    pass


class TRTPersonDetector:
    """YOLOv8n TensorRT engine restricted to COCO class 0.

    One engine, one execution context, one CUDA stream. The class is not
    thread-safe by design: a TensorRT execution context cannot be shared across
    threads, and pretending otherwise fails as sporadic garbage rather than as
    an exception.
    """

    def __init__(
        self,
        engine_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        import tensorrt as trt

        self.trt = trt
        self.cudart = _import_cudart()
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.engine_path = engine_path
        self.last_inference_ms = 0.0
        self.last_h2d_ms = 0.0
        self.last_d2h_ms = 0.0

        with open(engine_path, "rb") as handle:
            plan = handle.read()

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            # Overwhelmingly the cause, and worth naming: a serialized engine is
            # valid only for the exact TensorRT version and GPU it was built on.
            # A copied engine deserializes to None with nothing else to go on.
            raise RuntimeError(
                f"deserialize_cuda_engine failed for {engine_path}. A TensorRT "
                f"engine is not portable across devices or TensorRT versions -- "
                f"rebuild it on this device with tools/build_engine.sh."
            )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("create_execution_context failed")

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"expected one input and one output tensor, got {inputs} / {outputs}"
            )
        self.input_name, self.output_name = inputs[0], outputs[0]

        self.input_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        self.output_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        input_shape = (1, 3, self.input_size, self.input_size)
        if not self.context.set_input_shape(self.input_name, input_shape):
            raise RuntimeError(f"engine rejected input shape {input_shape}")
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if any(dim < 0 for dim in self.output_shape):
            raise RuntimeError(f"engine output shape unresolved: {self.output_shape}")

        err, self.stream = self.cudart.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")
        self.event_start = self._create_event()
        self.event_stop = self._create_event()

        # Pinned host staging on both ends. A pageable buffer silently turns
        # cudaMemcpyAsync into a synchronous copy, which both slows the transfer
        # and folds host stall time into the measured inference window.
        self.input_host = self._pinned(input_shape, self.input_dtype)
        self.output_host = self._pinned(self.output_shape, self.output_dtype)
        self.input_device = self._device_alloc(self.input_host.nbytes)
        self.output_device = self._device_alloc(self.output_host.nbytes)
        self.context.set_tensor_address(self.input_name, int(self.input_device))
        self.context.set_tensor_address(self.output_name, int(self.output_device))

        LOG.info(
            "engine %s loaded: %s%s %s -> %s%s %s",
            os.path.basename(engine_path),
            self.input_name, tuple(input_shape), self.input_dtype,
            self.output_name, tuple(self.output_shape), self.output_dtype,
        )

    # ------------------------------------------------------------- CUDA glue

    def _check(self, err, operation: str) -> None:
        if int(err) != 0:
            raise _CudaError(f"{operation} failed with CUDA status {int(err)}")

    def _create_event(self):
        err, event = self.cudart.cudaEventCreate()
        self._check(err, "cudaEventCreate")
        return event

    def _device_alloc(self, nbytes: int):
        err, pointer = self.cudart.cudaMalloc(nbytes)
        self._check(err, "cudaMalloc")
        return pointer

    def _pinned(self, shape, dtype) -> np.ndarray:
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        err, pointer = self.cudart.cudaHostAlloc(
            nbytes, self.cudart.cudaHostAllocDefault
        )
        self._check(err, "cudaHostAlloc")
        buffer = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint8 * nbytes)).contents
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)

    # -------------------------------------------------------------- pipeline

    @property
    def backend(self) -> str:
        return f"tensorrt-{self.trt.__version__}"

    def preprocess_into(self, rgb_canvas: np.ndarray) -> None:
        """HWC uint8 RGB letterbox canvas -> the pinned NCHW input buffer.

        Written straight into pinned memory: an intermediate array would be one
        more full-tensor copy per frame for nothing. ``np.divide`` with ``out=``
        does the uint8 -> float conversion, the /255 and the store in one pass.
        """
        if rgb_canvas.shape != (self.input_size, self.input_size, 3):
            raise ValueError(
                f"expected a {self.input_size}x{self.input_size} RGB canvas, "
                f"got {rgb_canvas.shape}"
            )
        np.divide(
            rgb_canvas.transpose(2, 0, 1),
            np.float32(255.0),
            out=self.input_host[0],
            dtype=self.input_dtype,
            casting="unsafe",
        )

    def infer(self) -> np.ndarray:
        """Run the engine on whatever is in the pinned input buffer."""
        cudart = self.cudart
        self._check(
            cudart.cudaMemcpyAsync(
                int(self.input_device), self.input_host.ctypes.data,
                self.input_host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
            )[0],
            "cudaMemcpyAsync(H2D)",
        )
        # The event pair brackets execute_async_v3 only. That is what the MQTT
        # contract means by inference_time_ms: the accelerator call, measured on
        # the device clock, not the host wall time around it.
        self._check(cudart.cudaEventRecord(self.event_start, self.stream)[0], "eventRecord")
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 failed")
        self._check(cudart.cudaEventRecord(self.event_stop, self.stream)[0], "eventRecord")
        self._check(
            cudart.cudaMemcpyAsync(
                self.output_host.ctypes.data, int(self.output_device),
                self.output_host.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
            )[0],
            "cudaMemcpyAsync(D2H)",
        )
        self._check(cudart.cudaStreamSynchronize(self.stream)[0], "cudaStreamSynchronize")
        err, elapsed = cudart.cudaEventElapsedTime(self.event_start, self.event_stop)
        self.last_inference_ms = float(elapsed) if int(err) == 0 else 0.0
        return self.output_host

    def detect(self, rgb_canvas: np.ndarray, tf: LetterboxTransform) -> list[Detection]:
        self.preprocess_into(rgb_canvas)
        raw = self.infer()
        return decode_person_head(
            raw.astype(np.float32, copy=False), tf, self.conf_threshold, self.iou_threshold
        )

    def close(self) -> None:
        cudart = getattr(self, "cudart", None)
        if cudart is None:
            return
        for pointer in (getattr(self, "input_device", None), getattr(self, "output_device", None)):
            if pointer is not None:
                cudart.cudaFree(pointer)
        for array in (getattr(self, "input_host", None), getattr(self, "output_host", None)):
            if array is not None:
                cudart.cudaFreeHost(array.ctypes.data)
        for event in (getattr(self, "event_start", None), getattr(self, "event_stop", None)):
            if event is not None:
                cudart.cudaEventDestroy(event)
        if getattr(self, "stream", None) is not None:
            cudart.cudaStreamDestroy(self.stream)
        # Release in dependency order: the context holds references into the
        # engine, and the engine into the runtime.
        self.context = None
        self.engine = None
        self.runtime = None
