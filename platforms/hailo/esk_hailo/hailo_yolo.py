"""HailoRT person detector (YOLOv8n split head, COCO class 0).

The HEF is compiled from the same ``yolov8n.onnx`` every other platform in this
repo runs, cut at the six detection-head convolutions. Three things live off
the graph as a result, and each is a deliberate choice rather than a
limitation:

* **``/255`` normalization is folded into the HEF input layer**
  (``normalization([0,0,0],[255,255,255])`` in the model script), so this
  module hands the accelerator raw uint8 NHWC RGB. Converting a 640x640x3
  frame to float32 on a Pi 5 costs more CPU than the whole inference call.
* **The class sigmoid runs on-chip** (``change_output_activation(convNN,
  sigmoid)``). That is not only a CPU saving: quantizing raw logits spends the
  int8 range on values no consumer ever reads, while a sigmoid output has a
  known ``[0, 1]`` range the optimizer can pin exactly. :meth:`detect` asserts
  the range on the first inference, so a HEF built without that model-script
  line fails loudly instead of publishing scores that are really logits.
* **DFL, the box decode and NMS run here in NumPy.** The arithmetic is a port
  of ``platforms/rknn``'s ``decode_zoo_head`` -- same DFL mean, same anchor
  centres, same greedy NMS -- because two platforms decoding the same head with
  two different implementations is how a coordinate bug hides on one of them.
  Cost is bounded by thresholding the person channel *first* and running DFL
  only on the surviving anchors, typically single digits out of 8400.

Branches are identified by tensor shape, never by output order or name. The
HailoRT infer model hands outputs back in a dict keyed by layer name
(``yolov8n/conv41`` ...), and those names change whenever the graph is
re-translated; a shape-driven mapping survives a recompile.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .letterbox import LetterboxTransform, letterbox, xyxy_to_frame_norm

PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"
DFL_BINS = 16
BOX_CHANNELS = 4 * DFL_BINS

# Anchors kept before NMS, highest score first. At the deployed threshold a
# frame yields single digits; this exists so an AP sweep at 0.001 cannot hand
# the quadratic NMS several thousand boxes.
MAX_PRE_NMS = 1000


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


def _dfl_distances(dist_logits: np.ndarray) -> np.ndarray:
    """``(4*bins, k)`` logits -> ``(4, k)`` expected distances, in grid units.

    Distribution Focal Loss regresses each box edge as a softmax over
    ``DFL_BINS`` discrete distances; the prediction is that distribution's
    mean. Identical to the RKNN platform's implementation.
    """
    reshaped = dist_logits.reshape(4, DFL_BINS, -1).astype(np.float32)
    reshaped -= reshaped.max(axis=1, keepdims=True)  # stabilize before exp
    weights = np.exp(reshaped)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.tensordot(np.arange(DFL_BINS, dtype=np.float32), weights, axes=([0], [1]))


def group_branches(outputs) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pair each stride's ``(box_distribution, class_map)`` by tensor shape.

    ``outputs`` is any iterable of NHWC (or squeezable NHWC-with-batch) arrays.
    Returned coarsest-grid-last, i.e. stride 8, 16, 32.
    """
    by_grid: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for raw in outputs:
        arr = np.asarray(raw)
        if arr.ndim == 4:
            arr = np.squeeze(arr, axis=0)
        if arr.ndim != 3:
            raise ValueError(f"expected an NHWC head output, got shape {arr.shape}")
        grid_h, grid_w, channels = arr.shape
        slot = by_grid.setdefault((grid_h, grid_w), {})
        if channels in slot:
            raise ValueError(
                f"two {channels}-channel tensors on the {grid_h}x{grid_w} grid: "
                "the box and class branches cannot be told apart"
            )
        slot[channels] = arr

    branches: list[tuple[np.ndarray, np.ndarray]] = []
    for grid, slot in sorted(by_grid.items(), reverse=True):
        if len(slot) != 2 or BOX_CHANNELS not in slot:
            raise ValueError(
                f"grid {grid[0]}x{grid[1]} has channels {sorted(slot)}, expected "
                f"exactly one {BOX_CHANNELS}-channel box tensor and one class map"
            )
        cls = next(v for c, v in slot.items() if c != BOX_CHANNELS)
        branches.append((slot[BOX_CHANNELS], cls))
    if len(branches) != 3:
        raise ValueError(f"expected 3 strides, got {len(branches)}")
    return branches


def decode_split_head(
    outputs,
    tf: LetterboxTransform,
    conf_threshold: float,
    iou_threshold: float,
    input_size: int = 640,
) -> list[Detection]:
    """Decode the six-output YOLOv8 head into ``frame_norm`` detections.

    The class map arrives with sigmoid already applied on-chip, so its values
    are probabilities and are thresholded directly.
    """
    boxes_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for dist, cls in group_branches(outputs):
        grid_h, grid_w = cls.shape[0], cls.shape[1]
        person = cls[:, :, PERSON_CLASS_ID]
        rows, cols = np.nonzero(person >= conf_threshold)
        if rows.size == 0:
            continue
        # (k, 64) -> (64, k) to match the DFL reshape, then decode only the
        # anchors that survived the threshold.
        distances = _dfl_distances(dist[rows, cols, :].T)
        stride = input_size / grid_h
        cx = cols.astype(np.float32) + 0.5
        cy = rows.astype(np.float32) + 0.5
        boxes_all.append(
            np.stack(
                [
                    (cx - distances[0]) * stride,
                    (cy - distances[1]) * stride,
                    (cx + distances[2]) * stride,
                    (cy + distances[3]) * stride,
                ],
                axis=1,
            )
        )
        scores_all.append(person[rows, cols].astype(np.float32))

    if not boxes_all:
        return []
    boxes = np.concatenate(boxes_all)
    scores = np.concatenate(scores_all)
    if scores.size > MAX_PRE_NMS:
        top = np.argpartition(scores, -MAX_PRE_NMS)[-MAX_PRE_NMS:]
        boxes, scores = boxes[top], scores[top]

    detections: list[Detection] = []
    for i in nms(boxes, scores, iou_threshold):
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
                score=float(scores[i]),
            )
        )
    return detections


class HailoPersonDetector:
    """YOLOv8n HEF restricted to COCO class 0, on a Hailo-8 / 8L NPU."""

    def __init__(
        self,
        hef_path: str,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        timeout_ms: int = 5000,
    ) -> None:
        from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = int(input_size)
        self.timeout_ms = int(timeout_ms)
        self.last_inference_ms = 0.0
        self._range_checked = False

        params = VDevice.create_params()
        # ROUND_ROBIN lets the HailoRT scheduler share the device with any other
        # process that has it open. The board this was measured on runs an
        # unrelated face-recognition service against the same /dev/hailo0.
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
        self.infer_model = self.vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        # FLOAT32 outputs: HailoRT dequantizes on the host. The alternative is
        # decoding uint8 with per-tensor scale/zero-point here, which buys a few
        # hundred microseconds and adds a second place for the quantization
        # parameters to be wrong.
        self.output_names = [vs.name for vs in self.infer_model.outputs]
        for name in self.output_names:
            self.infer_model.output(name).set_format_type(FormatType.FLOAT32)
        self.input_name = self.infer_model.inputs[0].name
        self.input_shape = tuple(self.infer_model.input(self.input_name).shape)
        self.configured = self.infer_model.configure()
        self._out_shapes = {
            name: tuple(self.infer_model.output(name).shape)
            for name in self.output_names
        }
        self.device_arch = self._device_arch()
        self.runtime_version = self._runtime_version()

    # ------------------------------------------------------------- identity

    @staticmethod
    def _runtime_version() -> str:
        try:
            from hailo_platform import __version__ as version

            return str(version)
        except Exception:  # pragma: no cover - depends on the installed wheel
            return "unknown"

    def _device_arch(self) -> str:
        try:
            devices = self.vdevice.get_physical_devices()
            return str(devices[0].get_architecture()).split(".")[-1].lower()
        except Exception:  # pragma: no cover - device-dependent
            return "hailo"

    @property
    def backend(self) -> str:
        return f"hailort-{self.runtime_version}-{self.device_arch}"

    @property
    def output_shapes(self) -> dict[str, tuple]:
        """Layer name -> NHWC shape, in the order HailoRT declares them."""
        return dict(self._out_shapes)

    # ------------------------------------------------------------ inference

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        """BGR frame -> uint8 NHWC RGB letterbox canvas plus its transform."""
        padded, tf = letterbox(frame, self.input_size, self.input_size)
        return np.ascontiguousarray(padded[:, :, ::-1]), tf

    def infer(self, rgb_canvas: np.ndarray) -> list[np.ndarray]:
        """Run one inference on a uint8 HWC RGB letterbox canvas."""
        if rgb_canvas.dtype != np.uint8:
            raise ValueError("Hailo input must stay uint8; the HEF normalizes")
        buffers = {
            name: np.empty(shape, dtype=np.float32)
            for name, shape in self._out_shapes.items()
        }
        bindings = self.configured.create_bindings(output_buffers=buffers)
        bindings.input().set_buffer(rgb_canvas)
        started = time.perf_counter()
        self.configured.run([bindings], self.timeout_ms)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        return [buffers[name] for name in self.output_names]

    def _check_class_range(self, outputs: list[np.ndarray]) -> None:
        """The class branch must be a probability, not a logit.

        ``change_output_activation(convNN, sigmoid)`` is a line in the model
        script, not a property of the ONNX. A HEF built without it still loads,
        still produces boxes, and publishes ``score`` values that are logits --
        wrong by a monotonic transform, so every threshold in the stack is
        silently off. Checked once, on real data, rather than trusted.
        """
        for _, cls in group_branches(outputs):
            lo, hi = float(cls.min()), float(cls.max())
            if lo < -1e-3 or hi > 1.0 + 1e-3:
                raise RuntimeError(
                    f"class branch range [{lo:.4f}, {hi:.4f}] is not a probability: "
                    "the HEF was compiled without change_output_activation(sigmoid)"
                )
        self._range_checked = True

    def detect(self, rgb_canvas: np.ndarray, tf: LetterboxTransform) -> list[Detection]:
        outputs = self.infer(rgb_canvas)
        if not self._range_checked:
            self._check_class_range(outputs)
        return decode_split_head(
            outputs, tf, self.conf_threshold, self.iou_threshold, self.input_size
        )

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        canvas, tf = self.preprocess(frame)
        return self.detect(canvas, tf)

    def close(self) -> None:
        """Tear down in dependency order, wrappers before the device.

        ``ConfiguredInferModel`` and ``InferModel`` hold handles into the
        ``VDevice``. Releasing the device first leaves their destructors to
        run against freed memory at interpreter exit, which segfaults after
        every message has already been published -- so the process prints a
        complete, correct result and then exits 139, and a supervisor cannot
        tell that from a real crash. Drop the wrappers first, then release.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.configured.shutdown()
        except Exception:  # pragma: no cover - teardown only
            pass
        # Drop the Python references so pybind11 runs both destructors here,
        # while the device they point into is still alive.
        self.configured = None
        self.infer_model = None
        try:
            self.vdevice.release()
        except Exception:  # pragma: no cover - teardown only
            pass
        self.vdevice = None
