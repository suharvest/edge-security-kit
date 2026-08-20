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
* **The head comes back as uint8 and is thresholded in the quantized domain.**
  HailoRT's FLOAT32 output format dequantizes all 1209600 elements of this head
  on the host, and the decoder reads 8400 of them -- the person channel -- to
  decide anything at all. The threshold is converted into a uint8 level instead
  (``q >= qp_zp + conf / qp_scale``), the comparison runs on the raw bytes, and
  only the surviving anchors' score and box distribution are converted. The
  parameters come from the same ``InferModel`` the tensors do, so they cannot
  describe a different HEF than the one loaded.

  **This moves work out of ``inference_time_ms``**, which times
  ``configured.run()`` alone: the dequantization used to happen inside that call
  and now happens in the decoder, outside it. The metric falls without the
  pipeline necessarily getting faster, so ``pipeline_ms`` is the figure to
  compare across this change and ``inference_time_ms`` is not.
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

import math
import os
import time
from dataclasses import dataclass

import numpy as np

from .letterbox import (
    LetterboxCanvas,
    LetterboxTransform,
    letterbox,
    xyxy_to_frame_norm,
)

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


def _as_hwc(raw) -> np.ndarray:
    """One head output as a plain HWC array, batch dimension squeezed."""
    arr = np.asarray(raw)
    if arr.ndim == 4:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim != 3:
        raise ValueError(f"expected an NHWC head output, got shape {arr.shape}")
    return arr


def group_branch_indices(outputs) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Pair each stride's ``(box_distribution, class_map)`` by tensor shape.

    Returns the squeezed arrays alongside ``(box_index, class_index)`` pairs
    *into that list*, coarsest-grid-last, i.e. stride 8, 16, 32. Indices rather
    than arrays because the quantized path needs to look each branch's
    scale/zero-point up by the same position, and re-deriving that pairing in a
    second place is how the two halves drift apart.
    """
    arrays = [_as_hwc(raw) for raw in outputs]
    by_grid: dict[tuple[int, int], dict[int, int]] = {}
    for index, arr in enumerate(arrays):
        grid_h, grid_w, channels = arr.shape
        slot = by_grid.setdefault((grid_h, grid_w), {})
        if channels in slot:
            raise ValueError(
                f"two {channels}-channel tensors on the {grid_h}x{grid_w} grid: "
                "the box and class branches cannot be told apart"
            )
        slot[channels] = index

    pairs: list[tuple[int, int]] = []
    for grid, slot in sorted(by_grid.items(), reverse=True):
        if len(slot) != 2 or BOX_CHANNELS not in slot:
            raise ValueError(
                f"grid {grid[0]}x{grid[1]} has channels {sorted(slot)}, expected "
                f"exactly one {BOX_CHANNELS}-channel box tensor and one class map"
            )
        cls_index = next(v for c, v in slot.items() if c != BOX_CHANNELS)
        pairs.append((slot[BOX_CHANNELS], cls_index))
    if len(pairs) != 3:
        raise ValueError(f"expected 3 strides, got {len(pairs)}")
    return arrays, pairs


def group_branches(outputs) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(box_distribution, class_map)`` per stride, coarsest-grid-last."""
    arrays, pairs = group_branch_indices(outputs)
    return [(arrays[b], arrays[c]) for b, c in pairs]


def quantized_threshold(conf_threshold: float, scale: float, zero_point: float):
    """Lowest uint8 level that can dequantize to ``conf_threshold`` or above.

    HailoRT dequantizes as ``(q - qp_zp) * qp_scale``, so the comparison
    ``value >= conf`` is ``q >= zp + conf / scale`` in the quantized domain.
    ``floor`` rather than ``ceil``: the boundary level is admitted and the exact
    float comparison is redone on the handful of survivors, so a level that
    rounds either way cannot be dropped here. Returns ``None`` when no uint8
    level can reach the threshold.
    """
    if scale <= 0.0:
        raise ValueError(f"non-positive quantization scale {scale}")
    level = math.floor(zero_point + conf_threshold / scale)
    if level > 255:
        return None
    return max(0, int(level))


def decode_split_head(
    outputs,
    tf: LetterboxTransform,
    conf_threshold: float,
    iou_threshold: float,
    input_size: int = 640,
    quants: list[tuple[float, float]] | None = None,
) -> list[Detection]:
    """Decode the six-output YOLOv8 head into ``frame_norm`` detections.

    The class map arrives with sigmoid already applied on-chip, so its values
    are probabilities and are thresholded directly.

    ``quants`` switches the whole function into the quantized domain: pass one
    ``(scale, zero_point)`` per entry of ``outputs``, positionally, and the
    tensors are expected to be the raw uint8 the accelerator wrote. Nothing is
    dequantized wholesale -- the person channel is thresholded as uint8, and
    only the surviving anchors' score and box distribution are converted. On
    this head that is 8400 uint8 comparisons and typically under 1300 floats,
    against 1209600 elements HailoRT would otherwise convert on the host.
    """
    arrays, pairs = group_branch_indices(outputs)
    boxes_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    for dist_index, cls_index in pairs:
        dist, cls = arrays[dist_index], arrays[cls_index]
        grid_h = cls.shape[0]
        person = cls[:, :, PERSON_CLASS_ID]

        if quants is None:
            rows, cols = np.nonzero(person >= conf_threshold)
            if rows.size == 0:
                continue
            scores = person[rows, cols].astype(np.float32)
            # (k, 64) -> decoded below as (64, k) to match the DFL reshape.
            selected = dist[rows, cols, :].astype(np.float32)
        else:
            cls_scale, cls_zp = quants[cls_index]
            box_scale, box_zp = quants[dist_index]
            level = quantized_threshold(conf_threshold, cls_scale, cls_zp)
            if level is None:
                continue
            rows, cols = np.nonzero(person >= level)
            if rows.size == 0:
                continue
            scores = (person[rows, cols].astype(np.float32) - cls_zp) * cls_scale
            # The uint8 threshold admits the boundary level; this is the exact
            # comparison the float32 path makes, on the survivors only, so the
            # two paths keep the same anchors and not merely similar ones.
            kept = scores >= conf_threshold
            if not kept.all():
                rows, cols, scores = rows[kept], cols[kept], scores[kept]
                if rows.size == 0:
                    continue
            # DFL is a softmax, which is not invariant to the quantization
            # scale, so the box distribution has to be dequantized before the
            # mean is taken. The zero point would cancel; the scale would not.
            selected = (dist[rows, cols, :].astype(np.float32) - box_zp) * box_scale

        distances = _dfl_distances(selected.T)
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
        scores_all.append(scores.astype(np.float32))

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
        optimizations: str | None = None,
    ) -> None:
        from hailo_platform import FormatType, HailoSchedulingAlgorithm, VDevice

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = int(input_size)
        self.timeout_ms = int(timeout_ms)
        self.last_inference_ms = 0.0
        self._range_checked = False

        # Which of the three host-side optimizations are on. All three are the
        # shipped path; the switch exists so a measurement run can turn them off
        # one at a time on the same binary, which is the only way to attribute a
        # change to one of them rather than to the board having warmed up.
        opts = optimizations
        if opts is None:
            opts = os.environ.get("ESK_HAILO_OPTS", "abc")
        opts = opts.lower()
        self.opt_uint8_output = "a" in opts
        self.opt_reuse_bindings = "b" in opts
        self.opt_reuse_canvas = "c" in opts
        self.optimizations = "".join(
            flag
            for flag, on in (
                ("a", self.opt_uint8_output),
                ("b", self.opt_reuse_bindings),
                ("c", self.opt_reuse_canvas),
            )
            if on
        )

        params = VDevice.create_params()
        # ROUND_ROBIN lets the HailoRT scheduler share the device with any other
        # process that has it open. The board this was measured on runs an
        # unrelated face-recognition service against the same /dev/hailo0.
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdevice = VDevice(params)
        self.infer_model = self.vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)

        # UINT8 outputs: the host never dequantizes a full tensor. HailoRT's
        # FLOAT32 format converts all 1209600 elements of this head on the CPU,
        # while the decoder reads 8400 of them to threshold and a few hundred
        # more for the boxes that survive. The quantization parameters come from
        # the same InferModel the tensors do (`quant_infos`, fields `qp_scale`
        # and `qp_zp`), so there is no second place for them to be wrong.
        self.output_names = [vs.name for vs in self.infer_model.outputs]
        fmt = FormatType.UINT8 if self.opt_uint8_output else FormatType.FLOAT32
        for name in self.output_names:
            self.infer_model.output(name).set_format_type(fmt)
        self.output_dtype = np.uint8 if self.opt_uint8_output else np.float32
        self.quants = self._read_quant_infos() if self.opt_uint8_output else None

        self.input_name = self.infer_model.inputs[0].name
        self.input_shape = tuple(self.infer_model.input(self.input_name).shape)
        self.configured = self.infer_model.configure()
        self._out_shapes = {
            name: tuple(self.infer_model.output(name).shape)
            for name in self.output_names
        }

        # One set of output buffers and one bindings object for the life of the
        # detector. Rebuilding them per frame allocates six arrays and crosses
        # into pybind11 six more times for work whose result never changes.
        self._buffers: dict[str, np.ndarray] | None = None
        self._bindings = None
        if self.opt_reuse_bindings:
            self._buffers = self._new_buffers()
            self._bindings = self.configured.create_bindings(
                output_buffers=self._buffers
            )

        self._canvas = (
            LetterboxCanvas(self.input_size, self.input_size)
            if self.opt_reuse_canvas
            else None
        )

        self.device_arch = self._device_arch()
        self.runtime_version = self._runtime_version()

    def _new_buffers(self) -> dict[str, np.ndarray]:
        return {
            name: np.empty(shape, dtype=self.output_dtype)
            for name, shape in self._out_shapes.items()
        }

    def _read_quant_infos(self) -> list[tuple[float, float]]:
        """``(qp_scale, qp_zp)`` per output, positionally by ``output_names``.

        ``InferModel.InferStream.quant_infos`` returns a list because a HEF may
        carry per-channel parameters. This decoder thresholds a whole channel
        map against one level, which only means anything with a single set, so
        a multi-parameter output is refused rather than silently decoded with
        the first entry.
        """
        quants: list[tuple[float, float]] = []
        for name in self.output_names:
            infos = self.infer_model.output(name).quant_infos
            if len(infos) != 1:
                raise RuntimeError(
                    f"output {name} carries {len(infos)} quantization infos; the "
                    "uint8 decode path needs exactly one per tensor. Re-run with "
                    "ESK_HAILO_OPTS excluding 'a' to fall back to FLOAT32 outputs."
                )
            info = infos[0]
            quants.append((float(info.qp_scale), float(info.qp_zp)))
        return quants

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
        """BGR frame -> uint8 NHWC RGB letterbox canvas plus its transform.

        With the canvas optimization on, the returned array is reused by the
        next call. The frame loop consumes it inside :meth:`detect` before it
        reads another frame, which is what makes that safe.
        """
        if self._canvas is not None:
            return self._canvas.convert(frame)
        padded, tf = letterbox(frame, self.input_size, self.input_size)
        return np.ascontiguousarray(padded[:, :, ::-1]), tf

    def infer(self, rgb_canvas: np.ndarray) -> list[np.ndarray]:
        """Run one inference on a uint8 HWC RGB letterbox canvas."""
        if rgb_canvas.dtype != np.uint8:
            raise ValueError("Hailo input must stay uint8; the HEF normalizes")
        if self._bindings is not None:
            buffers = self._buffers
            bindings = self._bindings
        else:
            buffers = self._new_buffers()
            bindings = self.configured.create_bindings(output_buffers=buffers)
        # Set every frame either way: the canvas is reused but its address is
        # not guaranteed, and re-binding an unchanged pointer is free.
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
        arrays, pairs = group_branch_indices(outputs)
        for _, cls_index in pairs:
            cls = arrays[cls_index]
            if self.quants is None:
                lo, hi = float(cls.min()), float(cls.max())
            else:
                scale, zero_point = self.quants[cls_index]
                # Two checks, and the first is the stronger one. A uint8 tensor
                # cannot show an out-of-range *observed* value however the HEF
                # was built -- the range is carried by the quantization
                # parameters, not by the samples -- so what settles it is the
                # span the parameters can represent at all. A branch left as
                # logits is pinned to something like [-20, 20] and fails here on
                # the first frame, whatever that frame happened to contain.
                span_lo = (0.0 - zero_point) * scale
                span_hi = (255.0 - zero_point) * scale
                if span_lo < -1e-3 or span_hi > 1.0 + 1e-3:
                    raise RuntimeError(
                        f"class branch quantization spans [{span_lo:.4f}, "
                        f"{span_hi:.4f}], not a probability: the HEF was compiled "
                        "without change_output_activation(sigmoid)"
                    )
                lo = (float(cls.min()) - zero_point) * scale
                hi = (float(cls.max()) - zero_point) * scale
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
            outputs,
            tf,
            self.conf_threshold,
            self.iou_threshold,
            self.input_size,
            quants=self.quants,
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
        # Drop the Python references so pybind11 runs the destructors here,
        # while the device they point into is still alive. Bindings first: they
        # hold a handle into ConfiguredInferModel the same way it holds one into
        # the VDevice, so the same teardown-order defect applies one level down.
        self._bindings = None
        self._buffers = None
        self.configured = None
        self.infer_model = None
        try:
            self.vdevice.release()
        except Exception:  # pragma: no cover - teardown only
            pass
        self.vdevice = None
