"""Letterbox resize and its exact inverse.

The model input is square; the source frame usually is not. Stretching a
1280x720 frame into 640x640 changes the aspect ratio, so every box comes back
horizontally compressed. We pad instead, and reverse the pad before the
coordinates leave the process -- ``coordinate_space`` in the MQTT contract is
always ``frame_norm``, i.e. normalized against the ORIGINAL frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    """Forward parameters, enough to invert the mapping exactly."""

    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        """Map a point from model-input pixels back to source-frame pixels."""
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale


def letterbox(
    frame: np.ndarray,
    dst_w: int,
    dst_h: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxTransform]:
    """Resize ``frame`` into ``dst_w`` x ``dst_h`` keeping the aspect ratio.

    The scaled image is centred; the remainder is padded with ``color``.
    """
    src_h, src_w = frame.shape[:2]
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
    interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    pad_x = (dst_w - new_w) / 2.0
    pad_y = (dst_h - new_h) / 2.0
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    bottom, right = dst_h - new_h - top, dst_w - new_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    # The inverse must use the integer pad actually applied, not the ideal half.
    return padded, LetterboxTransform(
        scale=scale,
        pad_x=float(left),
        pad_y=float(top),
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
    )


def xyxy_to_frame_norm(
    box: tuple[float, float, float, float], tf: LetterboxTransform
) -> list[float]:
    """Model-input xyxy -> ``frame_norm`` [cx, cy, w, h], clamped to [0, 1].

    Clamping happens in source-pixel space before the centre/size conversion,
    so a box clipped at the frame edge stays consistent (a clamped centre with
    an unclamped width would describe a box reaching outside the frame).
    """
    x1, y1 = tf.to_source(box[0], box[1])
    x2, y2 = tf.to_source(box[2], box[3])
    x1 = min(max(x1, 0.0), float(tf.src_w))
    x2 = min(max(x2, 0.0), float(tf.src_w))
    y1 = min(max(y1, 0.0), float(tf.src_h))
    y2 = min(max(y2, 0.0), float(tf.src_h))
    # Builtin floats, not numpy scalars: these values end up in json.dumps.
    return [
        float((x1 + x2) / 2.0) / tf.src_w,
        float((y1 + y2) / 2.0) / tf.src_h,
        float(x2 - x1) / tf.src_w,
        float(y2 - y1) / tf.src_h,
    ]


def frame_norm_to_pixels(
    bbox: list[float], width: int, height: int
) -> tuple[int, int, int, int]:
    """``frame_norm`` [cx, cy, w, h] -> pixel xyxy in the original frame."""
    cx, cy, w, h = bbox
    return (
        int(round((cx - w / 2) * width)),
        int(round((cy - h / 2) * height)),
        int(round((cx + w / 2) * width)),
        int(round((cy + h / 2) * height)),
    )


class LetterboxCanvas:
    """Reusable letterbox target that writes straight into an RGB canvas.

    :func:`letterbox` allocates a resized image, then ``copyMakeBorder``
    allocates the padded one, and the caller then allocates a third array to
    flip BGR to RGB. At 640x640x3 that is two extra 1.2 MB buffers and one full
    copy per frame, on a board where the whole inference call is ~10 ms.

    This class keeps one 640x640x3 uint8 canvas for the life of the detector:
    ``cv2.resize`` writes the scaled frame directly into the canvas ROI, and the
    BGR->RGB swap runs in place on that ROI. The pad border is painted once and
    then never touched again, because the ROI is the only region any frame
    writes to and its geometry only changes if the source resolution does.

    The arithmetic -- scale, rounded size, integer pad -- is copied from
    :func:`letterbox` rather than re-derived, so the two produce byte-identical
    canvases and the same :class:`LetterboxTransform`.
    """

    def __init__(
        self,
        dst_w: int,
        dst_h: int,
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> None:
        self.dst_w = int(dst_w)
        self.dst_h = int(dst_h)
        # Stored RGB, because the canvas is handed to the accelerator as RGB and
        # the pad has to be the same grey in both orders anyway.
        self.color = (int(color[2]), int(color[1]), int(color[0]))
        self.canvas = np.empty((self.dst_h, self.dst_w, 3), dtype=np.uint8)
        self._geometry: tuple[int, int] | None = None
        self._transform: LetterboxTransform | None = None
        self._roi: np.ndarray | None = None

    def _reshape_for(self, src_w: int, src_h: int) -> None:
        scale = min(self.dst_w / src_w, self.dst_h / src_h)
        new_w, new_h = int(round(src_w * scale)), int(round(src_h * scale))
        pad_x = (self.dst_w - new_w) / 2.0
        pad_y = (self.dst_h - new_h) / 2.0
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        # Repaint the whole canvas, not just the four border strips: this runs
        # once per source resolution, and a partial repaint is one more place
        # for an off-by-one to leave a stale stripe in the model input.
        self.canvas[:, :, :] = self.color
        self._roi = self.canvas[top : top + new_h, left : left + new_w]
        self._interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        self._new_size = (new_w, new_h)
        self._geometry = (src_w, src_h)
        self._transform = LetterboxTransform(
            scale=scale,
            pad_x=float(left),
            pad_y=float(top),
            src_w=src_w,
            src_h=src_h,
            dst_w=self.dst_w,
            dst_h=self.dst_h,
        )

    def convert(self, frame: np.ndarray) -> tuple[np.ndarray, LetterboxTransform]:
        """BGR frame -> the reused uint8 RGB canvas plus its transform.

        The returned array is the canvas itself and is overwritten by the next
        call, which is safe for the detector's frame loop (inference consumes it
        before the next frame is read) and is the whole point of the class.
        """
        src_h, src_w = frame.shape[:2]
        if self._geometry != (src_w, src_h):
            self._reshape_for(src_w, src_h)
        assert self._roi is not None and self._transform is not None
        # dst= writes into the canvas ROI. The view is row-strided but each row
        # is contiguous, which is exactly what OpenCV's step-based Mat wants.
        cv2.resize(frame, self._new_size, dst=self._roi, interpolation=self._interp)
        # In-place channel swap on the ROI only. The pad is grey in both orders.
        cv2.cvtColor(self._roi, cv2.COLOR_BGR2RGB, dst=self._roi)
        return self.canvas, self._transform
