"""Coordinate-space tests: the letterbox inverse must be exact on 16:9 input."""

from __future__ import annotations

import numpy as np
import pytest

from esk_hailo.letterbox import (
    LetterboxCanvas,
    frame_norm_to_pixels,
    letterbox,
    xyxy_to_frame_norm,
)


def test_letterbox_preserves_aspect_and_pads_to_square():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    padded, tf = letterbox(frame, 640, 640)
    assert padded.shape[:2] == (640, 640)
    assert tf.scale == pytest.approx(0.5)
    assert tf.pad_x == 0.0
    assert tf.pad_y == pytest.approx(140.0, abs=1.0)  # (640 - 360) / 2


@pytest.mark.parametrize("size", [(720, 1280), (1080, 1920), (480, 640), (600, 400)])
def test_roundtrip_pixel_box_survives_letterbox(size):
    """A pixel box mapped forward then inverted lands back within a pixel."""
    height, width = size
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _, tf = letterbox(frame, 640, 640)

    box_px = (0.25 * width, 0.30 * height, 0.40 * width, 0.85 * height)
    forward = tuple(
        v * tf.scale + (tf.pad_x if i % 2 == 0 else tf.pad_y)
        for i, v in enumerate(box_px)
    )
    bbox = xyxy_to_frame_norm(forward, tf)
    x1, y1, x2, y2 = frame_norm_to_pixels(bbox, width, height)
    assert (x1, y1, x2, y2) == pytest.approx(box_px, abs=1.0)


def test_aspect_ratio_is_not_squashed():
    """A square object in a 16:9 frame must stay square in frame_norm pixels.

    This is the regression that a naive blobFromImage resize introduces: the
    published box comes back stretched by 16/9 in x.
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, tf = letterbox(frame, 640, 640)
    side_px = 200.0
    x1 = 500.0 * tf.scale + tf.pad_x
    y1 = 250.0 * tf.scale + tf.pad_y
    forward = (x1, y1, x1 + side_px * tf.scale, y1 + side_px * tf.scale)
    bbox = xyxy_to_frame_norm(forward, tf)
    px1, py1, px2, py2 = frame_norm_to_pixels(bbox, 1280, 720)
    assert (px2 - px1) == pytest.approx(px2 - px1)
    assert (px2 - px1) == pytest.approx(side_px, abs=1.0)
    assert (py2 - py1) == pytest.approx(side_px, abs=1.0)


def test_bbox_values_are_json_serializable_floats():
    """numpy scalars must not reach the payload -- json.dumps cannot encode them."""
    import json

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, tf = letterbox(frame, 640, 640)
    box = tuple(np.array([100.0, 50.0, 300.0, 400.0], dtype=np.float32))
    bbox = xyxy_to_frame_norm(box, tf)
    assert all(type(v) is float for v in bbox)
    json.dumps({"bbox": bbox})


def test_bbox_clamped_into_unit_range():
    """Boxes hanging off the frame edge stay schema-valid (items in [0, 1])."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, tf = letterbox(frame, 640, 640)
    bbox = xyxy_to_frame_norm((-50.0, -50.0, 900.0, 900.0), tf)
    assert all(0.0 <= v <= 1.0 for v in bbox)
    cx, cy, w, h = bbox
    assert cx - w / 2 >= -1e-9 and cx + w / 2 <= 1.0 + 1e-9
    assert cy - h / 2 >= -1e-9 and cy + h / 2 <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# LetterboxCanvas: byte-identical to letterbox() + the BGR->RGB flip
# --------------------------------------------------------------------------


def test_canvas_matches_letterbox_byte_for_byte():
    """The reused canvas must equal the allocating path exactly, not closely.

    The optimization removes two 1.2 MB allocations and a full copy per frame.
    Anything short of byte equality here means the model sees different input
    than the path every published measurement was taken on, and every downstream
    difference would then be unattributable.
    """
    rng = np.random.default_rng(7)
    for src_w, src_h in ((1280, 720), (640, 640), (1920, 1080), (320, 240)):
        frame = rng.integers(0, 256, size=(src_h, src_w, 3), dtype=np.uint8)
        padded, want_tf = letterbox(frame, 640, 640)
        want = np.ascontiguousarray(padded[:, :, ::-1])

        canvas = LetterboxCanvas(640, 640)
        got, got_tf = canvas.convert(frame)

        assert got.shape == want.shape and got.dtype == want.dtype
        assert np.array_equal(got, want), f"{src_w}x{src_h} canvas differs"
        assert got_tf == want_tf


def test_canvas_reuses_one_buffer_and_survives_a_resolution_change():
    """Same array object every call, and no stale pixels after a reshape."""
    rng = np.random.default_rng(11)
    canvas = LetterboxCanvas(640, 640)

    wide = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    first, _ = canvas.convert(wide)
    second, _ = canvas.convert(wide)
    assert first is second is canvas.canvas

    # A taller source pads left/right instead of top/bottom. If the pad were
    # repainted only where the previous geometry had it, the old top band would
    # survive as image data.
    tall = rng.integers(0, 256, size=(1080, 720, 3), dtype=np.uint8)
    got, _ = canvas.convert(tall)
    padded, _ = letterbox(tall, 640, 640)
    assert np.array_equal(got, np.ascontiguousarray(padded[:, :, ::-1]))
