"""Coordinate-space tests: the letterbox inverse must be exact on 16:9 input.

The RK platform has a second failure mode the CPU platform does not: the
scaling is done by RGA during decode on the hardware path and by OpenCV on the
software path. If those disagree, the same scene yields different published
coordinates depending on which decoder happened to start, so the geometry is
pinned here rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from esk_rknn.letterbox import (
    LetterboxTransform,
    copy_strided_to_canvas,
    fit_geometry,
    frame_norm_to_pixels,
    pad_into_canvas,
    xyxy_to_frame_norm,
)


def test_720p_fits_with_symmetric_vertical_pad():
    scaled_w, scaled_h, pad_x, pad_y = fit_geometry(1280, 720, 640)
    assert (scaled_w, scaled_h) == (640, 360)
    assert (pad_x, pad_y) == (0, 140)


def test_transform_matches_fit_geometry():
    tf = LetterboxTransform.for_source(1280, 720, 640)
    assert tf.scale_x == pytest.approx(0.5)
    assert tf.scale_y == pytest.approx(0.5)
    assert tf.pad_x == 0.0
    assert tf.pad_y == pytest.approx(140.0)
    assert tf.active_region == (0, 140, 640, 360)


@pytest.mark.parametrize("size", [(720, 1280), (1080, 1920), (480, 640), (600, 400), (721, 1281)])
def test_roundtrip_pixel_box_survives_letterbox(size):
    """A pixel box mapped forward then inverted lands back within a pixel."""
    height, width = size
    tf = LetterboxTransform.for_source(width, height, 640)

    box_px = (0.25 * width, 0.30 * height, 0.40 * width, 0.85 * height)
    forward = (
        box_px[0] * tf.scale_x + tf.pad_x,
        box_px[1] * tf.scale_y + tf.pad_y,
        box_px[2] * tf.scale_x + tf.pad_x,
        box_px[3] * tf.scale_y + tf.pad_y,
    )
    bbox = xyxy_to_frame_norm(forward, tf)
    x1, y1, x2, y2 = frame_norm_to_pixels(bbox, width, height)
    assert (x1, y1, x2, y2) == pytest.approx(box_px, abs=1.0)


def test_aspect_ratio_is_not_squashed():
    """A square object in a 16:9 frame must stay square in frame_norm pixels.

    This is the regression a plain resize-to-square introduces: the published
    box comes back stretched by 16/9 in x. On a 1280x720 source with a 640x640
    input the pad is 140 px top and bottom, so skipping the inverse shortens the
    box by 261 px vertically while x still looks plausible.
    """
    tf = LetterboxTransform.for_source(1280, 720, 640)
    side_px = 200.0
    x1 = 500.0 * tf.scale_x + tf.pad_x
    y1 = 250.0 * tf.scale_y + tf.pad_y
    forward = (x1, y1, x1 + side_px * tf.scale_x, y1 + side_px * tf.scale_y)
    bbox = xyxy_to_frame_norm(forward, tf)
    px1, py1, px2, py2 = frame_norm_to_pixels(bbox, 1280, 720)
    assert (px2 - px1) == pytest.approx(side_px, abs=1.0)
    assert (py2 - py1) == pytest.approx(side_px, abs=1.0)


def test_bbox_values_are_json_serializable_floats():
    """numpy scalars must not reach the payload -- json.dumps cannot encode them."""
    import json

    tf = LetterboxTransform.for_source(1280, 720, 640)
    box = tuple(np.array([100.0, 50.0, 300.0, 400.0], dtype=np.float32))
    bbox = xyxy_to_frame_norm(box, tf)
    assert all(type(v) is float for v in bbox)
    json.dumps({"bbox": bbox})


def test_bbox_clamped_into_unit_range():
    """Boxes hanging off the frame edge stay schema-valid (items in [0, 1])."""
    tf = LetterboxTransform.for_source(1280, 720, 640)
    bbox = xyxy_to_frame_norm((-50.0, -50.0, 900.0, 900.0), tf)
    assert all(0.0 <= v <= 1.0 for v in bbox)
    cx, cy, w, h = bbox
    assert cx - w / 2 >= -1e-9 and cx + w / 2 <= 1.0 + 1e-9
    assert cy - h / 2 >= -1e-9 and cy + h / 2 <= 1.0 + 1e-9


def test_pad_into_canvas_centres_and_fills():
    scaled = np.full((360, 640, 3), 7, dtype=np.uint8)
    canvas = pad_into_canvas(scaled, 640)
    assert canvas.shape == (640, 640, 3)
    assert (canvas[140:500] == 7).all()
    assert (canvas[:140] == 114).all() and (canvas[500:] == 114).all()


def test_strided_copy_respects_row_alignment():
    """MPP/RGA aligns output rows; a tightly packed read shears the image.

    The buffer below is 640 px wide with a 704-byte-per-row-of-pixels stride
    (the alignment MPP actually applies). Reading it as if it were packed walks
    diagonally through the image, which is exactly the silent corruption this
    guards: the frame still looks like a frame, and the detector simply stops
    finding anything.
    """
    width, height, stride = 640, 360, 640 * 3 + 64
    row = np.arange(width, dtype=np.uint8)
    buffer = bytearray(stride * height)
    for y in range(height):
        base = y * stride
        for c in range(3):
            buffer[base + c: base + width * 3: 3] = bytes(row)

    canvas = copy_strided_to_canvas(bytes(buffer), width, height, stride, 640)
    assert canvas.shape == (640, 640, 3)
    # Every row of the active region must be identical -- shearing would make
    # each row start at a different offset.
    active = canvas[140:500, :, 0]
    assert (active == row[None, :]).all()


def test_strided_copy_rejects_a_short_buffer():
    with pytest.raises(ValueError):
        copy_strided_to_canvas(bytes(10), 640, 360, 640 * 3, 640)


def test_hardware_and_software_paths_agree_on_geometry():
    """The RGA output size and the OpenCV fallback must land on one geometry.

    ``fit_geometry`` is the single source of truth for both; this pins that they
    are not allowed to drift, because a one-pixel pad difference silently
    changes every published coordinate when a decoder falls back.
    """
    for src in ((1280, 720), (1920, 1080), (640, 480), (1281, 721)):
        scaled_w, scaled_h, pad_x, pad_y = fit_geometry(*src, 640)
        tf = LetterboxTransform.for_source(*src, 640)
        assert tf.active_region == (pad_x, pad_y, scaled_w, scaled_h)
        assert scaled_w <= 640 and scaled_h <= 640
        assert max(scaled_w, scaled_h) == 640
