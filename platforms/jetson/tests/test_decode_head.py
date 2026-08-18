"""The ultralytics detect head decodes to the same boxes on every platform.

This platform's decoder is the same function as ``platforms/generic``'s
postprocess, deliberately: ``coordinate_space: frame_norm`` is a contract
promise, and a platform carrying its own private decoder is a platform where a
coordinate bug can hide behind "well, it's a different backend".

The tests below build a synthetic ``(1, 84, 8400)`` head by hand, so they run on
any host with NumPy -- no engine, no GPU, no Jetson.
"""

from __future__ import annotations

import numpy as np
import pytest

from esk_jetson.letterbox import LetterboxTransform
from esk_jetson.trt_yolo import PERSON_CLASS_ID, decode_person_head

INPUT = 640
SRC_W, SRC_H = 1280, 720
ANCHORS = 8400
FEATURES = 84


def make_head(boxes_scores, layout: str = "feature_major") -> np.ndarray:
    """``[(cx, cy, w, h, person_score), ...]`` in model-input pixels -> a head."""
    pred = np.zeros((FEATURES, ANCHORS), dtype=np.float32)
    for index, (cx, cy, w, h, score) in enumerate(boxes_scores):
        pred[0, index], pred[1, index] = cx, cy
        pred[2, index], pred[3, index] = w, h
        pred[4 + PERSON_CLASS_ID, index] = score
    if layout == "anchor_major":
        return pred.T[None]
    return pred[None]


def transform() -> LetterboxTransform:
    return LetterboxTransform.for_source(SRC_W, SRC_H, INPUT)


def test_letterbox_inverse_is_applied_to_the_published_box():
    """A box high in the frame must come back at its SOURCE height, not the canvas one.

    With a 1280x720 source the letterbox puts 140 px of padding above and below
    the image. A decoder that skipped the inverse would report this box at
    ``cy_input / 640`` = 0.297 instead of 0.139 -- a plausible-looking number in
    the right range, which is exactly why this needs an assertion rather than an
    eyeball.
    """
    tf = transform()
    target_y_src, height_src = 100.0, 80.0    # source pixels, clear of both edges
    cx_input = 320.0
    cy_input = tf.pad_y + target_y_src * tf.scale_y
    head = make_head([(cx_input, cy_input, 100.0, height_src * tf.scale_y, 0.9)])
    detections = decode_person_head(head, tf, 0.35, 0.45)
    assert len(detections) == 1
    x1, y1, x2, y2 = detections[0].box
    assert (x1 + x2) / 2 == pytest.approx(0.5, abs=1e-3)
    assert (y1 + y2) / 2 == pytest.approx(target_y_src / SRC_H, abs=2e-3)
    assert (y2 - y1) == pytest.approx(height_src / SRC_H, abs=2e-3)
    # The naive no-inverse answer, asserted against so the test cannot pass by
    # accident if the two happened to converge.
    assert (y1 + y2) / 2 != pytest.approx(cy_input / INPUT, abs=1e-2)


def test_box_straddling_the_pad_edge_is_clamped_to_the_frame():
    """A detection reaching into the padding must not report a negative edge.

    The schema requires the box to lie inside the frame; clamping happens in
    source-pixel space before the centre/size conversion, so the reported centre
    moves with the clamp rather than describing a box that reaches outside.
    """
    tf = transform()
    head = make_head([(320.0, tf.pad_y + 5.0, 100.0, 120.0, 0.9)])
    x1, y1, x2, y2 = decode_person_head(head, tf, 0.35, 0.45)[0].box
    assert y1 == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= y2 <= 1.0 and 0.0 <= x1 <= x2 <= 1.0


def test_aspect_ratio_survives_the_round_trip():
    """A square box in canvas pixels is NOT square in normalized source units.

    1280x720 into a 640 square scales both axes by the same factor, so a square
    of canvas pixels covers twice the normalized width as height. Getting this
    backwards is exactly what happens when a decoder normalizes against the
    padded canvas instead of the source frame, and it is invisible unless the
    source is non-square -- hence 1280x720 here and in the acceptance fixture.
    """
    tf = transform()
    side = 90.0
    head = make_head([(320.0, 320.0, side, side, 0.8)])
    detections = decode_person_head(head, tf, 0.35, 0.45)
    x1, y1, x2, y2 = detections[0].box
    width, height = x2 - x1, y2 - y1
    assert height / width == pytest.approx(SRC_W / SRC_H, rel=1e-3)


def test_both_tensor_layouts_decode_identically():
    boxes = [(300.0, 310.0, 60.0, 120.0, 0.77)]
    tf = transform()
    a = decode_person_head(make_head(boxes, "feature_major"), tf, 0.35, 0.45)
    b = decode_person_head(make_head(boxes, "anchor_major"), tf, 0.35, 0.45)
    assert len(a) == len(b) == 1
    assert a[0].box == pytest.approx(b[0].box)


def test_only_the_person_class_is_kept():
    tf = transform()
    pred = np.zeros((FEATURES, ANCHORS), dtype=np.float32)
    pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0] = 320.0, 320.0, 50.0, 50.0
    pred[4 + 15, 0] = 0.99  # a very confident cat
    assert decode_person_head(pred[None], tf, 0.35, 0.45) == []


def test_nms_collapses_duplicates_and_keeps_the_best():
    tf = transform()
    head = make_head(
        [
            (320.0, 320.0, 100.0, 200.0, 0.90),
            (322.0, 318.0, 100.0, 200.0, 0.70),   # heavy overlap -> suppressed
            (100.0, 320.0, 100.0, 200.0, 0.60),   # disjoint      -> kept
        ]
    )
    detections = decode_person_head(head, tf, 0.35, 0.45)
    assert [round(d.score, 2) for d in detections] == [0.90, 0.60]


def test_below_threshold_yields_nothing():
    tf = transform()
    head = make_head([(320.0, 320.0, 80.0, 80.0, 0.20)])
    assert decode_person_head(head, tf, 0.35, 0.45) == []


def test_published_values_are_builtin_floats():
    """json.dumps has no encoder for np.float32, and the failure is at publish."""
    tf = transform()
    head = make_head([(320.0, 320.0, 80.0, 160.0, 0.9)])
    detection = decode_person_head(head, tf, 0.35, 0.45)[0]
    assert all(type(v) is float for v in detection.box)
    assert type(detection.score) is float
