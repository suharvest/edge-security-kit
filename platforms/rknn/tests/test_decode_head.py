"""The YOLOv8 head decode is shared with the generic platform in behaviour.

The RKNN model keeps the stock head, so a synthetic activation must produce the
same box either way. These run on the host: no NPU, no board.
"""

from __future__ import annotations

import numpy as np
import pytest

from esk_rknn.letterbox import LetterboxTransform
from esk_rknn.rknn_yolo import decode_person_head, nms


def _head_with_one_person(cx, cy, w, h, score, n_anchors=8400, n_classes=80):
    raw = np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)
    raw[0, 0, 0], raw[0, 1, 0], raw[0, 2, 0], raw[0, 3, 0] = cx, cy, w, h
    raw[0, 4, 0] = score  # class 0 == person
    return raw


def test_decode_maps_a_centred_box_back_through_the_letterbox():
    tf = LetterboxTransform.for_source(1280, 720, 640)
    # A person 100x300 model-input px at the canvas centre.
    raw = _head_with_one_person(320.0, 320.0, 100.0, 300.0, 0.9)
    dets = decode_person_head(raw, tf, conf_threshold=0.35, iou_threshold=0.45)
    assert len(dets) == 1
    x1, y1, x2, y2 = dets[0].box
    # Model input centre y=320 sits 180 px below the pad -> 360 px in source.
    assert ((x1 + x2) / 2) * 1280 == pytest.approx(640.0)
    assert ((y1 + y2) / 2) * 720 == pytest.approx(360.0)
    assert (x2 - x1) * 1280 == pytest.approx(200.0)   # 100 px / scale 0.5
    assert (y2 - y1) * 720 == pytest.approx(600.0)


def test_scores_below_threshold_are_dropped():
    tf = LetterboxTransform.for_source(1280, 720, 640)
    raw = _head_with_one_person(320.0, 320.0, 100.0, 300.0, 0.10)
    assert decode_person_head(raw, tf, 0.35, 0.45) == []


def test_only_class_zero_is_read():
    """A high score on a non-person class must not produce a detection."""
    tf = LetterboxTransform.for_source(1280, 720, 640)
    raw = _head_with_one_person(320.0, 320.0, 100.0, 300.0, 0.0)
    raw[0, 4 + 15, 0] = 0.99  # 'cat'
    assert decode_person_head(raw, tf, 0.35, 0.45) == []


def test_transposed_head_layout_is_accepted():
    """(1, N, 4+nc) must decode identically to (1, 4+nc, N)."""
    tf = LetterboxTransform.for_source(1280, 720, 640)
    raw = _head_with_one_person(320.0, 320.0, 100.0, 300.0, 0.9)
    a = decode_person_head(raw, tf, 0.35, 0.45)
    b = decode_person_head(np.ascontiguousarray(raw.transpose(0, 2, 1)), tf, 0.35, 0.45)
    assert a[0].box == b[0].box and a[0].score == b[0].score


def test_published_box_components_are_builtin_floats():
    import json

    tf = LetterboxTransform.for_source(1280, 720, 640)
    raw = _head_with_one_person(320.0, 320.0, 100.0, 300.0, 0.9)
    det = decode_person_head(raw, tf, 0.35, 0.45)[0]
    assert all(type(v) is float for v in det.box)
    assert type(det.score) is float
    json.dumps({"box": det.box, "score": det.score})


def test_nms_suppresses_a_duplicate_box():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]])
    scores = np.array([0.9, 0.8])
    assert nms(boxes, scores, 0.45) == [0]


def test_nms_keeps_two_disjoint_boxes():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]])
    scores = np.array([0.9, 0.8])
    assert sorted(nms(boxes, scores, 0.45)) == [0, 1]
