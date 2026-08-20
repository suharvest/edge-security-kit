"""The split-head decoder, checked against a hand-built head.

No NPU and no HEF: the layout the compiled graph produces is fully specified --
per stride, a 64-channel DFL box distribution and an 80-channel class map, both
NHWC, with sigmoid already applied on-chip -- so a synthetic head with one
planted box pins the arithmetic. What this catches is the class of bug that
does not raise: channel order inside the DFL block, stride derivation from the
grid, the sign convention on the two DFL halves, and the branch pairing. Each
of those yields a plausible box in the wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest

from esk_hailo.hailo_yolo import (
    BOX_CHANNELS,
    DFL_BINS,
    decode_split_head,
    group_branches,
)
from esk_hailo.letterbox import letterbox

STRIDES = (8, 16, 32)
INPUT = 640


def empty_head(input_size: int = INPUT) -> list[np.ndarray]:
    """Six NHWC outputs with no object anywhere: class maps at probability 0."""
    outputs: list[np.ndarray] = []
    for stride in STRIDES:
        grid = input_size // stride
        outputs.append(np.zeros((grid, grid, BOX_CHANNELS), np.float32))
        outputs.append(np.zeros((grid, grid, 80), np.float32))
    return outputs


def plant(outputs, branch: int, row: int, col: int, score: float, dist: int) -> None:
    """Put one person at (row, col) with all four edges ``dist`` cells away.

    A one-hot DFL distribution makes the expected distance exactly the bin
    index, so the box is analytically known and the test does not re-implement
    the softmax it is testing.
    """
    outputs[2 * branch + 1][row, col, 0] = score
    for edge in range(4):
        outputs[2 * branch][row, col, edge * DFL_BINS + dist] = 50.0


def tf_for(width: int, height: int):
    _, transform = letterbox(np.zeros((height, width, 3), np.uint8), INPUT, INPUT)
    return transform


def test_empty_head_yields_nothing():
    assert decode_split_head(empty_head(), tf_for(640, 640), 0.35, 0.45, INPUT) == []


@pytest.mark.parametrize(
    "branch,row,col,dist", [(0, 40, 40, 3), (1, 20, 25, 5), (2, 10, 10, 2)]
)
def test_box_lands_where_the_grid_cell_says(branch, row, col, dist):
    outputs = empty_head()
    plant(outputs, branch, row, col, 0.9, dist)
    detections = decode_split_head(outputs, tf_for(640, 640), 0.35, 0.45, INPUT)
    assert len(detections) == 1

    stride = STRIDES[branch]
    centre_x = (col + 0.5) * stride
    centre_y = (row + 0.5) * stride
    expected = [
        (centre_x - dist * stride) / INPUT,
        (centre_y - dist * stride) / INPUT,
        (centre_x + dist * stride) / INPUT,
        (centre_y + dist * stride) / INPUT,
    ]
    assert detections[0].box == pytest.approx(expected, abs=1e-3)
    assert detections[0].score == pytest.approx(0.9, abs=1e-6)


def test_only_the_person_channel_is_read():
    """A high score on class 1 must not become a person detection."""
    outputs = empty_head()
    outputs[1][40, 40, 1] = 0.99  # bicycle
    for edge in range(4):
        outputs[0][40, 40, edge * DFL_BINS + 3] = 50.0
    assert decode_split_head(outputs, tf_for(640, 640), 0.35, 0.45, INPUT) == []


def test_output_order_does_not_matter():
    """Branches are paired by shape, so a reordered outputs dict is harmless.

    HailoRT keys outputs by layer name and those names change on every
    re-translation of the graph; a decoder that depended on their order would
    break silently at the next recompile.
    """
    outputs = empty_head()
    plant(outputs, 1, 20, 25, 0.8, 4)
    straight = decode_split_head(outputs, tf_for(640, 640), 0.35, 0.45, INPUT)
    shuffled = decode_split_head(
        list(reversed(outputs)), tf_for(640, 640), 0.35, 0.45, INPUT
    )
    assert len(shuffled) == 1
    assert shuffled[0].box == pytest.approx(straight[0].box, abs=1e-9)


def test_batched_outputs_are_accepted():
    outputs = empty_head()
    plant(outputs, 0, 40, 40, 0.9, 3)
    flat = decode_split_head(outputs, tf_for(640, 640), 0.35, 0.45, INPUT)
    batched = decode_split_head(
        [o[None] for o in outputs], tf_for(640, 640), 0.35, 0.45, INPUT
    )
    assert batched[0].box == pytest.approx(flat[0].box, abs=1e-9)


def test_letterboxed_source_inverts_the_pad():
    """On a 1280x720 source the pad has to come back off, as in production.

    A square fixture would pass with the inverse missing entirely, which is why
    the deployed config points at a 16:9 stream.
    """
    outputs = empty_head()
    plant(outputs, 0, 40, 40, 0.9, 3)
    tf = tf_for(1280, 720)
    box = decode_split_head(outputs, tf, 0.35, 0.45, INPUT)[0].box
    # Model-space centre is (324, 324); undo pad then scale to get source px.
    cx_src, cy_src = tf.to_source(324.0, 324.0)
    assert (box[0] + box[2]) / 2 == pytest.approx(cx_src / 1280, abs=1e-4)
    assert (box[1] + box[3]) / 2 == pytest.approx(cy_src / 720, abs=1e-4)
    assert box[1] >= 0.0 and box[3] <= 1.0


def test_aspect_ratio_survives_the_inverse():
    """A square box in model space must stay square in source pixels.

    The pad is vertical on a 16:9 source, so an inverse that divided x and y by
    different factors -- or forgot the pad -- shows up here as a rectangle.
    """
    outputs = empty_head()
    plant(outputs, 1, 20, 25, 0.9, 4)
    tf = tf_for(1280, 720)
    box = decode_split_head(outputs, tf, 0.35, 0.45, INPUT)[0].box
    width_px = (box[2] - box[0]) * 1280
    height_px = (box[3] - box[1]) * 720
    assert width_px == pytest.approx(height_px, rel=1e-3)


def test_mismatched_branch_shapes_raise():
    """A head the decoder cannot pair must fail loudly, not decode half of it."""
    outputs = empty_head()[:4]
    with pytest.raises(ValueError):
        group_branches(outputs)


def test_two_box_tensors_on_one_grid_raise():
    grid = INPUT // 8
    with pytest.raises(ValueError):
        group_branches(
            [
                np.zeros((grid, grid, BOX_CHANNELS), np.float32),
                np.zeros((grid, grid, BOX_CHANNELS), np.float32),
            ]
        )


def test_nms_collapses_duplicates_on_one_object():
    """Two adjacent cells describing the same box must publish one detection."""
    outputs = empty_head()
    plant(outputs, 0, 40, 40, 0.9, 6)
    plant(outputs, 0, 40, 41, 0.8, 6)
    detections = decode_split_head(outputs, tf_for(640, 640), 0.35, 0.45, INPUT)
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9, abs=1e-6)


def test_scores_are_builtin_floats():
    """numpy scalars must not reach the payload -- json.dumps cannot encode them."""
    import json

    outputs = empty_head()
    plant(outputs, 0, 40, 40, 0.9, 3)
    det = decode_split_head(outputs, tf_for(1280, 720), 0.35, 0.45, INPUT)[0]
    assert type(det.score) is float
    assert all(type(v) is float for v in det.box)
    json.dumps({"box": det.box, "score": det.score})
