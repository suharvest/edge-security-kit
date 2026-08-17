"""The model-zoo head decoder, checked against a hand-built head.

No NPU and no model: the head layout is fully specified (per stride, a
64-channel DFL box distribution, an 80-channel sigmoid class map and a
1-channel score sum), so a synthetic head with one planted box is enough to pin
the arithmetic. What this catches is the class of bug that does not raise --
axis order, stride derivation, and the sign convention on the two DFL halves,
each of which yields a plausible box in the wrong place.
"""

from __future__ import annotations

import numpy as np
import pytest

from esk_rknn.letterbox import LetterboxTransform
from esk_rknn.rknn_yolo import DFL_BINS, decode_zoo_head

STRIDES = (8, 16, 32)
INPUT = 640


def empty_head(input_size: int = INPUT) -> list[np.ndarray]:
    """Nine outputs with no object anywhere: class maps at zero probability."""
    outputs: list[np.ndarray] = []
    for stride in STRIDES:
        grid = input_size // stride
        outputs.append(np.zeros((1, 4 * DFL_BINS, grid, grid), np.float32))
        outputs.append(np.zeros((1, 80, grid, grid), np.float32))
        outputs.append(np.zeros((1, 1, grid, grid), np.float32))
    return outputs


def plant(outputs, branch: int, row: int, col: int, score: float, dist: float):
    """Put one person at (row, col) with all four edges ``dist`` cells away.

    A one-hot DFL distribution makes the expected distance exactly the index,
    so the resulting box is analytically known and the test does not have to
    re-implement the softmax it is testing.
    """
    outputs[3 * branch + 1][0, 0, row, col] = score
    for edge in range(4):
        # Large positive logit in one bin, zeros elsewhere -> softmax mass ~1
        # on that bin, so the expectation is that bin's index.
        outputs[3 * branch][0, edge * DFL_BINS + int(dist), row, col] = 50.0


def square_tf() -> LetterboxTransform:
    """A 640x640 source: no padding, unit scale, so model px == source px."""
    return LetterboxTransform.for_source(INPUT, INPUT, INPUT)


def test_empty_head_yields_nothing():
    assert decode_zoo_head(empty_head(), square_tf(), 0.35, 0.45, INPUT) == []


@pytest.mark.parametrize("branch,row,col,dist", [(0, 40, 40, 3), (1, 20, 25, 5), (2, 10, 10, 2)])
def test_box_lands_where_the_grid_cell_says(branch, row, col, dist):
    outputs = empty_head()
    plant(outputs, branch, row, col, 0.9, dist)
    detections = decode_zoo_head(outputs, square_tf(), 0.35, 0.45, INPUT)
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
    outputs[1][0, 1, 40, 40] = 0.99  # bicycle
    for edge in range(4):
        outputs[0][0, edge * DFL_BINS + 3, 40, 40] = 50.0
    assert decode_zoo_head(outputs, square_tf(), 0.35, 0.45, INPUT) == []


def test_nhwc_outputs_decode_identically():
    """A runtime that hands back NHWC must not silently shift every box."""
    outputs = empty_head()
    plant(outputs, 0, 40, 40, 0.9, 3)
    nchw = decode_zoo_head(outputs, square_tf(), 0.35, 0.45, INPUT)
    nhwc = decode_zoo_head(
        [np.ascontiguousarray(o.transpose(0, 2, 3, 1)) for o in outputs],
        square_tf(), 0.35, 0.45, INPUT,
    )
    assert len(nhwc) == 1
    assert nhwc[0].box == pytest.approx(nchw[0].box, abs=1e-6)


def test_letterboxed_source_inverts_the_pad():
    """On a 1280x720 source the pad has to come back off, as in production."""
    outputs = empty_head()
    plant(outputs, 0, 40, 40, 0.9, 3)
    tf = LetterboxTransform.for_source(1280, 720, INPUT)
    box = decode_zoo_head(outputs, tf, 0.35, 0.45, INPUT)[0].box
    # Model-space centre is (324, 324); undo pad then scale to get source px.
    cx_src, cy_src = tf.to_source(324.0, 324.0)
    assert (box[0] + box[2]) / 2 == pytest.approx(cx_src / 1280, abs=1e-4)
    assert (box[1] + box[3]) / 2 == pytest.approx(cy_src / 720, abs=1e-4)
    assert box[1] >= 0.0 and box[3] <= 1.0


def test_two_per_branch_export_is_accepted():
    """The six-output export omits the score sum; the decoder ignores it anyway."""
    outputs = empty_head()
    plant(outputs, 1, 20, 25, 0.8, 4)
    six = [o for k, o in enumerate(outputs) if k % 3 != 2]
    nine = decode_zoo_head(outputs, square_tf(), 0.35, 0.45, INPUT)
    assert decode_zoo_head(six, square_tf(), 0.35, 0.45, INPUT)[0].box == pytest.approx(
        nine[0].box, abs=1e-6
    )
