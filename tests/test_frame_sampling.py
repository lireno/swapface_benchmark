from __future__ import annotations

import pytest

from swapface_benchmark.metrics.identity_strict import sample_eval_indices


def test_short_protocol_keeps_first_81_consecutive_frames() -> None:
    assert sample_eval_indices(120, 81, False, 42, frame_stride=1) == list(range(81))


def test_long_protocol_uses_fixed_stride_ten() -> None:
    assert sample_eval_indices(35, 0, False, 42, frame_stride=10) == [0, 10, 20, 30]


def test_frame_limit_counts_sampled_candidates() -> None:
    assert sample_eval_indices(100, 3, False, 42, frame_stride=10) == [0, 10, 20]


def test_frame_stride_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        sample_eval_indices(10, 0, False, 42, frame_stride=0)
