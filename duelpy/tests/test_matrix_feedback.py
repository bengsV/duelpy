"""Tests for MatrixFeedback, including a reproducibility regression."""

import numpy as np
import pytest

from duelpy.feedback.matrix_feedback import MatrixFeedback
from duelpy.stats.preference_matrix import PreferenceMatrix


_PREF = np.array(
    [
        [0.5, 0.7, 0.8],
        [0.3, 0.5, 0.6],
        [0.2, 0.4, 0.5],
    ]
)


def test_reproducibility_duel():
    """Same seed must produce identical duel outcomes."""
    fb1 = MatrixFeedback(_PREF, random_state=np.random.RandomState(42))
    fb2 = MatrixFeedback(_PREF, random_state=np.random.RandomState(42))
    results1 = [fb1.duel(0, 1) for _ in range(20)]
    results2 = [fb2.duel(0, 1) for _ in range(20)]
    assert results1 == results2


def test_reproducibility_duel_repeatedly():
    """duel_repeatedly must use self.random_state and be reproducible."""
    fb1 = MatrixFeedback(_PREF, random_state=np.random.RandomState(7))
    fb2 = MatrixFeedback(_PREF, random_state=np.random.RandomState(7))
    wins1 = fb1.duel_repeatedly(0, 1, 1000)
    wins2 = fb2.duel_repeatedly(0, 1, 1000)
    assert wins1 == wins2


def test_duel_repeatedly_rate_close_to_preference():
    """Win rate from duel_repeatedly should be close to the preference value."""
    fb = MatrixFeedback(_PREF, random_state=np.random.RandomState(0))
    wins = fb.duel_repeatedly(0, 1, 10_000)
    rate = wins / 10_000
    assert abs(rate - 0.7) < 0.02


def test_accepts_preference_matrix_object():
    pm = PreferenceMatrix(_PREF)
    fb = MatrixFeedback(pm)
    result = fb.duel(0, 2)
    assert isinstance(result, (bool, np.bool_))


def test_arm_count_mismatch_raises():
    with pytest.raises(ValueError):
        MatrixFeedback(_PREF, arms=[0, 1])


def test_default_arms_are_range():
    fb = MatrixFeedback(_PREF)
    assert fb.get_arms() == [0, 1, 2]


def test_custom_arms_label():
    fb = MatrixFeedback(_PREF, arms=["a", "b", "c"])
    assert fb.get_arms() == ["a", "b", "c"]
