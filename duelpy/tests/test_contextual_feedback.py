"""Tests for LinearContextFeedback."""

import numpy as np
import pytest

from duelpy.feedback.linear_context_feedback import LinearContextFeedback


_PARAMS = np.array(
    [
        [1.0, 0.0],  # arm 0: u = x[0]
        [0.0, 1.0],  # arm 1: u = x[1]
        [0.5, 0.5],  # arm 2: u = 0.5*(x[0]+x[1])
    ]
)
_X0 = np.array([1.0, 0.0])  # favours arm 0
_X1 = np.array([0.0, 1.0])  # favours arm 1


class TestLinearContextFeedbackStep:
    def test_get_best_arm_context_0(self):
        fb = LinearContextFeedback(_PARAMS, link="step")
        assert fb.get_best_arm(_X0) == 0

    def test_get_best_arm_context_1(self):
        fb = LinearContextFeedback(_PARAMS, link="step")
        assert fb.get_best_arm(_X1) == 1

    def test_duel_correct_winner(self):
        fb = LinearContextFeedback(_PARAMS, link="step")
        assert fb.duel(0, 1, _X0) is True   # arm 0 better in context x0
        assert fb.duel(1, 0, _X1) is True   # arm 1 better in context x1

    def test_duel_loser_returns_false(self):
        fb = LinearContextFeedback(_PARAMS, link="step")
        assert fb.duel(1, 0, _X0) is False

    def test_reproducibility(self):
        fb1 = LinearContextFeedback(_PARAMS, link="logistic",
                                     random_state=np.random.RandomState(42))
        fb2 = LinearContextFeedback(_PARAMS, link="logistic",
                                     random_state=np.random.RandomState(42))
        x = np.array([0.6, 0.4])
        r1 = [fb1.duel(0, 1, x) for _ in range(20)]
        r2 = [fb2.duel(0, 1, x) for _ in range(20)]
        assert r1 == r2

    def test_invalid_link_raises(self):
        with pytest.raises(ValueError):
            LinearContextFeedback(_PARAMS, link="softmax")


class TestLinearContextFeedbackInterface:
    def test_get_num_arms(self):
        fb = LinearContextFeedback(_PARAMS)
        assert fb.get_num_arms() == 3

    def test_get_context_dim(self):
        fb = LinearContextFeedback(_PARAMS)
        assert fb.get_context_dim() == 2

    def test_get_arms(self):
        fb = LinearContextFeedback(_PARAMS)
        assert fb.get_arms() == [0, 1, 2]

    def test_arm_parameters_are_copy(self):
        fb = LinearContextFeedback(_PARAMS)
        params = fb.arm_parameters
        params[0, 0] = 999.0
        assert fb.get_best_arm(_X0) == 0  # unchanged
