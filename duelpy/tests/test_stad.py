"""Tests for the StaD algorithm."""

import numpy as np
import pytest

from duelpy.algorithms.stad import StaD
from duelpy.feedback.linear_context_feedback import LinearContextFeedback


def _make_feedback(seed: int = 0) -> LinearContextFeedback:
    arm_params = np.array([
        [1.0, 0.0],   # arm 0: best for x ~ e_0
        [0.0, 1.0],   # arm 1: best for x ~ e_1
        [0.3, 0.3],   # arm 2: mediocre
    ])
    return LinearContextFeedback(
        arm_params, link="logistic", random_state=np.random.RandomState(seed)
    )


def _constant_context(x: np.ndarray):
    return lambda: x.copy()


class TestStaDBasic:
    def test_runs_without_error(self):
        fb = _make_feedback()
        alg = StaD(fb, time_horizon=20, failure_probability=0.1)
        rng = np.random.RandomState(0)
        alg.run(lambda: rng.randn(2))

    def test_duels_at_most_time_horizon(self):
        fb = _make_feedback()
        alg = StaD(fb, time_horizon=15)
        rng = np.random.RandomState(1)
        alg.run(lambda: rng.randn(2))
        assert alg._steps_taken <= 15

    def test_exactly_time_horizon_steps(self):
        fb = _make_feedback()
        alg = StaD(fb, time_horizon=10)
        rng = np.random.RandomState(2)
        alg.run(lambda: rng.randn(2))
        assert alg._steps_taken == 10

    def test_is_finished_after_run(self):
        fb = _make_feedback()
        alg = StaD(fb, time_horizon=8)
        rng = np.random.RandomState(3)
        alg.run(lambda: rng.randn(2))
        assert alg.is_finished()

    def test_get_copeland_winner_returns_valid_arm(self):
        fb = _make_feedback()
        alg = StaD(fb, time_horizon=20)
        rng = np.random.RandomState(4)
        alg.run(lambda: rng.randn(2))
        winner = alg.get_copeland_winner()
        assert winner is not None
        assert 0 <= winner < fb.get_num_arms()

    def test_get_copeland_winner_none_before_run(self):
        fb = _make_feedback()
        alg = StaD(fb, time_horizon=10)
        assert alg.get_copeland_winner() is None


class TestStaDCorrectness:
    def test_finds_best_arm_easy_instance(self):
        """With a very clear winner and many rounds, Sta'D should identify arm 0."""
        arm_params = np.array([
            [2.0, 0.0],   # arm 0: much better for x = e_0
            [0.0, 0.1],
            [-0.5, 0.0],
        ])
        fb = LinearContextFeedback(
            arm_params, link="logistic", random_state=np.random.RandomState(42)
        )
        alg = StaD(fb, time_horizon=100, failure_probability=0.1)
        alg.run(_constant_context(np.array([1.0, 0.0])))
        assert alg.get_copeland_winner() == 0

    def test_active_set_reduces_over_time(self):
        """Active set should shrink on easy instances with enough rounds."""
        arm_params = np.array([
            [3.0, 0.0],
            [-3.0, 0.0],
            [-6.0, 0.0],
        ])
        fb = LinearContextFeedback(
            arm_params, link="step", random_state=np.random.RandomState(0)
        )
        # Give enough rounds to complete at least 2 stages (len 2 + 4 = 6 rounds)
        alg = StaD(fb, time_horizon=512, failure_probability=0.001)
        alg.run(_constant_context(np.array([1.0, 0.0])))
        assert len(alg._active) < 3
