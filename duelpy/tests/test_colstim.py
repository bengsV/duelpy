"""Tests for the CoLSTIM algorithm."""

import numpy as np
import pytest

from duelpy.algorithms.colstim import CoLSTIM
from duelpy.feedback.arm_feature_feedback import ArmFeatureFeedback


def _make_feedback(seed: int = 0) -> ArmFeatureFeedback:
    features = np.eye(3)  # each arm is a unit basis vector
    theta = np.array([1.0, 0.5, 0.2])  # arm 0 is best
    return ArmFeatureFeedback(
        features, theta, link="logistic",
        random_state=np.random.RandomState(seed)
    )


class TestCoLSTIMBasic:
    def test_runs_without_error(self):
        fb = _make_feedback()
        alg = CoLSTIM(fb, time_horizon=20, random_state=np.random.RandomState(0))
        alg.run()

    def test_time_horizon_respected(self):
        fb = _make_feedback()
        alg = CoLSTIM(fb, time_horizon=15, random_state=np.random.RandomState(1))
        alg.run()
        assert alg.wrapped_feedback.duels_conducted == 15

    def test_is_finished_after_run(self):
        fb = _make_feedback()
        alg = CoLSTIM(fb, time_horizon=10, random_state=np.random.RandomState(2))
        alg.run()
        assert alg.is_finished()

    def test_get_copeland_winner_none_before_run(self):
        fb = _make_feedback()
        alg = CoLSTIM(fb, time_horizon=10, random_state=np.random.RandomState(3))
        assert alg.get_copeland_winner() is None

    def test_get_copeland_winner_returns_valid_arm(self):
        fb = _make_feedback()
        alg = CoLSTIM(fb, time_horizon=30, random_state=np.random.RandomState(4))
        alg.run()
        winner = alg.get_copeland_winner()
        assert winner is not None
        assert 0 <= winner < fb.get_num_arms()


class TestCoLSTIMCorrectness:
    def test_finds_best_arm_easy_instance(self):
        """With strong signal and many rounds, CoLSTIM should find arm 0."""
        features = np.eye(3)
        theta = np.array([5.0, 1.0, 0.1])  # arm 0 strongly dominates
        fb = ArmFeatureFeedback(
            features, theta, link="logistic",
            random_state=np.random.RandomState(42)
        )
        alg = CoLSTIM(
            fb, time_horizon=300,
            failure_probability=0.05,
            random_state=np.random.RandomState(42),
        )
        alg.run()
        assert alg.get_copeland_winner() == 0

    def test_theta_hat_direction_correct(self):
        """After many rounds, θ̂ should point in the direction of θ*."""
        features = np.eye(3)
        theta = np.array([5.0, 0.0, 0.0])  # strong signal on arm 0
        fb = ArmFeatureFeedback(
            features, theta, link="logistic",
            random_state=np.random.RandomState(0)
        )
        alg = CoLSTIM(
            fb, time_horizon=1000,
            random_state=np.random.RandomState(0),
        )
        alg.run()
        # The estimate should give arm 0 the highest utility
        utils = features @ alg._theta_hat
        assert int(np.argmax(utils)) == 0

    def test_reproducible_with_same_seed(self):
        fb1 = _make_feedback(seed=7)
        fb2 = _make_feedback(seed=7)
        alg1 = CoLSTIM(fb1, time_horizon=50, random_state=np.random.RandomState(99))
        alg2 = CoLSTIM(fb2, time_horizon=50, random_state=np.random.RandomState(99))
        alg1.run()
        alg2.run()
        assert alg1.get_copeland_winner() == alg2.get_copeland_winner()
        np.testing.assert_array_almost_equal(alg1._theta_hat, alg2._theta_hat)
