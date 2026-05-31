"""Tests for ArmFeatureFeedback."""

import numpy as np
import pytest

from duelpy.feedback.arm_feature_feedback import ArmFeatureFeedback


_FEATURES = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
_THETA = np.array([1.0, 0.0])  # arm 0 is best: u0=1, u1=0, u2=0.5


class TestArmFeatureFeedbackStep:
    def test_get_best_arm(self):
        fb = ArmFeatureFeedback(_FEATURES, _THETA, link="step")
        assert fb.get_best_arm() == 0

    def test_duel_correct_winner(self):
        fb = ArmFeatureFeedback(_FEATURES, _THETA, link="step")
        assert fb.duel(0, 1) is True
        assert fb.duel(1, 0) is False

    def test_duel_equal_utility_step(self):
        features = np.array([[0.5, 0.5], [0.5, 0.5]])
        theta = np.array([1.0, 1.0])
        fb = ArmFeatureFeedback(features, theta, link="step")
        # diff = 0.0 → step returns diff > 0 = False
        assert fb.duel(0, 1) is False


class TestArmFeatureFeedbackLogistic:
    def test_strong_preference_mostly_correct(self):
        fb = ArmFeatureFeedback(
            _FEATURES, _THETA * 10, link="logistic",
            random_state=np.random.RandomState(0)
        )
        wins = sum(fb.duel(0, 1) for _ in range(200))
        assert wins > 160  # arm 0 should win >>50% of the time

    def test_reproducibility(self):
        fb1 = ArmFeatureFeedback(_FEATURES, _THETA, link="logistic",
                                  random_state=np.random.RandomState(7))
        fb2 = ArmFeatureFeedback(_FEATURES, _THETA, link="logistic",
                                  random_state=np.random.RandomState(7))
        r1 = [fb1.duel(0, 1) for _ in range(30)]
        r2 = [fb2.duel(0, 1) for _ in range(30)]
        assert r1 == r2


class TestArmFeatureFeedbackInterface:
    def test_feature_dim(self):
        fb = ArmFeatureFeedback(_FEATURES, _THETA)
        assert fb.feature_dim == 2

    def test_arm_features_are_copy(self):
        fb = ArmFeatureFeedback(_FEATURES, _THETA)
        feats = fb.arm_features
        feats[0, 0] = 999.0
        assert fb.get_best_arm() == 0  # unchanged

    def test_invalid_link_raises(self):
        with pytest.raises(ValueError):
            ArmFeatureFeedback(_FEATURES, _THETA, link="relu")
