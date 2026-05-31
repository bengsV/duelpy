"""Tests for ContextualPreferenceEstimate."""

import numpy as np
import pytest

from duelpy.stats.contextual_preference_estimate import ContextualPreferenceEstimate


def _make_estimate(num_arms: int = 3, context_dim: int = 2) -> ContextualPreferenceEstimate:
    return ContextualPreferenceEstimate(num_arms=num_arms, context_dim=context_dim)


class TestInitialization:
    def test_theta_hat_zero(self):
        est = _make_estimate()
        np.testing.assert_array_equal(est.get_theta_hat(0), np.zeros(2))

    def test_no_samples_initially(self):
        est = _make_estimate()
        assert est.get_num_samples(0) == 0


class TestEnterSample:
    def test_increments_sample_count_both_arms(self):
        est = _make_estimate()
        est.enter_sample(0, 1, np.array([1.0, 0.0]), arm_i_won=True)
        assert est.get_num_samples(0) == 1
        assert est.get_num_samples(1) == 1

    def test_unrelated_arm_unaffected(self):
        est = _make_estimate()
        est.enter_sample(0, 1, np.array([1.0, 0.0]), arm_i_won=True)
        assert est.get_num_samples(2) == 0

    def test_theta_hat_updates_after_sample(self):
        est = _make_estimate()
        x = np.array([1.0, 0.0])
        est.enter_sample(0, 1, x, arm_i_won=True)
        theta = est.get_theta_hat(0)
        assert not np.allclose(theta, np.zeros(2))


class TestConfidenceBounds:
    def test_ucb_at_least_lcb(self):
        est = _make_estimate()
        x = np.array([1.0, 0.0])
        for _ in range(5):
            est.enter_sample(0, 1, x, arm_i_won=True)
        beta = 1.0
        assert est.get_upper_confidence_utility(0, x, beta) >= \
               est.get_lower_confidence_utility(0, x, beta)

    def test_mean_between_bounds(self):
        est = _make_estimate()
        x = np.array([0.5, 0.5])
        for _ in range(3):
            est.enter_sample(0, 1, x, arm_i_won=True)
        beta = 2.0
        mean = est.get_mean_utility(0, x)
        assert est.get_lower_confidence_utility(0, x, beta) <= mean
        assert mean <= est.get_upper_confidence_utility(0, x, beta)

    def test_confidence_shrinks_with_more_samples(self):
        est = _make_estimate()
        x = np.array([1.0, 0.0])
        beta = 1.0
        width_before = (est.get_upper_confidence_utility(0, x, beta)
                        - est.get_lower_confidence_utility(0, x, beta))
        for _ in range(50):
            est.enter_sample(0, 1, x, arm_i_won=bool(np.random.randint(2)))
        width_after = (est.get_upper_confidence_utility(0, x, beta)
                       - est.get_lower_confidence_utility(0, x, beta))
        assert width_after < width_before


class TestBeta:
    def test_beta_positive(self):
        est = _make_estimate()
        x = np.array([1.0, 0.0])
        for _ in range(10):
            est.enter_sample(0, 1, x, arm_i_won=True)
        assert est.compute_beta(delta=0.05) > 0

    def test_confidence_width_decreases_with_more_samples(self):
        """The confidence WIDTH beta*||x||_{V^{-1}} should shrink with more data."""
        x = np.array([1.0, 0.0])

        est_few = _make_estimate()
        for _ in range(5):
            est_few.enter_sample(0, 1, x, arm_i_won=True)

        est_many = _make_estimate()
        for _ in range(100):
            est_many.enter_sample(0, 1, x, arm_i_won=True)

        beta_few = est_few.compute_beta(0.05)
        beta_many = est_many.compute_beta(0.05)
        width_few = est_few.get_upper_confidence_utility(0, x, beta_few) \
                    - est_few.get_lower_confidence_utility(0, x, beta_few)
        width_many = est_many.get_upper_confidence_utility(0, x, beta_many) \
                     - est_many.get_lower_confidence_utility(0, x, beta_many)
        assert width_many < width_few
