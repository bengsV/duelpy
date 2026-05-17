"""Tests for PreferenceEstimate."""

import numpy as np
import pytest

from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelpy.stats.confidence_radius import TrivialConfidenceRadius
from duelpy.stats.preference_estimate import PreferenceEstimate


def _make_estimate(num_arms: int = 3) -> PreferenceEstimate:
    return PreferenceEstimate(
        num_arms=num_arms,
        confidence_radius=TrivialConfidenceRadius(radius=0.2),
    )


class TestInitialization:
    def test_mean_estimate_is_half(self):
        pe = _make_estimate(3)
        m = pe.get_mean_estimate_matrix()
        np.testing.assert_array_equal(m.preferences, np.full((3, 3), 0.5))

    def test_no_samples_initially(self):
        pe = _make_estimate(3)
        assert pe.get_num_samples(0, 1) == 0


class TestEnterSample:
    def test_single_win_updates_mean(self):
        pe = _make_estimate(3)
        pe.enter_sample(0, 1, first_won=True)
        assert pe.get_mean_estimate(0, 1) == pytest.approx(1.0)
        assert pe.get_mean_estimate(1, 0) == pytest.approx(0.0)

    def test_symmetric_update(self):
        pe = _make_estimate(3)
        pe.enter_sample(0, 1, first_won=True)
        pe.enter_sample(0, 1, first_won=False)
        assert pe.get_mean_estimate(0, 1) == pytest.approx(0.5)

    def test_num_samples_increments(self):
        pe = _make_estimate(3)
        pe.enter_sample(0, 1, first_won=True)
        pe.enter_sample(0, 1, first_won=False)
        assert pe.get_num_samples(0, 1) == 2
        assert pe.get_num_samples(1, 0) == 2

    def test_unrelated_pair_unaffected(self):
        pe = _make_estimate(3)
        pe.enter_sample(0, 1, first_won=True)
        assert pe.get_mean_estimate(0, 2) == pytest.approx(0.5)


class TestConfidenceIntervals:
    def test_upper_at_least_mean(self):
        pe = _make_estimate(3)
        pe.enter_sample(0, 1, first_won=True)
        assert pe.get_upper_estimate(0, 1) >= pe.get_mean_estimate(0, 1)

    def test_lower_at_most_mean(self):
        pe = _make_estimate(3)
        pe.enter_sample(0, 1, first_won=True)
        assert pe.get_lower_estimate(0, 1) <= pe.get_mean_estimate(0, 1)

    def test_bounds_within_zero_one(self):
        pe = _make_estimate(3)
        for _ in range(5):
            pe.enter_sample(0, 1, first_won=True)
        assert 0.0 <= pe.get_lower_estimate(0, 1) <= 1.0
        assert 0.0 <= pe.get_upper_estimate(0, 1) <= 1.0

    def test_set_confidence_radius_invalidates_cache(self):
        pe = _make_estimate(3)
        pe.enter_sample(0, 1, first_won=True)
        r1 = pe.get_radius_matrix()
        pe.set_confidence_radius(TrivialConfidenceRadius(radius=0.05))
        r2 = pe.get_radius_matrix()
        assert not np.array_equal(r1, r2)


class TestCopelandEstimates:
    def test_pessimistic_scores_shape(self):
        pe = _make_estimate(4)
        scores = pe.get_pessimistic_copeland_score_estimates()
        assert scores.shape == (4,)

    def test_optimistic_scores_shape(self):
        pe = _make_estimate(4)
        scores = pe.get_optimistic_copeland_score_estimates()
        assert scores.shape == (4,)

    def test_pessimistic_le_optimistic(self):
        pe = _make_estimate(3)
        pe.enter_sample(0, 1, first_won=True)
        pe.enter_sample(0, 2, first_won=True)
        pess = pe.get_pessimistic_copeland_score_estimates()
        opt = pe.get_optimistic_copeland_score_estimates()
        assert (pess <= opt).all()


class TestSamplePreferenceMatrix:
    def test_returns_preference_matrix(self):
        from duelpy.stats.preference_matrix import PreferenceMatrix

        pe = PreferenceEstimate(
            num_arms=3,
            confidence_radius=HoeffdingConfidenceRadius(failure_probability=0.1),
        )
        for _ in range(5):
            pe.enter_sample(0, 1, first_won=True)
        sampled = pe.sample_preference_matrix(np.random.RandomState(0))
        assert isinstance(sampled, PreferenceMatrix)
        assert sampled.preferences.shape == (3, 3)
