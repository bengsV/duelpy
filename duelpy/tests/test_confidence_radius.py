"""Tests for confidence radius implementations."""

import pytest

from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelpy.stats.confidence_radius import TrivialConfidenceRadius


class TestTrivialConfidenceRadius:
    def test_constant_return(self):
        cr = TrivialConfidenceRadius(radius=0.3)
        assert cr(0) == pytest.approx(0.3)
        assert cr(100) == pytest.approx(0.3)

    def test_default_radius_is_one(self):
        cr = TrivialConfidenceRadius()
        assert cr(50) == pytest.approx(1.0)


class TestHoeffdingConfidenceRadius:
    def test_zero_samples_returns_one(self):
        cr = HoeffdingConfidenceRadius(failure_probability=0.1)
        assert cr(0) == pytest.approx(1.0)

    def test_decreases_with_more_samples(self):
        cr = HoeffdingConfidenceRadius(failure_probability=0.05)
        prev = cr(1)
        for n in [5, 20, 100, 1000]:
            current = cr(n)
            assert current < prev
            prev = current

    def test_positive_radius(self):
        cr = HoeffdingConfidenceRadius(failure_probability=0.1)
        for n in range(1, 50):
            assert cr(n) >= 0

    def test_probability_scaling_factor(self):
        cr_scaled = HoeffdingConfidenceRadius(
            failure_probability=0.1,
            probability_scaling_factor=lambda n: n,
        )
        cr_base = HoeffdingConfidenceRadius(failure_probability=0.1)
        assert cr_scaled(10) != cr_base(10)

    def test_factor_scales_radius(self):
        cr1 = HoeffdingConfidenceRadius(failure_probability=0.1, factor=1)
        cr4 = HoeffdingConfidenceRadius(failure_probability=0.1, factor=4)
        assert cr4(10) == pytest.approx(cr1(10) * 2)
