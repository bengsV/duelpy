"""Tests for the BordaRegret metric."""

import numpy as np

from duelpy.stats.metrics import BordaRegret
from duelpy.stats.metrics import Cumulative
from duelpy.stats.preference_matrix import PreferenceMatrix


def _matrix() -> PreferenceMatrix:
    return PreferenceMatrix(
        np.array(
            [
                [0.5, 0.9, 0.9],
                [0.1, 0.5, 0.6],
                [0.1, 0.4, 0.5],
            ]
        )
    )


class TestBordaRegret:
    def test_best_arm_has_zero_regret(self) -> None:
        metric = BordaRegret(_matrix())
        # Arm 0 is the Borda winner, so dueling it against itself has no regret.
        np.testing.assert_allclose(metric(0, 0), 0.0, atol=1e-12)

    def test_regret_is_non_negative(self) -> None:
        metric = BordaRegret(_matrix())
        for first_arm in range(3):
            for second_arm in range(3):
                assert metric(first_arm, second_arm) >= -1e-12

    def test_matches_definition(self) -> None:
        preference_matrix = _matrix()
        borda_scores = preference_matrix.get_borda_scores()
        metric = BordaRegret(preference_matrix)
        expected = np.max(borda_scores) - 0.5 * (borda_scores[1] + borda_scores[2])
        np.testing.assert_allclose(metric(1, 2), expected)

    def test_accepts_numpy_array(self) -> None:
        metric = BordaRegret(
            np.array([[0.5, 0.9], [0.1, 0.5]])
        )
        np.testing.assert_allclose(metric(0, 0), 0.0, atol=1e-12)

    def test_cumulative_wrapper(self) -> None:
        metric = Cumulative(BordaRegret(_matrix()))
        first = metric(1, 2)
        second = metric(1, 2)
        np.testing.assert_allclose(second, 2 * first)
