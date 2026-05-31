"""Tests for the Dueling-EXP3 algorithms."""

import numpy as np

from duelpy.algorithms.dueling_exp3 import DuelingExp3
from duelpy.algorithms.dueling_exp3 import DuelingExp3HighProbability
from duelpy.feedback import MatrixFeedback


def _clear_borda_winner_feedback(seed: int = 42) -> MatrixFeedback:
    """A matrix where arm 0 is the unambiguous Borda winner."""
    preference_matrix = np.array(
        [
            [0.5, 0.9, 0.9],
            [0.1, 0.5, 0.6],
            [0.1, 0.4, 0.5],
        ]
    )
    return MatrixFeedback(preference_matrix, random_state=np.random.RandomState(seed))


class TestDuelingExp3Basic:
    def test_runs_without_error(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = DuelingExp3(
            feedback, time_horizon=100, random_state=np.random.RandomState(0)
        )
        algorithm.run()

    def test_exact_time_horizon(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = DuelingExp3(
            feedback, time_horizon=250, random_state=np.random.RandomState(1)
        )
        algorithm.run()
        assert algorithm.wrapped_feedback.duels_conducted == 250

    def test_distribution_is_valid(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = DuelingExp3(
            feedback, time_horizon=100, random_state=np.random.RandomState(2)
        )
        algorithm.run()
        assert np.all(algorithm._distribution >= 0)
        np.testing.assert_allclose(np.sum(algorithm._distribution), 1.0)

    def test_requires_time_horizon(self) -> None:
        feedback = _clear_borda_winner_feedback()
        try:
            DuelingExp3(feedback, time_horizon=None)
        except ValueError:
            return
        raise AssertionError("Expected a ValueError when no time horizon is given.")


class TestDuelingExp3Correctness:
    def test_finds_borda_winner(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = DuelingExp3(
            feedback, time_horizon=4000, random_state=np.random.RandomState(7)
        )
        algorithm.run()
        assert algorithm.get_borda_winner() == 0

    def test_high_probability_variant_finds_borda_winner(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = DuelingExp3HighProbability(
            feedback, time_horizon=4000, random_state=np.random.RandomState(7)
        )
        algorithm.run()
        assert algorithm.get_borda_winner() == 0

    def test_high_probability_exact_time_horizon(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = DuelingExp3HighProbability(
            feedback, time_horizon=300, random_state=np.random.RandomState(3)
        )
        algorithm.run()
        assert algorithm.wrapped_feedback.duels_conducted == 300
