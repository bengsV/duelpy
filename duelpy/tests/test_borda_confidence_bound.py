"""Tests for the Borda-Confidence-Bound algorithm."""

import numpy as np

from duelpy.algorithms.borda_confidence_bound import BordaConfidenceBound
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


class TestBordaConfidenceBoundBasic:
    def test_runs_without_error(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = BordaConfidenceBound(
            feedback, time_horizon=100, random_state=np.random.RandomState(0)
        )
        algorithm.run()

    def test_exact_time_horizon(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = BordaConfidenceBound(
            feedback, time_horizon=500, random_state=np.random.RandomState(1)
        )
        algorithm.run()
        assert algorithm.wrapped_feedback.duels_conducted == 500

    def test_winner_none_before_run(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = BordaConfidenceBound(feedback, time_horizon=100)
        assert algorithm.get_borda_winner() is None

    def test_requires_time_horizon(self) -> None:
        feedback = _clear_borda_winner_feedback()
        try:
            BordaConfidenceBound(feedback, time_horizon=None)
        except ValueError:
            return
        raise AssertionError("Expected a ValueError when no time horizon is given.")

    def test_single_arm_commits_immediately(self) -> None:
        feedback = MatrixFeedback(
            np.array([[0.5]]), random_state=np.random.RandomState(0)
        )
        algorithm = BordaConfidenceBound(
            feedback, time_horizon=5, random_state=np.random.RandomState(0)
        )
        algorithm.run()
        assert algorithm.get_borda_winner() == 0
        assert algorithm.wrapped_feedback.duels_conducted == 5


class TestBordaConfidenceBoundCorrectness:
    def test_finds_borda_winner(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = BordaConfidenceBound(
            feedback, time_horizon=5000, random_state=np.random.RandomState(7)
        )
        algorithm.run()
        assert algorithm.get_borda_winner() == 0

    def test_commits_on_easy_instance(self) -> None:
        feedback = _clear_borda_winner_feedback()
        algorithm = BordaConfidenceBound(
            feedback,
            time_horizon=10000,
            failure_probability=0.1,
            random_state=np.random.RandomState(7),
        )
        algorithm.run()
        assert algorithm._committed_arm == 0
