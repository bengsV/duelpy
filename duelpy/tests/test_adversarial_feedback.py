"""Tests for the adversarial dueling bandit feedback variants."""

import numpy as np

from duelpy.algorithms.dueling_exp3 import DuelingExp3
from duelpy.experiments.environments import FixedGapAdversarialMatrix
from duelpy.feedback import AdversarialMatrixFeedback


class TestAdversarialMatrixFeedback:
    def test_time_average_matrix(self) -> None:
        matrices = [
            np.array([[0.5, 0.8], [0.2, 0.5]]),
            np.array([[0.5, 0.6], [0.4, 0.5]]),
        ]
        feedback = AdversarialMatrixFeedback(
            matrices, random_state=np.random.RandomState(0)
        )
        np.testing.assert_allclose(feedback.preference_matrix[0, 1], 0.7)
        assert feedback.get_num_arms() == 2

    def test_round_advances_and_reuses_last_matrix(self) -> None:
        matrices = [
            np.array([[0.5, 1.0], [0.0, 0.5]]),  # round 0: arm 0 always wins
            np.array([[0.5, 0.0], [1.0, 0.5]]),  # round 1+: arm 1 always wins
        ]
        feedback = AdversarialMatrixFeedback(
            matrices, random_state=np.random.RandomState(0)
        )
        assert feedback.duel(0, 1) is True  # uses round-0 matrix
        assert feedback.duel(0, 1) is False  # uses round-1 matrix
        # The sequence has length 2; the last matrix is reused afterwards.
        assert feedback.duel(0, 1) is False

    def test_accepts_3d_array(self) -> None:
        matrices = np.stack(
            [
                np.array([[0.5, 0.9], [0.1, 0.5]]),
                np.array([[0.5, 0.5], [0.5, 0.5]]),
            ]
        )
        feedback = AdversarialMatrixFeedback(
            matrices, random_state=np.random.RandomState(0)
        )
        assert feedback.get_num_arms() == 2

    def test_empty_sequence_raises(self) -> None:
        try:
            AdversarialMatrixFeedback([])
        except ValueError:
            return
        raise AssertionError("Expected a ValueError for an empty sequence.")

    def test_dexp3_tracks_hindsight_borda_winner(self) -> None:
        """On a time-varying sequence, D-EXP3 finds the hindsight Borda winner."""
        time_horizon = 4000
        # Arm 0 is strong early, arm 1 strong late, but arm 0 wins on average.
        early = np.array([[0.5, 0.95, 0.95], [0.05, 0.5, 0.5], [0.05, 0.5, 0.5]])
        late = np.array([[0.5, 0.6, 0.9], [0.4, 0.5, 0.9], [0.1, 0.1, 0.5]])
        matrices = [early] * (time_horizon // 2) + [late] * (time_horizon // 2)
        feedback = AdversarialMatrixFeedback(
            matrices, random_state=np.random.RandomState(0)
        )
        hindsight_winner = int(
            np.argmax(feedback.preference_matrix.get_borda_scores())
        )
        algorithm = DuelingExp3(
            feedback, time_horizon=time_horizon, random_state=np.random.RandomState(0)
        )
        algorithm.run()
        assert algorithm.get_borda_winner() == hindsight_winner


class TestFixedGapAdversarialMatrix:
    def test_borda_gap_is_exact(self) -> None:
        feedback = FixedGapAdversarialMatrix(
            num_arms=5, random_state=np.random.RandomState(0), gap=0.15
        )
        borda_scores = feedback.preference_matrix.get_borda_scores()
        winner_score = np.max(borda_scores)
        runner_up_score = sorted(borda_scores)[-2]
        np.testing.assert_allclose(winner_score - runner_up_score, 0.15)

    def test_winner_is_condorcet_winner(self) -> None:
        feedback = FixedGapAdversarialMatrix(
            num_arms=4, random_state=np.random.RandomState(1), gap=0.2
        )
        borda_winner = int(np.argmax(feedback.preference_matrix.get_borda_scores()))
        assert feedback.preference_matrix.get_condorcet_winner() == borda_winner

    def test_rejects_too_large_gap(self) -> None:
        try:
            FixedGapAdversarialMatrix(num_arms=3, gap=0.9)
        except ValueError:
            return
        raise AssertionError("Expected a ValueError for an infeasible gap.")

    def test_rejects_single_arm(self) -> None:
        try:
            FixedGapAdversarialMatrix(num_arms=1)
        except ValueError:
            return
        raise AssertionError("Expected a ValueError for fewer than two arms.")
