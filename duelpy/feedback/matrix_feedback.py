"""Gather feedback from a ground-truth preference matrix."""

from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelpy.stats.preference_matrix import PreferenceMatrix


class MatrixFeedback(FeedbackMechanism):
    """Compare two arms based on a preference matrix.

    Parameters
    ----------
    arms
        It represents a list of arms from preference matrix. If not provided, the method will create its own list of
        arms from preference matrix.
    preference_matrix
        A quadratic matrix where :math:`p[i, j]` specifies the probability that arm :math:`i`
        wins against arm :math:`j`. This implies :math:`p[j, i] = 1 - p[i, j]` and :math:`p[i, i] = 0.5`.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.
    """

    def __init__(
        self,
        preference_matrix: Union[PreferenceMatrix, np.array],
        arms: Optional[list] = None,
        random_state: Optional[np.random.RandomState] = None,
    ):
        if not isinstance(preference_matrix, PreferenceMatrix):
            preference_matrix = PreferenceMatrix(preference_matrix)
        if arms is None:
            arms = list(range(preference_matrix.get_num_arms()))
        else:
            if preference_matrix.get_num_arms() != len(arms):
                raise ValueError("Labels and matrix size mismatch")
        super().__init__(arms)
        self.preference_matrix = preference_matrix
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.num_duels = 0

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel between two arms based on a given probability matrix.

        Parameters
        ----------
        arm_i_index
            The challenger arm.
        arm_j_index
            The arm to compare against.

        Returns
        -------
        bool
            True if ``arm_i_index`` wins.
        """
        self.num_duels += 1
        probability_i_wins = self.preference_matrix[arm_i_index][arm_j_index]
        i_wins = self.random_state.uniform() <= probability_i_wins
        return i_wins

    def multi_duels_step(
        self,
        arm_i_index: int,
        arm_j_index: int,
        duel_count: int,
        duel_limit: Optional[int] = None,
    ) -> Tuple:
        """Perform the duels between two arms multiple times.

        If the ``duel_count`` is a large number in the range of :math:`500000`, use this method.
        The binomial distribution save the computations on multiple comparisons.

        Parameters
        ----------
        arm_i_index
            The challenger arm.
        arm_j_index
            The arm to compare against.
        duel_count
            Number of duels that has to be performed.
        duel_limit
            Number of duels that the algorithm has a budget to allow. This is an Optional parameter.

        Returns
        -------
        bool
            ``True`` means ``arm_i_index`` beats ``arm_j_index`` and otherwise ``False``.
        mean_estimate
            Average of ``arm_i_index`` winning against ``arm_j_index``.
        """
        if duel_limit is not None and self.get_num_duels() + duel_count > duel_limit:
            self.num_duels = duel_limit
        else:
            self.num_duels += duel_count
        arm_i_win_frequency = np.random.binomial(
            int(duel_count),
            self.preference_matrix.preferences[arm_i_index, arm_j_index],
        )
        mean_estimate = arm_i_win_frequency / duel_count

        if mean_estimate > 0.5:
            return True, mean_estimate
        else:
            return False, mean_estimate

    def get_num_duels(self) -> int:
        """Get the number of duels that were already performed.

        Returns
        -------
        int
            The number of duels.
        """
        return self.num_duels

    def reset_duel_counter(self) -> None:
        """Reset the duel counter."""
        self.num_duels = 0
