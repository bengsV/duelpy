"""Gather feedback from a ground-truth preference matrix."""

from typing import Optional

import numpy as np

from duelpy.feedback.feedback_mechanism import FeedbackMechanism


class PreferenceMatrix(FeedbackMechanism):
    """Compare two arms based on a preference matrix.

    Parameters
    ----------
    preference_matrix
        A quadratic matrix where p[i, j] specifies the probability that arm i
        wins against arm j. This implies p[j, i] = 1 - p[i, j] and p[i, i] =
        0.5.

    random
        A numpy random state. Defaults to an unseeded state when not specified.
    """

    def __init__(
        self,
        preference_matrix: np.array,
        random: Optional[np.random.RandomState] = None,
    ):
        self.preference_matrix = preference_matrix
        self.random = random if random is not None else np.random.RandomState()

    def duel(self, arm_i: int, arm_j: int) -> bool:
        """Perform a duel between two arms based on a given probability matrix.

        Parameters
        ----------
        arm_i
            The challenger arm.
        arm_j
            The arm to compare against.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        probability_i_wins = self.preference_matrix[arm_i][arm_j]
        i_wins = self.random.random() <= probability_i_wins
        return i_wins
