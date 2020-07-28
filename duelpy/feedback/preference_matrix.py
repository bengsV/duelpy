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
        self.no_of_arms = len(self.preference_matrix)
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

    def get_condorcet_winner(self) -> Optional[int]:
        """Get the the index of the Condorcet winner if one exists.

        The Condorcet winner is the arm that is expected to beat every other
        arm in a pairwise comparison.

        Returns
        -------
        Optional[int]
            The index of the Condorcet winner if one exists.
        """
        # select one arm each time from the pool of total arms to check whether it is a Condorcet winner or not
        for arm_idx in range(self.no_of_arms):
            # preference_probabilities of selected arm with all arms present in pool of total arms.
            preference_probabilities = np.asarray(self.preference_matrix[arm_idx])
            # preference_probability of selected arm with itself is not required as arm are not compared with itself.
            preference_probabilities = np.delete(preference_probabilities, arm_idx)
            # The arm is the Condorcet winner if it is expected to win (win probability >1/2) against all other arms.
            if np.amin(preference_probabilities) > 0.5:
                return arm_idx
        return None
