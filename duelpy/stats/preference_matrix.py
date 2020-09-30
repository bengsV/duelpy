"""A preference matrix with associated utility functions."""

from typing import Any
from typing import Optional
from typing import Set

import numpy as np

from duelpy.util.utility_functions import argmax_set


class PreferenceMatrix:
    """Represents a preference matrix with associated utility functions.

    Parameters
    ----------
    preferences
        A quadratic matrix where p[i, j] specifies the probability that arm i
        wins against arm j. This implies p[j, i] = 1 - p[i, j] and p[i, i] =
        0.5.
    """

    def __init__(
        self, preferences: np.array,
    ):
        self.preferences = preferences

    # Unfortunately numpy's indexing is not typed, so we can't type this either
    # if we don't want to lose any of its power.
    def __getitem__(self, key: Any) -> Any:
        """Get a preference probability."""
        return self.preferences[key]

    def get_num_arms(self) -> int:
        """Get the number of arms in the preference matrix.

        Returns
        -------
        int
            The number of arms.
        """
        return self.preferences.shape[0]

    def get_condorcet_winner(self) -> Optional[int]:
        """Get the index of the Condorcet winner if one exists.

        The Condorcet winner is the arm that is expected to beat every other
        arm in a pairwise comparison.

        Returns
        -------
        Optional[int]
            The index of the Condorcet winner if one exists.
        """
        # select one arm each time from the pool of total arms to check whether it is a Condorcet winner or not
        for arm_idx in range(self.get_num_arms()):
            # preference_probabilities of selected arm with all arms present in pool of total arms.
            preference_probabilities = np.asarray(self.preferences[arm_idx])
            # preference_probability of selected arm with itself is not required as arm are not compared with itself.
            preference_probabilities = np.delete(preference_probabilities, arm_idx)
            # The arm is the Condorcet winner if it is expected to win (win probability >1/2) against all other arms.
            if np.amin(preference_probabilities) >= 0.5:
                return arm_idx
        return None

    def get_copeland_winners(self) -> Set[int]:
        """Get the set of Copeland winners.

        A Copeland winner is an arm that has the highest number of expected
        wins against all other arms. This does not need to be unique, since
        multiple arms can have the same number of expected wins.

        Returns
        -------
        Set[int]
            The indices of the Copeland winners.
        """
        num_arms = self.get_num_arms()
        expected_wins = np.zeros(num_arms)
        for first_arm_idx in range(num_arms):
            for second_arm_idx in range(first_arm_idx + 1, num_arms):
                if self.preferences[first_arm_idx, second_arm_idx] > 1 / 2:
                    expected_wins[first_arm_idx] += 1
                else:
                    expected_wins[second_arm_idx] += 1
        return argmax_set(expected_wins)
