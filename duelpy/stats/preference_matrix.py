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
        copeland_scores = self.get_copeland_scores()
        copeland_winners = argmax_set(copeland_scores)
        # condorcet winner is an arm that beats k-1 arms where k is the total number of the arms.
        # Also,  we can say that condorcet winner is an arm whose copeland score is equal to k-1 arms.
        if (
            len(copeland_winners) == 1
            and copeland_scores[copeland_winners[0]] == self.get_num_arms() - 1
        ):
            return copeland_winners[0]
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
        return set(argmax_set(self.get_copeland_scores()))

    def get_copeland_scores(self) -> np.array:
        """Calculate Copeland scores for each arm.

        The Copeland score of an arm is the number of other arms that the arm is expected to win against.

        Returns
        -------
        np.array
            A 1-D array with the Copeland scores.
        """
        return (self.preferences > 0.5).sum(axis=1)

    def get_normalized_copeland_scores(self) -> np.array:
        """Calculate the normalized Copeland scores for each arm.

        The normalized Copeland score of an arm is the fraction of other arms it is expected to win against.

        Returns
        -------
        np.array
            A 1-D array with the normalized Copeland scores.
        """
        return self.get_copeland_scores() / (self.get_num_arms() - 1)
