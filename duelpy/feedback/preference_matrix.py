"""Gather feedback from a ground-truth preference matrix."""

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

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
        super().__init__(len(preference_matrix))
        self.preference_matrix = preference_matrix
        self.random = random if random is not None else np.random.RandomState()
        self.history: List[Tuple[int, int]] = []

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
        self.history.append((arm_i, arm_j))
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
        for arm_idx in range(self.num_arms):
            # preference_probabilities of selected arm with all arms present in pool of total arms.
            preference_probabilities = np.asarray(self.preference_matrix[arm_idx])
            # preference_probability of selected arm with itself is not required as arm are not compared with itself.
            preference_probabilities = np.delete(preference_probabilities, arm_idx)
            # The arm is the Condorcet winner if it is expected to win (win probability >1/2) against all other arms.
            if np.amin(preference_probabilities) > 0.5:
                return arm_idx
        return None

    def reset_history(self) -> None:
        """Delete the regret history."""
        self.history.clear()

    def _calculate_regret(
        self, best_arm: int, aggregation_function: Callable[[float, float], float]
    ) -> Tuple[list, float]:
        """Calculate the regret.

        The regret is calculated with respect to the given arm. The regret type is specified by the aggregation function.

        Parameters
        ----------
        best_arm
            The arm on which the regret is based on.
        aggregation_function
            This function is used to calculate the regret. E.g. minimum for weak regret.

        Returns
        -------
        regret_history
            A list containing the regret per round.
        cumul_regret
            The cumulative regret.
        """
        regret_history = []
        cumul_regret = 0.0
        for arm_i, arm_j in self.history:
            weak_regret = (
                aggregation_function(
                    self.preference_matrix[best_arm, arm_i],
                    self.preference_matrix[best_arm, arm_j],
                )
                - 0.5
            )
            regret_history.append(weak_regret)
            cumul_regret += weak_regret
        return regret_history, cumul_regret

    def calculate_weak_regret(self, best_arm: int) -> Tuple[List[float], float]:
        """Calculate the weak regret with respect to an arm.

        The weak regret is defined as the distance from the best chosen arm to the best arm overall.

        Parameters
        ----------
        best_arm
            The arm with respect to which the regret is calculated

        Returns
        -------
        regret_history
            A list containing the weak regret per round.
        cumul_regret
            The cumulative weak regret.
        """
        return self._calculate_regret(best_arm, min)

    def calculate_strong_regret(self, best_arm: int) -> Tuple[List[float], float]:
        """Calculate the strong regret with respect to an arm.

        The strong regret is defined as the distance from the worst chosen arm to the best arm overall.

        Parameters
        ----------
        best_arm
            The arm with respect to which the regret is calculated

        Returns
        -------
        regret_history
            A list containing the weak regret per round.
        cumul_regret
            The cumulative strong regret.
        """
        return self._calculate_regret(best_arm, max)
