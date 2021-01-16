"""Gather feedback from a ground-truth preference matrix."""

from typing import List
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
        A quadratic matrix where p[i, j] specifies the probability that arm i
        wins against arm j. This implies p[j, i] = 1 - p[i, j] and p[i, i] =
        0.5.
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
        self.history: List[Tuple[int, int]] = []

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
            True if arm_i_index wins.
        """
        self.history.append((arm_i_index, arm_j_index))
        probability_i_wins = self.preference_matrix[arm_i_index][arm_j_index]
        i_wins = self.random_state.uniform() <= probability_i_wins
        return i_wins

    def get_num_duels(self) -> int:
        """Get the number of duels that were already performed.

        Returns
        -------
        int
            The number of duels.
        """
        return len(self.history)

    def reset_history(self) -> None:
        """Delete the regret history."""
        self.history.clear()

    def calculate_average_copeland_regret(self) -> Tuple[List[float], float]:
        """Calculate copeland regret with respect to normalized copeland score.

        The average Copeland regret of a single comparison is the difference between the average normalized Copeland score of
        the pulled arms and the maximum normalized Copeland score. It can only be 0 if a Copeland winner is compared against
        another Copeland winner. Copeland score is normalized by the number of Arms(i.e number_of_arms-1). Finally, This function
        calculates the normalized cumulative Copeland regret accumulated over all time steps.

        Returns
        -------
        regret_history
            A list containing the Copeland regret per round.
        cumulative_regret
            The cumulative average regret.
        """
        regret_history = []
        cumulative_regret = 0.0

        for arm_i, arm_j in self.history:
            regret = self.preference_matrix.calculate_average_copeland_regret_arms(
                arm_i, arm_j
            )
            regret_history.append(regret)
            cumulative_regret += regret
        return regret_history, cumulative_regret
