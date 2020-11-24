"""Multi-armed bandit environments for experiments."""

from typing import List
from typing import Optional

import numpy as np

from duelpy.feedback import MatrixFeedback
from duelpy.stats.preference_matrix import PreferenceMatrix
from duelpy.util.utility_functions import argmax_set


def _shuffle_preference_matrix(
    preference_matrix: PreferenceMatrix, random_state: np.random.RandomState
) -> PreferenceMatrix:
    """Shuffle the arm order in a preference matrix.

    Each relative arm probability will remain the same, but positions change.
    """
    num_arms = preference_matrix.get_num_arms()
    permutation = random_state.permutation(num_arms)
    result = np.copy(preference_matrix.preferences)
    for first_arm_idx in range(num_arms):
        for second_arm_idx in range(num_arms):
            perm_first = permutation[first_arm_idx]
            perm_second = permutation[second_arm_idx]
            result[perm_first][perm_second] = preference_matrix[first_arm_idx][
                second_arm_idx
            ]
    return PreferenceMatrix(result)


class HardCondorcetMatrix(MatrixFeedback):
    r"""A feedback-mechanism using a generated "hard Condorcet matrix".

    These matrices have a Condorcet winner, but it is hard to find since it has
    a tight margin. All generated matrices of the same size are equivalent,
    they are just randomly shuffled.

    As described in section 5.2.1 of :cite:`urvoy2013generic`:
    :math:`\mu` defined by :math:`mu_{(i, j)} = 1/2 + j/(2K)`. We add a small
    `epsilon` to the relative preference values to assure a Condorcet winner
    actually exists. This is necessary since our definition of a "win" (>0.5)
    slightly diverges from the definition used in :cite:`urvoy2013generic`
    (>=0.5).

    These matrices guarantee the existence of a Condorcet winner and include
    both easy and hard decisions: :math:`mu_{(1, 2)} = 0.51` is a hard
    decision, :math:`mu_(1, 100) = 1` is an easy one.

    Parameters
    ----------
    num_arms
        The size of the preference matrix to generate.
    random_state
        The numpy random state that will be used for shuffling.
    """

    def __init__(self, num_arms: int, random_state: np.random.RandomState):

        epsilon = 1e-10 / num_arms
        preferences = np.full((num_arms, num_arms), 0.5)
        for first_arm_idx in range(num_arms):
            for second_arm_idx in range(first_arm_idx):
                # Add epsilon since our definition of a Condorcet winner requires a
                # ">", not a ">=".
                relative_preference = 1 / 2 + second_arm_idx / (2 * num_arms) + epsilon
                # Clipping is necessary because of the epsilon, but has little
                # practical implication as long as epsilon is small.
                relative_preference = min(1.0, relative_preference)
                preferences[first_arm_idx][second_arm_idx] = relative_preference
                preferences[second_arm_idx][first_arm_idx] = 1 - relative_preference
        preference_matrix = PreferenceMatrix(preferences)
        shuffled_matrix = _shuffle_preference_matrix(preference_matrix, random_state)

        super().__init__(preference_matrix=shuffled_matrix, random_state=random_state)


class PlackettLuceModel(MatrixFeedback):
    r"""A feedback-mechanism based on the Plackett-Luce model.

    The probabilities are based on a non-negative 'skill' value for each arm. The probability of arm i winning against arm j is then based on their skills v: :math:`\frac{v_i}{v_i+v_j}`.

    Parameters
    ----------
    num_arms
        The size of the preference matrix to generate.
    random_state
        The numpy random state that will be used for sampling and generating skills, if they are not given.
    skill_vector
        Optional, contain scalars representing the skill of each arm. Must be of length `num_arms` and only contain non-negative values.
    """

    def __init__(
        self,
        num_arms: int,
        random_state: np.random.RandomState,
        skill_vector: Optional[List[float]] = None,
    ):

        if skill_vector is not None:
            if len(skill_vector) != num_arms:
                raise ValueError(
                    "The num_arms parameter needs to be equal to the length of the given skill_vector."
                )
            if any([x < 0 for x in skill_vector]):
                raise ValueError("The values in skill_vector must be non-negative.")
        else:
            skill_vector = random_state.uniform(size=num_arms)
        self._skill_vector = skill_vector

        preferences = np.full((num_arms, num_arms), 0.5)
        for first_arm_idx in range(num_arms):
            for second_arm_idx in range(first_arm_idx):
                relative_preference = skill_vector[first_arm_idx] / (
                    skill_vector[first_arm_idx] + skill_vector[second_arm_idx]
                )
                preferences[first_arm_idx][second_arm_idx] = relative_preference
                preferences[second_arm_idx][first_arm_idx] = 1 - relative_preference
        preference_matrix = PreferenceMatrix(preferences)

        super().__init__(preference_matrix=preference_matrix, random_state=random_state)

    def get_best_arms(self) -> List[int]:
        """Get a list of all best arms. This can (and usually is) only be the Condorcet winner. But if multiple arms have the same maximal skill value, they are returned as Copeland winners.

        Returns
        -------
        List[int]
            A list of all the best arms.
        """
        return argmax_set(self._skill_vector)

    def get_arbitrary_ranking(self) -> List[int]:
        """Get any correct ranking of the arms.

        Returns
        -------
        List[int]
            Any correct ranking of the arms, must not be the only correct one.
        """
        return sorted(
            list(range(len(self._skill_vector))), key=lambda i: self._skill_vector[i]
        )

    def test_ranking(self, ranking: List[int]) -> bool:
        r"""Check whether a ranking is admissible.

        Generating all correct rankings might take too long (possibly :math:`\mathit{num\_arms}!` many), so we simply allow checking of given rankings.

        Parameters
        ----------
        ranking
            The ranking that should be checked

        Returns
        -------
        bool
            Whether the ranking is correct.
        """
        current_skill = self._skill_vector[ranking[0]]
        for arm in ranking:
            if self._skill_vector[arm] > current_skill:
                return False
            current_skill = self._skill_vector[arm]
        return True
