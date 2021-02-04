"""An environment based on the Mallows model."""

from typing import List
from typing import Optional

import numpy as np

from duelpy.feedback import MatrixFeedback
from duelpy.stats.preference_matrix import PreferenceMatrix


class MallowsModel(MatrixFeedback):
    r"""A feedback-mechanism based on the Mallows model.

    The probability of a ranking depends on a spread parameter :math:`\phi \in (0,1]` and the Kendall distance to the ground truth ranking. For :math:`\phi=1`, a uniform distribution over all permutations results, lower values will have higher probabilities for rankings close to the ground truth.
    For an overview of the probabilities for rankings and the resulting marginal probabilities for arm duels, see :cite:`busa2014preference`.

    Parameters
    ----------
    num_arms
        The size of the preference matrix to generate.
    random_state
        The numpy random state that will be used for sampling and the ground truth or spread, if they are not given.
    ground_truth_ranking
        Optional, an ordering of ``num_arms`` indices from 0 to ``num_arms-1`. The arm indices are assumed to be ordered from best to worst.
    spread
        Optional, determines the spread of the resulting probability distribution from the ground truth.
    """

    def __init__(
        self,
        num_arms: int,
        random_state: np.random.RandomState,
        ground_truth_ranking: Optional[List[int]] = None,
        spread: float = 0.5,
    ):

        if ground_truth_ranking is not None:
            if len(ground_truth_ranking) != num_arms:
                raise ValueError(
                    "The num_arms parameter needs to be equal to the length of the given ground truth ranking."
                )
            if all(np.sort(ground_truth_ranking) == np.arange(num_arms)):
                raise ValueError(
                    "All indices from 0 to num_arms-1 must occur exactly once in the ground truth ranking."
                )
        else:
            ground_truth_ranking = list(self.random_state.permutation(num_arms))
        self._ground_truth_ranking = ground_truth_ranking
        arm_rank = np.argsort(
            ground_truth_ranking
        )  # invert the permutation to get the rank of an arm (via its index)
        self._best_arm = arm_rank[0]

        if spread <= 0 or spread > 1:
            raise ValueError(
                "The spread parameter must be larger than 0 and smaller or equal to 1."
            )
        self.spread = spread

        def h_function(k: int) -> float:
            """See h in the reference."""
            return k / (1 - np.pow(spread, k))

        def g_function(rank_1: int, rank_2: int) -> float:
            """See g in the reference."""
            difference = rank_2 - rank_1
            return h_function(difference + 1) - h_function(difference)

        preferences = np.full((num_arms, num_arms), 0.5)
        for first_arm_idx in range(num_arms):
            for second_arm_idx in range(first_arm_idx):
                relative_preference = g_function(
                    arm_rank[first_arm_idx], arm_rank[second_arm_idx]
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
        return [self._best_arm]

    def get_arbitrary_ranking(self) -> List[int]:
        """Get any correct ranking of the arms.

        Returns
        -------
        List[int]
            Any correct ranking of the arms, must not be the only correct one.
        """
        return self._ground_truth_ranking

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
        return all(np.array(ranking) == np.array(self._ground_truth_ranking))
