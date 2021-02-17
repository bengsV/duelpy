"""An implementation of the Multisort algorithm."""
from typing import List
from typing import Optional

import numpy as np

from duelpy.algorithms.interfaces import CopelandRankingProducer
from duelpy.feedback import FeedbackMechanism
from duelpy.util.exceptions import AlgorithmFinishedException
from duelpy.util.sorting import Quicksort


class Multisort(CopelandRankingProducer):
    r"""Implements the Multisort algorithm.

    The goal of the algorithm is to find a ranking using Copeland aggregation on a set of rankings returned by the
    Quicksort algorithm.

    It is assumed that comparison outcomes are noiseless, i.e., a total order over arms in a set of :math:`n`
    items. It is also assumed that there are no ties between arms, i.e. the probability of any arm winning against
    another is never 1/2.

    Theorem 2 in Section 3.1 in :cite:`maystre2017just` states that all but a vanishing fraction of the items are
    correctly ranked using :math:`O(\lambda^2 n\log^6n)` comparisons, where :math:`n` refers to the number of arms.

    This algorithm recursively builds a Copeland ranking over the arms by sorting them using Quicksort with random
    pivot element in each time step. Quicksort returns a partial ranking of the pairwise comparisons and termiates
    after sampling :math:`O(n\log n)` comparisons with high probability. After having an aggregated Copeland
    scores over time horizon :math:`T`, an aggregated Copeland Ranking is produced based on these scores.
    Multisort is neither a PAC (sample-complexity minimizing) algorithm nor a regret minimizing algorithm. Instead,
    it tries to come up with the best result possible in the given time horizon. This differs from the PAC setting,
    since it requires a time horizon. The probability of failure and the accuracy of the result are implicitly set by this time horizon.
    It differs from the regret-minimizing setting since it will never exploit its gathered knowledge. It will always
    "explore" and try to find a more accurate result, as long as the time horizon allows and regardless of the regret that is incurred during exploration.

    See :cite:`maystre2017just` for details.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment
    time_horizon
        The maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Used for the random pivot selection in the Quicksort mode.

    Attributes
    ----------
    feedback_mechanism
    random_state

    Examples
    --------
    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3, 0.2, 0.2],
    ...     [0.9, 0.7, 0.5, 0.8, 0.9],
    ...     [0.9, 0.8, 0.2, 0.5, 0.2],
    ...     [0.9, 0.8, 0.1, 0.8, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(3)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)
    >>> multisort = Multisort(feedback_mechanism=feedback_mechanism, time_horizon=1000, random_state=random_state)
    >>> multisort.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> ranking = multisort.get_ranking()
    >>> ranking, comparisons
    ([2, 4, 3, 1, 0], 1000)
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: Optional[np.random.RandomState] = None,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self._ranking: Optional[List[int]] = None
        self._copeland_agg_scores = np.zeros(
            self.feedback_mechanism.get_num_arms(), dtype=int
        )
        self._quicksort_instance = Quicksort(
            items=self.feedback_mechanism.get_arms(),
            compare_fn=self._determine_better_arm,
            random_state=random_state,
        )

    def explore(self) -> None:
        """Explore arms by advancing the sorting algorithm."""
        if self._quicksort_instance.is_finished():
            # Use the result to compute the Copeland aggregation
            self._ranking = self._quicksort_instance.get_result()
            if self._ranking is not None:
                for index, arm in enumerate(self._ranking):
                    self._copeland_agg_scores[arm] += (
                        self.feedback_mechanism.get_num_arms() - index
                    )
            # re-initialize quicksort with a new random seed to get a different ranking with a random pivot element.
            self._quicksort_instance = Quicksort(
                items=self.feedback_mechanism.get_arms(),
                compare_fn=self._determine_better_arm,
                random_state=self.random_state,
            )
        try:
            self._quicksort_instance.step()
        except AlgorithmFinishedException:
            pass

    def step(self) -> None:
        """Execute one step of the algorithm."""
        self.explore()

    def get_ranking(self) -> Optional[List[int]]:
        """Get the Copeland aggregation ranking.

        Returns
        -------
        Optional[List[int]]
            The ranking, None if it has not been calculated yet.
        """
        # return a sorted list in decreasing order of aggregated Copeland scores
        # this means that the arm at first position has the highest Copeland score i.e. the number of items that it
        # beats in a majority of the rankings after quicksort
        return list(np.argsort(self._copeland_agg_scores)[::-1])

    def _determine_better_arm(self, arm_1: int, arm_2: int) -> int:
        """Take a single sample of the pairwise preference between two arms.

        Parameters
        ----------
        arm_1
            The first arm.
        arm_2
            The second arm.

        Raises
        ------
        AlgorithmFinishedException
            If the comparison budget is reached.

        Returns
        -------
        int
            1 if the first arm is better, -1 if the second arm is better.
        """
        if self.is_finished():
            raise AlgorithmFinishedException()
        first_arm_won = self.feedback_mechanism.duel(arm_1, arm_2)
        if first_arm_won:
            return 1
        else:
            return -1
