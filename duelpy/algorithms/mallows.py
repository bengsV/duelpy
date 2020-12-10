"""Implementations of PB-MAB algorithms based on the Mallows model."""
from typing import List
from typing import Optional
from typing import Type

import numpy as np

from duelpy.algorithms.interfaces import CondorcetProducer
from duelpy.algorithms.interfaces import CopelandRankingProducer
from duelpy.feedback import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelpy.util.exceptions import AlgorithmFinishedException
from duelpy.util.sorting import MergeSort
from duelpy.util.sorting import SortingAlgorithm
from duelpy.util.utility_functions import pop_random


class MallowsMPI(CondorcetProducer):
    r"""Implementation of the Mallows Most Preferred Item algorithm.

    This algorithm finds the best arm with a given confidence. It is part of the epsilon-delta-PAC class of
    algorithms, with epsilon as 0. Best arm refers to the arm ranked first with the highest probability in the Mallows
    :math:`\phi`-model, this is also a Condorcet winner. The arms are assumed to be sampled from a Mallows distribution, which is stricter than the total order assumption. See :cite:`busa2014preference` for details on this distribution.
    It proceeds by selecting a random arm and comparing it against another arm until one of
    them can be considered worse than the other with sufficient confidence. The worse arm is discarded and the winner is compared
    against a new randomly chosen arm. This continues until only one arm is left, which is then
    returned. See :cite:`busa2014preference` for more details.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Optional, used for random choices in the algorithm.
    failure_probability
        An upper bound on the acceptable probability to fail, called delta in :cite:`busa2014preference`.


    Attributes
    ----------
    feedback_mechanism
    failure_probability
    random_state
    preference_estimate
        Estimates for arm preferences.


    Examples
    --------
    Find the Condorcet winner in this example:

    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5],
    ... ])
    >>> random_state = np.random.RandomState(3)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix,
    ...                                       random_state=random_state)
    >>> mallows = MallowsMPI(feedback_mechanism, random_state=random_state, failure_probability=0.9)
    >>> mallows.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> arm = mallows.get_condorcet_winner()
    >>> arm, comparisons
    (2, 18)
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        failure_probability: float = 0.05,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.failure_probability = failure_probability
        self.random_state = (
            np.random.RandomState() if random_state is None else random_state
        )

        num_arms = self.feedback_mechanism.get_num_arms()

        def probability_scaling(num_samples: int) -> float:
            return num_arms * (2 * num_samples) ** 2

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability, probability_scaling
        )
        self.preference_estimate = PreferenceEstimate(num_arms, confidence_radius)
        self._current_arms = list(range(num_arms))
        self._best_arm = pop_random(self._current_arms, self.random_state)[0]

    def explore(self) -> None:
        """Explore arms by advancing the sorting algorithm."""
        rival_arm = pop_random(self._current_arms, self.random_state)[0]

        while (
            self.preference_estimate.get_lower_estimate(self._best_arm, rival_arm)
            <= 1 / 2
            and self.preference_estimate.get_upper_estimate(self._best_arm, rival_arm)
            >= 1 / 2
        ):
            result = self.feedback_mechanism.duel(self._best_arm, rival_arm)
            self.preference_estimate.enter_sample(self._best_arm, rival_arm, result)
            if self.is_finished():
                return
        if (
            self.preference_estimate.get_upper_estimate(self._best_arm, rival_arm)
            < 1 / 2
        ):
            self._best_arm = rival_arm

    def exploit(self) -> None:
        """Exploit the found best arm by pulling it twice."""
        self.feedback_mechanism.duel(self._best_arm, self._best_arm)

    def step(self) -> None:
        """Execute one step of the algorithm."""
        if self.is_finished():
            return

        if len(self._current_arms) == 0:
            self.exploit()
        else:
            self.explore()

    def get_condorcet_winner(self) -> Optional[int]:
        """Get the arm with the highest probability of being the first in a ranking of the arms.

        Returns
        -------
        Optional[int]
            The best arm, None if has not been calculated yet.
        """
        if len(self._current_arms) == 0:
            return self._best_arm
        else:
            return None

    def is_finished(self) -> bool:
        """Determine whether the best arm has been found."""
        if self.time_horizon is not None:
            return self.feedback_mechanism.get_num_duels() >= self.time_horizon
        else:
            return len(self._current_arms) == 0


class MallowsMPR(CopelandRankingProducer):
    r"""Implementation of Mallows Most Probable Ranking Algorithm.

    This algorithm recursively builds a Copeland ranking over the arms by sorting them either using Mergesort or Quicksort.
    Arms are compared repeatedly until sufficient confidence is obtained, this confidence is based on the Mallows :math:`\phi`-model.
    The arms are assumed to be sampled from a Mallows distribution, which is stricter than the total order assumption. See :cite:`busa2014preference` for details.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment
    time_horizon
        Optional, the maximum amount of arm comparisons to execute. This may be exceeded, but will always be reached.
    random_state
        Used for the random pivot selection in the Quicksort mode.
    failure_probability
        An upper bound on the acceptable probability to fail, called delta in :cite:`busa2014preference`.
    sorting_mode
        Determines which sort algorithm should be used, 'merge' for Mergesort or 'quick' for Quicksort.


    Attributes
    ----------
    feedback_mechanism
    random_state
    failure_probability
    sorting_algorithm
    preference_estimate
        Estimates for arm preferences.

    Examples
    --------
    Find the Condorcet winner in this example:

    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3, 0.2, 0.2],
    ...     [0.9, 0.7, 0.5, 0.8, 0.9],
    ...     [0.9, 0.8, 0.2, 0.5, 0.2],
    ...     [0.9, 0.8, 0.1, 0.8, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(3)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix,
    ...                                       random_state=random_state)
    >>> mallows = MallowsMPR(feedback_mechanism, random_state=random_state, failure_probability=0.9)
    >>> mallows.run()
    >>> comparisons = feedback_mechanism.get_num_duels()
    >>> ranking = mallows.get_ranking()
    >>> ranking, comparisons
    ([2, 4, 3, 1, 0], 571)
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None,
        failure_probability: float = 0.05,
        sorting_algorithm: Optional[Type[SortingAlgorithm]] = None,
    ):
        super().__init__(feedback_mechanism, time_horizon)

        num_arms = self.feedback_mechanism.get_num_arms()

        self.failure_probability = failure_probability

        arms = list(range(num_arms))
        if sorting_algorithm is None:
            sorting_algorithm = MergeSort
        self._sorting_algorithm = sorting_algorithm(
            arms, self._determine_better_arm, random_state
        )

        self._ranking: Optional[List[int]] = None

        def probability_scaling(num_samples: int) -> float:
            return (
                4
                * self._sorting_algorithm.get_comparison_bound(num_arms)
                * num_samples ** 2
            )

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability, probability_scaling
        )

        self.preference_estimate = PreferenceEstimate(num_arms, confidence_radius)

        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

    def _determine_better_arm(self, arm_1: int, arm_2: int) -> int:
        """Determine whether the first arm is preferred to the second with the given confidence.

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
            1 if the first arm is better, -1 if the second arm is better, 0 if not sure yet.
        """
        if self.is_finished():
            raise AlgorithmFinishedException()
        first_arm_won = self.feedback_mechanism.duel(arm_1, arm_2)
        self.preference_estimate.enter_sample(arm_1, arm_2, first_arm_won)
        if self.preference_estimate.get_lower_estimate(arm_1, arm_2) > 0.5:
            return 1
        elif self.preference_estimate.get_upper_estimate(arm_1, arm_2) < 0.5:
            return -1
        else:
            return 0

    def explore(self) -> None:
        """Explore arms by advancing the sorting algorithm."""
        try:
            self._sorting_algorithm.step()
        except AlgorithmFinishedException:
            pass
        if self._sorting_algorithm.is_finished():
            self._ranking = self._sorting_algorithm.get_result()

    def exploit(self) -> None:
        """Exploit the found ranking by pulling the top-ranked arm twice."""
        assert self._ranking is not None
        self.feedback_mechanism.duel(self._ranking[0], self._ranking[0])

    def step(self) -> None:
        """Execute one step of the algorithm."""
        if self._ranking is None:
            self.explore()
        else:
            self.exploit()

    def is_finished(self) -> bool:
        """Determine whether the ranking has been found."""
        if self.time_horizon is not None:
            return self.time_horizon <= self.feedback_mechanism.get_num_duels()
        else:
            return self._ranking is not None

    def get_ranking(self) -> Optional[List[int]]:
        """Get the computed ranking.

        Returns
        -------
        Optional[List[int]]
            The ranking, None if it has not been calculated yet.
        """
        return self._ranking
