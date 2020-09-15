"""Find the Condorcet winner in a PB-MAB problem with Interleaved Filtering."""

from typing import List

import numpy as np

from duelpy.algorithms.algorithm import Algorithm
from duelpy.feedback import FeedbackMechanism
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelpy.stats.preference_estimate import PreferenceEstimate


class InterleavedFiltering(Algorithm):
    r"""Implements the Interleaved Filtering algorithm.

    This is an explore-then-exploit algorithm assuming a total order over arms, strong stochastic transitivity,
    and the stochastic triangle inequality. The Interleaved Filtering algorithm [1]_ gives the Condorcet winner which is the best arm in the provided set of arms.
    The algorithm is explained in [1]_.

    Exploration:

    Interleaved Filtering follows a sequential elimination approach in the exploration phase and thereby
    finds the best arm with probability at least  1-1/T, where T is the time horizon. At each time step, the algorithm
    selects a candidate arm and compares it with all the other arms in a one-versus-all manner.
    If the algorithm selects an arm "a" (candidate arm), then it compares all the other arms with "a". If there exists any arm, "b"
    such that upper confidence bound of "a" beating "b" is less than 1/2, then arm "a" is eliminated and arm "b" becomes
    the candidate arm and is compared to all other active arms. It applies a pruning technique to
    eliminate arm "b" if the lower confidence bound of "a" beating "b" is greater than 1/2,
    as it cannot be considered as the best arm with high probability. After the exploration, the candidate arm and the total number of comparisons are given as output.

    Exploitation:

    If the total number of comparisons is less than the given time horizon then the algorithm enters into the exploitation phase.
    In the exploitation phase, only the best arm from the exploration phase is pulled and compared to itself, assuming that the exploration found the best arm.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        For how many time steps the algorithm should run (must be >= the number of arms).
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.

    Attributes
    ----------
    failure_probability
        Allowed failure-probability (:math:`\delta`), i.e. probability that the actual value lies outside of the computed confidence interval.
        Derived from the Hoeffding bound.
    candidate_arm
        Randomly selected arm (corresponds to :math:`\hat{b}` in [1]_) from the list of arms.
    arms_without_candidate
        The remaining set of arms (corresponds to W in [1]_) after removing the candidate arm.
    preference_estimate
        Estimation of a preference matrix based on samples.
    total_comparisons
        Total number of comparisons (corresponds to :math:`\hat{T}` in [1]_) made to find the condorcet winner.
    feedback_mechanism
    time_horizon

    References
    ----------
    .. [1] Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. The K-armed Dueling Bandits Problem. Journal of Computer and System Sciences, 78(5):1538–1556, 2012.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelpy.feedback import PreferenceMatrix
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5],
    ... ])
    >>> random_state=np.random.RandomState(3)
    >>> feedback_mechanism = PreferenceMatrix(preference_matrix, random_state=random_state)
    >>> time_horizon = 750
    >>> interleaved_filtering = InterleavedFiltering(feedback_mechanism, time_horizon, random_state=random_state)
    >>> interleaved_filtering.run(time_horizon)
    >>> condorcet_winner = interleaved_filtering.get_condorcet_winner()
    >>> condorcet_winner
    2
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.RandomState = np.random.RandomState(),
    ) -> None:
        self.time_horizon = time_horizon
        self.feedback_mechanism = feedback_mechanism
        self.failure_probability = 1 / (
            self.time_horizon * (self.feedback_mechanism.get_num_arms() ** 2)
        )
        self.candidate_arm = random_state.choice(self.feedback_mechanism.get_arms())
        self.arms_without_candidate = self.feedback_mechanism.get_arms().copy()
        self.arms_without_candidate.remove(self.candidate_arm)
        # See the proof of Lemma 4 in [1]_. The factor `8` corresponds to `m` in the proof.
        self.preference_estimate = PreferenceEstimate(
            feedback_mechanism.get_num_arms(),
            confidence_radius=HoeffdingConfidenceRadius(
                self.failure_probability, factor=8,
            ),
        )
        self.total_comparisons = 0

    def get_condorcet_winner(self) -> int:
        """Return the estimated Condorcet winner, assuming the algorithm has already run.

        Returns
        -------
        candidate_arm
           The condorcet winner in the set of arms given to the algorithm.
        """
        return self.candidate_arm

    def explore(self) -> None:
        r"""Execute one round of exploration."""
        for arm in self.arms_without_candidate:
            self.preference_estimate.enter_sample(
                self.candidate_arm,
                arm,
                self.feedback_mechanism.duel(self.candidate_arm, arm),
            )
            self.total_comparisons += 1
            # Terminate explore
            if self.total_comparisons == self.time_horizon:
                break
        updated_arms_without_candidate = self._prune_arms()
        (self.arms_without_candidate) = self._find_candidate_arm(
            updated_arms_without_candidate
        )

    def _prune_arms(self) -> list:
        """Eliminate arms that cannot be expected to win against the candidate within the confidence interval.

        Returns
        -------
        updated_arms_without_candidate
           The remaining set of arms after eliminating all the arms which, do not satisfy the condition.
        """
        # A duplicate list of arms without candidate arm, in order to avoid the index out of bounds error while
        # removing an arm from the arms_without_candidate.
        duplicate_arms_without_candidate = np.copy(self.arms_without_candidate)
        for arm in duplicate_arms_without_candidate:
            # check whether probability_estimate is greater than 1/2 AND 1/2 is not in the confidence_bounds.
            if (
                self.preference_estimate.get_lower_estimate(self.candidate_arm, arm)
                > 1 / 2
            ):
                self.arms_without_candidate.remove(arm)
        updated_arms_without_candidate = self.arms_without_candidate
        return updated_arms_without_candidate

    def _find_candidate_arm(
        self, updated_arms_without_candidate: List[int]
    ) -> List[int]:
        """Find the candidate arm and remove the new candidate arm from the list of updated arms without candidate arm.

        Parameters
        ----------
        updated_arms_without_candidate
            The remaining set of arms after eliminating all the arms whose, lower confidence bound of candidate arm and
            each arm in the set of all arms except candidate arm is > 1/2.

        Returns
        -------
        updated_arms_without_candidate
            The updated list of arms after removing the new candidate arm.
        """
        candidate_found = False
        for arm in updated_arms_without_candidate:
            # check whether, if there is any arm whose probability_estimate is less than 1/2 AND 1/2 is not in
            # the confidence_bounds.
            if (
                self.preference_estimate.get_upper_estimate(self.candidate_arm, arm)
                < 1 / 2
            ):
                self.candidate_arm = arm
                candidate_found = True
                break
        if candidate_found:
            self.arms_without_candidate = updated_arms_without_candidate
            self.arms_without_candidate.remove(self.candidate_arm)
        return updated_arms_without_candidate

    def exploit(self) -> None:
        """Execute one round of exploitation.

        This simply compares the estimated Condorcet winner against itself, thereby making the best possible choice based on the available information.
        """
        self.feedback_mechanism.duel(self.candidate_arm, self.candidate_arm)

    def step(self) -> None:
        """Execute one step of the algorithm."""
        while len(self.arms_without_candidate) != 0:
            self.explore()
        self.exploit()

    def run(self, rounds: int) -> None:
        """Run the algorithm with a given comparison budget.

        Parameters
        ----------
        rounds
            The number of rounds for which the algorithm is run.
        """
        if self.total_comparisons < rounds:
            self.step()
