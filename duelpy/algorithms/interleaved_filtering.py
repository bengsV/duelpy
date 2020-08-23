"""Find the condorcet winner in a PB-MAB problem with Interleaved Filtering."""

from typing import Tuple

import numpy as np

from duelpy.feedback import FeedbackMechanism
from duelpy.stats.preference_estimate import PreferenceEstimate


class InterleavedFiltering:
    """Implements the Interleaved Filtering algorithm.

    This is an explore-then-exploit algorithm assuming a total order over arms, strong stochastic transitivity,
    and the stochastic triangle inequality. The Interleaved Filtering algorithm gives the Condorcet winner which is the best arm in the provided set of arms.
    The algorithm is explained in [1]_.

    Exploration:-
    The Interleaved Filtering follows a sequential elimination approach in the exploration phase and thereby
    finds the best arm with probability at least 1 - delta. At each time step, the algorithm
    selects an arm and compares it with all the other arms in a one-versus-all manner.
    If the algorithm selects an arm "a" (candidate arm), then it compares all the other arms with "a". If there exists any arm, "b"
    such that upper confidence bound of "a" beating "b" is less than 1/2, then arm "a" is eliminated and arm "b" becomes
    the candidate arm and is compared to all other active arms. It applies a pruning technique to
    eliminate arm "b" if the lower confidence bound of "a" beating "b" is greater than 1/2,
    as it cannot be considered as the best arm. After the exploration, the candidate arm and the total number of comparisons are given as output.
    It is possible that total number of comparisons is greater than the given time horizon.

    Exploitation:-
    If the total number of comparisons is less than the given time horizon then the algorithm enters into the exploitation phase.
    In the exploitation phase, only the best arm from the exploration phase is pulled and compared to itself, assuming that the exploration found the best arm.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        For how many time steps the algorithm should run (time horizon (T) greater than or equal to number of arms).
    arms
        The set of arms available to the algorithm.
    random_state
        Optional, used for random choices in the algorithm.

    Attributes
    ----------
    failure_probability
        Allowed failure-probability (delta), i.e. probability that the actual value lies outside of the computed confidence interval.
        Derived from the Hoeffding bound.
    candidate_arm
        Randomly selected arm (b^) from the list of arms.
    arms_without_candidate
        The remaining set of arms (W) after removing the candidate arm.
    confidence_bounds
        Upper and lower confidence bounds on the preference probability of the candidate arm against all other arms.
    preference_estimate
        Estimation of a preference matrix based on samples.
    feedback_mechanism
    arms
    time_horizon

    References
    ----------
    .. [1] Yisong Yue, Josef Broder, Robert Kleinberg, and Thorsten Joachims. The K-armed Dueling Bandits Problem.
    Journal of Computer and System Sciences, 78(5):1538–1556, 2012.\

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
    >>> feedback_mechanism = PreferenceMatrix(preference_matrix, random_state=np.random.RandomState(3))
    >>> time_horizon = 7  # time horizon greater than or equal to number of arms.
    >>> arms = list(range(len(preference_matrix)))
    >>> interleaved_filtering = InterleavedFiltering(time_horizon, arms, feedback_mechanism)
    >>> condorcet_winner = interleaved_filtering.interleaved_filter()
    >>> print("In this example the condorcet winner is the arm with index",condorcet_winner)
    In this example the condorcet winner is the arm with index 2

    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        time_horizon: int,
        arms: list,
        feedback_mechanism: FeedbackMechanism,
        random_state: np.random.RandomState = np.random.RandomState(),
    ) -> None:
        # Each step in the code refers to lines of Interleaved Filter algorithm presented in the paper.
        # Step 1: Input the arms and the time horizon.
        self.arms = arms
        self.time_horizon = time_horizon
        self.feedback_mechanism = feedback_mechanism
        # Step 2: Calculation of failure probability (delta).
        self.failure_probability = 1 / (time_horizon * (len(self.arms) ** 2))
        # Step 3: Choosing a random initial candidate arm (b^).
        self.candidate_arm = random_state.choice(self.arms)
        self.arms_without_candidate = self.arms.copy()
        # Step 4: Remaining arms after selecting the candidate arm.
        self.arms_without_candidate.remove(self.candidate_arm)
        # Step 5 and 6: Maintain estimates and confidence intervals.
        self.confidence_bounds = np.zeros((len(self.arms_without_candidate), 2))

        def confidence_radius(num_samples: int) -> float:
            if num_samples == 0:
                return 1 / 2
            return np.sqrt(
                (4 * np.math.log(1 / self.failure_probability)) / num_samples
            )

        self.preference_estimate = PreferenceEstimate(
            confidence_radius=confidence_radius, num_arms=feedback_mechanism.num_arms
        )

    def interleaved_filter(self) -> int:
        """Find the condorcet winner using Interleaved Filtering.

        Returns
        -------
        candidate_arm
           The condorcet winner in the set of arms given to the algorithm.
        """
        candidate_arm, total_comparisons = self.explore()
        if total_comparisons < self.time_horizon:
            self.exploit(total_comparisons)
        return candidate_arm

    def explore(self) -> Tuple[int, int]:  # pylint: disable=too-many-locals
        """Exploration phase of the Interleaved Filtering algorithm to find the candidate arm, which is the condorcet winner.

        Returns
        -------
        candidate_arm
           The condorcet winner in the set of arms given to the algorithm.
        total_comparisons
           The total number of comparisons (T^) that are made in finding the condorcet winner.
        """
        total_comparisons = 0
        # Step 7: Check number of arms without the candidate arm is not equal to zero.
        while len(self.arms_without_candidate) != 0:
            # Step 8: For every arm in the list of arms without candidate.
            for index, arm in enumerate(self.arms_without_candidate):
                # Step 9: Compare the arms with the candidate arm.
                self.preference_estimate.enter_sample(
                    self.candidate_arm,
                    arm,
                    self.feedback_mechanism.duel(self.candidate_arm, arm),
                )
                (
                    # Step 10: Update the confidence intervals.
                    self.confidence_bounds[index][0],
                    self.confidence_bounds[index][1],
                ) = self.preference_estimate.get_confidence_interval(
                    self.candidate_arm, arm
                )
                total_comparisons += 1
            # Step 11: End for.
            # Steps 12 to 14: Pruning of arms.
            updated_arms_without_candidate = self.prune_arms()
            # Steps 15 to 22: Finding the candidate arm and removing the new candidate arm from the list of arms.
            (self.arms_without_candidate) = self.find_candidate_arm(
                updated_arms_without_candidate
            )
        # Steps 23 to 24 : Return the candidate arm and the total comparisons made.
        return self.candidate_arm, total_comparisons

    def prune_arms(self) -> list:
        """Eliminate arms that cannot be expected to win against the candidate within the confidence interval.

        Returns
        -------
        updated_arms_without_candidate
           The remaining set of arms after eliminating all the arms which, do not satisfy the condition.
        """
        # A duplicate list of arms without candidate arm, in order to avoid the index out of bounds error while
        # removing an arm from the arms_without_candidate.
        duplicate_arms_without_candidate = np.copy(self.arms_without_candidate)
        for index, arm in enumerate(duplicate_arms_without_candidate):
            # check whether probability_estimate is greater than 1/2 AND 1/2 is not in the confidence_bounds.
            if self.confidence_bounds[index][0] > 1 / 2:
                self.arms_without_candidate.remove(arm)
        updated_arms_without_candidate = self.arms_without_candidate
        return updated_arms_without_candidate

    def find_candidate_arm(self, updated_arms_without_candidate: list) -> list:
        """Find the candidate arm and remove the new candidate arm from the list of updated arms without candidate arm.

        Parameters
        ----------
        updated_arms_without_candidate
            The remaining set of arms after eliminating all the arms whose, lower confidence bound of candidate arm and
            each arm in the set of all arms except candidate arm is greater than 1/2.

        Returns
        -------
        updated_arms_without_candidate
            The updated list of arms after removing the new candidate arm.
        """
        candidate_found = False
        for index, arm in enumerate(updated_arms_without_candidate):
            # check whether, if there is any arm whose probability_estimate is less than 1/2 AND 1/2 is not in
            # the confidence_bounds.
            if self.confidence_bounds[index][1] < 1 / 2:
                self.candidate_arm = arm
                candidate_found = True
                break
        if candidate_found:
            self.arms_without_candidate = updated_arms_without_candidate
            self.arms_without_candidate.remove(self.candidate_arm)
        return updated_arms_without_candidate

    def exploit(self, total_comparisons: int) -> None:
        """Exploitation phase of the Interleaved Filtering algorithm, repeatedly choosing the condorcet winner by comparing with itself.

        Parameters
        ----------
        total_comparisons
           The total number of comparisons (T^) that are made in finding the condorcet winner.
        """
        candidate_arm = self.candidate_arm
        for _ in range(total_comparisons + 1, self.time_horizon):
            if candidate_arm == self.candidate_arm:
                pass
