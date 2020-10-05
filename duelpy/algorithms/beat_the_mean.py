"""Find the Condorcet winner in a PB-MAB problem using the 'Beat the Mean Bandit' algorithm."""
from typing import List
from typing import Tuple

import numpy as np

from duelpy.feedback import FeedbackMechanism
from duelpy.stats.confidence_radius import ConfidenceRadius
from duelpy.stats.confidence_radius import TrivialConfidenceRadius
from duelpy.util.utility_functions import argmax_set
from duelpy.util.utility_functions import argmin_set


class BeatTheMeanBandit:
    """Implements the 'Beat the Mean Bandit' algorithm.

    This is an explore-then-exploit algorithm assuming a total order over arms, relaxed stochastic transitivity and the stochastic triangle inequality.
    The Beat the Mean algorithm :cite:`yue2011beat` gives the Condorcet winner which is the best arm in the provided set of arms.
    It proceeds in a sequence of rounds, and maintains a working set of active arms during each round.
    For each active arm, an empirical estimate is maintained for how likely an arm is to beat the mean bandit of the working set.
    In each iteration, an arm with the fewest recorded comparisons is selected for comparison.
    The algorithm terminates only when one active arm remains, or when time horizon is reached.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        The total number of rounds.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.

    Attributes
    ----------
    comparison_history
        A ComparisonHistory object which stores the history of the comparisons between the arms.
    feedback_mechanism
    time_horizon
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5],
    ... ])
    >>> random_state = np.random.RandomState(43)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=random_state)
    >>> time_horizon = 100  # time horizon greater than or equal to number of arms.
    >>> btm = BeatTheMeanBandit(feedback_mechanism=feedback_mechanism, time_horizon=time_horizon, random_state=random_state)
    >>> btm.run()
    >>> best_arm = btm.get_condorcet_winner()
    >>> best_arm
    2
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.RandomState,
    ) -> None:
        self.feedback_mechanism = feedback_mechanism
        self.time_horizon = time_horizon
        self.random_state = random_state
        self._excluding_indices: List[int] = []
        self._time_step = 0
        self.comparison_history = ComparisonHistory(
            number_of_arms=self.feedback_mechanism.get_num_arms()
        )

    def step(self) -> None:
        """Take a step in the algorithm.

        This includes choosing arms, comparing them and updating the estimates.
        """
        arm1, arm2 = self.comparison_history.get_arms(
            random_state=self.random_state, excluding_indices=self._excluding_indices,
        )
        if arm1 == arm2:
            return  # If both the arms are same, skip that iteration.
        first_won = self.feedback_mechanism.duel(arm_i_index=arm1, arm_j_index=arm2)
        self.comparison_history.enter_sample(arm1=arm1, arm2=arm2, first_won=first_won)
        self._time_step += 1
        if (
            self.comparison_history.get_lower_bound()
            <= self.comparison_history.get_upper_bound()
        ):
            # Check if the empirically worst bandit is separated from the empirically best one by a sufficient confidence margin.
            worst_arm = self.comparison_history.get_worst_arm(
                exclude_indices=self._excluding_indices
            )
            self.remove_from_working_set(worst_arm)

    def is_finished(self) -> bool:
        """Determine whether the termination conditions are met.

        Returns
        -------
        bool
            Whether the algorithm is finished.
        """
        # When making the Condorcet assumption, the algorithm terminates only when one active arm remains, or when time horizon is reached.
        return (len(self.comparison_history.working_set) <= 1) or (
            self._time_step >= self.time_horizon
        )

    def run(self) -> None:
        """Run the algorithm until it can make a prediction.

        The prediction can then be queried with ```get_condorcet_winner``` function
        """
        while not self.is_finished():
            self.step()

    def remove_from_working_set(self, worst_arm: int) -> None:
        """Remove the worst arm from the working set and update the comparison statistics.

        Parameters
        ----------
        worst_arm
            The index of the empirically worst arm.
        """
        self.comparison_history.remove_arm_history(worst_arm=worst_arm)
        self._excluding_indices.append(worst_arm)

    def get_condorcet_winner(self) -> int:
        """Return the arm with highest empirical estimate.

        Returns
        -------
        int
            The arm with the highest empirical estimate.
        """
        return argmax_set(
            array=self.comparison_history.probability_estimate,
            exclude_indexes=self._excluding_indices,
        )[0]


class ComparisonHistory:
    """Store the comparison history of the working set.

    Parameters
    ----------
    number_of_arms
        The number of arms in the estimated preference matrix.
    confidence_radius
        The confidence radius to use when computing confidence intervals.

    Attributes
    ----------
    working_set
        Stores the set of active arms. Corresponds to W1 in :cite:`yue2011beat`.
    comparisons
        Stores the total number of comparisons of a specific arm. Corresponds to n_b in :cite:`yue2011beat`.
    wins
        Stores the number of wins of each arm. Corresponds to w_b in :cite:`yue2011beat`.
    probability_estimate
        Stores the empirical estimate of arms versus the mean bandit. Corresponds to P_b in :cite:`yue2011beat`.
    arm_sum_comparisons
        A numpy array which stores the total number of comparisons of each arm in the corresponding index.
    """

    def __init__(
        self,
        number_of_arms: int,
        confidence_radius: ConfidenceRadius = TrivialConfidenceRadius(),  # for basic implementation only.
    ) -> None:
        self.number_of_arms = number_of_arms
        self.confidence_radius = confidence_radius
        self.working_set = np.arange(self.number_of_arms)
        self.comparisons = np.zeros((number_of_arms, number_of_arms), dtype=int)
        self.wins = np.zeros((number_of_arms, number_of_arms), dtype=int)
        self.probability_estimate = np.full(number_of_arms, 0.5)
        self.arm_sum_comparisons = np.zeros(number_of_arms, dtype=int)

    def get_arms(
        self, random_state: np.random.RandomState, excluding_indices: List[int],
    ) -> Tuple[int, int]:
        """Return the least sampled arm and a random challenger.

        Parameters
        ----------
        excluding_indices
            A list containing the worst_arms.
        random_state
            A numpy random state. Defaults to an unseeded state when not specified.

        Returns
        -------
        arm1
            The index of challenger arm from the working_set
        arm2
            The index of arm to compare against.
        """
        for arm in self.working_set:
            self.arm_sum_comparisons[arm] = np.sum(self.comparisons[arm, :])
        arm1 = random_state.choice(
            argmin_set(
                array=self.arm_sum_comparisons, exclude_indexes=excluding_indices,
            )
        )  # break ties randomly excluding the worst_arms

        arm2 = random_state.choice(
            self.working_set
        )  # select arm2 from the current working_set at random
        return arm1, arm2

    def enter_sample(self, arm1: int, arm2: int, first_won: bool) -> None:
        """Enter the result of a sampled duel.

        Parameters
        ----------
        arm1
            The index of the first arm of the duel.
        arm2
            The index of the second arm of the duel.
        first_won
            Whether the first arm won the duel.
        """
        if first_won:
            self.wins[arm1][arm2] += 1
        self.comparisons[arm1][arm2] += 1
        self.comparisons[arm2][arm1] += 1
        self.set_probability_estimate(arm=arm1)

    def set_probability_estimate(self, arm: int) -> None:
        """Set the estimate of the win probability of the arm.

        Parameters
        ----------
        arm
            The arm for which the win probability is to be estimated.
        """
        wins = np.sum(self.wins[arm, :])
        comparisons = np.sum(self.comparisons[arm, :])
        if np.sum(self.comparisons[arm, :]) == 0:
            self.probability_estimate[arm] = 0.5
        else:
            self.probability_estimate[arm] = wins / comparisons

    def remove_arm_history(self, worst_arm: int) -> None:
        """Remove the worst arm and the associated comparison history.

        Remove the empirically worst arm from the working set and update the comparisons, wins and probability estimate of each arm in the current working set.

        Parameters
        ----------
        worst_arm
            The index of the empirically worst arm.
        """
        self.comparisons[worst_arm, :] = 0
        self.comparisons[:, worst_arm] = 0
        self.wins[:, worst_arm] = 0
        self.wins[worst_arm, :] = 0
        for arm in self.working_set:
            self.set_probability_estimate(arm=arm)  # update the probability_estimate
        self.working_set = np.delete(
            self.working_set, np.argwhere(self.working_set == worst_arm)
        )

    def get_worst_arm(self, exclude_indices: List[int]) -> int:
        """Get the empirically worst arm from the current working_set.

        Parameters
        ----------
        exclude_indices
            A list containing the worst_arms.

        Returns
        -------
        int
            The index of the empirically worst arm in the current working_set.
        """
        return argmin_set(
            array=self.probability_estimate, exclude_indexes=exclude_indices
        )[0]

    def get_upper_bound(self) -> float:
        """Get the upper bound for empirically worst arm.

        Returns
        -------
        float
            The upper bound for empirically worst arm.
        """
        return min(self.probability_estimate) + self.confidence_radius(
            self.number_of_arms
        )

    def get_lower_bound(self) -> float:
        """Get the lower bound for empirically best arm.

        Returns
        -------
        float
            The lower bound for empirically best arm.
        """
        return max(self.probability_estimate) - self.confidence_radius(
            self.number_of_arms
        )
