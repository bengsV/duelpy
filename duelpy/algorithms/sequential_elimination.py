"""An implementation of the Sequential Elimination Algorithm."""

from typing import List
from typing import Optional

import numpy as np

from duelpy.algorithms.interfaces import SingleCopelandProducer
from duelpy.feedback import FeedbackMechanism
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelpy.stats.preference_estimate import PreferenceEstimate
from duelpy.util.exceptions import AlgorithmFinishedException
import duelpy.util.utility_functions as utility


# pylint: disable= too-many-instance-attributes
class SequentialElimination(SingleCopelandProducer):
    r"""Find an epsilon-maximum arm with Sequential Elimination.

    This module implements the Sequential Elimination algorithm which was introduced in :cite:`falahatgar2018limits` to
    find the most preferred arms of a preference-based multi-armed bandit problem.
    The algorithm computes a probably-approximately-correct estimation of the Copeland winner.
    An arm is epsilon maximum (where :math:`\epsilon = \epsilon_u-\epsilon_l`), if it is preferable to other arms with
    probability at least :math:`0.5-\epsilon`.
    If the anchor arm provided to the algorithm is a good anchor element, then there are only m elements
    for which element a is not :math:`\epsilon_l` preferable. This means, all other elements will be eliminated but
    among these m elements, there can be at most m changes of anchor element. Thus, there can be at most m rounds and
    hence we can bound total comparison rounds by :math:`\mathcal{O}(\lvert S \rvert + m^2)`.
    Thus this PAC algorithm reduces the comparisons to at most m elements which are not :math:`\epsilon_l` preferable and
    the remaining n-m elements are :math:`\epsilon_l` perferable and hence are removed with comparison complexity of
    :math:`\mathcal{O}(\lvert S \rvert)`.

    Parameters
    ----------
    feedback_mechanism
        Object used for gathering the feedback of drawing arms.
    time_horizon
        The number of steps that the algorithm is supposed to be run. Specify None for an infinite time horizon.
    failure_probability
        Determines the number of iterations that both arms are compared against. Corresponds to :math:`\delta` in
        :cite:`falahatgar2018limits`. Default value is taken from :cite:`falahatgar2018limits` in section 6 is 0.1.
    bias_lower
        Default value is 0.0. Refer to section 3.1.2 in :cite:`falahatgar2018limits`.
    bias_upper
        Corresponds to :math:`\epsilon` with default value is 0.5, as given in section 6 of paper
        :cite:`falahatgar2018limits`.
    arms_subset
        Represents the list of arms which is sent by other algorithms and is the subset from list of arms
        fetched from feedback_mechanism.
    anchor_arm
        If none is provided, it is selected randomly from the list of arms provided to the algorithm.
        Otherwise, it represents the anchor arm extracted from feedback_mechanism.get_arms().
        A good anchor element is an arm for which every other arm r (being :math:`\epsilon_l` preferable) is deemed bad
        and gets eliminated.

    Attributes
    ----------
    feedback_mechanism
    failure_probability
    preference_estimate

    Raises
    ------
    ValueError
        Raised when the value of upper bias is not greater than lower bias.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> import numpy as np
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.5],
    ...     [0.9, 0.5, 0.5],
    ... ])
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=np.random.RandomState())
    >>> sequential_elimination = SequentialElimination(feedback_mechanism, random_state=np.random.RandomState())
    >>> sequential_elimination.run()
    >>> preferred_arms = [1,2]
    >>> best_arm = sequential_elimination.get_copeland_winner()
    >>> best_arm in preferred_arms
    True
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        random_state: np.random.RandomState = np.random.RandomState(),
        time_horizon: Optional[int] = None,
        failure_probability: float = 0.1,
        bias_lower: float = 0.0,
        bias_upper: float = 0.05,
        arms_subset: Optional[List] = None,
        anchor_arm: Optional[int] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        if bias_upper <= bias_lower:
            raise ValueError("upper bias must be bigger than lower bias.")
        self._bias_upper = bias_upper
        self._bias_lower = bias_lower
        self.failure_probability = failure_probability
        self.feedback_mechanism = feedback_mechanism
        self._random_state = random_state
        self.preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms()
        )

        if arms_subset is None:
            self._remaining_arms: list = self.feedback_mechanism.get_arms()
        else:
            self._remaining_arms = arms_subset.copy()

        if anchor_arm is None:
            random_arms = utility.pop_random(
                self._remaining_arms, random_state=self._random_state
            )
            self._anchor_arm = random_arms.pop(0)

        else:
            self._anchor_arm = anchor_arm
            if self._anchor_arm in self._remaining_arms:
                self._remaining_arms.remove(self._anchor_arm)

    def is_finished(self) -> bool:
        """Determine whether algorithm has completed execution.

        Returns
        -------
        bool
            Whether the algorithm is finished.
        """
        if self.time_horizon is not None:
            return self.feedback_mechanism.get_num_duels() >= self.time_horizon
        else:
            return self.exploration_finished()

    def step(self) -> None:
        """Take a step in the algorithm.

        Includes determining the next sample, asking for feedback once and
        updating the environment candidates based on this new data.
        """
        if not self.exploration_finished():
            self.explore()
        else:
            self.exploit()

    def exploit(self) -> None:
        """Run one step of exploitation."""
        winner = self.get_copeland_winner()
        assert winner is not None
        self.feedback_mechanism.duel(winner, winner)

    def explore(self) -> None:
        """Compare the current anchor arm against a randomly selected arm.

        The anchor arm is updated with the arm beating the current anchor arm and the new anchor arm is compared against
        the remaining arms step by step. Refer to section 3.1.2 in paper :cite:`falahatgar2018limits`.
        """
        # randomly select a competing arm.
        random_competing_arm = utility.pop_random(
            self._remaining_arms, random_state=np.random.RandomState()
        )[0]

        try:
            comparison_result = self.determine_better_arm(
                competing_arm=random_competing_arm,
            )
            if comparison_result:
                # competing arm beats the anchor arm.
                self._anchor_arm = random_competing_arm
        except AlgorithmFinishedException:
            # Algorithm was terminated, explore will not be called anymore.
            pass

    def exploration_finished(self) -> bool:
        """Determine whether the exploration phase is finished.

        If no time horizon is provided, this coincides with is_finished. Once
        this function returns ``True``, the algorithm will have finished
        computing a PAC Copeland winner.
        """
        return len(self._remaining_arms) == 0

    def get_copeland_winner(self) -> Optional[int]:
        """Return the copeland winner arm selected by the algorithm.

        Returns
        -------
        None
            If the algorithm has not concluded.
        int
            If the algorithm has found the winner.
        """
        if self.exploration_finished():
            return self._anchor_arm
        else:
            return None

    def determine_better_arm(self, competing_arm: int) -> bool:
        r"""Determine if competing arm beats anchor arm.

        The calibrated preference probability estimate (:math:`\hat{p}_{i,j}`) for competing arm against anchor arm
        refers to the probability that competing arm is preferred to anchor arm and is calculated based on the number of
        times competing arm has won divided by total number of duels between both arms. This value is updated after each
         duel between them.

        The confidence radius (:math:`\hat{c}`) is calculated such that with proof >= :math:`1-\delta`,
        :math:`\lvert \hat{p}_{i,j} - p_{i,j} \rvert < \hat{c}` after any number of comparisons. Here :math:`1-\delta`
        as mentioned in the paper :cite:`falahatgar2017assumption`, is called confidence value but we have referred it
        as the failure probability.

        The method returns 1 if :math:`\hat{p}_{i,j}` >= :math:`(\epsilon_u + \epsilon_l)/2` otherwise -1 is returned.
        For more details, please refer to appendix section A.1 Compare Algorithm in :cite:`falahatgar2018limits`.

        Parameters
        ----------
        competing_arm
            Arm that challenges the current anchor arm.

        Raises
        ------
        AlgorithmFinishedException
            If the comparison budget is exceeded before the better arm could be
            determined.

        Returns
        -------
        bool
            Whether competing arm is better than anchor arm or not.

        """
        epsilon = (
            self._bias_upper - self._bias_lower
        )  # refer to :math:`\epsilon` in paper :cite:`falahatgar2018limits`.
        bias_mean = (self._bias_upper + self._bias_lower) / 2
        confidence_radius = 0.5
        current_iteration_count = (
            0  # refer to variable 't' in paper :cite:`falahatgar2018limits`
        )
        calibrated_preference_estimate = 0.0

        # number of rounds is selected in such a way that compare method selects the winner with :math:`1-\delta`
        # confidence. See Algorithm 9 of paper :cite:`falahatgar2018limits`.
        rounds_for_iteration = int(
            2 * np.log(2 / self.failure_probability) / (np.power(epsilon, 2))
        )

        def prob_scaling(num_iteration: int) -> float:
            return np.square(2 * num_iteration)

        confidence_radius_fn = HoeffdingConfidenceRadius(
            self.failure_probability, prob_scaling
        )

        # compare two arms multiple times to get an estimate of their winnings. Refer to Algorithm 9 of paper
        # :cite:`falahatgar2018limits`.
        while (
            current_iteration_count < rounds_for_iteration
            and np.absolute(calibrated_preference_estimate - bias_mean)
            <= confidence_radius
        ):
            if self.is_finished():
                raise AlgorithmFinishedException()
            current_iteration_count += 1
            feedback_result = self.feedback_mechanism.duel(
                competing_arm, self._anchor_arm
            )

            self.preference_estimate.enter_sample(
                competing_arm, self._anchor_arm, feedback_result
            )

            calibrated_preference_estimate = (
                self.preference_estimate.get_mean_estimate(
                    competing_arm, self._anchor_arm
                )
                - 0.5
            )
            confidence_radius = confidence_radius_fn(
                self.preference_estimate.get_num_samples(
                    competing_arm, self._anchor_arm
                )
            )

        # refer to algorithm of COMPARE of Algorithm 9 in paper :cite:`falahatgar2018limits`.
        return calibrated_preference_estimate >= bias_mean
