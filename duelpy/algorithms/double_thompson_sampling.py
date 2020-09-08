"""An implementation of the Double Thompson Sampling algorithm for Dueling Bandits."""
from typing import Optional

import numpy as np

from duelpy.algorithms.interfaces import SingleCopelandProducer
from duelpy.feedback import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelpy.util.utility_functions import argmax_set


class DoubleThompsonSampling(SingleCopelandProducer):
    """Implementation of the Double Thompson Sampling algorithm.

    As proposed in paper :cite:`huasen2016dts`, the Double Thompson Sampling (D-TS) algorithm for dueling bandits
    includes both Condorcet dueling bandits and general Copeland dueling bandits. D-TS uses a double sampling structure
    where the first as well as the second candidates are selected according to independently drawn samples
    from the beta posterior distribution and then dueled. The confidence bounds are used to eliminate the
    arms unlikely to be winner arms and thus avoids suboptimal comparisons. While selecting the first arm and the
    second arm, the confidence bounds are used to eliminate non-likely winners.
    The main goal of the algorithm is to minimize regret after each comparison. For analysis of regret, Copeland regret
    is used by this algorithm.

    Compared to prior studies on dueling bandits, D-TS has the following advantages.

    - First, the double sampling structure of D-TS is better suited for dueling bandits nature. Unlike RCS, launching
    two independent rounds of sampling provide us the opportunity to select the same arm in both rounds.
    This allows to compare the winners against themselves which significantly reduced regret. D-TS is more robust
    in practice.

    - Second, this double sampling structure enables us to obtain theoretical bounds for the regret of D-TS.
    D-TS achieves O(K^2 log T) regret for a general K-armed Copeland dueling bandit. Also, D-TS achieves
    O(K log T + K^2 log log T) in Condorcet dueling bandits and many practical Copeland dueling bandits.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    time_horizon
        This states the number of comparision to be done before the algorithm terminates.
    random_state
        Used for random choices in the algorithm.
    exploratory_constant
        Optional, the confidence radius grows proportional to the square root of this value. Corresponds to `alpha` in
        :cite:`huasen2016dts`. The value of exploratory_constant must be greater than 0.5. Default value is 0.51

    Attributes
    ----------
    feedback_mechanism
    exploratory_constant
    random_state
    time_horizon
    preference_estimate
        Stores estimates of arm preferences

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix=preference_matrix, random_state=random_state)
    >>> test_object = DoubleThompsonSampling(feedback_mechanism, random_state=random_state, time_horizon=100)
    >>> test_object.run()
    >>> test_object.get_copeland_winner()
    2
    >>> regret_history, cumul_regret = feedback_mechanism.calculate_average_copeland_regret()
    >>> np.round(cumul_regret, 2)
    17.5
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: int,
        random_state: np.random.RandomState = None,
        exploratory_constant: float = 0.51,
    ):
        super().__init__(feedback_mechanism, time_horizon)
        self.time_step = 0
        self.exploratory_constant = exploratory_constant
        self.preference_estimate = PreferenceEstimate(
            num_arms=self.feedback_mechanism.get_num_arms()
        )
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

    def _update_confidence_radius(self) -> None:
        """Update the confidence radius using latest failure probability.

        Failure probability for the upper confidence bound is `1/t^(2 * alpha)`
        where `t` is the current round of the algorithm and `alpha` is the
        exploratory constant.
        Refer to :cite:`huasen2016dts` for further details.
        """
        failure_probability = 1 / (self.time_step ** (2 * self.exploratory_constant))
        confidence_radius = HoeffdingConfidenceRadius(failure_probability)
        self.preference_estimate.set_confidence_radius(confidence_radius)

    def _choose_first_candidate(self) -> int:
        """Choose a champion arm whose Copeland score is high in a sample.

        Select an arm_c from the potential champion arms whose copeland score is high based on the preference matrix
        computed under beta distribution. If there exist a tie between arms, arm_c is selected randomly.
        Also, upper confidence bound is used to estimate the preference between the arms. So, potential champions
        arms are selected upon normalized copeland scores computed based upon the preference estimate of arms using
        upper confidence bound.

        Return
        ------
        int
            The champion arm with high copeland score.
        """
        # select the potential champion according to their normalized copeland scores from the preference estimate upper confidence bound
        potential_champion = (
            self.preference_estimate.get_upper_estimate_matrix().get_copeland_winners()
        )

        non_potential_champion = (
            set(self.feedback_mechanism.get_arms()) - potential_champion
        )

        # sample preference matrix between the arm through beta distribution
        sample_preference_matrix = self.preference_estimate.sample_preference_matrix(
            self.random_state
        )

        # calculate copeland score for all the arms based on sample preference matrix to remove non-likely winner arms
        # Ties between the arms are broken randomly
        arm_c = self.random_state.choice(
            argmax_set(
                sample_preference_matrix.get_copeland_scores(),
                exclude_indexes=list(non_potential_champion),
            )
        )

        return arm_c

    def _choose_second_candidate(self, champion: int) -> int:
        """Choose challenger arm which is likely to win against the champion.

        Select an arm_d from the potential challenger arms whose preference is high compared with the champion arm
        (arm_c). The preference between the challenger arms with champion arm is based on. If there exist ties between
        arms, arm_d is selected randomly. Also, lower confidence bound is used to estimate the preference between the
        arms, so arms whose lower preference estimate is less than 0.5 are selected as potential challengers.

        Return
        ------
        int
            The challenger arm with high preference than champion arm.
        """
        # preference estimate between the arms using lower confidence bound.
        # select the potential challenger (possible for arm_d)
        potential_challenger = argmax_set(
            self.preference_estimate.get_lower_estimate_matrix().preferences[:][
                champion
            ]
            <= 0.5
        )
        non_potential_challenger = set(self.feedback_mechanism.get_arms()) - set(
            potential_challenger
        )

        # sample preference matrix between the arm through beta distribution
        # sample the preference with champion arm from the sampled preference matrix.
        sample_preference_with_champion = self.preference_estimate.sample_preference_matrix(
            self.random_state
        ).preferences[
            :
        ][
            champion
        ]

        #  Choosing only from uncertain pairs (potential challenger). Ties are broken randomly.
        arm_d = self.random_state.choice(
            argmax_set(sample_preference_with_champion, list(non_potential_challenger))
        )
        return arm_d

    def get_copeland_winner(self) -> Optional[int]:
        """Compute single Copeland winner as per the estimated preference matrix.

        Returns
        -------
        Optional[int]
            Copeland winner from preference estimate.
        """
        if self.is_finished():
            return list(
                self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
            )[0]

        return None

    def step(self) -> None:
        """Run one round of an algorithm."""
        self.time_step += 1
        self._update_confidence_radius()
        arm_c = self._choose_first_candidate()
        arm_d = self._choose_second_candidate(arm_c)
        self.preference_estimate.enter_sample(
            arm_c, arm_d, self.feedback_mechanism.duel(arm_c, arm_d)
        )
