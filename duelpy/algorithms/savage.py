"""PAC best arm selection with the SAVAGE algorithm."""


from typing import Set
from typing import Tuple

import numpy as np

from duelpy.feedback import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius


class Savage:
    r"""Determine the PAC-best arm with the SAVAGE algorithm.

    SAVAGE is a general algorithm that can infer some information about an
    environment from samples. It works by repeatedly sampling possible
    environments (in the case of PB-MAB, an environment is specified by a
    preference matrix) and eliminating

    - those environment candidates that fall outside of the current confidence
      interval (for example the sets of preference matrices that would make our
      previous samples too unlikely) and
    - those environment variables (preference matrix entries) that are no
      longer relevant on the current environment candidates (for example the
      arms that cannot be the Copeland winner). See Figure 1 in [1]_ for an
      illustration. In this case :math:`\mu` is the preference matrix while
      :math:`x_1` and :math:`x_2` are two entries of the matrix (without loss
      of generality it is sufficient to estimate the upper-right triangle of
      the matrix). If we already know that arm i is strictly better than arm j,
      it is no longer necessary to test arm i and we can stop trying to improve
      our estimate on :math:`q_{ik}`.

    Environment parameters in the PB-MAB case are the upper triangle of the preference matrix.
    The goal goal is to design a sequence of pairwise experiments (samples of
    random variables) / duels to find the best arm (according to ranking
    procedure). This is called "voting bandits" since we use pairwise election
    criterion to find best bandit (such as "beating" -> Copeland, "better
    expectation" -> Borda).

    Parameters
    ----------
    feedback_mechanism
        The feedback mechanism that specifies the underlying problem.
    failure_probability
        Upper bound on the probability of failure (the "delta" in
        epsilon-delta-PAC).
    verbose
        Whether to log the internal state. This is only for the testbed, we
        should come up with a more elegant method of accessing the internal
        state during algorithm execution for interactive applications.

    Attributes
    ----------
    feedback_mechanism
    failure_probability
    verbose
    preference_estimate
        The current estimate of the preference matrix.

    References
    ----------
    This implements the Sensitivity Analysis of VAriables for Generic
    Exploration (SAVAGE) [1]_ algorithm.

    .. [1] Urvoy, Tanguy, et al. "Generic exploration and k-armed voting
           bandits." International Conference on Machine Learning. 2013.
           (http://proceedings.mlr.press/v28/urvoy13-supp.pdf)

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
    >>> feedback_mechanism = PreferenceMatrix(preference_matrix, random_state=np.random.RandomState(42))

    Obviously, the last arm (index 2) is expected to win against the most other
    arms. That makes it the Copeland winner, as SAVAGE is correctly able to
    determine:

    >>> algorithm = Savage(feedback_mechanism)
    >>> algorithm.run()
    >>> list(algorithm.get_pac_copeland_winners())[0]
    2
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        failure_probability: float = 0.1,
        verbose: bool = False,
    ):
        self.feedback_mechanism = feedback_mechanism
        self.failure_probability = failure_probability
        self.verbose = verbose

        # The number of random variables that we attempt to estimate
        # (corresponds to the upper triangle of the preference matrix).
        num_arms = self.feedback_mechanism.get_num_arms()
        num_random_variables = num_arms * (num_arms - 1) / 2

        # The failure probability of each individual confidence interval must be
        # scaled appropriately so that the probability that *any* estimate fails is
        # sufficiently low (as set by failure_probability). That is achieved by a
        # naive Union bound. See page 10 of
        # http://proceedings.mlr.press/v28/urvoy13-supp.pdf for a detailed
        # derivation of this bound. Intuitively, we scale the allowed failure
        # probability down proportional the number of random variables we are
        # estimating. Since the time horizon is unknown (infinite-horizon case) we
        # additionally scale by the square of the current sample to make sure the
        # infinite sum converges.
        def union_bound_scaling_factor(num_samples: int) -> float:
            return 3 / (np.pi ** 2 * num_random_variables * num_samples ** 2)

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability, probability_scaling_factor=union_bound_scaling_factor
        )

        # Estimate the preference matrix based on past samples. Keeps track of the
        # t_i and \hat\mu_i variables in the paper.
        self.preference_estimate = PreferenceEstimate(
            feedback_mechanism.get_num_arms(), confidence_radius
        )

        # Initialize with all possible arm pairings, without loss of generality the
        # first arm has the lower index. This maintains a list of all pairwise win
        # probabilities we are not sufficiently sure about yet, i.e. which may
        # still influence our result. Corresponds to the W set array in the
        # reference paper.
        self._relevant_arm_combinations = {
            (i, i + j) for i in range(num_arms) for j in range(1, num_arms - i)
        }

    def copeland_independence_test(self, arm_pair: Tuple[int, int]) -> bool:
        """Test if the result of a duel can still influence our estimate of the Copeland winner.

        This corresponds to the "IndepTest" in the paper.

        Parameters
        ----------
        arm_pair
            The pair of arms in question.

        Returns
        -------
        bool
            Whether more information about the arm pair is still needed.
        """
        # Set of viable hypotheses is represented implicitly by a set of confidence
        # intervals.
        (lower_bound, upper_bound) = self.preference_estimate.get_confidence_interval(
            *arm_pair
        )
        if lower_bound > 1 / 2 or upper_bound < 1 / 2:
            # We already know which arm is expected to win. How probable
            # its win is is not important for the Copeland score.
            return True

        # The remainder of the function corresponds to the "Cop" check in the
        # paper.
        # Determines whether we already know that some other arm has a
        # higher Copeland score (with at least 1-failure_probability probability).
        # Compute pessimistic estimates for all Copeland scores.
        num_arms = self.feedback_mechanism.get_num_arms()
        expected_wins = np.zeros(num_arms)
        for arm in range(num_arms):
            for other_arm in range(arm + 1, num_arms):
                (
                    lower_bound,
                    upper_bound,
                ) = self.preference_estimate.get_confidence_interval(arm, other_arm)
                if lower_bound > 1 / 2:
                    expected_wins[arm] += 1
                elif upper_bound < 1 / 2:
                    expected_wins[other_arm] += 1
        most_certain_wins = np.max(expected_wins)

        # Compute optimistic estimates for the arm pair.
        for arm in arm_pair:
            possible_wins = 0
            for other_arm in range(num_arms):
                if other_arm == arm:
                    continue
                (_, upper_bound) = self.preference_estimate.get_confidence_interval(
                    arm, other_arm
                )
                if upper_bound > 1 / 2:
                    possible_wins += 1
            # There is still something interesting to learn.
            if possible_wins > most_certain_wins:
                return False

        return True

    def step(self) -> None:
        """Take a step in the algorithm.

        Includes determining the next sample, asking for feedback once and
        updating the environment candidates based on this new data.
        """
        # Find the next arm to sample. This could probably be optimized by choosing
        # a better data structure, but I'm trying to keep it simple and relatively
        # close to the paper for now.
        next_sample = None
        current_lowest_sample_count = np.infty
        for arm_pair in self._relevant_arm_combinations:
            if (
                self.preference_estimate.get_num_samples(*arm_pair)
                < current_lowest_sample_count
            ):
                next_sample = arm_pair
                current_lowest_sample_count = self.preference_estimate.get_num_samples(
                    *arm_pair
                )

        # To keep mypy happy. Cannot happen due to initialization of
        # current_lowest_sample_count.
        assert next_sample is not None

        # Sample a duel and keep track of the results.
        self.preference_estimate.enter_sample(
            *next_sample, self.feedback_mechanism.duel(*next_sample)
        )
        if self.verbose:
            # Printing for the interactive test. This is not ideal and should
            # not be done in the final implementation. Maybe we should
            # generally implement a way to run an algorithm step-by-step.
            print("Preference estimate is now")
            print(self.preference_estimate)

        self._relevant_arm_combinations.difference_update(
            {
                arm_pair
                for arm_pair in self._relevant_arm_combinations
                if self.copeland_independence_test(arm_pair)
            }
        )

    def is_finished(self) -> bool:
        """Determine whether enough data for a PAC prediction is available.

        Once this function returns ``True``, you can query the
        probably-approximately-correct result with the
        ``get_pac_copeland_winner`` function.

        Returns
        -------
        bool
            Whether the algorithm is finished.
        """
        # When making the Condorcet assumption, the termination condition could be
        # replaced by one allowing for an epsilon-approximation. See Section 4.1.2
        # in the reference paper.
        return len(self._relevant_arm_combinations) > 0

    def run(self) -> None:
        """Run the algorithm until it can make a prediction.

        The prediction can then be queried with the ``get_pac_copeland_winner`` function.
        """
        while not self.is_finished():
            self.step()

    def get_pac_copeland_winners(self) -> Set[int]:
        """Find a Copeland winner with the SAVAGE algorithm.

        Note that only the correctness of any one of the Copeland winners is
        covered by the failure probability. The probability that all arms in
        the set are actually Copeland winners is lower. We still return the
        full set of arms for convenience.

        Returns
        -------
        Set[int]
            The indices of the delta-PAC best (Copeland) arms. The "delta"
            failure probability refers to any individual arm, but not all arms
            together.
        """
        return (
            self.preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )
