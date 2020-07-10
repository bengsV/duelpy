"""PAC best arm selection with the SAVAGE algorithm."""


from typing import Tuple

import numpy as np

from duelpy.feedback import FeedbackMechanism
from duelpy.stats import PreferenceEstimate


# corresponds to "IndepTest" in the paper.
def copeland_independence_test(
    preference_estimate: PreferenceEstimate, num_arms: int, arm_pair: Tuple[int, int]
) -> bool:
    """Test if the result of a duel can still influence our estimate of the Copeland winner.

    Parameters
    ----------
    preference_estimate
        The current state of knowledge about the preference matrix.
    num_arms
        The number of arms in the PB-MAB problem.
    arm_pair
        The pair of arms in question.

    Returns
    -------
    bool
        Whether more information about the arm pair is still needed.
    """
    # Set of viable hypotheses is represented implicitly by a set of confidence
    # intervals.
    (lower_bound, upper_bound) = preference_estimate.get_confidence_interval(*arm_pair)
    if lower_bound > 1 / 2 or upper_bound < 1 / 2:
        # We already know which arm is expected to win. How probable
        # its win is is not important for the Copeland score.
        return True

    # The remainder of the function corresponds to the "Cop" check in the
    # paper.
    # Determines whether we already know that some other arm has a
    # higher Copeland score (with at least 1-delta probability).
    # Compute pessimistic estimates for all Copeland scores.
    expected_wins = np.zeros(num_arms)
    for arm in range(num_arms):
        for other_arm in range(arm + 1, num_arms):
            (lower_bound, upper_bound) = preference_estimate.get_confidence_interval(
                arm, other_arm
            )
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
            (_, upper_bound) = preference_estimate.get_confidence_interval(
                arm, other_arm
            )
            if upper_bound > 1 / 2:
                possible_wins += 1
        # There is still something interesting to learn.
        if possible_wins > most_certain_wins:
            return False

    return True


def savage(
    num_arms: int,
    feedback_mechanism: FeedbackMechanism,
    delta: float = 0.1,
    verbose: bool = False,
) -> int:
    r"""Determine the probably-approximately-correct best arm.

    This algorithm works by repeatedly sampling possible environments (in the
    case of PB-MAB an environment is specified by a preference matrix) and
    eliminating

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

    Environment parameters here are K[i, j] (K preference matrix) with i < j
    (chosen arbitrarily to reduce redundancy) Goal: sequence of pairwise
    experiments (samples of random variables) / duels to find the best arm
    (according to ranking procedure) "voting" means we use pairwise election
    criterion to find best bandit (such as "beating" -> copeland, "better
    expectation" -> borda).

    Parameters
    ----------
    num_arms
        The number of arms in the PB-MAB problem.
    delta
        Upper bound on the probability of failure.
    verbose
        Whether to log the internal state. This is only for the testbed, we
        should come up with a more elegant method of accessing the internal
        state during algorithm execution for interactive applications.


    Returns
    -------
    int
        The index of the epsilon-delta-PAC best (Copeland) arm.

    References
    ----------
    This implements the Sensitivity Analysis of VAriables for Generic
    Exploration (SAVAGE) [1]_ algorithm.

    .. [1] Urvoy, Tanguy, et al. "Generic exploration and k-armed voting
           bandits." International Conference on Machine Learning. 2013.
           (http://proceedings.mlr.press/v28/urvoy13.pdf)

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
    >>> feedback_mechanism = PreferenceMatrix(preference_matrix, random=np.random.RandomState(42))

    Obviously, the last arm (index 2) is expected to win against the most other
    arms. That makes it the copeland winner:
    >>> savage(num_arms=3, feedback_mechanism=feedback_mechanism)
    2
    """
    # Based on Hoeffding + Union Bound. Might be interesting to experiment with
    # more advanced methods, such as https://arxiv.org/pdf/1905.06208.pdf.
    def confidence_radius(num_samples: int) -> float:
        if num_samples == 0:
            return 1
        # Possible number of combinations of two arms
        num_random_variables = num_arms * (num_arms - 1) / 2
        # eta = 2NT when finite, pi^2 N t^2/3 otherwise
        eta = (np.pi ** 2) * num_random_variables * (num_samples ** (2 / 3))

        # Radius in which where the true value lies with probability at
        # least (1-delta) (according to the Hoeffding bound).
        return np.sqrt(1 / (2 * num_samples) * np.log(eta / delta))

    # Initialize with all possible arm pairings, without loss of generality the
    # first arm has the lower index. This maintains a list of all pairwise win
    # probabilities we are not sufficiently sure about yet, i.e. which may
    # still influence our result. Corresponds to the W set array in the
    # reference paper.
    relevant_arm_combinations = {
        (i, i + j) for i in range(num_arms) for j in range(1, num_arms - i)
    }
    # Estimate the preference matrix based on past samples. Keeps track of the
    # t_i and \hat\mu_i variables in the paper.
    preference_estimate = PreferenceEstimate(
        confidence_radius=confidence_radius, num_arms=num_arms
    )

    # When making the Condorcet assumption, the termination condition could be
    # replaced by one allowing for an epsilon-approximation. See Section 4.1.2
    # in the reference paper.
    while len(relevant_arm_combinations) > 0:
        # Find the next arm to sample. This could probably be optimized by choosing
        # a better data structure, but I'm trying to keep it simple and relatively
        # close to the paper for now.
        next_sample = None
        current_lowest_sample_count = np.infty
        for arm_pair in relevant_arm_combinations:
            if (
                preference_estimate.get_num_samples(*arm_pair)
                < current_lowest_sample_count
            ):
                next_sample = arm_pair
                current_lowest_sample_count = preference_estimate.get_num_samples(
                    *arm_pair
                )

        # To keep mypy happy. Cannot happen due to initialization of
        # current_lowest_sample_count.
        assert next_sample is not None

        # Sample a duel and keep track of the results.
        preference_estimate.enter_sample(
            *next_sample, feedback_mechanism.duel(*next_sample)
        )
        if verbose:
            # Printing for the interactive test. This is not ideal and should
            # not be done in the final implementation. Maybe we should
            # generally implement a way to run an algorithm step-by-step.
            print("Preference estimate is now")
            print(preference_estimate)

        relevant_arm_combinations.difference_update(
            {
                arm_pair
                for arm_pair in relevant_arm_combinations
                if copeland_independence_test(preference_estimate, num_arms, arm_pair)
            }
        )

    # Assuming the mean arm estimates are true, find a Copeland winner
    expected_wins = np.zeros(num_arms)
    for arm in range(num_arms):
        for other_arm in range(arm + 1, num_arms):
            if preference_estimate.get_mean_estimate(arm, other_arm) > 1 / 2:
                expected_wins[arm] += 1
            else:
                expected_wins[other_arm] += 1
    return np.argmax(expected_wins)
