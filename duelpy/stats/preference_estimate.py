"""Utilities for estimating preference matrices based on samples."""

from typing import Callable
from typing import Tuple

import numpy as np

from duelpy.stats.confidence_radius import ConfidenceRadius
from duelpy.stats.confidence_radius import TrivialConfidenceRadius
from duelpy.stats.preference_matrix import PreferenceMatrix


class PreferenceEstimate:
    """An estimation of a preference matrix based on samples.

    Consider this example:

    >>> preference_estimate = PreferenceEstimate(
    ...     num_arms = 3,
    ...     confidence_radius=TrivialConfidenceRadius(0.5)
    ... )

    In the beginning, nothing is known yet.

    >>> preference_estimate.get_mean_estimate_matrix()
    array([[0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5]])
    >>> preference_estimate.get_upper_estimate_matrix()
    array([[0.5, 1. , 1. ],
           [1. , 0.5, 1. ],
           [1. , 1. , 0.5]])
    >>> preference_estimate.get_lower_estimate_matrix()
    array([[0.5, 0. , 0. ],
           [0. , 0.5, 0. ],
           [0. , 0. , 0.5]])

    If we enter a sampled win, the estimated probability of that arm increases
    and the inverse probability decreases accordingly.

    >>> preference_estimate.enter_sample(0, 1, first_won=True)
    >>> preference_estimate.get_mean_estimate_matrix()
    array([[0.5, 1. , 0.5],
           [0. , 0.5, 0.5],
           [0.5, 0.5, 0.5]])

    When entering more samples, the probability keeps adjusting. Let's make it
    one win out of four.
    >>> preference_estimate.enter_sample(0, 1, first_won=False)
    >>> preference_estimate.enter_sample(0, 1, first_won=True)
    >>> preference_estimate.enter_sample(0, 1, first_won=True)
    >>> preference_estimate.get_mean_estimate_matrix()
    array([[0.5 , 0.75, 0.5 ],
           [0.25, 0.5 , 0.5 ],
           [0.5 , 0.5 , 0.5 ]])

    Meanwhile the confidence intervals have adjusted as well:

    >>> preference_estimate.get_upper_estimate_matrix()
    array([[0.5 , 1.  , 1.  ],
           [0.75, 0.5 , 1.  ],
           [1.  , 1.  , 0.5 ]])
    >>> preference_estimate.get_lower_estimate_matrix()
    array([[0.5 , 0.25, 0.  ],
           [0.  , 0.5 , 0.  ],
           [0.  , 0.  , 0.5 ]])

    And if we tighten the confidence radius, they get changed yet again:

    >>> preference_estimate.set_confidence_radius(TrivialConfidenceRadius(0.1))
    >>> preference_estimate.get_upper_estimate_matrix()
    array([[0.5 , 0.85, 0.6 ],
           [0.35, 0.5 , 0.6 ],
           [0.6 , 0.6 , 0.5 ]])
    >>> preference_estimate.get_lower_estimate_matrix()
    array([[0.5 , 0.65, 0.4 ],
           [0.15, 0.5 , 0.4 ],
           [0.4 , 0.4 , 0.5 ]])

    We can now also sample a complete preference matrix from a beta
    distribution:

    >>> preference_estimate.sample_preference_matrix(
    ...     random_state=np.random.RandomState(42)
    ... )
    array([[0.5       , 0.72606244, 0.4978376 ],
           [0.27393756, 0.5       , 0.44364733],
           [0.5021624 , 0.55635267, 0.5       ]])

    Parameters
    ----------
    num_arms
        The number of arms in the estimated preference matrix.
    confidence_radius
        The confidence radius to use when computing confidence intervals.
    """

    def __init__(
        self,
        num_arms: int,
        confidence_radius: ConfidenceRadius = TrivialConfidenceRadius(),
    ) -> None:
        self.num_arms = num_arms
        self.wins = np.zeros((num_arms, num_arms))
        self.confidence_radius = confidence_radius

    def set_confidence_radius(self, confidence_radius: ConfidenceRadius) -> None:
        """Set the confidence radius to the given parameter.

        Parameters
        ----------
        confidence_radius
            The confidence radius to be set as the new `confidence_radius`.
        """
        self.confidence_radius = confidence_radius

    def enter_sample(
        self, first_arm_index: int, second_arm_index: int, first_won: bool
    ) -> None:
        """Enter the result of a sampled duel.

        Parameters
        ----------
        first_arm_index
            The index of the first arm of the duel.
        second_arm_index
            The index of the second arm of the duel.
        first_won
            Whether the first arm won the duel.
        """
        # It would be possible to normalize the order instead of duplicating
        # the information. That would restrict us to comparable arm
        # representations though.
        if first_won:
            self.wins[first_arm_index][second_arm_index] += 1
        else:
            self.wins[second_arm_index][first_arm_index] += 1

    def get_mean_estimate(self, first_arm_index: int, second_arm_index: int) -> float:
        """Get the estimate of the win probability of `first_arm_index` against `second_arm_index`.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        float
            The estimated probability that `first_arm_index` wins against `second_arm_index`.
        """
        samples = self.get_num_samples(first_arm_index, second_arm_index)
        wins = self.wins[first_arm_index, second_arm_index]
        if samples == 0 or first_arm_index == second_arm_index:
            return 1 / 2
        else:
            return wins / samples

    def get_confidence_interval(
        self, first_arm_index: int, second_arm_index: int
    ) -> Tuple[float, float]:
        """Get the bounds of the confidence interval on the win probability.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        Tuple[float, float]
            The lower and upper bound of the confidence estimate for the
            probability that `first_arm_index` wins against `second_arm_index`.
        """
        if first_arm_index == second_arm_index:
            return 0.5, 0.5
        mean = self.get_mean_estimate(first_arm_index, second_arm_index)
        confidence_radius = self.confidence_radius(
            self.get_num_samples(first_arm_index, second_arm_index)
        )
        return max(mean - confidence_radius, 0), min(mean + confidence_radius, 1)

    def get_upper_estimate(self, first_arm_index: int, second_arm_index: int) -> float:
        """Get the upper estimate of the win probability of `first_arm_index` against `second_arm_index`.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        float
            The upper bound of the confidence estimate for the probability that `first_arm_index` wins against `second_arm_index`.
        """
        if first_arm_index == second_arm_index:
            return 1 / 2
        mean = self.get_mean_estimate(first_arm_index, second_arm_index)
        confidence_radius = self.confidence_radius(
            self.get_num_samples(first_arm_index, second_arm_index)
        )
        return min(mean + confidence_radius, 1)

    def get_lower_estimate(self, first_arm: int, second_arm: int) -> float:
        """Get the lower estimate of the win probability of `first_arm` against `second_arm`.

        Parameters
        ----------
        first_arm
            The first arm of the duel.
        second_arm
            The second arm of the duel.

        Returns
        -------
        float
            The lower bound of the confidence estimate for the probability that `first_arm` wins against `second_arm`.
        """
        if first_arm == second_arm:
            return 1 / 2
        mean = self.get_mean_estimate(first_arm, second_arm)
        confidence_radius = self.confidence_radius(
            self.get_num_samples(first_arm, second_arm)
        )
        return max(mean - confidence_radius, 0)

    def get_num_samples(self, first_arm_index: int, second_arm_index: int) -> int:
        """Get the number of times a duel between first_arm and second_arms was sampled.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        int
            The number of times a duel between the two arms was sampled,
            regardless of the arm order.
        """
        return (
            self.wins[first_arm_index][second_arm_index]
            + self.wins[second_arm_index][first_arm_index]
        )

    def _estimate_to_matrix(
        self, estimate_function: Callable[[int, int], float]
    ) -> PreferenceMatrix:
        matrix = np.zeros((self.num_arms, self.num_arms))
        for first_arm_idx in range(self.num_arms):
            for second_arm_idx in range(self.num_arms):
                matrix[first_arm_idx, second_arm_idx] = estimate_function(
                    first_arm_idx, second_arm_idx
                )
        return PreferenceMatrix(matrix)

    def get_mean_estimate_matrix(self) -> PreferenceMatrix:
        """Get the current mean estimates as a PreferenceMatrix.

        Returns
        -------
        PreferenceMatrix
            The current mean estimate.
        """
        return self._estimate_to_matrix(self.get_mean_estimate)

    def get_upper_estimate_matrix(self) -> PreferenceMatrix:
        """Get the current upper estimates as a PreferenceMatrix.

        Returns
        -------
        PreferenceMatrix
            The current mean estimate.
        """
        return self._estimate_to_matrix(self.get_upper_estimate)

    def get_lower_estimate_matrix(self) -> PreferenceMatrix:
        """Get the current lower estimates as a PreferenceMatrix.

        Returns
        -------
        PreferenceMatrix
            The current mean estimate.
        """
        return self._estimate_to_matrix(self.get_lower_estimate)

    def sample_preference_matrix(
        self, random_state: np.random.RandomState
    ) -> PreferenceMatrix:
        """Sample a preference matrix based on a Beta distribution.

        The outcome is a PreferenceMatrix object which is initialized from a sampled
        preference matrix. In this preference matrix, each pairwise preference is
        drawn from a beta-distribution which is parameterized on the results of prior
        duels.

        Parameters
        ----------
        random_state
            A numpy random state.

        Returns
        -------
        PreferenceMatrix
            A PreferenceMatrix object which is initialized from a preference matrix which
            is sampled on a Beta distribution.
        """
        # Construct the parameters of a beta distribution to sample preference
        # probabilities.
        beta_a = self.wins + 1
        beta_b = beta_a.T
        # Only the upper triangle is important, the rest is adjusted afterwards.
        upper_triangle_preferences = random_state.beta(beta_a, beta_b)
        return PreferenceMatrix.from_upper_triangle(upper_triangle_preferences)

    def __str__(self) -> str:
        """Produce a string representation of the estimate."""
        result = ""
        for first_arm_index in range(self.num_arms):
            row = f"{first_arm_index} |"
            for second_arm_index in range(self.num_arms):
                mean = self.get_mean_estimate(first_arm_index, second_arm_index)
                radius = self.confidence_radius(
                    self.get_num_samples(first_arm_index, second_arm_index)
                )
                row += "  {:.2f}+-{:.2f}".format(mean, radius)
            result += row
            result += "\n"
        return result
