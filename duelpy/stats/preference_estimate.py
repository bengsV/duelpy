"""Utilities for estimating preference matrices based on samples."""

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
        self._cached_mean_estimate = np.full((num_arms, num_arms), 0.5)
        self._cached_radius = np.full(
            (num_arms, num_arms), confidence_radius(0), dtype=np.float64
        )
        np.fill_diagonal(self._cached_radius, 0.0)

    def set_confidence_radius(self, confidence_radius: ConfidenceRadius) -> None:
        """Set the confidence radius to the given parameter.

        Parameters
        ----------
        confidence_radius
            The confidence radius to be set as the new `confidence_radius`.
        """
        self.confidence_radius = confidence_radius
        self._cached_radius = np.full((self.num_arms, self.num_arms), np.nan)
        np.fill_diagonal(self._cached_radius, 0.0)

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

        if first_arm_index == second_arm_index:
            # Nothing to update, the preference is known.
            return

        # based on wins array, already updated
        samples = self.get_num_samples(first_arm_index, second_arm_index)

        prev = self._cached_mean_estimate[first_arm_index][second_arm_index]
        win_indicator = 1 if first_won else 0
        new_mean = prev + (win_indicator - prev) / samples

        self._cached_mean_estimate[first_arm_index][second_arm_index] = new_mean
        self._cached_mean_estimate[second_arm_index][first_arm_index] = 1 - new_mean
        # Confidence radius estimates are computed on-demand, since they are
        # not always necessary and need to be changed when the confidence
        # radius changes.
        self._cached_radius[first_arm_index][second_arm_index] = np.nan
        self._cached_radius[second_arm_index][first_arm_index] = np.nan

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
        return self._cached_mean_estimate[first_arm_index][second_arm_index]

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
        return (
            self.get_lower_estimate(first_arm_index, second_arm_index),
            self.get_upper_estimate(first_arm_index, second_arm_index),
        )

    def _get_confidence_radius(self, first_arm_idx: int, second_arm_idx: int) -> float:
        """Get the confidence radius and fill the cache if necessary.

        Parameters
        ----------
        first_arm_idx
            The first arm of the duel.
        second_arm_idx
            The second arm of the duel.

        Returns
        -------
        float
            The current confidence value.
        """
        if np.isnan(self._cached_radius[first_arm_idx][second_arm_idx]):
            num_samples = self.get_num_samples(first_arm_idx, second_arm_idx)
            radius = self.confidence_radius(num_samples)
            self._cached_radius[first_arm_idx][second_arm_idx] = radius
            self._cached_radius[second_arm_idx][first_arm_idx] = radius
        return self._cached_radius[first_arm_idx][second_arm_idx]

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
        return min(
            self._cached_mean_estimate[first_arm_index][second_arm_index]
            + self._get_confidence_radius(first_arm_index, second_arm_index),
            1,
        )

    def get_lower_estimate(self, first_arm_index: int, second_arm_index: int) -> float:
        """Get the lower estimate of the win probability of `first_arm` against `second_arm`.

        Parameters
        ----------
        first_arm_index
            The first arm of the duel.
        second_arm_index
            The second arm of the duel.

        Returns
        -------
        float
            The lower bound of the confidence estimate for the probability that `first_arm` wins against `second_arm`.
        """
        return max(
            self._cached_mean_estimate[first_arm_index][second_arm_index]
            - self._get_confidence_radius(first_arm_index, second_arm_index),
            0,
        )

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

    def _get_radius_matrix(self) -> np.array:
        """Seed the confidence radius cache and return it.

        Returns
        -------
        np.array
            A numpy matrix containing the current confidence radius values.
        """
        for (first_idx, second_idx) in np.argwhere(np.isnan(self._cached_radius)):
            # Seed the cache.
            self._get_confidence_radius(first_idx, second_idx)
        return self._cached_radius

    def get_mean_estimate_matrix(self) -> PreferenceMatrix:
        """Get the current mean estimates as a PreferenceMatrix.

        Returns
        -------
        PreferenceMatrix
            The current mean estimate.
        """
        return PreferenceMatrix(self._cached_mean_estimate)

    def get_upper_estimate_matrix(self) -> PreferenceMatrix:
        """Get the current upper estimates as a PreferenceMatrix.

        Returns
        -------
        PreferenceMatrix
            The current mean estimate.
        """
        return PreferenceMatrix(
            np.clip(
                self._cached_mean_estimate + self._get_radius_matrix(), a_min=0, a_max=1
            )
        )

    def get_lower_estimate_matrix(self) -> PreferenceMatrix:
        """Get the current lower estimates as a PreferenceMatrix.

        Returns
        -------
        PreferenceMatrix
            The current mean estimate.
        """
        return PreferenceMatrix(
            np.clip(
                self._cached_mean_estimate - self._get_radius_matrix(), a_min=0, a_max=1
            )
        )

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
