"""Utilities for estimating preference matrices based on samples."""

from typing import Dict
from typing import FrozenSet
from typing import Tuple

import numpy as np

from duelpy.stats.confidence_radius import ConfidenceRadius
from duelpy.stats.confidence_radius import TrivialConfidenceRadius


class PreferenceEstimate:
    """An estimation of a preference matrix based on samples.

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
        self.wins: Dict[Tuple[int, int], int] = dict()
        self.num_samples: Dict[FrozenSet[int], int] = dict()
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
            self.wins[(first_arm_index, second_arm_index)] = (
                self.wins.get((first_arm_index, second_arm_index), 0) + 1
            )
        else:
            self.wins[(second_arm_index, first_arm_index)] = (
                self.wins.get((second_arm_index, first_arm_index), 0) + 1
            )
        # Order does not matter here, hence index with a set.
        self.num_samples[frozenset((second_arm_index, first_arm_index))] = (
            self.num_samples.get(frozenset((second_arm_index, first_arm_index)), 0) + 1
        )

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
        wins = self.wins.get((first_arm_index, second_arm_index), 0)
        if samples == 0:
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
        return self.num_samples.get(frozenset((first_arm_index, second_arm_index)), 0)

    def get_upper_estimate_matrix(self) -> np.array:
        """Compute the current upper confidence matrix.

        Returns
        -------
        np.array
            2D-array representing the upper confidence bounds of the preference probabilities.
        """
        upper_estimate_matrix: np.array = np.zeros((self.num_arms, self.num_arms))
        for first_arm in range(self.num_arms):
            for second_arm in range(self.num_arms):
                upper_estimate_matrix[first_arm][second_arm] = self.get_upper_estimate(
                    first_arm, second_arm
                )
        return upper_estimate_matrix

    def get_lower_estimate_matrix(self) -> np.array:
        """Compute the current lower confidence matrix.

        Returns
        -------
        np.array
            2D-array representing the lower confidence bounds of the preference probabilities.
        """
        lower_estimate_matrix: np.array = np.zeros((self.num_arms, self.num_arms))
        for first_arm in range(self.num_arms):
            for second_arm in range(self.num_arms):
                lower_estimate_matrix[first_arm][second_arm] = self.get_lower_estimate(
                    first_arm, second_arm
                )
        return lower_estimate_matrix

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
