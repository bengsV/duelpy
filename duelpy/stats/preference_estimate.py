"""Utilities for estimating preference matrices based on samples."""

from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Tuple

import numpy as np

from duelpy.feedback.preference_matrix import PreferenceMatrix
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
        # The diagonal values remain 0.5 whereas the other values change after sampling.
        preference_matrix_sample = np.full((self.num_arms, self.num_arms), 0.5)
        random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

        # Fill the preference matrix `preference_matrix_sample`(denoted by q).
        # Sample the values for q[i][j] such that i < j and fill q[j][i] = 1 - q[i][j].
        for first_arm in range(self.num_arms):
            for second_arm in range(first_arm + 1, self.num_arms):
                preference_matrix_sample[first_arm][second_arm] = random_state.beta(
                    self.wins.get((first_arm, second_arm), 0) + 1,
                    self.wins.get((second_arm, first_arm), 0) + 1,
                )
                preference_matrix_sample[second_arm][first_arm] = (
                    1 - preference_matrix_sample[first_arm][second_arm]
                )

        return PreferenceMatrix(preference_matrix_sample, random_state=random_state)

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
