"""Utilities for estimating preference matrices based on samples."""

from typing import Callable
from typing import Dict
from typing import FrozenSet
from typing import Tuple


class PreferenceEstimate:
    """An estimation of a preference matrix based on samples.

    Parameters
    ----------
    num_arms
        The number of arms in the estimated preference matrix.
    confidence_radius
        A function that computes the radius of a confidence interval given the
        number of samples that were already performed.
    """

    def __init__(
        self, num_arms: int, confidence_radius: Callable[[int], float]
    ) -> None:
        self.num_arms = num_arms
        self.wins: Dict[Tuple[int, int], int] = dict()
        self.num_samples: Dict[FrozenSet[int], int] = dict()
        self.confidence_radius = confidence_radius

    def enter_sample(self, first_arm: int, second_arm: int, first_won: bool) -> None:
        """Enter the result of a sampled duel.

        Parameters
        ----------
        first_arm
            The first arm of the duel.
        second_arm
            The second arm of the duel.
        first_won
            Whether the first arm won the duel.
        """
        # It would be possible to normalize the order instead of duplicating
        # the information. That would restrict us to comparable arm
        # representations though.
        if first_won:
            self.wins[(first_arm, second_arm)] = (
                self.wins.get((first_arm, second_arm), 0) + 1
            )
        else:
            self.wins[(second_arm, first_arm)] = (
                self.wins.get((second_arm, first_arm), 0) + 1
            )
        # Order does not matter here, hence index with a set.
        self.num_samples[frozenset((second_arm, first_arm))] = (
            self.num_samples.get(frozenset((second_arm, first_arm)), 0) + 1
        )

    def get_mean_estimate(self, first_arm: int, second_arm: int) -> float:
        """Get the estimate of the win probability of `first_arm` against `second_arm`.

        Parameters
        ----------
        first_arm
            The first arm of the duel.
        second_arm
            The second arm of the duel.

        Returns
        -------
        float
            The estimated probability that `first_arm` wins against `second_arm`.
        """
        samples = self.get_num_samples(first_arm, second_arm)
        wins = self.wins.get((first_arm, second_arm), 0)
        if samples == 0:
            return 1 / 2
        else:
            return wins / samples

    def get_confidence_interval(
        self, first_arm: int, second_arm: int
    ) -> Tuple[float, float]:
        """Get the bounds of the confidence interval on the win probability.

        Parameters
        ----------
        first_arm
            The first arm of the duel.
        second_arm
            The second arm of the duel.

        Returns
        -------
        Tuple[float, float]
            The lower and upper bound of the confidence estimate for the
            probability that `first_arm` wins against `second_arm`.
        """
        if first_arm == second_arm:
            return (0.5, 0.5)
        mean = self.get_mean_estimate(first_arm, second_arm)
        confidence_radius = self.confidence_radius(
            self.get_num_samples(first_arm, second_arm)
        )
        return max(mean - confidence_radius, 0), min(mean + confidence_radius, 1)

    def get_upper_estimate(self, first_arm: int, second_arm: int) -> float:
        """Get the upper estimate of the win probability of `first_arm` against `second_arm`.

        Parameters
        ----------
        first_arm
            The first arm of the duel.
        second_arm
            The second arm of the duel.

        Returns
        -------
        float
            The upper bound of the confidence estimate for the probability that `first_arm` wins against `second_arm`.
        """
        if first_arm == second_arm:
            return 1 / 2
        mean = self.get_mean_estimate(first_arm, second_arm)
        confidence_radius = self.confidence_radius(
            self.get_num_samples(first_arm, second_arm)
        )
        return min(mean + confidence_radius, 1)

    def get_num_samples(self, first_arm: int, second_arm: int) -> int:
        """Get the number of times a duel between first_arm and second_arms was sampled.

        Parameters
        ----------
        first_arm
            The first arm of the duel.
        second_arm
            The second arm of the duel.

        Returns
        -------
        int
            The number of times a duel between the two arms was sampled,
            regardless of the arm order.
        """
        return self.num_samples.get(frozenset((first_arm, second_arm)), 0)

    def __str__(self) -> str:
        """Produce a string representation of the estimate."""
        result = ""
        for first_arm in range(self.num_arms):
            row = f"{first_arm} |"
            for second_arm in range(self.num_arms):
                mean = self.get_mean_estimate(first_arm, second_arm)
                radius = self.confidence_radius(
                    self.get_num_samples(first_arm, second_arm)
                )
                row += "  {:.2f}+-{:.2f}".format(mean, radius)
            result += row
            result += "\n"
        return result
