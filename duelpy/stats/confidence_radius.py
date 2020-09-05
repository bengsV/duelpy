"""Implementation of various confidence intervals with different assumptions."""

from typing import Callable

import numpy as np


class ConfidenceRadius:
    """An abstract superclass for confidence-radius definitions."""

    def __call__(self, num_samples: int) -> float:
        """Compute the confidence radius.

        Parameters
        ----------
        num_samples
            The number of samples that were already taken.

        Returns
        -------
        float
            The confidence radius around the empirical mean.
        """
        raise NotImplementedError


class TrivialConfidenceRadius(ConfidenceRadius):
    """A trivial confidence radius that contains no information.

    Only useful as a place-holder. Always returns a confidence-radius of 1.
    """

    def __call__(self, num_samples: int) -> float:
        """Compute the confidence radius.

        Parameters
        ----------
        num_samples
            The number of samples that were already taken.

        Returns
        -------
        float
            The confidence radius around the empirical mean.
        """
        return 1


class HoeffdingConfidenceRadius(ConfidenceRadius):
    """A confidence radius based on Hoeffding's inequality and the Union bound.

    Parameters
    ----------
    failure_probability
        The probability that the actual value does not lie within the computed
        confidence interval.
    probability_scaling_factor
        A factor by which to scale the failure_probability, dependent on the
        number of samples that were already taken. This is often useful when
        multiple random variables are estimated using this confidence interval
        and we want to bound the union of their failures. In that case, it can
        be necessary to scale the failure probability of any individual
        estimate.

    Attributes
    ----------
    failure_probability
    probability_scaling_factor
    """

    def __init__(
        self,
        failure_probability: float,
        probability_scaling_factor: Callable[[int], float] = lambda num_samples: 1,
    ):
        self.failure_probability = failure_probability
        self.probability_scaling_factor = probability_scaling_factor

    def __call__(self, num_samples: int) -> float:
        """Compute the confidence radius.

        The random variable will deviate by at most this radius from the
        empirical mean estimate with high probability.

        For more details about the Hoeffding inequality and the derivation of
        this confidence radius, refer to `wikipedia`_ and re-solve for ``t``.

        .. _wikipedia: https://en.wikipedia.org/wiki/Hoeffding%27s_inequality#Confidence_intervals

        Parameters
        ----------
        num_samples
            The number of samples that were already taken.

        Returns
        -------
        float
            The confidence radius around the empirical mean.
        """
        if num_samples == 0:
            return 1
        adjusted_probability = (
            self.probability_scaling_factor(num_samples) / self.failure_probability
        )
        return np.sqrt(1 / (2 * num_samples) * np.log(adjusted_probability))
