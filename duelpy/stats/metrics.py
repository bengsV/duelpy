"""Common metrics for algorithm performance."""


from typing import Callable
from typing import Union

import numpy as np

from duelpy.stats.preference_matrix import PreferenceMatrix

__all__ = [
    "Metric",
    "AverageRegret",
    "StrongRegret",
    "WeakRegret",
]


class Metric:
    """A metric that measures the performance of a PB-MAB algorithm."""

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Compute the metric value for a duel."""
        raise NotImplementedError()


class Regret(Metric):
    """The regret compared to the Condorcet winner.

    The regret of pulling an arm is defined by the calibrated preference
    probability of the best arm over the pulled one.

    This metric computes the per-duel regret. It is common practice to report
    the cumulative regret instead. You can combine this class with the
    ``Cumulative`` wrapper for that purpose.

    Parameters
    ----------
    preference_matrix
        The true preferences.
    aggregation_function
        A function that aggregates the regret of the two arms that are pulled
        in a single sample. Commonly mean, max or min. Also see
        ``AverageRegret``, ``StrongRegret`` and ``WeakRegret``.

    Examples
    --------
    >>> import numpy as np
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.9],
    ...     [0.1, 0.5],
    ... ]))
    >>> Regret(preference_matrix, max)(0, 1)
    0.4
    """

    def __init__(
        self,
        preference_matrix: Union[PreferenceMatrix, np.array],
        aggregation_function: Callable[[float, float], float],
    ):
        # Accept simple numpy arrays for convenience.
        if isinstance(preference_matrix, np.ndarray):
            preference_matrix = PreferenceMatrix(preference_matrix)
        self.preference_matrix = preference_matrix
        self.aggregation_function = aggregation_function
        self.best_arm = self.preference_matrix.get_condorcet_winner()
        assert (
            self.best_arm is not None
        ), "The regret can only be computed if a Condorcet winner exists."

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Compute the regret of a duel."""
        return (
            self.aggregation_function(
                self.preference_matrix[self.best_arm, arm_i_index],
                self.preference_matrix[self.best_arm, arm_j_index],
            )
            - 0.5
        )


class AverageRegret(Regret):
    """The average regret compared to the Condorcet winner.

    The regret of pulling an arm is defined by the calibrated preference
    probability of the best arm over the pulled one. This metric takes the
    average of the regret of the two pulled arms. Also see ``Regret``,
    ``StrongRegret`` and ``WeakRegret``.

    This metric computes the per-duel regret. It is common practice to report
    the cumulative regret instead. You can combine this class with the
    ``Cumulative`` wrapper for that purpose.

    Parameters
    ----------
    preference_matrix
        The true preferences.

    Examples
    --------
    >>> import numpy as np
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.9],
    ...     [0.1, 0.5],
    ... ]))
    >>> round(AverageRegret(preference_matrix)(0, 1), 2)
    0.2
    """

    def __init__(self, preference_matrix: Union[np.array, PreferenceMatrix]) -> None:
        super().__init__(
            preference_matrix, aggregation_function=lambda a, b: (a + b) / 2
        )


class StrongRegret(Regret):
    """The strong regret compared to the Condorcet winner.

    The regret of pulling an arm is defined by the calibrated preference
    probability of the best arm over the pulled one. This metric takes the
    maximum of the regret of the two pulled arms. Also see ``Regret``,
    ``AverageRegret`` and ``WeakRegret``.

    This metric computes the per-duel regret. It is common practice to report
    the cumulative regret instead. You can combine this class with the
    ``Cumulative`` wrapper for that purpose.

    Parameters
    ----------
    preference_matrix
        The true preferences.

    Examples
    --------
    >>> import numpy as np
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.9],
    ...     [0.1, 0.5],
    ... ]))
    >>> StrongRegret(preference_matrix)(0, 1)
    0.4
    """

    def __init__(self, preference_matrix: Union[np.array, PreferenceMatrix]) -> None:
        super().__init__(preference_matrix, aggregation_function=max)


class WeakRegret(Regret):
    """The weak regret compared to the Condorcet winner.

    The regret of pulling an arm is defined by the calibrated preference
    probability of the best arm over the pulled one. This metric takes the
    minimum of the regret of the two pulled arms. Also see ``Regret``,
    ``AverageRegret`` and ``StrongRegret``.

    This metric computes the per-duel regret. It is common practice to report
    the cumulative regret instead. You can combine this class with the
    ``Cumulative`` wrapper for that purpose.

    Parameters
    ----------
    preference_matrix
        The true preferences.

    Examples
    --------
    >>> import numpy as np
    >>> preference_matrix = PreferenceMatrix(np.array([
    ...     [0.5, 0.9],
    ...     [0.1, 0.5],
    ... ]))
    >>> WeakRegret(preference_matrix)(0, 1)
    0.0
    """

    def __init__(self, preference_matrix: Union[np.array, PreferenceMatrix]) -> None:
        super().__init__(preference_matrix, aggregation_function=min)
