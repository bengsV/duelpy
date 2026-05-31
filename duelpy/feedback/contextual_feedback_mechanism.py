"""Abstract base class for context-aware feedback mechanisms."""

from typing import List
from typing import Optional

import numpy as np


class ContextualFeedbackMechanism:
    """A feedback mechanism where duel outcomes depend on a context vector.

    This is a parallel hierarchy to :class:`FeedbackMechanism<duelpy.feedback.FeedbackMechanism>`.
    It does not inherit from the static-matrix base class because the preference
    structure here is context-dependent rather than fixed.

    Parameters
    ----------
    arms
        The pool of arms available.
    context_dim
        The dimensionality of the context vectors that will be passed to
        :meth:`duel`.
    """

    def __init__(self, arms: list, context_dim: int) -> None:
        self._arms = arms
        self._context_dim = context_dim

    def duel(self, arm_i_index: int, arm_j_index: int, context: np.ndarray) -> bool:
        """Compare two arms under a given context.

        Parameters
        ----------
        arm_i_index
            Index of the first arm.
        arm_j_index
            Index of the second arm.
        context
            A context vector of shape ``(context_dim,)``.

        Returns
        -------
        bool
            True if the first arm wins.
        """
        raise NotImplementedError

    def get_arms(self) -> List:
        """Return the pool of arms."""
        return self._arms.copy()

    def get_num_arms(self) -> int:
        """Return the number of arms."""
        return len(self._arms)

    def get_context_dim(self) -> int:
        """Return the dimensionality of the context vector."""
        return self._context_dim
