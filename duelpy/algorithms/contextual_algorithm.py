"""Abstract base class for contextual dueling bandit algorithms."""

from typing import Callable
from typing import Optional

import numpy as np

from duelpy.feedback.contextual_feedback_mechanism import ContextualFeedbackMechanism


class ContextualAlgorithm:
    r"""Parent class for algorithms that receive a context vector each round.

    At each round the caller supplies a context :math:`x_t \in \mathbb{R}^d`
    to :meth:`step`.  The algorithm selects a pair of arms, performs a duel
    via the feedback mechanism, and updates its internal state.

    Unlike :class:`Algorithm<duelpy.algorithms.Algorithm>`, this class does
    **not** wrap the feedback mechanism in a
    :class:`BudgetedFeedbackMechanism<duelpy.util.feedback_decorators.BudgetedFeedbackMechanism>`.
    The time horizon is enforced by counting calls to :meth:`step` directly,
    since the contextual run loop owns the iteration.

    Parameters
    ----------
    feedback_mechanism
        The context-aware environment.
    time_horizon
        Total number of rounds (calls to :meth:`step`).  ``None`` means no
        limit; algorithms must then provide their own :meth:`is_finished`
        implementation.
    """

    def __init__(
        self,
        feedback_mechanism: ContextualFeedbackMechanism,
        time_horizon: Optional[int] = None,
    ) -> None:
        self.feedback_mechanism = feedback_mechanism
        self.time_horizon = time_horizon
        self._steps_taken: int = 0

    def step(self, context: np.ndarray) -> None:
        """Execute one round of the algorithm.

        Parameters
        ----------
        context
            Context vector for this round, shape ``(context_dim,)``.
        """
        raise NotImplementedError

    def is_finished(self) -> bool:
        """Return True when the algorithm has completed all rounds.

        Raises
        ------
        NotImplementedError
            If no ``time_horizon`` was given and the subclass does not
            override this method.
        """
        if self.time_horizon is None:
            raise NotImplementedError(
                "No time horizon set and no custom termination condition implemented."
            )
        return self._steps_taken >= self.time_horizon

    def run(self, context_generator: Callable[[], np.ndarray]) -> None:
        """Run the algorithm to completion.

        At each round, a context is obtained from ``context_generator`` and
        passed to :meth:`step`.

        Parameters
        ----------
        context_generator
            A callable that returns a fresh context vector each time it is
            called.
        """
        while not self.is_finished():
            context = context_generator()
            self.step(context)
