"""Regret metrics for contextual dueling bandits."""

import numpy as np

from duelpy.feedback.linear_context_feedback import LinearContextFeedback


class ContextualLinearRegret:
    r"""Per-round regret relative to the best arm for the current context.

    At each round the regret is:

    .. math::

        r_t = u_{i^*(x_t)}(x_t)
              - \frac{u_{i_t}(x_t) + u_{j_t}(x_t)}{2}

    where :math:`i^*(x_t) = \arg\max_i \theta_i^{*\top} x_t` is the optimal
    arm and :math:`u_k(x) = \theta_k^{*\top} x` is the utility of arm
    :math:`k`.

    Parameters
    ----------
    feedback_mechanism
        The underlying :class:`LinearContextFeedback<duelpy.feedback.linear_context_feedback.LinearContextFeedback>`
        that holds the true arm parameters.

    Examples
    --------
    >>> arm_params = np.array([[1.0, 0.0], [0.0, 0.0]])
    >>> fb = LinearContextFeedback(arm_params, link="step")
    >>> metric = ContextualLinearRegret(fb)
    >>> context = np.array([1.0, 0.0])
    >>> metric(0, 0, context)  # optimal vs optimal
    0.0
    >>> metric(1, 1, context)  # worst vs worst
    1.0
    """

    def __init__(self, feedback_mechanism: LinearContextFeedback) -> None:
        self._feedback = feedback_mechanism

    def __call__(
        self,
        arm_i_index: int,
        arm_j_index: int,
        context: np.ndarray,
    ) -> float:
        """Compute the average regret of the dueled pair for a given context.

        Parameters
        ----------
        arm_i_index
            Index of the first dueled arm.
        arm_j_index
            Index of the second dueled arm.
        context
            Context vector at this round, shape ``(context_dim,)``.

        Returns
        -------
        float
            Non-negative regret value.
        """
        params = self._feedback.arm_parameters
        utilities = params @ context
        best_utility = float(np.max(utilities))
        avg_utility = (float(utilities[arm_i_index]) + float(utilities[arm_j_index])) / 2.0
        return max(best_utility - avg_utility, 0.0)
