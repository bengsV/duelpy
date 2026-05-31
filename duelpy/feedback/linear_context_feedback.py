"""Feedback mechanism based on a linear utility model over contexts."""

from typing import Optional

import numpy as np

from duelpy.feedback.contextual_feedback_mechanism import ContextualFeedbackMechanism


class LinearContextFeedback(ContextualFeedbackMechanism):
    r"""Pairwise preferences driven by linear utilities over a context.

    Each arm :math:`i` has a parameter vector :math:`\theta_i^* \in \mathbb{R}^d`.
    The utility of arm :math:`i` under context :math:`x` is
    :math:`u_i(x) = \theta_i^{*\top} x`.

    The preference model is:

    .. math::

        P(\text{arm } i \text{ beats arm } j \mid x) =
        \sigma\!\left((\theta_i^* - \theta_j^*)^\top x\right)

    where :math:`\sigma` is the chosen link function.  With ``link="step"``
    (the default) the winner is the arm with strictly higher utility, making
    the feedback deterministic.  With ``link="logistic"`` the outcome is
    sampled from a Bernoulli distribution.

    Parameters
    ----------
    arm_parameters
        Array of shape ``(num_arms, context_dim)`` where row ``i`` is
        :math:`\theta_i^*`.
    link
        ``"step"`` for noiseless, deterministic feedback (winner = arm with
        higher utility) or ``"logistic"`` for stochastic feedback sampled
        from a logistic model.
    random_state
        Random state for stochastic feedback.  Unused when ``link="step"``.

    Examples
    --------
    >>> arm_params = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    >>> fb = LinearContextFeedback(arm_params, link="step")
    >>> context = np.array([1.0, 0.0])  # favours arm 0
    >>> fb.get_best_arm(context)
    0
    >>> fb.duel(0, 1, context)
    True
    """

    def __init__(
        self,
        arm_parameters: np.ndarray,
        link: str = "step",
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        num_arms, context_dim = arm_parameters.shape
        super().__init__(arms=list(range(num_arms)), context_dim=context_dim)
        self._arm_parameters = arm_parameters.copy()
        if link not in ("step", "logistic"):
            raise ValueError("link must be 'step' or 'logistic'")
        self._link = link
        self._random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )

    @property
    def arm_parameters(self) -> np.ndarray:
        """Return a copy of the arm parameter matrix."""
        return self._arm_parameters.copy()

    def _utility(self, arm: int, context: np.ndarray) -> float:
        return float(self._arm_parameters[arm] @ context)

    def duel(self, arm_i_index: int, arm_j_index: int, context: np.ndarray) -> bool:
        """Return True if arm_i wins against arm_j under the given context.

        Parameters
        ----------
        arm_i_index
            Index of the first arm.
        arm_j_index
            Index of the second arm.
        context
            Context vector of shape ``(context_dim,)``.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        diff = self._utility(arm_i_index, context) - self._utility(arm_j_index, context)
        if self._link == "step":
            return diff > 0
        # logistic link
        prob = 1.0 / (1.0 + np.exp(-diff))
        return bool(self._random_state.uniform() < prob)

    def get_best_arm(self, context: np.ndarray) -> int:
        """Return the index of the arm with highest utility under *context*.

        Parameters
        ----------
        context
            Context vector of shape ``(context_dim,)``.

        Returns
        -------
        int
            Index of the arm with the highest utility.
        """
        utilities = self._arm_parameters @ context
        return int(np.argmax(utilities))
