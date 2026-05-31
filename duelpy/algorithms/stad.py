"""Stagewise Adaptive Duel (Sta'D) algorithm for contextual dueling bandits."""

from typing import List
from typing import Optional

import numpy as np

from duelpy.algorithms.contextual_algorithm import ContextualAlgorithm
from duelpy.feedback.contextual_feedback_mechanism import ContextualFeedbackMechanism
from duelpy.stats.contextual_preference_estimate import ContextualPreferenceEstimate


class StaD(ContextualAlgorithm):
    r"""Stagewise Adaptive Duel for contextual dueling bandits.

    At each round :math:`t` an external context :math:`x_t \in \mathbb{R}^d`
    is revealed.  Each arm :math:`i` has a latent parameter vector
    :math:`\theta_i^* \in \mathbb{R}^d`; its utility is
    :math:`u_i(x_t) = \theta_i^{*\top} x_t`.

    The algorithm divides the time horizon into **stages** of doubling length.
    Within each stage it maintains per-arm ridge-regression estimates and
    confidence ellipsoids.  At the end of a stage it **eliminates** arms
    whose upper confidence bound on utility falls below the best lower
    confidence bound, then enters the next stage with a reduced active set.

    **Pair selection**: at each round within stage :math:`s` the algorithm
    selects

    .. math::

        (i_t, j_t) = \arg\max_{i \in A_s} \text{UCB}_i(x_t),\;
                     \arg\max_{j \in A_s,\, j \ne i_t} \text{UCB}_j(x_t)

    (the two most optimistic active arms), duel them, and update both
    estimates.

    **Elimination**: at the end of stage :math:`s`, arm :math:`i` is
    removed from :math:`A_s` if

    .. math::

        \text{UCB}_i(x_{\text{last}}) <
        \max_{j \in A_s} \text{LCB}_j(x_{\text{last}})

    where :math:`x_{\text{last}}` is the final context seen in that stage.

    This is an approximate implementation of the algorithm introduced in the
    NeurIPS 2021 paper on contextual dueling bandits.

    Parameters
    ----------
    feedback_mechanism
        The contextual environment.
    time_horizon
        Total number of rounds.
    failure_probability
        Confidence parameter :math:`\delta \in (0, 1)`.
    random_state
        For reproducibility.  Currently unused (Sta'D is deterministic in
        arm selection), but kept for API consistency.

    Examples
    --------
    >>> import numpy as np
    >>> from duelpy.feedback.linear_context_feedback import LinearContextFeedback
    >>> arm_params = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    >>> feedback = LinearContextFeedback(arm_params, link="step",
    ...                                  random_state=np.random.RandomState(0))
    >>> algorithm = StaD(feedback, time_horizon=30, failure_probability=0.1)
    >>> rng = np.random.RandomState(0)
    >>> algorithm.run(lambda: rng.randn(2))
    >>> algorithm.get_copeland_winner() is not None
    True
    """

    def __init__(
        self,
        feedback_mechanism: ContextualFeedbackMechanism,
        time_horizon: int,
        failure_probability: float = 0.05,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self._failure_probability = failure_probability
        self._estimate = ContextualPreferenceEstimate(
            num_arms=feedback_mechanism.get_num_arms(),
            context_dim=feedback_mechanism.get_context_dim(),
        )
        # Active set — starts as all arms
        self._active: List[int] = list(range(feedback_mechanism.get_num_arms()))
        # Stage bookkeeping
        self._stage: int = 1
        self._stage_length: int = 2  # 2^stage
        self._rounds_in_stage: int = 0
        self._last_context: Optional[np.ndarray] = None
        self._copeland_winner: Optional[int] = None

    def _current_beta(self) -> float:
        return self._estimate.compute_beta(delta=self._failure_probability)

    def _select_pair(self, context: np.ndarray):
        """Return (arm_i, arm_j) from the active set by highest UCB."""
        beta = self._current_beta()
        ucbs = [
            self._estimate.get_upper_confidence_utility(a, context, beta)
            for a in self._active
        ]
        sorted_idx = sorted(range(len(self._active)), key=lambda k: ucbs[k], reverse=True)
        arm_i = self._active[sorted_idx[0]]
        arm_j = self._active[sorted_idx[1]] if len(self._active) > 1 else arm_i
        return arm_i, arm_j

    def _eliminate(self, context: np.ndarray) -> None:
        """Remove dominated arms from the active set."""
        if len(self._active) <= 1:
            return
        beta = self._current_beta()
        # Best LCB across active arms
        best_lcb = max(
            self._estimate.get_lower_confidence_utility(a, context, beta)
            for a in self._active
        )
        self._active = [
            a
            for a in self._active
            if self._estimate.get_upper_confidence_utility(a, context, beta) >= best_lcb
        ]

    def step(self, context: np.ndarray) -> None:
        """Execute one round of Sta'D.

        Parameters
        ----------
        context
            Context vector for this round, shape ``(context_dim,)``.
        """
        if self.is_finished():
            return

        self._last_context = context

        if len(self._active) == 1:
            # Only one arm left — no more eliminations possible
            self._copeland_winner = self._active[0]
            self._steps_taken += 1
            self._rounds_in_stage += 1
            self._advance_stage_if_needed(context)
            return

        arm_i, arm_j = self._select_pair(context)
        arm_i_won = self.feedback_mechanism.duel(arm_i, arm_j, context)
        self._estimate.enter_sample(arm_i, arm_j, context, arm_i_won)

        self._steps_taken += 1
        self._rounds_in_stage += 1
        self._advance_stage_if_needed(context)

    def _advance_stage_if_needed(self, context: np.ndarray) -> None:
        if self._rounds_in_stage >= self._stage_length:
            self._eliminate(context)
            self._stage += 1
            self._stage_length = 2 ** self._stage
            self._rounds_in_stage = 0

    def get_copeland_winner(self) -> Optional[int]:
        """Return the estimated best arm.

        If elimination has narrowed the active set to one arm that arm is
        returned.  Otherwise the arm with the highest mean utility estimate
        over the last observed context is returned.

        Returns
        -------
        Optional[int]
            Index of the estimated Copeland winner, or ``None`` if no steps
            have been taken yet.
        """
        if self._copeland_winner is not None:
            return self._copeland_winner
        if self._last_context is None:
            return None
        if len(self._active) == 1:
            return self._active[0]
        # Return arm with highest mean utility over the last context seen
        mean_utils = {
            a: self._estimate.get_mean_utility(a, self._last_context)
            for a in self._active
        }
        return max(mean_utils, key=mean_utils.get)  # type: ignore[arg-type]
