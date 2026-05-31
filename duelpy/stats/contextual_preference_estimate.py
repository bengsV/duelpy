"""Per-arm ridge regression for contextual dueling bandit preference estimation."""

from typing import Optional

import numpy as np


class ContextualPreferenceEstimate:
    r"""Estimates per-arm utility parameters from contextual duel outcomes.

    For each arm :math:`i`, maintains an online ridge-regression estimate
    :math:`\hat{\theta}_i` of the true utility parameter :math:`\theta_i^*`
    based on the observed duel history.  The design matrix and response
    vector are updated incrementally.

    **Data structures per arm** :math:`i`:

    - :math:`V_i = \lambda I + \sum_t x_t x_t^\top` — regularised design matrix, shape ``(d, d)``
    - :math:`b_i = \sum_t y_{it} x_t` — weighted response vector, shape ``(d,)``
    - :math:`\hat{\theta}_i = V_i^{-1} b_i` — current parameter estimate

    where :math:`y_{it} \in \{0, 1\}` is 1 if arm :math:`i` won its duel at
    round :math:`t` and the arm was involved in that round.

    **Confidence width** for arm :math:`i` at context :math:`x`:

    .. math::

        w_i(x) = \beta_t \|x\|_{V_i^{-1}}
                = \beta_t \sqrt{x^\top V_i^{-1} x}

    where :math:`\beta_t = \sqrt{d \log(1 + n_i / (d \lambda)) + 2 \log(1 / \delta)} + \sqrt{\lambda}`.

    Parameters
    ----------
    num_arms
        Number of arms.
    context_dim
        Dimensionality of the context vectors.
    regularization
        Ridge regularization coefficient :math:`\lambda > 0`.  Defaults to 1.

    Examples
    --------
    >>> estimate = ContextualPreferenceEstimate(num_arms=2, context_dim=2)
    >>> x = np.array([1.0, 0.0])
    >>> estimate.enter_sample(0, 1, context=x, arm_i_won=True)
    >>> estimate.get_theta_hat(0)
    array([0.5, 0. ])
    """

    def __init__(
        self,
        num_arms: int,
        context_dim: int,
        regularization: float = 1.0,
    ) -> None:
        self._num_arms = num_arms
        self._context_dim = context_dim
        self._regularization = regularization
        # Per-arm regularised design matrices V_i = λI + X_i^T X_i
        self._V = np.stack(
            [regularization * np.eye(context_dim) for _ in range(num_arms)]
        )  # shape (num_arms, d, d)
        # Per-arm response vectors b_i = X_i^T y_i
        self._b = np.zeros((num_arms, context_dim))
        # Per-arm sample counts
        self._n = np.zeros(num_arms, dtype=int)
        # Cached parameter estimates
        self._theta_hat = np.zeros((num_arms, context_dim))

    def enter_sample(
        self,
        arm_i_index: int,
        arm_j_index: int,
        context: np.ndarray,
        arm_i_won: bool,
    ) -> None:
        """Record the outcome of a duel and update both arms' estimates.

        Parameters
        ----------
        arm_i_index
            Index of the first arm (challenger).
        arm_j_index
            Index of the second arm.
        context
            Context vector at this round, shape ``(context_dim,)``.
        arm_i_won
            True if the first arm won.
        """
        outer = np.outer(context, context)
        # Update design matrices for both arms
        self._V[arm_i_index] += outer
        self._V[arm_j_index] += outer
        # Update response vectors: y=1 for the winner, y=0 for the loser
        if arm_i_won:
            self._b[arm_i_index] += context
        else:
            self._b[arm_j_index] += context
        # Increment sample counts
        self._n[arm_i_index] += 1
        self._n[arm_j_index] += 1
        # Recompute parameter estimates for both arms
        self._theta_hat[arm_i_index] = np.linalg.solve(
            self._V[arm_i_index], self._b[arm_i_index]
        )
        self._theta_hat[arm_j_index] = np.linalg.solve(
            self._V[arm_j_index], self._b[arm_j_index]
        )

    def _confidence_width(self, arm: int, context: np.ndarray, beta: float) -> float:
        """Compute beta * ||context||_{V_arm^{-1}}."""
        V_inv = np.linalg.inv(self._V[arm])
        norm_sq = float(context @ V_inv @ context)
        return beta * np.sqrt(max(norm_sq, 0.0))

    def get_upper_confidence_utility(
        self, arm: int, context: np.ndarray, beta: float
    ) -> float:
        r"""Return the upper confidence bound on arm utility.

        .. math::

            \hat{\theta}_i^\top x + \beta \|x\|_{V_i^{-1}}

        Parameters
        ----------
        arm
            Arm index.
        context
            Context vector, shape ``(context_dim,)``.
        beta
            Confidence radius scaling factor.

        Returns
        -------
        float
            Upper confidence utility estimate.
        """
        mean = float(self._theta_hat[arm] @ context)
        return mean + self._confidence_width(arm, context, beta)

    def get_lower_confidence_utility(
        self, arm: int, context: np.ndarray, beta: float
    ) -> float:
        r"""Return the lower confidence bound on arm utility.

        .. math::

            \hat{\theta}_i^\top x - \beta \|x\|_{V_i^{-1}}

        Parameters
        ----------
        arm
            Arm index.
        context
            Context vector, shape ``(context_dim,)``.
        beta
            Confidence radius scaling factor.

        Returns
        -------
        float
            Lower confidence utility estimate.
        """
        mean = float(self._theta_hat[arm] @ context)
        return mean - self._confidence_width(arm, context, beta)

    def get_mean_utility(self, arm: int, context: np.ndarray) -> float:
        r"""Return the point-estimate utility :math:`\hat{\theta}_i^\top x`.

        Parameters
        ----------
        arm
            Arm index.
        context
            Context vector, shape ``(context_dim,)``.

        Returns
        -------
        float
            Current mean utility estimate.
        """
        return float(self._theta_hat[arm] @ context)

    def get_theta_hat(self, arm: int) -> np.ndarray:
        """Return the current parameter estimate for *arm*.

        Parameters
        ----------
        arm
            Arm index.

        Returns
        -------
        np.ndarray
            Shape ``(context_dim,)``.
        """
        return self._theta_hat[arm].copy()

    def get_num_samples(self, arm: int) -> int:
        """Return the number of duels in which *arm* was involved.

        Parameters
        ----------
        arm
            Arm index.

        Returns
        -------
        int
            Number of duels arm has participated in.
        """
        return int(self._n[arm])

    def compute_beta(self, delta: float, Optional_n: Optional[int] = None) -> float:
        r"""Compute the confidence radius scaling factor.

        .. math::

            \beta = \sqrt{d \log\!\left(1 + \frac{n}{d \lambda}\right)
                         + 2 \log\!\frac{1}{\delta}} + \sqrt{\lambda}

        Parameters
        ----------
        delta
            Failure probability.
        Optional_n
            Override for the sample count.  If ``None``, uses the maximum
            sample count across all arms.

        Returns
        -------
        float
            The confidence radius scaling factor :math:`\beta`.
        """
        n = int(np.max(self._n)) if Optional_n is None else Optional_n
        d = self._context_dim
        lam = self._regularization
        inside = d * np.log(1.0 + n / (d * lam)) + 2.0 * np.log(1.0 / delta)
        return np.sqrt(max(inside, 0.0)) + np.sqrt(lam)
