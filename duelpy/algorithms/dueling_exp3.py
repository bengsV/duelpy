"""Dueling-EXP3 algorithms for adversarial dueling bandits."""

from typing import Optional

import numpy as np

from duelpy.algorithms.interfaces import BordaProducer
from duelpy.feedback import FeedbackMechanism


class DuelingExp3(BordaProducer):
    r"""The Dueling-EXP3 (D-EXP3) algorithm for adversarial dueling bandits.

    This is *Algorithm 1* of :cite:`saha2021adversarial`.  It tackles the
    general adversarial setting in which the sequence of preference matrices
    :math:`P_1, \dots, P_T` may be arbitrary, and minimizes regret with respect
    to the :term:`Borda winner` in hindsight.

    The algorithm is an adaptation of the classical EXP3 algorithm for
    adversarial multi-armed bandits.  It maintains a distribution :math:`q_t`
    over the arms.  In every round it samples both duel arms
    :math:`x_t, y_t \sim q_t` independently (with replacement), observes the
    binary preference feedback, and constructs an unbiased estimate of the
    *shifted* :term:`Borda score` of every arm:

    .. math::

        \tilde{s}_t(i) = \frac{\mathbb{1}(x_t = i)\, o_t}
                              {K\, q_t(x_t)\, q_t(y_t)},

    where :math:`o_t \in \{0, 1\}` indicates whether :math:`x_t` won the duel.
    Only the sampled first arm :math:`x_t` receives a non-zero estimate.  The
    distribution is then updated with an exponential weight update and mixed
    with the uniform distribution for :math:`\gamma`-exploration:

    .. math::

        \tilde{q}_{t+1}(i) \propto
            \exp\!\Bigl(\eta \sum_{\tau \le t} \tilde{s}_\tau(i)\Bigr),
        \qquad
        q_{t+1}(i) = (1 - \gamma)\,\tilde{q}_{t+1}(i) + \frac{\gamma}{K}.

    With the parameter choices :math:`\eta = (\log K / (T \sqrt{K}))^{2/3}` and
    :math:`\gamma = \sqrt{\eta K}` the expected regret is bounded by
    :math:`\mathcal{O}\bigl((K \log K)^{1/3} T^{2/3}\bigr)`.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.  The
        environment may be stationary or adversarial; the algorithm makes no
        assumption about it.
    time_horizon
        The number of duels to perform.  Required, since it determines the
        learning rate and exploration parameters.
    learning_rate
        The exponential-weights learning rate :math:`\eta`.  Defaults to
        :math:`(\log K / (T \sqrt{K}))^{2/3}` as in Theorem 2 of
        :cite:`saha2021adversarial`.
    exploration
        The uniform-exploration parameter :math:`\gamma \in (0, 1]`.  Defaults
        to :math:`\sqrt{\eta K}` (clipped to ``1``).
    random_state
        A numpy random state.  Defaults to an unseeded state when not
        specified.

    Attributes
    ----------
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference
    matrix in which arm ``0`` is the clear :term:`Borda winner`:

    >>> import numpy as np
    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.9, 0.9],
    ...     [0.1, 0.5, 0.6],
    ...     [0.1, 0.4, 0.5],
    ... ])
    >>> random_state = np.random.RandomState(42)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)
    >>> algorithm = DuelingExp3(
    ...     feedback_mechanism, time_horizon=3000, random_state=random_state
    ... )
    >>> algorithm.run()
    >>> algorithm.get_borda_winner()
    0
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        learning_rate: Optional[float] = None,
        exploration: Optional[float] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        if time_horizon is None:
            raise ValueError("DuelingExp3 requires a time horizon.")
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            np.random.RandomState() if random_state is None else random_state
        )
        num_arms = self.wrapped_feedback.get_num_arms()
        self._num_arms = num_arms
        if learning_rate is None:
            learning_rate = (
                (np.log(num_arms) / (time_horizon * np.sqrt(num_arms))) ** (2 / 3)
                if num_arms > 1
                else 0.0
            )
        self._learning_rate = learning_rate
        if exploration is None:
            exploration = np.sqrt(learning_rate * num_arms)
        self._exploration = min(1.0, exploration)
        # Cumulative estimated (shifted) Borda scores, one entry per arm.
        self._cumulative_estimate = np.zeros(num_arms)
        # Current sampling distribution q_t, initialised uniformly.
        self._distribution = np.full(num_arms, 1.0 / num_arms)

    def _update_estimate(
        self, first_arm: int, second_arm: int, first_won: bool
    ) -> None:
        """Update the cumulative score estimate after a duel."""
        outcome = 1.0 if first_won else 0.0
        distribution = self._distribution
        self._cumulative_estimate[first_arm] += outcome / (
            self._num_arms * distribution[first_arm] * distribution[second_arm]
        )

    def _update_distribution(self) -> None:
        """Recompute the sampling distribution from the cumulative estimates."""
        exponents = self._learning_rate * self._cumulative_estimate
        # Subtract the maximum for numerical stability; this does not change the
        # resulting (normalised) distribution.
        exponents -= np.max(exponents)
        weights = np.exp(exponents)
        smoothed = weights / np.sum(weights)
        distribution = (
            1 - self._exploration
        ) * smoothed + self._exploration / self._num_arms
        # Guard against floating point drift so the result is a valid
        # distribution for ``np.random.choice``.
        self._distribution = distribution / np.sum(distribution)

    def step(self) -> None:
        """Run one round of Dueling-EXP3."""
        first_arm = int(self.random_state.choice(self._num_arms, p=self._distribution))
        second_arm = int(self.random_state.choice(self._num_arms, p=self._distribution))
        first_won = self.wrapped_feedback.duel(first_arm, second_arm)
        self._update_estimate(first_arm, second_arm, first_won)
        self._update_distribution()

    def get_borda_winner(self) -> Optional[int]:
        """Return the arm with the highest estimated cumulative Borda score.

        Returns
        -------
        Optional[int]
            The index of the estimated :term:`Borda winner`.
        """
        return int(np.argmax(self._cumulative_estimate))


class DuelingExp3HighProbability(DuelingExp3):
    r"""High-probability variant of Dueling-EXP3.

    This is *Algorithm 2* of :cite:`saha2021adversarial`.  It is identical to
    :class:`DuelingExp3` except that the score estimate is shifted by a bias
    term :math:`\beta / q_t(i)` for **every** arm :math:`i` (not only the
    sampled one):

    .. math::

        s'_t(i) = \frac{\mathbb{1}(x_t = i)\, o_t}{K\, q_t(x_t)\, q_t(y_t)}
                  + \frac{\beta}{q_t(i)}.

    This optimistic bias yields a regret bound of
    :math:`\tilde{\mathcal{O}}(K^{1/3} T^{2/3})` that holds with probability at
    least :math:`1 - \delta`, at the cost of the estimate no longer being
    unbiased.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        The number of duels to perform.
    failure_probability
        The probability :math:`\delta` that the regret bound does not hold.
        Used to derive the default bias term.
    bias
        The bias term :math:`\beta`.  Defaults to
        :math:`\sqrt{\log(K / \delta) / (K T)}`.
    learning_rate
        The exponential-weights learning rate :math:`\eta`.  See
        :class:`DuelingExp3`.
    exploration
        The uniform-exploration parameter :math:`\gamma`.  See
        :class:`DuelingExp3`.
    random_state
        A numpy random state.  Defaults to an unseeded state when not
        specified.

    Examples
    --------
    >>> import numpy as np
    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.9, 0.9],
    ...     [0.1, 0.5, 0.6],
    ...     [0.1, 0.4, 0.5],
    ... ])
    >>> random_state = np.random.RandomState(42)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)
    >>> algorithm = DuelingExp3HighProbability(
    ...     feedback_mechanism, time_horizon=3000, random_state=random_state
    ... )
    >>> algorithm.run()
    >>> algorithm.get_borda_winner()
    0
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        failure_probability: float = 0.1,
        bias: Optional[float] = None,
        learning_rate: Optional[float] = None,
        exploration: Optional[float] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(
            feedback_mechanism,
            time_horizon,
            learning_rate=learning_rate,
            exploration=exploration,
            random_state=random_state,
        )
        assert time_horizon is not None  # guaranteed by the parent constructor
        if bias is None:
            num_arms = self._num_arms
            bias = (
                np.sqrt(
                    np.log(num_arms / failure_probability)
                    / (num_arms * time_horizon)
                )
                if num_arms > 1
                else 0.0
            )
        self._bias = bias

    def _update_estimate(
        self, first_arm: int, second_arm: int, first_won: bool
    ) -> None:
        """Update the cumulative score estimate with the optimistic bias."""
        # The bias term applies to every arm.
        self._cumulative_estimate += self._bias / self._distribution
        super()._update_estimate(first_arm, second_arm, first_won)
