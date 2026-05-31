"""Borda-Confidence-Bound algorithm for fixed-gap adversarial dueling bandits."""

from typing import Optional

import numpy as np

from duelpy.algorithms.interfaces import BordaProducer
from duelpy.feedback import FeedbackMechanism


class BordaConfidenceBound(BordaProducer):
    r"""The Borda-Confidence-Bound (BCB) algorithm.

    This is *Algorithm 3* of :cite:`saha2021adversarial`.  It targets the
    *fixed-gap* adversarial setting, which bridges stationary and fully
    adversarial preferences: there is a fixed item :math:`i^*` whose average
    :term:`Borda score` exceeds that of every other arm by at least a gap
    :math:`\Delta` at all rounds, while the individual preference matrices may
    otherwise change adversarially.

    The algorithm is an explore-then-commit scheme.  During exploration it
    samples a pair of distinct arms :math:`x_t \ne y_t` uniformly at random,
    duels them, and maintains an unbiased estimate of the :term:`Borda score`
    of every arm,

    .. math::

        \hat{b}_t(i) = K\, o_t\, \mathbb{1}(x_t = i),
        \qquad
        \tilde{b}_t(i) = \frac{1}{t} \sum_{\tau \le t} \hat{b}_\tau(i),

    together with confidence bounds

    .. math::

        \text{LCB}(i; t),\, \text{UCB}(i; t)
            = \tilde{b}_t(i) \mp 2 \sqrt{\frac{K}{t} \log \frac{2 K T}{\delta}}.

    As soon as some arm :math:`\hat{i}` has a lower confidence bound exceeding
    the upper confidence bound of all others, the algorithm commits to
    :math:`\hat{i}` and plays the duel :math:`(\hat{i}, \hat{i})` for all
    remaining rounds.  This achieves a regret of
    :math:`\tilde{\mathcal{O}}(K / \Delta^2)`.

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        The number of duels to perform.  Required, as it appears in the
        confidence bounds.
    failure_probability
        The confidence parameter :math:`\delta \in (0, 1)`.
    random_state
        A numpy random state.  Defaults to an unseeded state when not
        specified.

    Attributes
    ----------
    random_state

    Examples
    --------
    Define a preference-based multi-armed bandit problem with a clear
    fixed-gap :term:`Borda winner` (arm ``0``):

    >>> import numpy as np
    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.9, 0.9],
    ...     [0.1, 0.5, 0.6],
    ...     [0.1, 0.4, 0.5],
    ... ])
    >>> random_state = np.random.RandomState(42)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)
    >>> algorithm = BordaConfidenceBound(
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
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        if time_horizon is None:
            raise ValueError("BordaConfidenceBound requires a time horizon.")
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            np.random.RandomState() if random_state is None else random_state
        )
        self._failure_probability = failure_probability
        num_arms = self.wrapped_feedback.get_num_arms()
        self._num_arms = num_arms
        # Cumulative unbiased Borda-score estimates, one entry per arm.
        self._cumulative_estimate = np.zeros(num_arms)
        self._rounds: int = 0
        # The arm we have committed to, or ``None`` while still exploring.
        self._committed_arm: Optional[int] = None
        # With a single arm there is nothing to explore.
        if num_arms == 1:
            self._committed_arm = 0

    def _sample_distinct_pair(self) -> tuple:
        """Sample two distinct arms uniformly at random."""
        first_arm = int(self.random_state.randint(self._num_arms))
        second_arm = int(self.random_state.randint(self._num_arms - 1))
        if second_arm >= first_arm:
            second_arm += 1
        return first_arm, second_arm

    def _check_commit(self) -> None:
        """Commit to an arm if its LCB exceeds every other arm's UCB."""
        assert self.time_horizon is not None  # guaranteed by the constructor
        num_arms = self._num_arms
        mean_estimate = self._cumulative_estimate / self._rounds
        width = 2 * np.sqrt(
            (num_arms / self._rounds)
            * np.log(2 * num_arms * self.time_horizon / self._failure_probability)
        )
        lower_bounds = mean_estimate - width
        upper_bounds = mean_estimate + width
        candidate = int(np.argmax(lower_bounds))
        # The highest upper bound among all arms other than the candidate.
        competitor_upper = max(
            upper_bounds[arm] for arm in range(num_arms) if arm != candidate
        )
        if lower_bounds[candidate] > competitor_upper:
            self._committed_arm = candidate

    def step(self) -> None:
        """Run one round of Borda-Confidence-Bound."""
        if self._committed_arm is not None:
            # Exploitation: play the committed arm against itself.
            self.wrapped_feedback.duel(self._committed_arm, self._committed_arm)
            return
        first_arm, second_arm = self._sample_distinct_pair()
        first_won = self.wrapped_feedback.duel(first_arm, second_arm)
        if first_won:
            self._cumulative_estimate[first_arm] += self._num_arms
        self._rounds += 1
        self._check_commit()

    def get_borda_winner(self) -> Optional[int]:
        """Return the committed arm, or the current Borda-score leader.

        Returns
        -------
        Optional[int]
            The index of the committed arm if the algorithm has committed,
            otherwise the arm with the highest estimated :term:`Borda score`.
            ``None`` if no exploration round has been performed yet.
        """
        if self._committed_arm is not None:
            return self._committed_arm
        if self._rounds == 0:
            return None
        return int(np.argmax(self._cumulative_estimate))
