"""A fixed-gap Borda environment for adversarial dueling bandits."""

from typing import Optional

import numpy as np

from duelpy.feedback import MatrixFeedback
from duelpy.stats.preference_matrix import PreferenceMatrix


class FixedGapAdversarialMatrix(MatrixFeedback):
    r"""A stationary instance of the fixed-gap adversarial dueling setting.

    This environment realises the *fixed-gap* setting of
    :cite:`saha2021adversarial` as a stationary preference matrix, i.e. a
    constant (and therefore trivially fixed-gap) preference sequence.  A single
    designated arm is the :term:`Borda winner`, beating every other arm with a
    fixed probability, while all remaining arms are indistinguishable (they
    tie with one another).  The resulting :term:`Borda score` gap between the
    best arm and any other arm equals exactly ``gap``.

    Concretely, with :math:`K` arms, designated winner :math:`i^*`, and a win
    probability :math:`p = \tfrac{1}{2} + \tfrac{(K - 1)}{K}\,\Delta`:

    .. math::

        P(i^*, j) = p \quad \forall j \ne i^*, \qquad
        P(i, j) = \tfrac{1}{2} \quad \forall i, j \ne i^*.

    The winner is also a :term:`Condorcet winner` (since :math:`p > 1/2`), so
    this environment is compatible with both Borda- and Condorcet-based
    metrics.

    Parameters
    ----------
    num_arms
        The number of arms :math:`K`.  Must be at least ``2``.
    random_state
        The numpy random state used to pick the winning arm and to sample
        duel outcomes.
    gap
        The Borda-score gap :math:`\Delta` between the best arm and every
        other arm.  Must satisfy :math:`0 < \Delta \le K / (2 (K - 1))` so that
        the implied win probability does not exceed ``1``.

    Examples
    --------
    >>> import numpy as np
    >>> feedback_mechanism = FixedGapAdversarialMatrix(
    ...     num_arms=4, random_state=np.random.RandomState(0), gap=0.2
    ... )
    >>> borda_scores = feedback_mechanism.preference_matrix.get_borda_scores()
    >>> winner = int(np.argmax(borda_scores))
    >>> runner_up = sorted(borda_scores)[-2]
    >>> round(borda_scores[winner] - runner_up, 6)
    0.2
    """

    def __init__(
        self,
        num_arms: int,
        random_state: Optional[np.random.RandomState] = None,
        gap: float = 0.2,
    ) -> None:
        if num_arms < 2:
            raise ValueError("FixedGapAdversarialMatrix requires at least two arms.")
        max_gap = num_arms / (2 * (num_arms - 1))
        if not 0 < gap <= max_gap:
            raise ValueError(
                f"gap must lie in (0, {max_gap}] for {num_arms} arms, got {gap}."
            )
        random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        win_probability = 0.5 + (num_arms - 1) / num_arms * gap
        winner = int(random_state.randint(num_arms))
        preferences = np.full((num_arms, num_arms), 0.5)
        for arm in range(num_arms):
            if arm == winner:
                continue
            preferences[winner][arm] = win_probability
            preferences[arm][winner] = 1 - win_probability
        super().__init__(
            preference_matrix=PreferenceMatrix(preferences),
            random_state=random_state,
        )
