"""Feedback mechanism for adversarial (time-varying) dueling bandits."""

from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelpy.stats.preference_matrix import PreferenceMatrix


class AdversarialMatrixFeedback(FeedbackMechanism):
    r"""A feedback mechanism with a time-varying sequence of preference matrices.

    This implements the general *adversarial* dueling bandit model of
    :cite:`saha2021adversarial`, in which the preference matrices
    :math:`P_1, \dots, P_T` form an arbitrary, possibly adversarial, sequence.
    On the :math:`t`-th duel the outcome is sampled from :math:`P_t`, and an
    internal round counter advances by one.  When more duels are requested than
    matrices were supplied, the last matrix is reused.

    The time-averaged preference matrix
    :math:`\bar{P} = \frac{1}{T} \sum_t P_t` is exposed through the
    ``preference_matrix`` attribute.  Its :term:`Borda winner` coincides with
    the hindsight Borda winner :math:`\arg\max_i \sum_t b_t(i)` against which
    regret is measured, so it can be used directly with
    :class:`BordaRegret<duelpy.stats.metrics.BordaRegret>`.

    Parameters
    ----------
    preference_matrices
        The sequence of per-round preference matrices.  May be a list of
        :class:`PreferenceMatrix<duelpy.stats.preference_matrix.PreferenceMatrix>`
        objects, a list of square ``numpy`` arrays, or a single 3-D array of
        shape ``(num_rounds, num_arms, num_arms)``.
    random_state
        A numpy random state.  Defaults to an unseeded state when not
        specified.

    Attributes
    ----------
    preference_matrix
        The time-averaged preference matrix, as a
        :class:`PreferenceMatrix<duelpy.stats.preference_matrix.PreferenceMatrix>`.
    random_state

    Examples
    --------
    >>> import numpy as np
    >>> matrices = [
    ...     np.array([[0.5, 0.9], [0.1, 0.5]]),
    ...     np.array([[0.5, 0.7], [0.3, 0.5]]),
    ... ]
    >>> feedback_mechanism = AdversarialMatrixFeedback(
    ...     matrices, random_state=np.random.RandomState(0)
    ... )
    >>> feedback_mechanism.get_num_arms()
    2
    >>> feedback_mechanism.preference_matrix[0, 1]  # average of 0.9 and 0.7
    0.8
    >>> isinstance(feedback_mechanism.duel(0, 1), bool)
    True
    """

    def __init__(
        self,
        preference_matrices: Union[
            np.ndarray, Sequence[Union[np.ndarray, PreferenceMatrix]]
        ],
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        matrices: List[np.ndarray] = [
            matrix.preferences
            if isinstance(matrix, PreferenceMatrix)
            else np.asarray(matrix, dtype=float)
            for matrix in preference_matrices
        ]
        if len(matrices) == 0:
            raise ValueError("At least one preference matrix is required.")
        self._matrices = matrices
        num_arms = matrices[0].shape[0]
        super().__init__(arms=list(range(num_arms)))
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self._round = 0
        # Time-averaged matrix, used for hindsight-Borda-winner based metrics.
        self.preference_matrix = PreferenceMatrix(
            np.mean(np.stack(matrices, axis=0), axis=0)
        )

    def _current_matrix(self) -> np.ndarray:
        """Return the preference matrix for the current round."""
        index = min(self._round, len(self._matrices) - 1)
        return self._matrices[index]

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Perform a duel using the current round's preference matrix.

        Parameters
        ----------
        arm_i_index
            The index of the challenger arm.
        arm_j_index
            The index of the arm to compare against.

        Returns
        -------
        bool
            True if the first arm wins.
        """
        probability_i_wins = self._current_matrix()[arm_i_index][arm_j_index]
        i_wins = self.random_state.uniform() <= probability_i_wins
        self._round += 1
        return bool(i_wins)
