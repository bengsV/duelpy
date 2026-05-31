"""Contextual dueling bandit environment based on linear arm utilities."""

from typing import Callable
from typing import Optional

import numpy as np

from duelpy.feedback.linear_context_feedback import LinearContextFeedback


class LinearContextEnvironment:
    r"""A contextual experiment environment with linear arm utilities.

    Each arm :math:`i` is assigned a parameter vector
    :math:`\theta_i^* \in \mathbb{R}^d` drawn uniformly from the unit sphere.
    At each round, a context :math:`x_t` is drawn i.i.d. from a Gaussian
    distribution and then projected onto the unit sphere.

    This environment is designed for use with the
    :class:`StaD<duelpy.algorithms.stad.StaD>` algorithm.

    Parameters
    ----------
    num_arms
        Number of arms :math:`K`.
    context_dim
        Dimensionality :math:`d` of the context vectors.
    random_state
        Controls both the arm parameter initialisation and the context
        generation stream.

    Examples
    --------
    >>> env = LinearContextEnvironment(num_arms=3, context_dim=4,
    ...                                random_state=np.random.RandomState(0))
    >>> fb = env.get_feedback_mechanism()
    >>> fb.get_num_arms()
    3
    >>> gen = env.get_context_generator()
    >>> x = gen()
    >>> x.shape
    (4,)
    """

    def __init__(
        self,
        num_arms: int,
        context_dim: int,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        self._num_arms = num_arms
        self._context_dim = context_dim
        self._random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        # Draw arm parameters uniformly from the unit sphere
        raw = self._random_state.randn(num_arms, context_dim)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        self._arm_parameters = raw / norms

        # Separate random state for context generation so it can be used
        # independently after construction
        seed = self._random_state.randint(0, 2**31)
        self._context_rng = np.random.RandomState(seed)

        self._feedback = LinearContextFeedback(
            self._arm_parameters,
            link="logistic",
            random_state=np.random.RandomState(self._random_state.randint(0, 2**31)),
        )

    def get_feedback_mechanism(self) -> LinearContextFeedback:
        """Return the :class:`LinearContextFeedback` for this environment.

        Returns
        -------
        LinearContextFeedback
            The environment's feedback mechanism.
        """
        return self._feedback

    def get_context_generator(self) -> Callable[[], np.ndarray]:
        """Return a callable that generates i.i.d. unit-sphere contexts.

        Returns
        -------
        Callable
            Each call returns a fresh context vector of shape
            ``(context_dim,)`` on the unit sphere.
        """

        def _generate() -> np.ndarray:
            raw = self._context_rng.randn(self._context_dim)
            return raw / np.linalg.norm(raw)

        return _generate

    def get_best_arm(self, context: np.ndarray) -> int:
        """Return the index of the arm with the highest utility for *context*.

        Parameters
        ----------
        context
            A context vector of shape ``(context_dim,)``.

        Returns
        -------
        int
            Index of the optimal arm.
        """
        return self._feedback.get_best_arm(context)
