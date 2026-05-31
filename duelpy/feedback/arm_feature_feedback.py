"""Feedback mechanism based on arm-specific feature vectors and a shared latent parameter."""

from typing import Optional

import numpy as np

from duelpy.feedback.feedback_mechanism import FeedbackMechanism


class ArmFeatureFeedback(FeedbackMechanism):
    r"""Pairwise preferences determined by arm features and a shared latent parameter.

    Each arm :math:`i` has a fixed feature vector :math:`\phi_i \in \mathbb{R}^d`.
    A shared latent parameter :math:`\theta^* \in \mathbb{R}^d` governs
    preferences:

    .. math::

        P(\text{arm } i \text{ beats arm } j) =
        \sigma\!\left((\phi_i - \phi_j)^\top \theta^*\right)

    This is the model underlying the **CoLSTIM** algorithm
    (Bengs, Saha & Hüllermeier, ICML 2022).  Unlike
    :class:`LinearContextFeedback<duelpy.feedback.linear_context_feedback.LinearContextFeedback>`,
    the preferences here are **fixed** — there is no time-varying external context.
    The arm features serve as the static context.

    Parameters
    ----------
    arm_features
        Array of shape ``(num_arms, feature_dim)`` where row ``i`` is
        :math:`\phi_i`.
    latent_parameter
        Shape ``(feature_dim,)``.  The true :math:`\theta^*`.
    link
        ``"logistic"`` (default) for stochastic Bradley–Terry feedback or
        ``"step"`` for noiseless / deterministic feedback.
    random_state
        Random state used for stochastic feedback.  Unused when
        ``link="step"``.

    Examples
    --------
    >>> features = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    >>> theta = np.array([1.0, 0.0])
    >>> fb = ArmFeatureFeedback(features, theta, link="step")
    >>> fb.get_best_arm()
    0
    >>> fb.duel(0, 1)
    True
    """

    def __init__(
        self,
        arm_features: np.ndarray,
        latent_parameter: np.ndarray,
        link: str = "logistic",
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        num_arms = arm_features.shape[0]
        super().__init__(arms=list(range(num_arms)))
        self._arm_features = arm_features.copy()
        self._latent_parameter = latent_parameter.copy()
        if link not in ("step", "logistic"):
            raise ValueError("link must be 'step' or 'logistic'")
        self._link = link
        self._random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self._utilities: np.ndarray = arm_features @ latent_parameter

    @property
    def arm_features(self) -> np.ndarray:
        """Return a copy of the arm feature matrix."""
        return self._arm_features.copy()

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of the feature vectors."""
        return self._arm_features.shape[1]

    def duel(self, arm_i_index: int, arm_j_index: int) -> bool:
        """Return True if arm_i beats arm_j.

        Parameters
        ----------
        arm_i_index
            Index of the first arm.
        arm_j_index
            Index of the second arm.

        Returns
        -------
        bool
            True if arm_i wins.
        """
        diff = float(self._utilities[arm_i_index] - self._utilities[arm_j_index])
        if self._link == "step":
            return diff > 0
        prob = 1.0 / (1.0 + np.exp(-diff))
        return bool(self._random_state.uniform() < prob)

    def get_best_arm(self) -> int:
        """Return the index of the arm with the highest utility.

        Returns
        -------
        int
            Index of the arm with the highest utility under the latent parameter.
        """
        return int(np.argmax(self._utilities))
