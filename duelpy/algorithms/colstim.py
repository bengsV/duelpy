"""CoLSTIM: Contextual Linear Stochastic Transitivity via Imitation."""

from typing import Optional

import numpy as np

from duelpy.algorithms.algorithm import Algorithm
from duelpy.algorithms.interfaces import SingleCopelandProducer
from duelpy.feedback.arm_feature_feedback import ArmFeatureFeedback


class CoLSTIM(SingleCopelandProducer):
    r"""Contextual dueling bandit algorithm from Bengs, Saha & Hüllermeier (ICML 2022).

    Arms have fixed feature vectors :math:`\phi_i \in \mathbb{R}^d`.  A
    shared latent parameter :math:`\theta^* \in \mathbb{R}^d` governs
    preferences:

    .. math::

        P(\text{arm } i \text{ beats arm } j)
        = \sigma\!\left((\phi_i - \phi_j)^\top \theta^*\right)

    At each round :math:`t` the algorithm:

    1. Draws a perturbed parameter
       :math:`\tilde{\theta}_t \sim \mathcal{N}(\hat{\theta}_t, \beta_t^2 V_t^{-1})`
       (Thompson-sampling style).
    2. Selects the **imitation arm**
       :math:`i^* = \arg\max_i \phi_i^\top \tilde{\theta}_t`
       (best arm under the perturbed parameter).
    3. Selects the **exploitation arm**
       :math:`i^{**} = \arg\max_i \phi_i^\top \hat{\theta}_t`
       (best arm under the current estimate).
    4. If :math:`i^* = i^{**}` the arm with the second-highest perturbed utility
       is used as the imitation arm.
    5. Duels :math:`i^*` vs :math:`i^{**}`.
    6. Updates the ridge-regression estimate using the feature contrast
       :math:`\phi_{i^*} - \phi_{i^{**}}` and the binary outcome.

    The ridge-regression estimator solves:

    .. math::

        \hat{\theta}_t = V_t^{-1} b_t, \quad
        V_t = \lambda I + \sum_{s=1}^{t} z_s z_s^\top, \quad
        b_t = \sum_{s=1}^{t} y_s z_s

    where :math:`z_s = \phi_{i^*} - \phi_{i^{**}}` is the feature contrast
    and :math:`y_s \in \{0, 1\}` is 1 if the imitation arm won.

    This achieves :math:`\tilde{O}(\sqrt{dT})` weak regret.

    Parameters
    ----------
    feedback_mechanism
        The arm-feature-based dueling environment.
    time_horizon
        Number of duels to perform.
    failure_probability
        Confidence parameter :math:`\delta`.  Controls the perturbation scale.
    regularization
        Ridge coefficient :math:`\lambda > 0`.
    random_state
        For reproducibility of the Thompson-sampling perturbations.

    Examples
    --------
    >>> import numpy as np
    >>> from duelpy.feedback.arm_feature_feedback import ArmFeatureFeedback
    >>> features = np.eye(3)                        # each arm is a unit basis vector
    >>> theta = np.array([1.0, 0.5, 0.2])          # arm 0 is best
    >>> fb = ArmFeatureFeedback(features, theta, link="logistic",
    ...                          random_state=np.random.RandomState(0))
    >>> alg = CoLSTIM(fb, time_horizon=200, random_state=np.random.RandomState(0))
    >>> alg.run()
    >>> alg.get_copeland_winner()
    0
    """

    def __init__(
        self,
        feedback_mechanism: ArmFeatureFeedback,
        time_horizon: int,
        failure_probability: float = 0.05,
        regularization: float = 1.0,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self._arm_features = feedback_mechanism.arm_features
        self._feature_dim = feedback_mechanism.feature_dim
        self._failure_probability = failure_probability
        self._regularization = regularization
        self._random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        # Ridge regression state (shared single parameter θ*)
        self._V = regularization * np.eye(self._feature_dim)  # (d, d)
        self._b = np.zeros(self._feature_dim)                 # (d,)
        self._theta_hat = np.zeros(self._feature_dim)         # (d,)
        self._num_samples: int = 0

    def _compute_beta(self) -> float:
        n = self._num_samples
        d = self._feature_dim
        lam = self._regularization
        inside = d * np.log(1.0 + n / (d * lam)) + 2.0 * np.log(1.0 / self._failure_probability)
        return np.sqrt(max(inside, 0.0)) + np.sqrt(lam)

    def step(self) -> None:
        """Execute one round of CoLSTIM."""
        beta = self._compute_beta()

        # Thompson-sample a perturbed parameter θ̃ ~ N(θ̂, β² V⁻¹)
        try:
            V_inv = np.linalg.inv(self._V)
            L = np.linalg.cholesky(V_inv)
            noise = self._random_state.randn(self._feature_dim)
            theta_tilde = self._theta_hat + beta * (L @ noise)
        except np.linalg.LinAlgError:
            theta_tilde = self._theta_hat.copy()

        # Perturbed utilities: φ_i^T θ̃
        perturbed_utils = self._arm_features @ theta_tilde
        # Current best estimates: φ_i^T θ̂
        hat_utils = self._arm_features @ self._theta_hat

        # Exploitation arm i** = argmax φ_i^T θ̂
        arm_exploit = int(np.argmax(hat_utils))

        # Imitation arm i* = argmax φ_i^T θ̃
        arm_imitate = int(np.argmax(perturbed_utils))

        # If they coincide, pick second-highest perturbed utility
        if arm_imitate == arm_exploit:
            perturbed_copy = perturbed_utils.copy()
            perturbed_copy[arm_imitate] = -np.inf
            arm_imitate = int(np.argmax(perturbed_copy))

        # Duel: imitation arm vs exploitation arm
        imitate_won = self.wrapped_feedback.duel(arm_imitate, arm_exploit)

        # Update ridge regression
        phi_diff = self._arm_features[arm_imitate] - self._arm_features[arm_exploit]
        self._V += np.outer(phi_diff, phi_diff)
        self._b += float(imitate_won) * phi_diff
        self._theta_hat = np.linalg.solve(self._V, self._b)
        self._num_samples += 1

    def get_copeland_winner(self) -> Optional[int]:
        """Return the arm with the highest estimated utility.

        Returns
        -------
        Optional[int]
            Index of the arm with the highest mean utility under the current
            estimate, or ``None`` if no duels have been conducted.
        """
        if self._num_samples == 0:
            return None
        return int(np.argmax(self._arm_features @ self._theta_hat))
