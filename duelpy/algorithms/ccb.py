"""An implementation of the Copeland Confidence Bound algorithm."""
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from duelpy.algorithms.algorithm import Algorithm
from duelpy.feedback.feedback_mechanism import FeedbackMechanism
from duelpy.stats import PreferenceEstimate
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelpy.util.utility_functions import argmax_set


class CopelandConfidenceBound(Algorithm):
    """Implement the Copeland Confidence Bound(CCB) algorithm.

    The Copeland Confidence Bound algorithm is based on [1]_.
    CCB is designed for smaller number of arms. The goal of the CCB algorithm
    is to minimize the Copeland regret. This is done by conducting duels which
    are most informative about the precedence of the participating arms in terms
    of their Copeland scores. The confirmed non-Copeland winners are eliminated
    based on the results of the prior duels. After conducting sufficient rounds,
    the set of possible Copeland winners will converge which will result in
    minimal increment in Copeland regret and thus the goal would be achieved.

    CCB runs continuously and in each time step it follows these steps:
    1. Optimisic and Pessimistic estimates (viz. `U` and `L` respectively) of the
    Preference matrix are calculated.
    2. A Copeland winner candidate `a_c` is chosen using the optimistic estimate
    `U` such that it has a chance of being a true Copeland winner. `a_c` is chosen
    from a set of top scorers from `U`, especially those which are present in a
    list `B_t`. `B_t` contains the arms which have a higher chance of being a Copeland
    winner.
    3. A suitable opponent `a_d` is chosen using the pessimistic estimate `L` such
    that it can beat the notion of `a_c` being the true Copeland winner. By definition,
    `U` and `L` define a confidence interval around the original preference matrix.
    Using these confidence intervals, `a_d` is chosen such that a duel between `a_c`
    and `a_d` provides maximum information about their precedence over each other in
    terms of their Copeland scores. The historically proven strong opponents to `a_c`
    are maintained in a shortlist `B_t_i`. The arms in this list are preferred while
    selecting an opponent. Such a list is maintained for every participating arm.
    The `B_t_i` lists for non-Copeland winners will contain a large number of opponents
    and thus help in their quick elimination from the list of possible winners.
    4. Finally, a duel is conducted between `a_c` and `a_d` and the result is recorded.

    References
    ----------
    This implements Copeland Confidence Bound [1]_ algorithm.

     .. [1] Masrour Zoghi, Zohar S. Karnin, Shimon Whiteson, and Maarten de Rijke.
     "Copeland Dueling Bandits" In Proceedings of Advances in Neural Information
     Processing Systems (NIPS), pages 307–315, 2015a.

    Parameters
    ----------
    feedback_mechanism
        A FeedbackMechanism object describing the environment.
    exploratory_constant
        A parameter which is used in calculating the upper confidence bounds.
        The confidence radius grows proportional to the square root of this value.
        A higher upper confidence bound results in more exploration.
        Corresponds to `alpha` in [1]_. The value of exploratory_constant must
        be greater than 0.5.
    time_horizon
        Number of times the method ``run`` executes.
    random_state
        A numpy random state. Defaults to an unseeded state when not specified.

    Attributes
    ----------
    copeland_winner_candidates
        The arms which have a higher possibilty of becoming a Copeland winner.
        Corresponds to `B_t` in [1]_.
    max_allowed_losses
        Maximum allowed losses for a Copeland winner.
        Corresponds to `L_C` in [1]_.
    respective_opponents
        A dictionary which has every arm as a key. Each value in this dictionary
        is a list. This list consists of the arms which are strong opponents to the
        arm present as the key.
        Corresponds to `B_t_i` in [1]_.
    time_step
        Number of rounds the algorithm has executed.
        Corresponds to `t` in [1]_.

    Examples
    --------
    Define a preference-based multi-armed bandit problem through a preference matrix:

    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3],
    ...     [0.9, 0.7, 0.5]
    ... ])
    >>> arms = list(range(len(preference_matrix)))
    >>> random_state = np.random.RandomState(20)
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, arms, random_state)
    >>> ccb = CopelandConfidenceBound(feedback_mechanism=feedback_mechanism, exploratory_constant=0.6, time_horizon=100, random_state=random_state)
    >>> ccb.run()

    The best arm in this case is the last arm (index 2)

    >>> regret_history, cumul_regret = feedback_mechanism.calculate_average_copeland_regret()
    >>> np.round(cumul_regret, 2)
    47.25
    >>> ccb.get_copeland_winner()
    2
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        exploratory_constant: float,
        time_horizon: int,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        super().__init__(feedback_mechanism, time_horizon)
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self.time_step = 0
        if exploratory_constant <= 0.5:
            raise ValueError("Value of exploratory constant must be greater than 0.5")
        self.exploratory_constant = exploratory_constant
        self.copeland_winner_candidates: List[int] = list()
        self.respective_opponents: Dict[int, List[int]] = dict()
        self.max_allowed_losses: int = 0
        # Initialize the Copeland winner candidates, their respective opponents and
        # the maximum number of losses allowed for a Copeland winner
        self._reset_copeland_attributes()

        self._preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms()
        )

    def step(self) -> None:
        """Run one round of the algorithm."""
        self.time_step += 1

        # Update and set the new confidence radius in `_preference_estimate`
        # as per the current `time_step`
        self._update_confidence_radius()

        optimistic_matrix = self._preference_estimate.get_upper_estimate_matrix()
        pessimistic_matrix = self._preference_estimate.get_lower_estimate_matrix()
        pessimistic_copeland_scores = pessimistic_matrix.get_copeland_scores()
        optimistic_copeland_scores = optimistic_matrix.get_copeland_scores()
        optimistic_copeland_winners = list(optimistic_matrix.get_copeland_winners())

        # Initialize a dictionary which contains the temporary values required
        # throughout the implementation to execute one round of the algorithm
        current_values: Dict[str, Any] = dict()
        current_values["optimistic_matrix"] = optimistic_matrix
        current_values["pessimistic_matrix"] = pessimistic_matrix
        current_values["pessimistic_copeland_scores"] = pessimistic_copeland_scores
        current_values["optimistic_copeland_scores"] = optimistic_copeland_scores
        current_values["optimistic_copeland_winners"] = optimistic_copeland_winners

        # Update the Copeland winner candidates and opponents for each arm
        # using the following three methods.
        self._reset_disproven_hypotheses_using(current_values)
        self._remove_non_copeland_winners_using(current_values)
        self._add_copeland_winner_candidates_using(current_values)

        # Find best duel candidates
        (
            copeland_winner_candidate,
            suitable_opponent,
        ) = self._find_best_duel_candidates_using(current_values)
        copeland_winner_candidate_won = self.feedback_mechanism.duel(
            copeland_winner_candidate, suitable_opponent
        )
        self._preference_estimate.enter_sample(
            copeland_winner_candidate, suitable_opponent, copeland_winner_candidate_won
        )

    def get_copeland_winner(self) -> Optional[int]:
        """Determine a Copeland winner using CCB algorithm.

        Returns
        -------
        Optional[int]
            The index of a Copeland winner among the given arms.
        """
        copeland_winners = list(
            self._preference_estimate.get_mean_estimate_matrix().get_copeland_winners()
        )
        if len(copeland_winners) > 0:
            return copeland_winners[0]
        return None

    def _reset_copeland_attributes(self) -> None:
        """Reset the attributes for finding a Copeland winner."""
        self._reset_copeland_winner_candidates()
        self._reset_respective_opponents()
        self._reset_max_allowed_losses()

    def _update_confidence_radius(self) -> None:
        """Update the confidence radius using latest failure probability.

        Failure probability for the upper confidence bound is `1/t^(2 * alpha)`
        where `t` is the current round of the algorithm and `alpha` is the
        exploratory constant.

        Refer Appendix D in https://arxiv.org/pdf/1506.00312.pdf for further
        details.
        """
        failure_probability = 1 / (self.time_step ** (2 * self.exploratory_constant))
        confidence_radius = HoeffdingConfidenceRadius(failure_probability)
        self._preference_estimate.set_confidence_radius(confidence_radius)

    def _reset_copeland_winner_candidates(self) -> None:
        """Reset the winner candidates list to include all the arms."""
        self.copeland_winner_candidates = list(
            range(self.feedback_mechanism.get_num_arms())
        )

    def _reset_respective_opponents(self) -> None:
        """Clear the opponents list for all the arms."""
        for arm in range(self.feedback_mechanism.get_num_arms()):
            self.respective_opponents[arm] = list()

    def _reset_max_allowed_losses(self) -> None:
        """Set maximum allowed losses for a Copeland winner."""
        self.max_allowed_losses = self.feedback_mechanism.get_num_arms()

    def _reset_disproven_hypotheses_using(self, current_values: Dict[str, Any]) -> None:
        """Check if the Copeland winner attributes need a reset.

        Reset the Copeland attributes if the lower estimate of an arm winning
        against any of its corresponding opponents is in favour of the arm.

        Parameters
        ----------
        current_values
            Dictionary with all the values calculated in this round.
        """
        reset = False
        pessimistic_matrix = current_values["pessimistic_matrix"]
        for arm, opponents in self.respective_opponents.items():
            for opponent in opponents:
                if pessimistic_matrix[arm][opponent] > 0.5:
                    reset = True
                    break
            if reset:
                break

        if reset:
            self._reset_copeland_attributes()

    def _remove_non_copeland_winners_using(
        self, current_values: Dict[str, Any]
    ) -> None:
        """Drop the candidate arms which cannot become Copeland winner.

        Parameters
        ----------
        current_values
            Dictionary with all the values calculated in this round.
        """
        non_copeland_winners = list()
        optimistic_matrix = current_values["optimistic_matrix"]
        pessimistic_copeland_scores = current_values["pessimistic_copeland_scores"]
        optimistic_copeland_scores = current_values["optimistic_copeland_scores"]
        for candidate in self.copeland_winner_candidates:
            for score in pessimistic_copeland_scores:
                if optimistic_copeland_scores[candidate] < score:
                    non_copeland_winners.append(candidate)
                    break
            if len(self.respective_opponents[candidate]) != self.max_allowed_losses + 1:
                new_opponents = list()
                for arm in range(self.feedback_mechanism.get_num_arms()):
                    if optimistic_matrix[candidate][arm] < 0.5:
                        new_opponents.append(arm)
                self.respective_opponents[candidate] = new_opponents

        for candidate in non_copeland_winners:
            self.copeland_winner_candidates.remove(candidate)

        if not self.copeland_winner_candidates:
            self._reset_copeland_attributes()

    def _add_copeland_winner_candidates_using(
        self, current_values: Dict[str, Any]
    ) -> None:
        """Add Copeland winner candidate arms.

        Parameters
        ----------
        current_values
            Dictionary with all the values calculated in this round.
        """
        optimistic_copeland_winners = current_values["optimistic_copeland_winners"]
        pessimistic_copeland_scores = current_values["pessimistic_copeland_scores"]
        optimistic_copeland_scores = current_values["optimistic_copeland_scores"]
        for arm in optimistic_copeland_winners:
            if optimistic_copeland_scores[arm] == pessimistic_copeland_scores[arm]:
                self.copeland_winner_candidates.append(arm)
                self.respective_opponents[arm].clear()
                self.max_allowed_losses = (
                    self.feedback_mechanism.get_num_arms()
                    - 1
                    - optimistic_copeland_scores[arm]
                )
                for other_arm in range(self.feedback_mechanism.get_num_arms()):
                    if other_arm != arm:
                        if (
                            len(self.respective_opponents[other_arm])
                            < self.max_allowed_losses + 1
                        ):
                            self.respective_opponents[other_arm].clear()
                        elif (
                            len(self.respective_opponents[other_arm])
                            > self.max_allowed_losses + 1
                        ):
                            self.respective_opponents[
                                other_arm
                            ] = self.random_state.choice(
                                self.respective_opponents[other_arm],
                                (self.max_allowed_losses + 1),
                                replace=False,
                            ).tolist()

    def _find_best_duel_candidates_using(
        self, current_values: Dict[str, Any]
    ) -> Tuple[int, int]:
        """Find the arms whose duel is the most informative.

        Parameters
        ----------
        current_values
            Dictionary with all the values calculated in this round.

        Returns
        -------
        Tuple[int, int]
            A pair of arms where the first arm is a Copeland winner candidate
            and the second one is a suitable opponent.
        """
        if self.random_state.random() < 0.25:
            duel_pair_list = self._obtain_duel_pair_list_using(current_values)
            if duel_pair_list:
                random_index = self.random_state.choice(len(duel_pair_list))
                return duel_pair_list[random_index]

        copeland_winner_candidate = self._obtain_copeland_winner_candidate_using(
            current_values
        )
        current_values["copeland_winner_candidate"] = copeland_winner_candidate

        opponent = self._obtain_suitable_opponent_using(current_values)
        return copeland_winner_candidate, opponent

    def _obtain_duel_pair_list_using(
        self, current_values: Dict[str, Any]
    ) -> List[Tuple[int, int]]:
        """Calculate and return the pairs of arms for a duel.

        Parameters
        ----------
        current_values : Dict[str, Any]
            Dictionary with all the values calculated in this round.

        Returns
        -------
        List[Tuple[int, int]]
            A list of pairs of arms which can be good dueling candidates.
        """
        pessimistic_matrix = current_values["pessimistic_matrix"]
        optimistic_matrix = current_values["optimistic_matrix"]
        duel_pair_list: List[Tuple[int, int]] = list()
        for arm, opponents in self.respective_opponents.items():
            for opponent in opponents:
                if (
                    pessimistic_matrix[arm][opponent]
                    <= 0.5
                    <= optimistic_matrix[arm][opponent]
                ):
                    duel_pair_list.append((arm, opponent))
        return duel_pair_list

    def _obtain_copeland_winner_candidate_using(
        self, current_values: Dict[str, Any]
    ) -> int:
        """Obtain a good Copeland winner candidate.

        Parameters
        ----------
        current_values : Dict[str, Any]
            Dictionary with all the values calculated in this round.

        Returns
        -------
        int
            A Copeland winner candidate
        """
        optimistic_copeland_winners = current_values["optimistic_copeland_winners"]
        common_candidates = list(
            set(self.copeland_winner_candidates) & set(optimistic_copeland_winners)
        )
        if common_candidates:
            if self.random_state.random() < 0.67:
                optimistic_copeland_winners = common_candidates

        return self.random_state.choice(optimistic_copeland_winners)

    def _obtain_suitable_opponent_using(self, current_values: Dict[str, Any]) -> int:
        """Obtain a suitable opponent to the Copeland winner candidate.

        Parameters
        ----------
        current_values : Dict[str, Any]
            Dictionary with all the values calculated in this round.

        Returns
        -------
        int
            An appropriate opponent for the Copeland winner candidate.
        """
        pessimistic_matrix = current_values["pessimistic_matrix"]
        optimistic_matrix = current_values["optimistic_matrix"]
        copeland_winner_candidate = current_values["copeland_winner_candidate"]
        opponent_list = list()
        if self.random_state.random() < 0.5:
            opponent_list = self.respective_opponents[copeland_winner_candidate]
        else:
            opponent_list = list(range(self.feedback_mechanism.get_num_arms()))

        candidate_opponents = np.zeros(self.feedback_mechanism.get_num_arms())
        for j in opponent_list:
            if pessimistic_matrix[j][copeland_winner_candidate] <= 0.5:
                candidate_opponents[j] = optimistic_matrix[j][copeland_winner_candidate]

        suitable_opponents = argmax_set(
            candidate_opponents, exclude_indexes=[copeland_winner_candidate]
        )
        return suitable_opponents[0]
