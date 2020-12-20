"""Algorithm of the Binary Search Ranking."""
from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from duelpy.algorithms.interfaces import CopelandRankingProducer
from duelpy.algorithms.interfaces import PacAlgorithm
from duelpy.feedback import FeedbackMechanism
from duelpy.stats.confidence_radius import HoeffdingConfidenceRadius
from duelpy.stats.preference_estimate import PreferenceEstimate
from duelpy.util.exceptions import AlgorithmFinishedException
from duelpy.util.sorting import MergeSort
import duelpy.util.utility_functions as utility

# pylint: disable=too-many-branches
# pylint: disable=too-many-nested-blocks
# pylint: disable=too-many-statements


class Node:
    """Utility for implementation of Binary Search Tree.

    Refer to Algorithm 9 for more details.
    """

    def __init__(self, value_1: int, value_2: int):
        self.value1 = value_1
        self.value2 = value_2
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.parent: Optional[Node] = None

    def insert_child(self, value1: int, value2: int) -> Tuple:
        """Add children to the current node and also set the current node as the parent in the child nodes."""
        self.left = Node(value1, int(np.ceil(np.mean([value1, value2]))))
        self.left.parent = self
        self.right = Node(int(np.ceil(np.mean([value1, value2]))), value2)
        self.right.parent = self
        return self.left, self.right

    def get_child(self) -> Tuple:
        """Return the left and right children of current node."""
        return self.left, self.right

    def get_parent(self) -> Optional[Node]:
        """Return the parent of the current node."""
        return self.parent


class BinarySearchRanking(CopelandRankingProducer, PacAlgorithm):
    r"""Implementation of the Binary Search Ranking algorithm.

    This algorithm finds an :math:`\epsilon`-ranking among the given arms.

    The algorithm does :math:`\mathcal{O}\left(\frac{n \log n}{\epsilon^2}\right)` comparisons. The algorithm assumes
    the Rank-3 algorithm that performs :math:`\mathcal{O}\left(\frac{n}{\epsilon^2} * \log(n)^3 * \log\frac{n}{\delta}\right)`
     comparisons to output an :math:`\epsilon`-ranking for any :math:`\delta > 0, \epsilon > 0` with probability at
     least :math:`1-\delta`.

    The paper assumes Rank-3 algorithm which is also called MergeRank algorithm.

    1. Binary Search Ranking algorithm first selects randomly :math:`a = \frac{n}{(\log n)^3}` arms which it calls
    anchor arms.
    2. These randomly selected anchor arms are ranked using Rank-3 algorithm.
    3. Bins are added between the consecutive ranked anchor arms. As there are ``a`` ranked arms, hence there will be
       ``a-1`` bins, each bin between two ranked anchor. As there can be arms which are lower ranked as compared to the
       lowest ranked anchor arm or there can be arms which are higher ranked as compared to the highest ranked anchor
       arm, hence we consider 2 extra bins around these extreme ends in the ranked anchor arms set.
    4. The algorithm then place the remaining elements in these bins, such that some bins can have multiple elements
       while some bins can also have zero elements.
    5. Sort the elements in each bin using Rank-3 algorithm. In this way all the elements are sorted.

    Refer to paper :cite:`falahatgar2017maximum`

    Parameters
    ----------
    feedback_mechanism
        A ``FeedbackMechanism`` object describing the environment.
    time_horizon
        This states the number of comparision to be done before the algorithm terminates.
    random_state
        Used for random choices in the algorithm.
    epsilon
        The bias term referred to as :math:`\epsilon`.

    Examples
    --------
    >>> from duelpy.feedback import MatrixFeedback
    >>> preference_matrix = np.array([
    ...     [0.5, 0.1, 0.1, 0.1, 0.1],
    ...     [0.9, 0.5, 0.3, 0.2, 0.2],
    ...     [0.9, 0.7, 0.5, 0.8, 0.9],
    ...     [0.9, 0.8, 0.2, 0.5, 0.2],
    ...     [0.9, 0.8, 0.1, 0.8, 0.5]
    ... ])
    >>> random_state = np.random.RandomState()
    >>> feedback_mechanism = MatrixFeedback(preference_matrix, random_state=random_state)
    >>> rank = BinarySearchRanking(feedback_mechanism)
    >>> rank.run()
    >>> rank.get_ranking()
    [0, 1, 3, 4, 2]
    """

    def __init__(
        self,
        feedback_mechanism: FeedbackMechanism,
        time_horizon: Optional[int] = None,
        epsilon: float = 0.1,
        random_state: np.random = None,
    ):
        super().__init__(feedback_mechanism, time_horizon=time_horizon)
        self._epsilon = epsilon
        self.random_state = (
            random_state if random_state is not None else np.random.RandomState()
        )
        self._remaining_arms = self.feedback_mechanism.get_arms()
        self.preference_estimate = PreferenceEstimate(
            self.feedback_mechanism.get_num_arms()
        )
        self._final_result: List = list()

    def exploration_finished(self) -> bool:
        """Return True if ranking of arms has been created."""
        return bool(self.get_ranking())

    def get_ranking(self) -> Optional[list]:
        """Return the ranked list of arms as decided by the algorithm."""
        if self._final_result is not None:
            return self._final_result
        else:
            return None

    def _binary_search(
        self,
        sorted_list: List[int],
        ordered_node_list: List[int],
        search_arm: int,
        epsilon: float,
    ) -> int:
        """Implement BinarySearch Algorithm.

        Using the nodes ordered using IntervalBinarySearch Algorithm(Step 4 of Algorithm 8), indices of sorted_list are
        used.
        Refer to Algorithm 10 for more details.

        Parameters
        ----------
        sorted_list
            List of arms sorted using merge ranking.
        ordered_node_list
            List of nodes obtained from Binary Search Tree.
        search_arm
            Element that is used to search node in ordered_node_list.
        epsilon
            Bias provided to the algorithm.

        Returns
        -------
        int
            Node from ordered_node_list.
        """
        start_index = 1
        end_index = len(ordered_node_list)

        while end_index - start_index > 0:
            # call COMPARE2 method.
            _, compare_result = self._is_arm1_better(
                search_arm,
                sorted_list[ordered_node_list[int(np.mean([start_index, end_index]))]],
                10 * np.log(self.feedback_mechanism.get_num_arms() / pow(epsilon, 2)),
            )

            if compare_result < (1 / 2 - (3 * epsilon)):
                # moving left in the ordered_node_list.
                end_index = int(np.ceil(np.mean([start_index, end_index])))

            elif compare_result <= (1 / 2 + (3 * epsilon)):
                return ordered_node_list[
                    int(np.ceil(np.mean([start_index, end_index])))
                ]

            else:
                # moving right in the ordered list
                start_index = int(np.ceil(np.mean([start_index, end_index])))

        return ordered_node_list[end_index]

    def explore(self) -> None:
        """Find the ranked set of arms.

        Implement the Algorithm 4 (Binary Search Ranking).
        """
        try:
            # 1. create anchors randomly
            final_sorted_list = list()

            elements_near_anchor: Dict[int, List] = dict()  # refer to ``C_j``.
            elements_far_anchor: Dict[int, List] = dict()  # refer to ``B_j``.

            # refer to step 1.
            ranked_anchor_arms = utility.pop_random(
                self._remaining_arms,
                self.random_state,
                amount=int(
                    np.floor(
                        self.feedback_mechanism.get_num_arms()
                        / pow(np.log(self.feedback_mechanism.get_num_arms()), 3)
                    )
                ),
            )

            if len(ranked_anchor_arms) > 1:
                merge_sort = MergeSort(
                    ranked_anchor_arms,
                    lambda a1, a2: self._merge_compare_function(
                        a1,
                        a2,
                        self._epsilon / 16,
                        1 / pow(self.feedback_mechanism.get_num_arms(), 6),
                    ),
                    self.random_state,
                )
                while not merge_sort.is_finished():
                    merge_sort.step()
                temp_list = merge_sort.get_result()
                assert temp_list is not None
                ranked_anchor_arms = temp_list

            # refer to step 3.
            # -1 refers to weak arm which will lose against any arm.
            # -2 refers to strong arm which will beat all the arms.
            ranked_anchor_arms.insert(0, -1)
            ranked_anchor_arms.append(-2)

            bin_list: Dict[int, List] = dict()  # refer to ``S_j``.

            # this fetch the root node from the binary tree. Refer to :math:`\alpha` in algorithm 8.
            root_node = build_binary_search_tree(len(ranked_anchor_arms))

            # refer to step 4.
            # remaining_arms doesn't contain previously selected anchor arms.
            for arm in self._remaining_arms:
                bin_index = self._search_binary_interval(
                    ranked_anchor_arms, arm, self._epsilon / 15, root_node
                )
                # if dictionary contains any list at index of the bin, then add the arm to that list otherwise, assign new
                # list to the bin_list.
                if bin_list.get(bin_index) is not None:
                    bin_list[bin_index].append(arm)
                else:
                    bin_list[bin_index] = [arm]

            # Refer to step 5. ``index`` refers to ``j``. Iterate through all the bins.
            for bin_index in range(
                int(
                    np.floor(
                        self.feedback_mechanism.get_num_arms()
                        / pow(np.log(self.feedback_mechanism.get_num_arms()), 3)
                    )
                )
                + 2
            ):
                # element refers to ``e`` in step 5.a.
                # Iterate through all the elements in a bin.
                if bin_list.get(bin_index) is not None:
                    for element in bin_list[bin_index]:
                        _, mean_value = self._is_arm1_better(
                            element,
                            ranked_anchor_arms[bin_index],
                            10
                            * pow(self._epsilon / 15, -2)
                            * np.log(self.feedback_mechanism.get_num_arms()),
                        )
                        if (
                            (1 / 2 - 6 * self._epsilon / 15)
                            <= mean_value
                            <= (1 / 2 + 6 * self._epsilon / 15)
                        ):
                            # if the prob. estimate of the element in bin beating anchor arm at bin i is within the range,
                            # then add that element into :math:`C_j`.
                            if elements_near_anchor.get(bin_index) is not None:
                                elements_near_anchor[bin_index].append(element)
                            else:
                                elements_near_anchor[bin_index] = [element]
                        else:
                            _, mean_value = self._is_arm1_better(
                                element,
                                ranked_anchor_arms[bin_index + 1],
                                10
                                * pow(self._epsilon / 15, -2)
                                * np.log(self.feedback_mechanism.get_num_arms()),
                            )

                            if (
                                (1 / 2 - 6 * self._epsilon / 15)
                                <= mean_value
                                <= (1 / 2 + 6 * self._epsilon / 15)
                            ):
                                # if estimate of element in bin beating element at bin i+1 is within the range then append
                                # element into :math:`C_(j+1)`.
                                if elements_near_anchor.get(bin_index + 1) is not None:
                                    elements_near_anchor[bin_index + 1].append(element)
                                else:
                                    elements_near_anchor[bin_index + 1] = [element]
                            else:
                                # otherwise append this element into :math:`B_j`
                                if elements_far_anchor.get(bin_index) is not None:
                                    elements_far_anchor[bin_index].append(element)
                                else:
                                    elements_far_anchor[bin_index] = [element]

                # Refer to step 5.b.
                # Rank list at element_far_anchor[bin_index].
                if (
                    elements_far_anchor.get(bin_index) is not None
                    and len(elements_far_anchor[bin_index]) > 1
                ):
                    merge_sort = MergeSort(
                        elements_far_anchor[bin_index],
                        lambda a1, a2: self._merge_compare_function(
                            a1,
                            a2,
                            self._epsilon / 15,
                            1 / pow(self.feedback_mechanism.get_num_arms(), 4),
                        ),
                        self.random_state,
                    )

                    while not merge_sort.is_finished():
                        merge_sort.step()
                    temp_list = merge_sort.get_result()
                    assert temp_list is not None
                    elements_far_anchor[bin_index] = temp_list

                # Refer to step 5.c.
                if ranked_anchor_arms[bin_index] in [-1, -2]:
                    pass
                else:
                    final_sorted_list.append(ranked_anchor_arms[bin_index])
                if elements_near_anchor.get(bin_index) is not None:
                    for each in elements_near_anchor[bin_index]:
                        final_sorted_list.append(each)
                if elements_far_anchor.get(bin_index) is not None:
                    for each in elements_far_anchor[bin_index]:
                        final_sorted_list.append(each)

            self._final_result = final_sorted_list

        except AlgorithmFinishedException:
            # the time_horizon is reached.
            pass

    def _search_binary_interval(
        self,
        sorted_list: List[int],
        search_element: int,
        epsilon: float,
        root_node: Node,
    ) -> int:
        r"""Implement Interval Binary Search.

        This algorithm find which bin each element belongs to.
        According to the nodes in binary tree, it finds out the bin in which ``search_element`` should be placed.

        Refer to Algorithm 8 in the paper :cite:`falahatgar2017maximum`.

        Parameters
        ----------
        sorted_list
            List of elements sorted using Merge Ranking.
        search_element
            Element that is to be searched.
        epsilon
            Bias term. Refer to :math:`\epsilon` in the algorithm.
        root_node
            Refer to the root node of the binary tree.

        Returns
        -------
        int
            Value of node.
        """
        current_node = root_node
        node_values = list()  # alias for ``Q`` in Algorithm 8.
        count = 0

        # refer to step 3.
        for i in range(int(30 * np.log(self.feedback_mechanism.get_num_arms()))):
            i += 1
            if (
                current_node.value2 - current_node.value1 > 1
            ):  # implies there are child nodes of this node.
                if current_node is None:
                    break
                if current_node.value1 not in node_values:
                    node_values.append(current_node.value1)
                if current_node.value2 not in node_values:
                    node_values.append(current_node.value2)
                if (
                    int(np.mean([current_node.value2, current_node.value1]))
                    not in node_values
                ):
                    node_values.append(
                        int(np.mean([current_node.value2, current_node.value1]))
                    )

                # check if anchor arm at position value1 of sorted list beats search arm. Refer to Algorithm 8 for more
                # details.
                anchor_duel_result, _ = self._is_arm1_better(
                    sorted_list[current_node.value1],
                    search_element,
                    int(10 / pow(epsilon, 2)),
                )
                # check if search arm beats anchor arm at position value2 of sorted list.
                search_arm_duel_result, _ = self._is_arm1_better(
                    search_element,
                    sorted_list[current_node.value2],
                    int(10 / pow(epsilon, 2)),
                )

                # refer to 3.(a).ii
                if anchor_duel_result or search_arm_duel_result:
                    temp_node = current_node.get_parent()
                    assert temp_node is not None
                    current_node = temp_node

                # refer to 3.(a).iii.
                else:

                    left_child, right_child = current_node.get_child()
                    search_arm_duel_result, _ = self._is_arm1_better(
                        sorted_list[
                            int(np.mean([current_node.value1, current_node.value2]))
                        ],
                        search_element,
                        int(10 / pow(epsilon, 2)),
                    )

                    current_node = left_child if search_arm_duel_result else right_child
            # refer to step 3.b.
            else:
                search_arm_duel_result, _ = self._is_arm1_better(
                    search_element,
                    sorted_list[current_node.value1],
                    int(10 / pow(epsilon, 2)),
                )
                anchor_duel_result, _ = self._is_arm1_better(
                    sorted_list[current_node.value2],
                    search_element,
                    int(10 / pow(epsilon, 2)),
                )

                if search_arm_duel_result and anchor_duel_result:
                    count += 1  # this increases our confidence that search arm belongs to current bin.
                else:
                    if count == 0:
                        temp_node = (
                            current_node.get_parent()
                        )  # search arm doesn't belong to current bin.
                        assert temp_node is not None
                        current_node = temp_node
                    else:
                        count -= 1  # this decreases our confidence that search arm belongs to current bin.

        # refer to Step 4
        if count > 10 * np.log(self.feedback_mechanism.get_num_arms()):
            return (
                current_node.value1
            )  # we are confident that search arm belong to the current bin.
        else:
            # sort node_value_list
            node_values.sort()
            # call binary search method at step 4.b.ii.
            return self._binary_search(
                sorted_list, node_values, search_element, self._epsilon
            )

    def _is_arm1_better(self, arm1: int, arm2: int, duel_count: float) -> Tuple:
        """Compare two arms enough number of times to be confident of the result..

        Compare two arms ``m`` number of times to get an estimate on the winner among the two arms.
        Binomial distribution is used as often ``m`` is a large number.
        As referred in Algorithm 4, step 3, if the arm1 is best arm then arm2 will not beat it. Similarly, if arm1 is
        weak arm, then arm2 will beat it.

        Refer to Algorithm 5 Compare2 in the paper :cite:`falahatgar2017maximum`.

        Parameters
        ----------
        arm1
        arm2
            Arm elements provided by the user which would compete against each other.
        duel_count
            Number of duels that arm 1 compete against arm 2.

        Returns
        -------
        bool
            True if arm1 wins otherwise False.
        float
            Average probability estimate of arm1 beating arm2.

        Raises
        ------
        AlgorithmFinishedException
            Number of duels has equaled the given time_horizon by the user.
        """
        if -1 in [arm1, arm2]:
            if arm1 == -1:
                return (
                    False,
                    0.0,
                )  # as weak arm looses against all the arms. Refer to step 3 in algorithm 4.
            else:
                return True, 1.0
        elif -2 in [arm1, arm2]:
            if arm1 == -2:
                return (
                    True,
                    1.0,
                )  # as best arm beats all other arms. Refer to step 3 in algorithm 4.
            else:
                return False, 0.0
        self.feedback_mechanism.duel(0, 0)  # to include the duel conducted above.

        arm1_win, estimate = self.feedback_mechanism.multi_duels_step(
            arm1, arm2, int(duel_count), duel_limit=self.time_horizon
        )
        if self.is_finished():
            # as time_horizon reached, exception is raised.
            raise AlgorithmFinishedException

        return arm1_win, estimate

    def _merge_compare_function(
        self,
        arm1: int,
        arm2: int,
        epsilon: float,
        failure_probability: float,
    ) -> int:
        r"""Compare 2 arms with respect to confidence radius and the budget.

        Refer to algorithm 1 Compare1 in the paper.

        Parameters
        ----------
        arm1
            Index of the first arm.
        arm2
            Index of the second arm.
        epsilon
            Refer to :math:`\epsilon` which is called bias in the paper :cite:`falahatgar2017maximum`.
        failure_probability
            Refer to :math:`\delta` which is called confidence in the paper :cite:`falahatgar2017maximum`.

        Raises
        ------
        AlgorithmFinishedException
            If the comparison budget is reached.

        Returns
        -------
        int
            1 if arm2 beats arm1 else -1.
        """

        def probability_scaling(num_samples: int) -> float:
            return 4 * num_samples ** 2

        confidence_radius = HoeffdingConfidenceRadius(
            failure_probability=failure_probability,
            probability_scaling_factor=probability_scaling,
        )
        estimate_probability_arm1 = 0.5

        # Refer to Algorithm 1 in the paper :cite:`falahatgar2017maximum`.
        comparison_budget = (1 / (2 * pow(epsilon, 2))) * np.log(
            2 / failure_probability
        )

        while (
            np.abs(estimate_probability_arm1 - 0.5)
            <= confidence_radius(self.preference_estimate.get_num_samples(arm1, arm2))
            - epsilon
            and self.preference_estimate.get_num_samples(arm1, arm2)
            <= comparison_budget
        ):
            if self.is_finished():
                raise AlgorithmFinishedException()

            # update information about the preferred and eliminated arms
            self.preference_estimate.enter_sample(
                arm1, arm2, self.feedback_mechanism.duel(arm1, arm2)
            )

            estimate_probability_arm1 = self.preference_estimate.get_mean_estimate(
                arm1, arm2
            )

        # if estimate_probability_arm1 <= 0.5, means that arm2 beats arm1. Hence -1 is returned.
        return 1 if estimate_probability_arm1 <= 0.5 else -1


def build_binary_search_tree(size: int) -> Node:
    """Implement Binary Search Tree.

    For more details refer to Algorithm 9 of the paper :cite:`falahatgar2017maximum`.

    Parameters
    ----------
    size
        Number of leaves in the tree.

    Returns
    -------
    Node
        Root node of the tree.
    """
    root = Node(0, size - 1)
    tree_temp = list()
    tree_temp.append(root)
    for node in tree_temp:
        if node.value2 - node.value1 > 1:
            left, right = node.insert_child(node.value1, node.value2)
            tree_temp.append(left)
            tree_temp.append(right)
    return root
