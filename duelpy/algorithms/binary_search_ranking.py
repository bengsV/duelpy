"""Algorithm of the Binary Search Ranking."""
from typing import Optional, Collection

from duelpy.algorithms.interfaces import CopelandRankingProducer


class BinarySearchRanking(CopelandRankingProducer):
    r"""Implementation of the Binary Search Ranking algorithm.

    This algorithm finds an :math:`\epsilon`-ranking among the given arms.

    Unlike Packet-Luce model which requires :mathCal:`n^2`, this algorithm takes :mathCal:`n` memory requirement.
    Binary Search Ranking algorithm first selects a = :math:`n/(\log n)^x` arms which it calls anchor arms/elements.
    Between every two successively ranked anchor elements, this algorithm will consider that it has created a bin so in
    total there are 'a' bins. For each element(non anchor elements), the algorithm considers the bin that element
    belongs to and each bin can have multiple elements. The algorithm then calls Rank-x method to rank the elements
    within each bin.

    The complexity/number of comparisons the algorithm is bounded by is :mathCal:`n \times (\log n)^2 / \epsilon^2`.

    The Rank-x method determines ranking with probability at least 1-:math:`\Delta` and uses
    :mathCal:`n/\epsilon \times (\log n)^x \times (\log n/)\Delta`.

    """

    def step(self) -> None:
        pass

    def get_ranking(self) -> Optional[Collection[int]]:
        pass
