"""Standardized interfaces for different kinds of Algorithms."""

from typing import Collection
from typing import Optional

from duelpy.algorithms.algorithm import Algorithm


class CondorcetProducer(Algorithm):
    """An Algorithm that computes or estimates a Condorcet winner."""

    def get_condorcet_winner(self) -> Optional[int]:
        """Return the computed Condorcet winner if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError


class SingleCopelandProducer(Algorithm):
    """An Algorithm that computes or estimates one of the Copeland winners."""

    def get_copeland_winner(self) -> Optional[int]:
        """Return the computed Copeland winner if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError


class AllCopelandProducer(Algorithm):
    """An Algorithm that computes or estimates all Copeland winners."""

    def get_copeland_winners(self) -> Optional[Collection[int]]:
        """Return the computed Copeland winner if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError


class CopelandRankingProducer(Algorithm):
    """An Algorithm that computes or estimates the Copeland ranking over the arms."""

    def get_ranking(self) -> Optional[Collection[int]]:
        """Return the computed Copeland ranking if it is ready.

        This will only return a result when ``step`` has been called a
        sufficient amount of times. If this is a PAC algorithm, the result
        might be approximate.
        """
        raise NotImplementedError
