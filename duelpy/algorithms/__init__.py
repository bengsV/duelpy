"""Various algorithms to solve Preference-Based Multi-Armed Bandit Problems."""

from duelpy.algorithms.algorithm import Algorithm
from duelpy.algorithms.beat_the_mean import BeatTheMeanBandit
from duelpy.algorithms.rcs import RelativeConfidenceSampling
from duelpy.algorithms.relative_ucb import RelativeUCB
from duelpy.algorithms.savage import Savage
from duelpy.algorithms.winner_stays import WinnerStaysStrongRegret
from duelpy.algorithms.winner_stays import WinnerStaysWeakRegret

__all__ = [
    "Savage",
    "WinnerStaysWeakRegret",
    "WinnerStaysStrongRegret",
    "BeatTheMeanBandit",
    "RelativeConfidenceSampling",
    "RelativeUCB",
    "Algorithm",
]
