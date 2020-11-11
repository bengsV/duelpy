"""Various algorithms to solve Preference-Based Multi-Armed Bandit Problems."""

from duelpy.algorithms.algorithm import Algorithm
from duelpy.algorithms.beat_the_mean import BeatTheMeanBandit
from duelpy.algorithms.copeland_confidence_bound import CopelandConfidenceBound
from duelpy.algorithms.double_thompson_sampling import DoubleThompsonSampling
from duelpy.algorithms.interleaved_filtering import InterleavedFiltering
from duelpy.algorithms.knockout_tournament import KnockoutTournament
from duelpy.algorithms.mallows import MallowsMPI
from duelpy.algorithms.mallows import MallowsMPR
from duelpy.algorithms.relative_confidence_sampling import RelativeConfidenceSampling
from duelpy.algorithms.relative_ucb import RelativeUCB
from duelpy.algorithms.savage import Savage
from duelpy.algorithms.sequential_elimination import SequentialElimination
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
    "InterleavedFiltering",
    "KnockoutTournament",
    "CopelandConfidenceBound",
    "SequentialElimination",
    "MallowsMPI",
    "MallowsMPR",
    "DoubleThompsonSampling",
]
