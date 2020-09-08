"""Various algorithms to solve Preference-Based Multi-Armed Bandit Problems."""

from duelpy.algorithms.algorithm import Algorithm
from duelpy.algorithms.beat_the_mean import BeatTheMeanBandit
from duelpy.algorithms.beat_the_mean import BeatTheMeanBanditPAC
from duelpy.algorithms.copeland_confidence_bound import CopelandConfidenceBound
from duelpy.algorithms.double_thompson_sampling import DoubleThompsonSampling
from duelpy.algorithms.double_thompson_sampling import DoubleThompsonSamplingPlus
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

# Pylint insists that algorithm_list and interfaces are constants and should be
# named in UPPER_CASE. Technically that is correct, but it doesn't feel quite
# right for this use case. Its not a typical constant. A similar use-case would
# be numpy's np.core.numerictypes.allTypes, which is also not names in
# UPPER_CASE.
# pylint: disable=invalid-name

# Make the actual algorithm classes available for easy enumeration in
# experiments and tests.
algorithm_list = [
    Savage,
    WinnerStaysWeakRegret,
    WinnerStaysStrongRegret,
    BeatTheMeanBandit,
    BeatTheMeanBanditPAC,
    RelativeConfidenceSampling,
    RelativeUCB,
    InterleavedFiltering,
    KnockoutTournament,
    CopelandConfidenceBound,
    MallowsMPI,
    MallowsMPR,
    SequentialElimination,
    DoubleThompsonSampling,
    DoubleThompsonSamplingPlus,
]
# This is not really needed, but otherwise zimports doesn't understand the
# __all__ construct and complains that the Algorithm import is unnecessary.
interfaces = [Algorithm]

# Generate __all__ for tab-completion etc.
__all__ = ["Algorithm"] + [algorithm.__name__ for algorithm in algorithm_list]
