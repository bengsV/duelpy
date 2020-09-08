"""Various algorithms to solve Preference-Based Multi-Armed Bandit Problems."""

from duelpy.algorithms.beat_the_mean import BeatTheMeanBandit
from duelpy.algorithms.rcs import RelativeConfidenceSampling
from duelpy.algorithms.savage import Savage

__all__ = ["Savage", "BeatTheMeanBandit", "RelativeConfidenceSampling"]
