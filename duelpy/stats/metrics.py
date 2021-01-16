"""Common metrics for algorithm performance."""


__all__ = [
    "Metric",
]


class Metric:
    """A metric that measures the performance of a PB-MAB algorithm."""

    def __call__(self, arm_i_index: int, arm_j_index: int) -> float:
        """Compute the metric value for a duel."""
        raise NotImplementedError()
