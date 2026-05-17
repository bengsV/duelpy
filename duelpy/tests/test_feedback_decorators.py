"""Tests for BudgetedFeedbackMechanism and MetricKeepingFeedbackMechanism."""

import numpy as np
import pytest

from duelpy.feedback.matrix_feedback import MatrixFeedback
from duelpy.stats.metrics import AverageRegret
from duelpy.util.exceptions import AlgorithmFinishedException
from duelpy.util.feedback_decorators import BudgetedFeedbackMechanism
from duelpy.util.feedback_decorators import MetricKeepingFeedbackMechanism


_PREF = np.array(
    [
        [0.5, 0.8],
        [0.2, 0.5],
    ]
)


def _make_fb() -> MatrixFeedback:
    return MatrixFeedback(_PREF, random_state=np.random.RandomState(0))


class TestBudgetedFeedbackMechanism:
    def test_counts_duels(self):
        fb = BudgetedFeedbackMechanism(_make_fb(), max_duels=5)
        for _ in range(5):
            fb.duel(0, 1)
        assert fb.duels_conducted == 5

    def test_raises_after_budget_exhausted(self):
        fb = BudgetedFeedbackMechanism(_make_fb(), max_duels=2)
        fb.duel(0, 1)
        fb.duel(0, 1)
        with pytest.raises(AlgorithmFinishedException):
            fb.duel(0, 1)

    def test_duel_repeatedly_counts_as_multiple(self):
        fb = BudgetedFeedbackMechanism(_make_fb(), max_duels=10)
        fb.duel_repeatedly(0, 1, 5)
        assert fb.duels_conducted == 5

    def test_get_arms_delegates(self):
        fb = BudgetedFeedbackMechanism(_make_fb(), max_duels=5)
        assert fb.get_arms() == [0, 1]


class TestMetricKeepingFeedbackMechanism:
    def test_records_metric_per_duel(self):
        metric = AverageRegret(_PREF)
        fb = MetricKeepingFeedbackMechanism(
            _make_fb(), metrics={"regret": metric}
        )
        fb.duel(0, 1)
        fb.duel(0, 1)
        assert len(fb.results["regret"]) == 2

    def test_metric_values_are_floats(self):
        metric = AverageRegret(_PREF)
        fb = MetricKeepingFeedbackMechanism(
            _make_fb(), metrics={"regret": metric}
        )
        fb.duel(0, 1)
        assert isinstance(fb.results["regret"][0], float)

    def test_multiple_metrics(self):
        from duelpy.stats.metrics import StrongRegret

        fb = MetricKeepingFeedbackMechanism(
            _make_fb(),
            metrics={"avg": AverageRegret(_PREF), "strong": StrongRegret(_PREF)},
        )
        for _ in range(3):
            fb.duel(0, 1)
        assert len(fb.results["avg"]) == 3
        assert len(fb.results["strong"]) == 3
