"""Tests for PreferenceMatrix utility methods."""

import numpy as np
import pytest

from duelpy.stats.preference_matrix import PreferenceMatrix


_PREF = np.array(
    [
        [0.5, 0.9, 0.8],
        [0.1, 0.5, 0.4],
        [0.2, 0.6, 0.5],
    ]
)
_PM = PreferenceMatrix(_PREF)


def test_get_num_arms():
    assert _PM.get_num_arms() == 3


def test_condorcet_winner():
    assert _PM.get_condorcet_winner() == 0


def test_condorcet_winner_none_when_absent():
    no_condorcet = PreferenceMatrix(
        np.array([[0.5, 0.6, 0.4], [0.4, 0.5, 0.6], [0.6, 0.4, 0.5]])
    )
    assert no_condorcet.get_condorcet_winner() is None


def test_copeland_scores():
    scores = _PM.get_copeland_scores()
    assert scores[0] == 2  # arm 0 beats both others
    assert scores[1] == 0
    assert scores[2] == 1


def test_normalized_copeland_scores():
    scores = _PM.get_normalized_copeland_scores()
    np.testing.assert_allclose(scores[0], 1.0)
    np.testing.assert_allclose(scores[1], 0.0)


def test_borda_scores_in_zero_one_range():
    scores = _PM.get_borda_scores()
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_get_winners_against():
    winners = _PM.get_winners_against(1)
    assert 0 in winners
    assert 1 not in winners


def test_get_losers_against():
    losers = _PM.get_losers_against(0)
    assert 1 in losers
    assert 2 in losers


def test_winners_losers_are_disjoint():
    for arm in range(_PM.get_num_arms()):
        winners = set(_PM.get_winners_against(arm))
        losers = set(_PM.get_losers_against(arm))
        assert winners.isdisjoint(losers)
        assert arm not in winners
        assert arm not in losers


def test_from_upper_triangle():
    upper = np.array([[-1, 0.7, 0.6], [0, -1, 0.4], [0, 0, -1]], dtype=float)
    pm = PreferenceMatrix.from_upper_triangle(upper)
    assert pm[0, 1] == pytest.approx(0.7)
    assert pm[1, 0] == pytest.approx(0.3)
    assert pm[0, 0] == pytest.approx(0.5)


def test_copeland_winners_set():
    winners = _PM.get_copeland_winners()
    assert winners == {0}


def test_borda_winners_set():
    winners = _PM.get_borda_winners()
    assert 0 in winners
