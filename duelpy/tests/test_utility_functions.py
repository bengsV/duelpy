"""Tests for helper functions in util/utility_functions.py."""

import math

import numpy as np
import pytest

from duelpy.util.utility_functions import argmax_set
from duelpy.util.utility_functions import argmin_set
from duelpy.util.utility_functions import newton_raphson
from duelpy.util.utility_functions import pop_random


class TestArgminSet:
    def test_single_minimum(self):
        arr = np.array([3.0, 1.0, 2.0])
        result = argmin_set(arr)
        assert list(result) == [1]

    def test_multiple_minima(self):
        arr = np.array([1.0, 3.0, 1.0, 2.0])
        result = argmin_set(arr)
        assert set(result) == {0, 2}

    def test_exclude_indexes(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = argmin_set(arr, exclude_indexes=[0])
        assert list(result) == [1]

    def test_all_equal(self):
        arr = np.array([5.0, 5.0, 5.0])
        result = argmin_set(arr)
        assert set(result) == {0, 1, 2}

    def test_negative_values(self):
        arr = np.array([-3.0, -1.0, -3.0])
        result = argmin_set(arr)
        assert set(result) == {0, 2}


class TestArgmaxSet:
    def test_single_maximum(self):
        arr = np.array([1.0, 3.0, 2.0])
        result = argmax_set(arr)
        assert list(result) == [1]

    def test_multiple_maxima(self):
        arr = np.array([3.0, 1.0, 3.0])
        result = argmax_set(arr)
        assert set(result) == {0, 2}

    def test_exclude_indexes(self):
        arr = np.array([3.0, 2.0, 1.0])
        result = argmax_set(arr, exclude_indexes=[0])
        assert list(result) == [1]

    def test_negative_values(self):
        arr = np.array([-1.0, -2.0, -1.0])
        result = argmax_set(arr)
        assert set(result) == {0, 2}


class TestPopRandom:
    def test_removes_element_from_list(self):
        lst = [1, 2, 3, 4]
        rs = np.random.RandomState(0)
        removed = pop_random(lst, rs)
        assert len(removed) == 1
        assert len(lst) == 3
        assert removed[0] not in lst

    def test_clears_list_when_amount_exceeds_size(self):
        lst = [1, 2]
        rs = np.random.RandomState(0)
        removed = pop_random(lst, rs, amount=5)
        assert len(removed) == 2
        assert lst == []

    def test_multiple_amount(self):
        lst = list(range(10))
        rs = np.random.RandomState(1)
        removed = pop_random(lst, rs, amount=3)
        assert len(removed) == 3
        assert len(lst) == 7

    def test_reproducible_with_same_seed(self):
        lst1 = [10, 20, 30, 40, 50]
        lst2 = [10, 20, 30, 40, 50]
        removed1 = pop_random(lst1, np.random.RandomState(5))
        removed2 = pop_random(lst2, np.random.RandomState(5))
        assert removed1 == removed2


class TestNewtonRaphson:
    def test_finds_root_of_x_squared_minus_two(self):
        root = newton_raphson(
            reference_point=1.5,
            function=lambda x: x**2 - 2,
            derivative_of_function=lambda x: 2 * x,
        )
        assert abs(root**2 - 2) < 1e-6

    def test_returns_inf_for_zero_derivative(self):
        result = newton_raphson(
            reference_point=5.0,
            function=lambda x: x - 2,
            derivative_of_function=lambda x: 0.0,
        )
        assert math.isinf(result)

    def test_returns_inf_for_nan_function(self):
        result = newton_raphson(
            reference_point=0.0,
            function=lambda x: float("nan"),
            derivative_of_function=lambda x: 1.0,
        )
        assert math.isinf(result)
