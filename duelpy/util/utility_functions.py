"""Implementation of various helper functions."""

import numpy as np


def argmax_set(array: np.array) -> np.array:
    """Calculate the argmax set of the input array.

    Parameters
    ----------
    array
        The array for which the argmax set should be calculated.

    Returns
    -------
    indices
        A 1-D array containing all indices which point to the maximum value.
    """
    # np.argmax only returns the first index, to get the whole set,
    # we first find the maximum and then search for all indices which point
    # to a value equal to this maximum
    max_value = array.max()
    indices = np.argwhere(array == max_value).flatten()
    return indices


def argmin_set(array: np.array) -> np.array:
    """Calculate the complete argmin set, returning an array with all indices.

    Parameters
    ----------
    array
        The array for which the argmin set should be calculated

    Returns
    -------
    indices
        A 1-D array containing all indices which point to the minimum value.
    """
    # np.argmax only returns the first index, to get the whole set,
    # we first find the maximum and then search for all indices which point
    # to a value equal to this maximum
    min_value = array.min()
    indices = np.argwhere(array == min_value).flatten()
    return indices
