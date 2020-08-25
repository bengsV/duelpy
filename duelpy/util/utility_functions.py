"""Implementation of various helper functions."""

import numpy as np


def argmax_set(array: np.array) -> np.array:
    """Calculate the complete argmax set, returning an array with all indices.

    Parameters
    ----------
    array
        The array for which the argmax should be calculated

    Returns
    -------
    indices
        An 1-D array containing the argmax set
    """
    # np.argmax returns the first index, to get the whole set we search for all indices which point to a value equal to the maximum
    max_value = array.max()
    indices = np.argwhere(array == max_value).flatten()
    return indices


def argmin_set(array: np.array) -> np.array:
    """Calculate the complete argmin set, returning an array with all indices.

    Parameters
    ----------
    array
        The array for which the argmin should be calculated

    Returns
    -------
    indices
        An 1-D array containing the argmin set
    """
    # np.argmin returns the first index, to get the whole set we search for all indices which point to a value equal to the minimum
    min_value = array.min()
    indices = np.argwhere(array == min_value).flatten()
    return indices
