"""Implementation of various helper functions."""
from typing import List
from typing import Optional
from typing import Set

import numpy as np


def argmin_set(array: np.array, exclude_indexes: Optional[Set[int]] = None) -> np.array:
    """Calculate the complete argmin set, returning an array with all indices.

    It removes the ``exclude_indexes`` from the  array and calculates the indices set with minimum value from remaining
    indexes of array.

    Parameters
    ----------
    array
        The 1-D array for which the argmin should be calculated.
    exclude_indexes
        Indices to exclude in the argmin operation.

    Returns
    -------
    indices
        A 1-D array containing all indices which point to the minimum value.
    """
    # np.argmin only returns the first index, to get the whole set,
    # we first find the minimum and then search for all indices which point
    # to a value equal to this minimum
    if exclude_indexes is None:
        # For this case the simpler implementation is more efficient, although
        # the other one with a trivial mask would also work.
        return np.argwhere(array == np.amin(array)).flatten()
    mask = np.zeros(array.size, dtype=bool)
    mask[exclude_indexes] = True
    min_value = np.min(np.ma.array(array, mask=mask))
    indices = set(np.ndarray.flatten(np.argwhere(array == min_value)))
    indices = indices - set(exclude_indexes) if exclude_indexes is not None else indices
    return list(indices)


def argmax_set(array: np.array, exclude_indexes: Optional[Set[int]] = None) -> np.array:
    """Calculate the complete argmax set, returning an array with all indices..

    It removes the ``exclude_indexes`` from the  array and calculates the indices set with maximum value from the remaining
    indexes of array.

    Parameters
    ----------
    array
        The 1-D array for which the argmax should be calculated
    exclude_indexes
        Indices to exclude in the argmax operation.

    Returns
    -------
    indices
        A 1-D array containing all indices which point to the minimum value.
    """
    # np.argmax only returns the first index, to get the whole set,
    # we first find the maximum and then search for all indices which point
    # to a value equal to this maximum
    if exclude_indexes is None:
        # For this case the simpler implementation is more efficient, although
        # the other one with a trivial mask would also work.
        return np.argwhere(array == np.amax(array)).flatten()
    mask = np.zeros(array.size, dtype=bool)
    mask[exclude_indexes] = True
    max_value = np.max(np.ma.array(array, mask=mask))
    indices = set(np.ndarray.flatten(np.argwhere(array == max_value)))
    indices = indices - set(exclude_indexes) if exclude_indexes is not None else indices
    return list(indices)


def pop_random(
    input_list: List[int], random_state: np.random.RandomState, amount: int = 1
) -> List[int]:
    """Remove randomly chosen elements from a given list and return them.

    If the list contains less than or exactly `amount` elements, all elements are chosen.

    Parameters
    ----------
    input_list
        The list from which an arm should be removed.
    random_state
        The random state to use.
    amount
        The amount of elements to pick, defaults to 1.

    Returns
    -------
    List[int]
        The list containing the removed elements.
    """
    if len(input_list) <= amount:
        list_copy = input_list.copy()
        input_list.clear()
        return list_copy
    else:
        picked = []
        left_over = input_list
        for _ in range(amount):
            random_index = random_state.randint(0, len(input_list))
            removed_element = left_over.pop(random_index)
            picked.append(removed_element)
        return picked
