#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Compute the average excluding the top N values."""

import numpy as np

def average_excluding_top_n(arr, n, axis=0):
    """
    Compute average along specified axis, excluding the N largest values.

    Note that np.partition is more efficient than full sorting 
    when you only need to separate the N largest values.
    The function handles arbitrary axis selection.
    Time complexity is O(n) average case, versus O(n log n) for 
    full sorting
    
    Parameters
    ----------
    arr : ndarray
        Input array
    n : int
        Number of largest values to exclude
    axis : int, optional
        Axis along which to compute the average (default: 0)
        
    Returns
    -------
    ndarray
        Array with reduced dimensionality along specified axis
    """
    if n >= arr.shape[axis]:
        raise ValueError(f"n ({n}) must be smaller than array size along axis {axis} ({arr.shape[axis]})")
    
    # Use partition to find the (n+1)th largest value
    # Everything before this index will be the smaller values
    kth = arr.shape[axis] - n
    
    # Partition the array along the specified axis
    partitioned = np.partition(arr, kth, axis=axis)
    
    # Take only the values up to (not including) the N largest
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(None, kth)
    
    return np.mean(partitioned[tuple(slices)], axis=axis)
