#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Remove isolated pixels in a 2D array."""
import numpy as np


def remove_isolated_pixels(h):
    """Remove isolated pixels in a 2D array corresponding to a 2D histogram.

    Parameters
    ----------
    h : 2D numpy array
        The input 2D array (e.g., a histogram).

    Returns
    -------
    hclean : 2D numpy array
        The cleaned 2D array with isolated pixels removed.
    """
    if not isinstance(h, np.ndarray) or h.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    naxis2, naxis1 = h.shape
    flag = np.zeros_like(h, dtype=float)
    # Fill isolated holes
    for i in range(naxis2):
        for j in range(naxis1):
            if h[i, j] == 0:
                k, ktot = 0, 0
                fsum = 0.0
                for ii in [i - 1, i, i + 1]:
                    if 0 <= ii < naxis2:
                        for jj in [j - 1, j, j + 1]:
                            if 0 <= jj < naxis1:
                                ktot += 1
                                if h[ii, jj] != 0:
                                    k += 1
                                    fsum += h[ii, jj]
                if k == ktot - 1:
                    flag[i, j] = fsum / k
    hclean = h.copy()
    hclean[flag > 0] = flag[flag > 0]

    # Remove pixels with less than 4 neighbors
    flag = np.zeros_like(h, dtype=np.uint8)
    for i in range(naxis2):
        for j in range(naxis1):
            if h[i, j] != 0:
                k = 0
                for ii in [i - 1, i, i + 1]:
                    if 0 <= ii < naxis2:
                        for jj in [j - 1, j, j + 1]:
                            if 0 <= jj < naxis1:
                                if h[ii, jj] != 0:
                                    k += 1
                if k < 5:
                    flag[i, j] = 1
    hclean[flag == 1] = 0

    """
    # Remove pixels with no neighbor on the left hand side
    # when moving from left to right in each row
    for i in range(naxis2):
        j = 0
        loop = True
        while loop:
            j += 1
            if j < naxis1:
                loop = h[i, j] != 0
            else:
                loop = False
        if j < naxis1:
            pass
            hclean[i, j:] = 0
    """

    return hclean
