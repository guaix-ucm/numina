#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Define a piecewise linear function."""
import numpy as np


def define_piecewise_linear_function(xarray, yarray):
    """Define a piecewise linear function.

    Parameters
    ----------
    xarray : 1D numpy array
        The x coordinates of the points.
    yarray : 1D numpy array
        The y coordinates of the points.

    """
    isort = np.argsort(xarray)
    xfit = xarray[isort]
    yfit = yarray[isort]

    def function(x):
        y = np.interp(x, xfit, yfit)
        return y

    return function
