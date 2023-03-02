#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import math

from ..utils import coor_to_pix
from ._traces import tracing


def trace(arr, x, y, axis=0, background=0.0,
          step=4, hs=1, tol=2, maxdis=2.0, gauss=1):
    """Trace peak in array starting in (x,y).

    Trace a peak feature in an array starting in position (x,y).

    Parameters
    ----------
    arr : array
         A 2D array
    x : float
        x coordinate of the initial position
    y : float
        y coordinate of the initial position
    axis : {0, 1}
           Spatial axis of the array (0 is Y, 1 is X).
    background: float
           Background level
    step : int, optional
           Number of pixels to move (left and rigth)
           in each iteration
    gauss : bint, optional
            If 1 --> Gaussian interpolation method
            if 0 --> Linear interpolation method

    Returns
    -------
    ndarray
        A nx3 array, with x,y,p of each point in the trace
    """

    i,j = coor_to_pix([x, y], order='xy')
    value = arr[i,j]

    # If arr is not in native byte order, the C-extension won't work

    if arr.dtype.byteorder != '=':
        arr2 = arr.byteswap().newbyteorder()
    else:
        arr2 = arr

    if axis == 0:
        arr3 = arr2
    elif axis == 1:
        arr3 = arr2.t
    else:
        raise ValueError("'axis' must be 0 or 1")

    result = tracing(arr3, x, y, value, background=background,
                     step=step, hs=hs, tol=tol, maxdis=maxdis, gauss=gauss)

    if axis == 1:
        # Flip X,Y columns
        return result[:,::-1]

    return result


def tracing_limits(size, col, step, hs):
    m = col % step
    k0 = ((hs + step - m) / step)
    r0 = int(math.floor(k0))
    xx_start = r0 * step + m
    k1 = ((size - hs - step - m) / step)
    r1 = int(math.ceil(k1))
    xx_end = r1 * step + m
    return xx_start, xx_end


def axis_to_dispaxis(axis):
    """Obtain the dispersion axis from the spatial axis."""
    if axis == 0:
        dispaxis = 1
    elif axis == 1:
        dispaxis = 0
    else:
        raise ValueError("'axis' must be 0 or 1")
    return dispaxis
