#
# Copyright 2015-2022 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""
Functions to find peaks in 1D arrays and subpixel positions
"""


import numpy
from numpy import dot
from numpy.linalg import inv
from scipy.ndimage import generic_filter
from ._kernels import kernel_peak_function


WW = dict()
WW[3] = numpy.array([[1.11022302e-16, 1.00000000e+00, 1.11022302e-16],
                     [-5.00000000e-01, 0.00000000e+00, 5.00000000e-01],
                     [5.00000000e-01, -1.00000000e+00, 5.00000000e-01]])

WW[5] = numpy.array([[-0.08571429, 0.34285714, 0.48571429, 0.34285714, -0.08571429],
                     [-0.4, -0.2, 0, 0.2, 0.4],
                     [0.57142857, -0.28571429, -0.57142857, -0.28571429, 0.57142857]])

WW[7] = numpy.array([[-9.52380952e-02, 1.42857143e-01, 2.85714286e-01, 3.33333333e-01,
                      2.85714286e-01, 1.42857143e-01, -9.52380952e-02],
                     [-3.21428571e-01, -2.14285714e-01, -1.07142857e-01, -6.79728382e-18,
                      1.07142857e-01, 2.14285714e-01, 3.21428571e-01],
                     [5.35714286e-01, 5.55111512e-17, -3.21428571e-01, -4.28571429e-01,
                      -3.21428571e-01, -2.22044605e-16, 5.35714286e-01]])

WW[9] = numpy.array([[-0.09090909, 0.06060606, 0.16883117, 0.23376623, 0.25541126,
                      0.23376623, 0.16883117, 0.06060606, -0.09090909],
                     [-0.26666667, -0.2, -0.13333333, -0.06666667, 0,
                      0.06666667, 0.13333333, 0.2, 0.26666667],
                     [0.48484848, 0.12121212, -0.13852814, -0.29437229, -0.34632035,
                      -0.29437229, -0.13852814, 0.12121212, 0.48484848]])


def _check_window_width(window_width):
    """Check `window_width` is odd and >=3"""
    if (window_width < 3) or (window_width % 2 == 0):
        raise ValueError('Window width must be an odd number and >=3')


def filter_array_margins(arr, ipeaks, window_width=5):
    _check_window_width(window_width)

    max_number = (len(arr)-1) - (window_width // 2)
    min_number = window_width // 2
    return ipeaks[(ipeaks >= min_number) & (ipeaks <= max_number)]


def find_peaks_indexes(arr, window_width=5, threshold=0.0, fpeak=0):
    """Find indexes of peaks in a 1d array.

    Note that window_width must be an odd number. The function imposes that the
    fluxes in the window_width /2 points to the left (and right) of the peak
    decrease monotonously as one moves away from the peak, except that
    it allows fpeak constant values around the peak.

    Parameters
    ----------
    arr : 1d numpy array
        Input 1D spectrum.
    window_width : int
        Width of the window where the peak must be found. This number must be
        odd.
    threshold : float
        Minimum signal in the peak (optional).
    fpeak: int
        Number of equal values around the peak

    Returns
    -------
    ipeaks : 1d numpy array (int)
        Indices of the input array arr in which the peaks have been found.


    """

    _check_window_width(window_width)

    if fpeak < 0 or fpeak + 1 >= window_width:
        raise ValueError('fpeak must be in the range 0- window_width - 2')

    kernel_peak = kernel_peak_function(threshold, fpeak)
    out = generic_filter(arr, kernel_peak, window_width, mode="reflect")
    result, =  numpy.nonzero(out)

    return filter_array_margins(arr, result, window_width)


def return_weights(window_width):
    """

    Parameters
    ----------
    window_width : int
       Odd number greater than 3, width of the window to seek for the peaks.
    Returns
    -------
    ndarray :
           Matrix needed to interpolate 'window_width' points
    """
    _check_window_width(window_width)

    try:
        return WW[window_width]
    except KeyError:
        final_ww = generate_weights(window_width)
        WW[window_width] = final_ww
        return final_ww


def generate_weights(window_width):
    """

    Parameters
    ----------
    window_width : int
                Odd number greater than 3, width of the window to seek for the peaks.

    Returns
    -------
    ndarray :
           Matrix needed to interpolate 'window_width' points
    """
    _check_window_width(window_width)

    evenly_spaced = numpy.linspace(-1, 1, window_width)
    pow_matrix = numpy.fliplr(numpy.vander(evenly_spaced, N=3))
    final_ww = dot(inv(dot(pow_matrix.T, pow_matrix)), pow_matrix.T)
    return final_ww


def refine_peaks(arr, ipeaks, window_width):
    """Refine the peak location previously found by find_peaks_indexes

    Parameters
    ----------
    arr : 1d numpy array, float
        Input 1D spectrum.
    ipeaks : 1d numpy array (int)
        Indices of the input array arr in which the peaks were initially found.
    window_width : int
        Width of the window where the peak must be found.

    Returns
    -------
     xc, yc: tuple
        X-coordinates in which the refined peaks have been found,
        interpolated Y-coordinates

    """
    _check_window_width(window_width)

    step = window_width // 2

    ipeaks = filter_array_margins(arr, ipeaks, window_width)

    winoff = numpy.arange(-step, step+1, dtype='int')
    peakwin = ipeaks[:, numpy.newaxis] + winoff
    ycols = arr[peakwin]

    ww = return_weights(window_width)

    coff2 = numpy.dot(ww, ycols.T)

    uc = -0.5 * coff2[1] / coff2[2]
    yc = coff2[0] + uc * (coff2[1] + coff2[2] * uc)
    xc = ipeaks + 0.5 * (window_width-1) * uc

    return xc, yc
