
import numpy
from numpy.linalg import inv
from scipy.ndimage.filters import generic_filter
from ._kernels import kernel_peak_function

WW = {}
WW[3] = numpy.array([[1.11022302e-16, 1.00000000e+00, 1.11022302e-16],
                     [-5.00000000e-01, 0.00000000e+00, 5.00000000e-01],
                     [5.00000000e-01, -1.00000000e+00, 5.00000000e-01]])


def find_peaks_indexes(arr, window_width=5, threshold=0.0):
    """Find indexes of peaks in a 1d array.

    Note that window_width must be an odd number. The function imposes that the
    fluxes in the window_width /2 points to the left (and right) of the peak
    decrease monotonously as one moves away from the peak.

    Parameters
    ----------
    arr : 1d numpy array
        Input 1D spectrum.
    window_width : int
        Width of the window where the peak must be found. This number must be
        odd.
    threshold : float
        Minimum signal in the peak (optional).

    Returns
    -------
    ipeaks : 1d numpy array (int)
        Indices of the input array arr in which the peaks have been found.


    """

    kernel_peak = kernel_peak_function(threshold)
    out = generic_filter(arr, kernel_peak, window_width)
    result, =  numpy.nonzero(out)

    return result


def generate_weights(window_width):
    """

    :param window_width: Int, odd number
    Width of the window (greater or equal than 3) to seek for the peaks.
    :return: ndarray
    Matrix needed to interpolate 'window_width' points
    """

    try:
        return WW[window_width]
    except KeyError:
        evenly_spaced = numpy.linspace(-1, 1, window_width)
        pow_matrix = numpy.vander(evenly_spaced, N=3, increasing=True)
        final_ww = numpy.dot(inv(numpy.dot(pow_matrix.T, pow_matrix)), pow_matrix.T)
        WW[window_width] = final_ww
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

    step = window_width // 2

    winoff = numpy.arange(-step, step+1)
    peakwin = ipeaks[:, numpy.newaxis] + winoff
    ycols = arr[peakwin]

    ww = generate_weights(window_width)

    coff2 = numpy.dot(ww, ycols.T)

    uc = -0.5 * coff2[1] / coff2[2]
    yc = coff2[0] + uc * (coff2[1] + coff2[2] * uc)
    xc = ipeaks + 0.5 * (window_width-1) * uc

    return xc, yc
