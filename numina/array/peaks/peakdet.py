
import numpy
from numpy.linalg import inv
from scipy.ndimage.filters import generic_filter

from ._kernels import kernel_peak_function


def get_peak_indexes(original_data, window_width=5, threshold=0.0):

    kernel_peak2 = kernel_peak_function(threshold)
    out = generic_filter(original_data, kernel_peak2, window_width)
    result, =  numpy.nonzero(out)

    return result


def generate_kernel(window_width):
    """Auxiliary kernel."""
    evenly_spaced = numpy.linspace(-1, 1, window_width)
    pow_matrix = numpy.vander(evenly_spaced, N=3, increasing=True)
    return numpy.dot(inv(numpy.dot(pow_matrix.T, pow_matrix)), pow_matrix.T)


WW = {}
WW[3] = generate_kernel(3)
WW[5] = generate_kernel(5)
WW[7] = generate_kernel(7)
WW[9] = generate_kernel(9)


def accurated_peaks_spectrum(original_data, ipeaks, window_width):

    step = window_width // 2

    winoff = numpy.arange(-step, step+1)
    peakwin = ipeaks[:, numpy.newaxis] + winoff
    ycols = original_data[peakwin]

    if window_width in WW:
        ww = WW[window_width]
    else:
        ww = generate_kernel(window_width)

    coff2 = numpy.dot(ww, ycols.T)

    uc = -0.5 * coff2[1] / coff2[2]
    yc = coff2[0] + uc * (coff2[1] + coff2[2] * uc)
    xc = ipeaks + 0.5 * (window_width-1) * uc

    return xc, yc

