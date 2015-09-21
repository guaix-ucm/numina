
import numpy
from numpy.linalg import inv
from scipy.ndimage.filters import generic_filter
from ._kernels import kernel_peak_function

WW = {}
WW[3] = numpy.array([[  1.11022302e-16, 1.00000000e+00, 1.11022302e-16],
                     [ -5.00000000e-01, 0.00000000e+00, 5.00000000e-01],
                     [  5.00000000e-01, -1.00000000e+00, 5.00000000e-01]])


def get_peak_indexes(original_data, window_width=5, threshold=0.0):
    '''

    :param original_data: Array
    Has all original values of the graphic
    :param window_width: int,
    Width of the window to seek for the peaks.
    :param threshold: float
    Value used to trim values to seek for the peaks
    :return: ndarray
    Integer indexes of the original_data where a peak has been detected
    '''

    kernel_peak2 = kernel_peak_function(threshold)
    out = generic_filter(original_data, kernel_peak2, window_width)
    result, =  numpy.nonzero(out)
    numero_maximo = numpy.amax(original_data) - (window_width // 2)
    numero_minimo = window_width // 2
    result = result[(result >= numero_minimo) & (result <= numero_maximo)]
    return result


def generate_weights(window_width):
    '''

    :param window_width: Int, odd number
    Width of the window (greater or equal than 3) to seek for the peaks.
    :return: ndarray
    Matrix needed to interpolate 'window_width' points
    '''
    try:
        return WW[window_width]
    except:
        evenly_spaced = numpy.linspace(-1, 1, window_width)
        pow_matrix = numpy.vander(evenly_spaced, N=3, increasing=True)
        final_ww =  numpy.dot(inv(numpy.dot(pow_matrix.T, pow_matrix)), pow_matrix.T)
        WW[window_width] = final_ww
        return final_ww


def refine_peaks(original_data, ipeaks, window_width):
    '''

    :param original_data: array
    Has all original values of the graphic
    :param ipeaks: array (int)
    Has the indexes of original_data where a peak is found
    :param window_width: int,
    Width of the window to seek for the peaks.
    :return: ndarray, ndarry
    Values which correspond with the OX and the OY axis respectively
    '''

    step = window_width // 2

    winoff = numpy.arange(-step, step+1)
    peakwin = ipeaks[:, numpy.newaxis] + winoff
    ycols = original_data[peakwin]

    ww = generate_weights(window_width)

    coff2 = numpy.dot(ww, ycols.T)

    uc = -0.5 * coff2[1] / coff2[2]
    yc = coff2[0] + uc * (coff2[1] + coff2[2] * uc)
    xc = ipeaks + 0.5 * (window_width-1) * uc

    return xc, yc

