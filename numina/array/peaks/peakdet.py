
import numpy
from numpy.linalg import inv
from scipy.ndimage.filters import generic_filter

from ._kernels import kernel_peak_function

def kernel_peak1(buff, threshold):
    """Kernel for peak finding"""
    filter_size = buff.shape[0]
    nmed = filter_size // 2

    if buff[nmed] < threshold:
        return 0.0

    for i in range(nmed, filter_size-1):
        if buff[i] < buff[i+1]:
            return 0.0

    for i in range(0, nmed):
        if buff[i] > buff[i+1]:

            return 0.0

    return 1.0


def find_peaks_index1(arr, window=5, threshold=0.0):
    out = generic_filter(arr, kernel_peak1, window, extra_arguments=(threshold,))
    result, =  numpy.nonzero(out)
    return result


def find_peaks_index2(arr, window=5, threshold=0.0):

    kernel_peak2 = kernel_peak_function(threshold)

    out = generic_filter(arr, kernel_peak2, window)
    result, =  numpy.nonzero(out)
    return result


# Auxiliary kernels
def generate_kernel(window):
    """Auxiliary kernel."""
    xm = numpy.linspace(-1, 1, window)
    xv = numpy.vander(xm, N=3, increasing=True)
    return numpy.dot(inv(numpy.dot(xv.T, xv)), xv.T)


WW = {}
WW[3] = generate_kernel(3)
WW[5] = generate_kernel(5)
WW[7] = generate_kernel(7)
WW[9] = generate_kernel(9)


def peakfun(xx, yy, ww):

    # map x to [-1:1]
    b1 = xx[-1]
    a1 = xx[0]
    alp1 = 2.0 / (b1-a1)
    bet1 = - (b1+a1) / (b1-a1)

    # map y to [0:1]
    b2 = yy.max()
    a2 = yy.min()
    alp2 = 1.0 / (b2-a2)
    bet2 = -a2 / (b2-a2)

    # um = alp1 * xx + bet1 # um is always [-1,1] with N steps
    vm = (yy - a2) / (b2-a2)

    thetap = numpy.dot(ww, vm)

    # Coordinates of the center
    uc = -thetap[1] / (2*thetap[2])
    vc = thetap[0] + uc * (thetap[1]+thetap[2]*uc)

    # Convert c,b,a, just for checking
    # trn = np.array([[1, bet1, bet1**2], [0, alp1, 2 * alp1 * bet1], [0, 0, alp1**2]])
    # theta2 = np.dot(trn, thetap)
    # theta = (theta2 - [bet2, 0, 0]) / alp2

    #xc = -theta[1] / (2*theta[2])
    #yc = theta[0] + theta[1]*xc+theta[2]*xc**2
    xc = (uc - bet1) / alp1
    yc = (vc - bet2) / alp2
    return xc, yc


def refine_peaks1(arr, ipeaks, window):

    step = window // 2

    xfpeaks = numpy.zeros(len(ipeaks))

    if window in WW:
        ww = WW[window]
    else:
        ww = generate_kernel(window)

    for idx, ipeak in enumerate(ipeaks):
        xx = numpy.arange(ipeak-step, ipeak+step+1)
        yy = arr[ipeak-step:ipeak+step+1].copy()
        xc, yc = peakfun(xx, yy, ww)
        xfpeaks[idx] = xc
    return xfpeaks, xfpeaks


def refine_peaks2(arr, ipeaks, window):

    step = window // 2

    winoff = numpy.arange(-step, step+1)
    peakwin = ipeaks[:, numpy.newaxis] + winoff

    ycols = arr[peakwin]
    # numpy.take takes the same time
    # ycols = np.take(arr, peakwin)

    if window in WW:
        ww = WW[window]
    else:
        ww = generate_kernel(window)

    coff2 = numpy.dot(ww, ycols.T)

    uc = -0.5 * coff2[1] / coff2[2]

    # Evaluate yc
    #
    yc = coff2[0] + uc * (coff2[1] + coff2[2] * uc)
    #
    xc = ipeaks + 0.5 * (window-1) * uc
    return xc, yc


def refine_peaks3(arr, ipeaks, window):

    step = window // 2

    winoff = numpy.arange(-step, step+1)
    peakwin = ipeaks[:, numpy.newaxis] + winoff

    ycols = arr[peakwin]

    ww = generate_kernel(window)

    coff2 = numpy.dot(ww, ycols.T)

    uc = -0.5 * coff2[1] / coff2[2]

    # Evaluate yc
    yc = coff2[0] + uc * (coff2[1] + coff2[2] * uc)
    #

    xc = ipeaks + 0.5 * (window-1) * uc
    return xc, yc


def refine_peaks3b(arr, ipeaks, window):

    step = window // 2

    winoff = numpy.arange(-step, step+1)
    peakwin = ipeaks[:, numpy.newaxis] + winoff

    # Divide between peak value
    ycols = arr[peakwin]
    ypeak = ycols[:,step]
    ycols = ycols / ypeak[:, numpy.newaxis]

    ww = generate_kernel(window)

    coff2 = numpy.dot(ww, ycols.T)

    uc = -0.5 * coff2[1] / coff2[2]

    # Evaluate yc
    vc = coff2[0] + uc * (coff2[1] + coff2[2] * uc)
    yc = ypeak * vc
    #

    xc = ipeaks + 0.5 * (window-1) * uc
    return xc, yc


def refine_peaks4(arr, ipeaks, window):

    step = window // 2

    winoff = numpy.arange(-step, step+1)
    peakwin = ipeaks[:, numpy.newaxis] + winoff

    ycols = arr[peakwin]
    coff2 = numpy.polyfit(winoff, ycols.T, 2)

    # Higher order goes first
    uc = -0.5 * coff2[1] / coff2[0]

    # Evaluate yc
    yc = coff2[2] + uc * (coff2[1] + coff2[0] * uc)
    #

    xc = ipeaks +  uc
    return xc, yc


