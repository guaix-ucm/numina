#
# Copyright 2008-2021 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import math

import numpy


def fowler_array(fowlerdata, ti=0.0, ts=0.0, gain=1.0, ron=1.0,
                 badpixels=None, dtype='float64',
                 saturation=65631, blank=0, normalize=False):
    """Loop over the first axis applying Fowler processing.

    *fowlerdata* is assumed to be a 3D numpy.ndarray containing the
    result of a nIR observation in Fowler mode (Fowler and Gatley 1991).
    The shape of the array must be of the form 2N_p x M x N, with N_p being
    the number of pairs in Fowler mode.

    The output signal is just the mean value of the differences between the
    last N_p values (S_i) and the first N_p values (R-i).

    .. math::

        S_F = \\frac{1}{N_p}\\sum\\limits_{i=0}^{N_p-1} S_i - R_i


    If the source has a radiance F, then the measured signal is equivalent
    to:

    .. math::

        S_F = F T_I - F T_S (N_p -1) = F T_E

    being T_I the integration time (*ti*), the time since the first
    productive read to the last productive read for a given pixel and T_S the
    time between samples (*ts*). T_E is the time between correlated reads
    :math:`T_E = T_I - T_S (N_p - 1)`.

    The variance of the signnal is the sum of two terms, one for the readout
    noise:

    .. math::

        \\mathrm{var}(S_{F1}) =\\frac{2\sigma_R^2}{N_p}

    and other for the photon noise:

    .. math::

        \\mathrm{var}(S_{F2}) = F T_E - F T_S \\frac{1}{3}(N_p-\\frac{1}{N_p})
        = F T_I - F T_S (\\frac{4}{3} N_p -1 -  \\frac{1}{3N_p})


    :param fowlerdata: Convertible to a 3D numpy.ndarray with first axis even
    :param ti: Integration time.
    :param ts: Time between samples.
    :param gain: Detector gain.
    :param ron: Detector readout noise in counts.
    :param badpixels: An optional MxN mask of dtype 'uint8'.
    :param dtype: The dtype of the float outputs.
    :param saturation: The saturation level of the detector.
    :param blank: Invalid values in output are substituted by *blank*.
    :returns: A tuple of (signal, variance of the signal, numper of pixels used
        and badpixel mask.
    :raises: ValueError

    """
    import numina.array._nirproc as _nirproc

    if gain <= 0:
        raise ValueError("invalid parameter, gain <= 0.0")

    if ron <= 0:
        raise ValueError("invalid parameter, ron < 0.0")

    if ti < 0:
        raise ValueError("invalid parameter, ti < 0.0")

    if ts < 0:
        raise ValueError("invalid parameter, ts < 0.0")

    if saturation <= 0:
        raise ValueError("invalid parameter, saturation <= 0")

    fowlerdata = numpy.asarray(fowlerdata)

    if fowlerdata.ndim != 3:
        raise ValueError('fowlerdata must be 3D')

    npairs = fowlerdata.shape[0] // 2
    if 2 * npairs != fowlerdata.shape[0]:
        raise ValueError('axis-0 in fowlerdata must be even')

    # change byteorder
    ndtype = fowlerdata.dtype.newbyteorder('=')
    fowlerdata = numpy.asarray(fowlerdata, dtype=ndtype)
    # type of the output
    fdtype = numpy.result_type(fowlerdata.dtype, dtype)
    # Type of the mask
    mdtype = numpy.dtype('uint8')

    fshape = (fowlerdata.shape[1], fowlerdata.shape[2])

    if badpixels is None:
        badpixels = numpy.zeros(fshape, dtype=mdtype)
    else:
        if badpixels.shape != fshape:
            raise ValueError('shape of badpixels is not '
                             'compatible with shape of fowlerdata')
        if badpixels.dtype != mdtype:
            raise ValueError('dtype of badpixels must be uint8')

    result = numpy.empty(fshape, dtype=fdtype)
    var = numpy.empty_like(result)
    npix = numpy.empty(fshape, dtype=mdtype)
    mask = badpixels.copy()

    _nirproc._process_fowler_intl(
        fowlerdata, ti, ts,  gain, ron,
        badpixels, saturation, blank,
        result, var, npix, mask
    )
    return result, var, npix, mask


def ramp_array(rampdata, ti, gain=1.0, ron=1.0,
               badpixels=None, dtype='float64',
               saturation=65631, blank=0, nsig=None, normalize=False):
    """Loop over the first axis applying ramp processing.

    *rampdata* is assumed to be a 3D numpy.ndarray containing the
    result of a nIR observation in folow-up-the-ramp mode.
    The shape of the array must be of the form N_s x M x N, with N_s being
    the number of samples.

    :param fowlerdata: Convertible to a 3D numpy.ndarray
    :param ti: Integration time.
    :param gain: Detector gain.
    :param ron: Detector readout noise in counts.
    :param badpixels: An optional MxN mask of dtype 'uint8'.
    :param dtype: The dtype of the float outputs.
    :param saturation: The saturation level of the detector.
    :param blank: Invalid values in output are substituted by *blank*.
    :returns: A tuple of signal, variance of the signal, numper of pixels used
        and badpixel mask.
    :raises: ValueError
    """

    import numina.array._nirproc as _nirproc
    if ti <= 0:
        raise ValueError("invalid parameter, ti <= 0.0")

    if gain <= 0:
        raise ValueError("invalid parameter, gain <= 0.0")

    if ron <= 0:
        raise ValueError("invalid parameter, ron < 0.0")

    if saturation <= 0:
        raise ValueError("invalid parameter, saturation <= 0")

    rampdata = numpy.asarray(rampdata)
    if rampdata.ndim != 3:
        raise ValueError('rampdata must be 3D')

    # change byteorder
    ndtype = rampdata.dtype.newbyteorder('=')
    rampdata = numpy.asarray(rampdata, dtype=ndtype)
    # type of the output
    fdtype = numpy.result_type(rampdata.dtype, dtype)
    # Type of the mask
    mdtype = numpy.dtype('uint8')
    fshape = (rampdata.shape[1], rampdata.shape[2])

    if badpixels is None:
        badpixels = numpy.zeros(fshape, dtype=mdtype)
    else:
        if badpixels.shape != fshape:
            msg = 'shape of badpixels is not compatible with shape of rampdata'
            raise ValueError(msg)
        if badpixels.dtype != mdtype:
            raise ValueError('dtype of badpixels must be uint8')

    result = numpy.empty(fshape, dtype=fdtype)
    var = numpy.empty_like(result)
    npix = numpy.empty(fshape, dtype=mdtype)
    mask = badpixels.copy()

    _nirproc._process_ramp_intl(
        rampdata, ti, gain, ron, badpixels,
        saturation, blank, result, var, npix, mask
    )
    return result, var, npix, mask


# This is not used...
# Old code used to detect cosmic rays in the ramp
def _ramp(data, saturation, dt, gain, ron, nsig):
    nsdata = data[data < saturation]

    # Finding glitches in the pixels
    intervals, glitches = _rglitches(nsdata, gain=gain, ron=ron, nsig=nsig)
    vals = numpy.asarray([_slope(nsdata[intls], dt=dt, gain=gain, ron=ron)
                          for intls in intervals if len(nsdata[intls]) >= 2])
    weights = (1.0 / vals[:, 1])
    average = numpy.average(vals[:, 0], weights=weights)
    variance = 1.0 / weights.sum()
    return average, variance, vals[:, 2].sum(), glitches


def _rglitches(nsdata, gain, ron, nsig):
    diffs = nsdata[1:] - nsdata[:-1]
    psmedian = numpy.median(diffs)
    sigma = math.sqrt(abs(psmedian / gain) + 2 * ron * ron)

    start = 0
    intervals = []
    glitches = []
    for idx, diff in enumerate(diffs):
        if not (psmedian - nsig * sigma < diff < psmedian + nsig * sigma):
            intervals.append(slice(start, idx + 1))
            start = idx + 1
            glitches.append(start)

    intervals.append(slice(start, None))

    return intervals, glitches


def _slope(nsdata, dt, gain, ron):

    if len(nsdata) < 2:
        raise ValueError('Two points needed to compute the slope')

    nn = len(nsdata)
    delt = dt * nn * (nn + 1) * (nn - 1) / 12
    ww = numpy.arange(1, nn + 1) - (nn + 1) / 2

    final = (ww * nsdata).sum() / delt

    # Readout limited case
    delt2 = dt * delt
    var1 = (ron / gain)**2 / delt2
    # Photon limiting case
    var2 = (6 * final * (nn * nn + 1)) / (5 * nn * dt * (nn * nn - 1) * gain)
    variance = var1 + var2
    return final, variance, nn
