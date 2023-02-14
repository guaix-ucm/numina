#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Resampling functions related to wavelength calibration"""

import numpy as np
from numpy.polynomial.polynomial import polyval
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplotxy import ximplotxy
from numina.array.interpolation import SteffenInterpolator


def oversample1d(sp, crval1, cdelt1, oversampling=1, debugplot=0):
    """Oversample spectrum.

    Parameters
    ----------
    sp : numpy array
        Spectrum to be oversampled.
    crval1 : float
        Abscissae of the center of the first pixel in the original
        spectrum 'sp'.
    cdelt1 : float
        Abscissae increment corresponding to 1 pixel in the original
        spectrum 'sp'.
    oversampling : int
        Oversampling value per pixel.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    sp_over : numpy array
        Oversampled data array.
    crval1_over : float
        Abscissae of the center of the first pixel in the oversampled
        spectrum.
    cdelt1_over : float
        Abscissae of the center of the last pixel in the oversampled
        spectrum.

    """

    if sp.ndim != 1:
        raise ValueError('Unexpected array dimensions')

    naxis1 = sp.size
    naxis1_over = naxis1 * oversampling
    cdelt1_over = cdelt1 / oversampling
    xmin = crval1 - cdelt1/2   # left border of first pixel
    crval1_over = xmin + cdelt1_over / 2

    sp_over = np.zeros(naxis1_over)
    for i in range(naxis1):
        i1 = i * oversampling
        i2 = i1 + oversampling
        sp_over[i1:i2] = sp[i]

    if abs(debugplot) in (21, 22):
        crvaln = crval1 + (naxis1 - 1) * cdelt1
        crvaln_over = crval1_over + (naxis1_over - 1) * cdelt1_over
        xover = np.linspace(crval1_over, crvaln_over, naxis1_over)
        ax = ximplotxy(np.linspace(crval1, crvaln, naxis1), sp, 'bo',
                       label='original', show=False)
        ax.plot(xover, sp_over, 'r+', label='resampled')
        pause_debugplot(debugplot, pltshow=True)

    return sp_over, crval1_over, cdelt1_over


def rebin(a, *args):
    """See http://scipy-cookbook.readthedocs.io/items/Rebinning.html

    Note: integer division in the computation of 'factor' has been
    included to avoid the following runtime message:
    VisibleDeprecationWarning: using a non-integer number instead of
    an integer will result in an error in the future


    """

    shape = a.shape
    len_shape = len(shape)
    factor = np.asarray(shape) // np.asarray(args)
    # ev_list = ['a.reshape('] + \
    #           ['args[%d], factor[%d], ' % (i, i) for i in range(len_shape)] + \
    #           [')'] + ['.mean(%d)' % (i+1) for i in range(len_shape)]
    # FIXME: this construction is weird
    ev_list = ['a.reshape('] + \
              [f'args[{idx}], factor[{idx}], ' for idx in range(len_shape)] + \
              [')'] + [f'.mean({idx + 1})' for idx in range(len_shape)]
    # print(''.join(ev_list))
    return eval(''.join(ev_list))


def shiftx_image2d_flux(image2d_orig, xoffset):
    """Resample 2D image using a shift in the x direction (flux is preserved).

    Parameters
    ----------
    image2d_orig : numpy array
        2D image to be resampled.
    xoffset : float
        Offset to be applied.

    Returns
    -------
    image2d_resampled : numpy array
        Resampled 2D image.

    """

    if image2d_orig.ndim == 1:
        naxis1 = image2d_orig.size
    elif image2d_orig.ndim == 2:
        naxis2, naxis1 = image2d_orig.shape
    else:
        print('>>> image2d_orig.shape:', image2d_orig.shape)
        raise ValueError('Unexpected number of dimensions')

    return resample_image2d_flux(image2d_orig,
                                 naxis1=naxis1,
                                 cdelt1=1,
                                 crval1=1,
                                 crpix1=1,
                                 coeff=[xoffset, 1])


def resample_image2d_flux(image2d_orig,
                          naxis1, cdelt1, crval1, crpix1, coeff):
    """Resample a 1D/2D image using NAXIS1, CDELT1, CRVAL1, and CRPIX1.

    The same NAXIS1, CDELT1, CRVAL1, and CRPIX1 are employed for all
    the scans (rows) of the original 'image2d'. The wavelength
    calibrated output image has dimensions NAXIS1 * NSCAN, where NSCAN
    is the original number of scans (rows) of the original image.

    Flux is preserved.

    Parameters
    ----------
    image2d_orig : numpy array
        1D or 2D image to be resampled.
    naxis1 : int
        NAXIS1 of the resampled image.
    cdelt1 : float
        CDELT1 of the resampled image.
    crval1 : float
        CRVAL1 of the resampled image.
    crpix1 : float
        CRPIX1 of the resampled image.
    coeff : numpy array
        Coefficients of the wavelength calibration polynomial.

    Returns
    -------
    image2d_resampled : numpy array
        Wavelength calibrated 1D or 2D image.

    """

    # duplicate input array, avoiding problems when using as input
    # 1d numpy arrays with shape (nchan,) instead of a 2d numpy
    # array with shape (1,nchan)
    if image2d_orig.ndim == 1:
        nscan = 1
        nchan = image2d_orig.size
        image2d = np.zeros((nscan, nchan))
        image2d[0, :] = np.copy(image2d_orig)
    elif image2d_orig.ndim == 2:
        nscan, nchan = image2d_orig.shape
        image2d = np.copy(image2d_orig)
    else:
        print('>>> image2d_orig.shape:', image2d_orig.shape)
        raise ValueError('Unexpected number of dimensions')

    new_x = np.arange(naxis1)
    new_wl = crval1 + cdelt1 * new_x

    old_x_borders = np.arange(-0.5, nchan)
    old_x_borders += crpix1  # following FITS criterium

    new_borders = map_borders(new_wl)

    accum_flux = np.empty((nscan, nchan + 1))
    accum_flux[:, 1:] = np.cumsum(image2d, axis=1)
    accum_flux[:, 0] = 0.0
    image2d_resampled = np.zeros((nscan, naxis1))

    old_wl_borders = polyval(old_x_borders, coeff)

    for iscan in range(nscan):
        # We need a monotonic interpolator
        # linear would work, we use a cubic interpolator
        interpolator = SteffenInterpolator(
            old_wl_borders,
            accum_flux[iscan],
            extrapolate='border'
        )
        fl_borders = interpolator(new_borders)
        image2d_resampled[iscan] = fl_borders[1:] - fl_borders[:-1]

    if image2d_orig.ndim == 1:
        return image2d_resampled[0, :]
    else:
        return image2d_resampled


def map_borders(wls):
    """Compute borders of pixels for interpolation.

    The border of the pixel is assumed to be midway of the wls
    """
    midpt_wl = 0.5 * (wls[1:] + wls[:-1])
    all_borders = np.zeros((wls.shape[0] + 1,))
    all_borders[1:-1] = midpt_wl
    all_borders[0] = 2 * wls[0] - midpt_wl[0]
    all_borders[-1] = 2 * wls[-1] - midpt_wl[-1]
    return all_borders
