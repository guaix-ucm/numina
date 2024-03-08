#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Broad spectrum (linear wavelength scale) applying a Gaussian"""

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.ndimage import gaussian_filter1d, convolve


def broadsp_gaussian_linearwv(crval1, cdelt1, flux, fwhm_velocity, tsigma=5.0):
    """Apply a Gaussian broadening to a spectrum.

    The method assumes that the spectrum is linearly calibrated
    in wavelength. Since the Gaussian function is parametrized by a
    constant FWHM in velocity units, a varying kernel width (in pixels)
    is computed as we move in the wavelength direction.

    Parameters
    ----------
    crval1 : `~astroppy.units.Quantity`
        Wavelength of the first spectrum pixel.
    cdelt1 : `~astroppy.units.Quantity`
        Constant wavelength increment/pixel.
    flux : array_like
        Array-like object containing the spectrum flux
    fwhm_velocity : `~astroppy.units.Quantity`
        FWHM of the Gaussian kernel in velocity units.
    tsigma : float
        Times sigma to extend the computation of the Gaussian
        broadening.

    Returns
    -------
    broadened_flux : ndarray
        Broadened spectrum.

    """

    # initial checks
    if not isinstance(crval1, u.Quantity):
        raise ValueError(f'{crval1=} is not a Quantity')
    if not crval1.unit.is_equivalent(u.m):
        raise ValueError(f"Unexpected units for 'crval1': {crval1.unit}")

    if not isinstance(cdelt1, u.Quantity):
        raise ValueError(f'{cdelt1=} is not a Quantity')
    if not cdelt1.unit.is_equivalent(u.m / u.pix):
        raise ValueError(f"Unexpected units for 'cdelt1': {cdelt1.unit}")

    if not isinstance(fwhm_velocity, u.Quantity):
        raise ValueError(f'{fwhm_velocity=} is not a Quantity')
    if not fwhm_velocity.unit.is_equivalent(u.m / u.s):
        raise ValueError(f"Unexpected units for 'fwhm_velocity': {fwhm_velocity.unit}")

    # generate wavelength array
    flux = np.asarray(flux)
    naxis1 = len(flux)
    wave = crval1 + np.arange(naxis1) * u.pix * cdelt1

    # conversion factor between FWHM and sigma
    factor_fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))

    # varying sigma for each pixel (wavelength units)
    sigma_wave = fwhm_velocity.to(u.m / u.s) / const.c.to(u.m / u.s) * factor_fwhm_to_sigma * wave

    # varying sigma for each pixel (pixel units)
    sigma_pix = (sigma_wave / cdelt1).value

    # kernel size (pixel units; integer values)
    initial_kernel_size_intpix = (2 * sigma_pix * tsigma).astype(int) + 1
    # force the kernel size to be odd
    kernel_size_intpix = np.where(
        initial_kernel_size_intpix % 2 == 0,
        initial_kernel_size_intpix + 1,
        initial_kernel_size_intpix
    )

    # broaden spectrum
    broadened_flux = np.zeros(naxis1)
    for i in range(naxis1):
        # select kernel size
        kernel_size = kernel_size_intpix[i]
        # location of the kernel center
        kcenter = kernel_size // 2
        # generate array of zeros an insert 1.0 in the central pixel
        delta_filter = np.zeros(kernel_size)
        delta_filter[kcenter] = 1.0
        # generate Gaussian kernel
        gauss_filter = gaussian_filter1d(delta_filter, sigma_pix[i])
        # redistribute the signal from the current pixel to the
        # neighboring pixels, following the kernel calculated in the
        # current pixel
        redistributed_pixel = gauss_filter * flux[i]
        i1 = max(0, i - kcenter)
        i2 = min(naxis1, i + kcenter + 1)
        ii1 = max(0, kcenter - i)
        ii2 = ii1 + (i2 - i1)
        broadened_flux[i1:i2] += redistributed_pixel[ii1:ii2]

    return broadened_flux
