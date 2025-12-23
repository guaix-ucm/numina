#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Calculate elliptical 2D Gaussian kernel."""
import math
import numpy as np


def gausskernel2d_elliptical(fwhm_x, fwhm_y, kernsize):
    """Calculate elliptical 2D Gaussian kernel.

    This kernel can be used as the psfk parameter of
    ccdproc.cosmicray_lacosmic.

    Parameters
    ----------
    fwhm_x : float
        Full width at half maximum in the x direction.
    fwhm_y : float
        Full width at half maximum in the y direction.
    kernsize : int
        Size of the kernel (must be odd).

    Returns
    -------
    kernel : 2D numpy array
        The elliptical Gaussian kernel. It is normalized
        so that the sum of all its elements is 1. It
        is returned as a float32 array (required by
        ccdproc.cosmicray_lacosmic).
    """

    if kernsize % 2 == 0 or kernsize < 3:
        raise ValueError("kernsize must be an odd integer >= 3.")
    if fwhm_x <= 0 or fwhm_y <= 0:
        raise ValueError("fwhm_x and fwhm_y must be positive numbers.")

    sigma_x = fwhm_x / (2 * math.sqrt(2 * math.log(2)))
    sigma_y = fwhm_y / (2 * math.sqrt(2 * math.log(2)))
    halfsize = kernsize // 2
    y, x = np.mgrid[-halfsize : halfsize + 1, -halfsize : halfsize + 1]
    kernel = np.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))
    kernel /= np.sum(kernel)

    # reverse the psf kernel as that is what it is used in the convolution
    kernel = kernel[::-1, ::-1]
    return kernel.astype(np.float32)
