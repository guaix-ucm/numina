#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compute offsets between 2d images using 2D cross-correlation."""

import numpy as np
from scipy.signal import correlate2d

import numina.array.imsurfit as imsurfit
from numina.array.imsurfit import vertex_of_quadratic
import numina.array.utils as utils


def yx_offsets_correlate2d(reference_image, moving_image, refine_box=3):
    """Compute (Y, X) offsets between two 2D images using cross-correlation.

    Here we use two calls to correlate2d, and this is itentional.
    The first call computes the autocorrelation of the reference image
    with itself. The second call computes the actual cross-correlation
    between the reference image and the moving image. Both correlation
    maps (corr_self and corr) are locally fitted with a 2D quadratic
    surface (via imsurfit) around their respective peaks. The difference
    between the two fitted peaks provides a high-precision (subpixel)
    estimate of the offsets between the two images. This approach helps
    to mitigate potential biases that could arise from asymmetries or
    noise in the images, leading to a more robust determination of the
    offsets.

    Parameters
    ----------
    reference_image : 2D array
        The reference image (e.g., a template or a fixed image).
    moving_image : 2D array
        The image to be aligned (e.g., a target or moving image).
    refine_box : int, optional
        The size of the box (in pixels) around the peak of the correlation
        to use for the quadratic fit. Default is 3.

    Returns
    -------
    yx_offsets : array
        A 2-element array containing the (Y, X) offsets between the two images.

    """

    corr_self = correlate2d(
        in1=reference_image,
        in2=reference_image,
        mode='full',
        boundary='fill',
        fillvalue=0
    )
    corr = correlate2d(
        in1=reference_image,
        in2=moving_image,
        mode='full',
        boundary='fill',
        fillvalue=0
    )
    maxindex_self = np.unravel_index(np.argmax(corr_self), corr_self.shape)
    maxindex = np.unravel_index(np.argmax(corr), corr.shape)
    region_refine_self = utils.image_box(
        maxindex_self,
        corr_self.shape,
        box=(refine_box, refine_box)
    )
    region_refine = utils.image_box(
        maxindex,
        corr.shape,
        box=(refine_box, refine_box)
    )
    coeffs_self, = imsurfit.imsurfit(corr_self[region_refine_self], order=2)
    coeffs, = imsurfit.imsurfit(corr[region_refine], order=2)
    xm, ym = vertex_of_quadratic(coeffs_self)
    maxindex_self += np.asarray([ym, xm])
    xm, ym = vertex_of_quadratic(coeffs)
    maxindex += np.asarray([ym, xm])
    yx_offsets = np.asarray(maxindex) - np.asarray(maxindex_self)

    return yx_offsets
