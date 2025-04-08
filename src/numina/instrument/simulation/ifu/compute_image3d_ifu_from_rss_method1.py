#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Compute image3d IFU from RSS image"""
import numpy as np
from scipy.signal import convolve2d

from numina.tools.ctext import ctext


def compute_image3d_ifu_from_rss_method1(
    image2d_rss_method1,
    naxis1_detector,
    naxis2_ifu,
    naxis1_ifu,
    nslices,
    verbose=False   
):
    """
    Compute the image3d IFU from the image2d RSS using method 1.
    
    Parameters
    ----------
    image2d_rss_method1 : numpy.ndarray
        The input RSS image.
    naxis1_detector : int
        Number of pixels in the first axis of the detector image
        (dispersion direction).
    naxis2_ifu : int
        Number of pixels in the second axis of the IFU image.
    naxis1_ifu : int
        Number of pixels in the first axis of the IFU image.
    nslices : int
        Number of slices in the IFU image.
    verbose : bool, optional
        If True, print verbose output. Default is False.
    
    Returns
    -------
    image3d_ifu_method1 : numpy.ndarray
        The output IFU image.
    """

    if verbose:
        print(ctext('\n* Computing image3d IFU from image2d RSS method 1', fg='green'))

    # kernel in the spectral direction
    # (bidimensional to be applied to a bidimensional image)
    # TODO: this is valid only for FRIDA
    kernel = np.array([[0.25, 0.50, 0.25]])

    # convolve RSS image
    convolved_data = convolve2d(image2d_rss_method1, kernel, boundary='fill', fillvalue=0, mode='same')

    # TODO: the second dimension in the following array should be 2*nslices
    # (check what to do for another IFU, like TARSIS)
    image3d_ifu_method1 = np.zeros((naxis1_detector.value, naxis2_ifu.value, naxis1_ifu.value))
    if verbose:
        print(f'(debug): {image3d_ifu_method1.shape=}')

    for islice in range(nslices):
        i1 = islice * 2
        j1 = islice * naxis1_ifu.value
        j2 = j1 + naxis1_ifu.value
        image3d_ifu_method1[:, i1, :] = convolved_data[j1:j2, :].T
        image3d_ifu_method1[:, i1+1, :] = convolved_data[j1:j2, :].T

    image3d_ifu_method1 /= 2
    if verbose:
        print(f'(debug): {np.sum(image2d_rss_method1)=}')
        print(f'(debug):      {np.sum(convolved_data)=}')
        print(f'(debug): {np.sum(image3d_ifu_method1)=}')

    return image3d_ifu_method1
