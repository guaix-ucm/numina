#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Compute RSS image from detector image"""
from joblib import Parallel, delayed
import numpy as np
import time

from numina.tools.ctext import ctext

from .update_image2d_rss_method1 import update_image2d_rss_method1


def compute_image2d_rss_from_detector_method1(
        image2d_detector_method0,
        naxis1_detector,
        naxis1_ifu,
        nslices,
        dict_ifu2detector,
        wv_crpix1,
        wv_crval1,
        wv_cdelt1,
        noparallel_computation=False,
        verbose=False
):
    """
    Compute the RSS image from the detector image using method 1.
    
    Parameters
    ----------
    image2d_detector_method0 : numpy.ndarray
        The input detector image.
    naxis1_detector : int
        Number of pixels in the first axis of the detector image
        (dispersion direction)
    naxis1_ifu : int
        Number of pixels in the first axis of the IFU image.
    nslices : int
        Number of slices in the IFU image.
    dict_ifu2detector : dict
        Dictionary mapping IFU slices to detector pixels.
    wv_crpix1 : float
        Reference pixel for the first axis of the wavelength image.
    wv_crval1 : float
        Reference value for the first axis of the wavelength image.
    wv_cdelt1 : float
        Increment for the first axis of the wavelength image.
    noparallel_computation : bool, optional
        If True, do not use parallel computation. Default is False.
    verbose : bool, optional
        If True, print verbose output. Default is False.

    Returns
    -------
    image2d_rss_method1 : numpy.ndarray
        The computed RSS image.
    """

    if verbose:
        print(ctext('\n* Computing image2d RSS (method 1)', fg='green'))

    # initialize image
    image2d_rss_method1 = np.zeros((naxis1_ifu.value * nslices, naxis1_detector.value))

    if verbose:
        print('Rectifying...')
    t0 = time.time()
    if noparallel_computation:
        # explicit loop in slices
        for islice in range(nslices):
            if verbose:
                print(f'{islice=}')
            update_image2d_rss_method1(
                islice=islice,
                image2d_detector_method0=image2d_detector_method0,
                dict_ifu2detector=dict_ifu2detector,
                naxis1_detector=naxis1_detector,
                naxis1_ifu=naxis1_ifu,
                wv_crpix1=wv_crpix1,
                wv_crval1=wv_crval1,
                wv_cdelt1=wv_cdelt1,
                image2d_rss_method1=image2d_rss_method1,
                debug=False
            )
    else:
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(update_image2d_rss_method1)(
                islice=islice,
                image2d_detector_method0=image2d_detector_method0,
                dict_ifu2detector=dict_ifu2detector,
                naxis1_detector=naxis1_detector,
                naxis1_ifu=naxis1_ifu,
                wv_crpix1=wv_crpix1,
                wv_crval1=wv_crval1,
                wv_cdelt1=wv_cdelt1,
                image2d_rss_method1=image2d_rss_method1,
                debug=False
            ) for islice in range(nslices))

    t1 = time.time()
    if verbose:
        print(f'Delta time: {t1 - t0}')

    return image2d_rss_method1
