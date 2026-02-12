#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Compute RSS image from detector image"""

from joblib import Parallel, delayed
import logging
import numpy as np
import time

from numina.instrument.simulation.ifu.define_3d_wcs import get_wvparam_from_wcs3d

from .update_image2d_rss_method1 import update_image2d_rss_method1


def compute_image2d_rss_from_detector_method1(
    image2d_detector_method0,
    naxis1_detector,
    naxis1_ifu,
    nslices,
    dict_ifu2detector,
    wcs3d,
    parallel_computation=False,
    logger=None,
):
    """
    Compute the RSS image from the detector image using method 1.

    Parameters
    ----------
    image2d_detector_method0 : numpy.ndarray
        The input detector image.
    naxis1_detector : `~astropy.units.Quantity`
        Number of pixels in the first axis of the detector image
        (dispersion direction)
    naxis1_ifu : `~astropy.units.Quantity`
        Number of pixels in the first axis of the IFU image.
    nslices : int
        Number of slices in the IFU image.
    dict_ifu2detector : dict
        Dictionary mapping IFU slices to detector pixels.
    wcs3d : `~astropy.wcs.WCS`
        WCS object for the 3D image.
    parallel_computation : bool, optional
        If True, use parallel computation. Default is False.
    verbose : bool, optional
        If True, print verbose output. Default is False.
    logger : `~logging.Logger`, optional
        Logger for logging messages. If None, the root logger is used.

    Returns
    -------
    image2d_rss_method1 : numpy.ndarray
        The computed RSS image.
    """

    if logger is None:
        logger = logging.getLogger()

    logger.debug("[green]* Computing image2d RSS (method 1)[/green]")

    # get WCS parameters
    wv_cunit1, wv_crpix1, wv_crval1, wv_cdelt1 = get_wvparam_from_wcs3d(wcs3d)

    # initialize image
    image2d_rss_method1 = np.zeros((naxis1_ifu.value * nslices, naxis1_detector.value))

    logger.debug("Rectifying...")
    t0 = time.time()
    if not parallel_computation:
        # explicit loop in slices
        for islice in range(nslices):
            logger.debug(f"{islice=}")
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
                debug=False,
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
                debug=False,
            )
            for islice in range(nslices)
        )

    t1 = time.time()
    logger.debug(f"Delta time: {t1 - t0}")

    return image2d_rss_method1
