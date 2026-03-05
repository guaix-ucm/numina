#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Compute RSS image from detector image.

See notes in ifu_simulator.py about parallelisation. 
In this function, the same approach is used as in ifu_simulator.py:
"""

import logging
from multiprocessing import Pool
import numpy as np
import time

from numina.instrument.simulation.ifu.define_3d_wcs import get_wvparam_from_wcs3d

from .update_image2d_rss_method1 import update_image2d_rss_method1


_worker_inputs_method1 = {}

def _worker_init_method1(image2d_detector_method0, 
                         dict_ifu2detector,
                         naxis1_detector, 
                         naxis1_ifu, 
                         nslices,
                         wv_crpix1, 
                         wv_crval1, 
                         wv_cdelt1):
    """Initialiser executed once per worker process at Pool startup.

    Stores all large input arrays in the module-level global so that
    individual tasks only need to receive the slice index.
    """
    global _worker_inputs_method1
    _worker_inputs_method1 = {
        "image2d_detector_method0": image2d_detector_method0,
        "dict_ifu2detector": dict_ifu2detector,
        "naxis1_detector": naxis1_detector,
        "naxis1_ifu": naxis1_ifu,
        "nslices": nslices,
        "wv_crpix1": wv_crpix1,
        "wv_crval1": wv_crval1,
        "wv_cdelt1": wv_cdelt1,
    }


def _worker_method1(islice):
    """Worker function for parallel computation of a single slice.
    
    All large input arrays are accessed from the global _worker_inputs_method1. 
    Allocates private output arrays, calls the update function, 
    and returns results."""
    local_rss = np.zeros(
        (_worker_inputs_method1['naxis1_ifu'].value * _worker_inputs_method1['nslices'], 
         _worker_inputs_method1['naxis1_detector'].value)
    )
    update_image2d_rss_method1(
        islice=islice,
        image2d_detector_method0=_worker_inputs_method1['image2d_detector_method0'],
        dict_ifu2detector=_worker_inputs_method1['dict_ifu2detector'],
        naxis1_detector=_worker_inputs_method1['naxis1_detector'],
        naxis1_ifu=_worker_inputs_method1['naxis1_ifu'],
        wv_crpix1=_worker_inputs_method1['wv_crpix1'],
        wv_crval1=_worker_inputs_method1['wv_crval1'],
        wv_cdelt1=_worker_inputs_method1['wv_cdelt1'],
        image2d_rss_method1=local_rss,
        debug=False,
    )
    return local_rss


def compute_image2d_rss_from_detector_method1(
    image2d_detector_method0,
    naxis1_detector,
    naxis1_ifu,
    nslices,
    dict_ifu2detector,
    wcs3d,
    ncores=1,
    logger=None,
    console=None,
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
    ncores : int, optional
        Number of CPU cores to be used for parallel processing. If ncores=1,
        no parallel processing is used. If ncores > 1, the computation is
        parallelised using multiprocessing.Pool. Default is 1 (no parallelisation).
    verbose : bool, optional
        If True, print verbose output. Default is False.
    logger : `~logging.Logger`, optional
        Logger for logging messages. If None, the root logger is used.
    console : `~rich.console.Console` or None, optional
        Rich console for rich printing. If None, the default console will be used.

    Returns
    -------
    image2d_rss_method1 : numpy.ndarray
        The computed RSS image.
    """

    if logger is None:
        logger = logging.getLogger()
    logger_level_in_use = logger.getEffectiveLevel()

    logger.debug("[green]* Computing image2d RSS (method 1)[/green]")

    # Get WCS parameters
    wv_cunit1, wv_crpix1, wv_crval1, wv_cdelt1 = get_wvparam_from_wcs3d(wcs3d)

    # Initialize image
    image2d_rss_method1 = np.zeros((naxis1_ifu.value * nslices, naxis1_detector.value))

    logger.debug("Rectifying...")
    t0 = time.time()

    if ncores == 1:
        # Explicit loop in slices
        for islice in range(nslices):
            if logger_level_in_use <= logging.DEBUG:
                logger.debug(f"slice: {islice + 1}/{nslices}")
            else:
                if (islice + 1) % 10 == 0:
                    console.print(f"{islice + 1}", end="")
                else:
                    console.print(".", end="")
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
        if logger_level_in_use > logging.DEBUG:
            console.print("")  # new line after the progress dots
    else:
        # Large input arrays are loaded into each worker once via the initializer,
        # so only the slice index is passed per task, minimising serialisation overhead.
        with Pool(
            processes=ncores,
            initializer=_worker_init_method1,
            initargs=(
                image2d_detector_method0,
                dict_ifu2detector,
                naxis1_detector,
                naxis1_ifu,
                nslices,
                wv_crpix1,
                wv_crval1,
                wv_cdelt1,
            ),
        ) as pool:
            results = pool.map(_worker_method1, range(nslices))
        # Final reduction (single-threaded, always safe)
        for local_rss in results:
            image2d_rss_method1 += local_rss

    t1 = time.time()
    logger.info(f"Delta time: {t1 - t0}")

    return image2d_rss_method1
