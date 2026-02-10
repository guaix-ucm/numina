#
# Copyright 2024-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import logging

from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import os

from .raise_valueerror import raise_ValueError


def simulate_image2d_from_fitsfile(
        infile,
        diagonal_fov_arcsec,
        plate_scale_x,
        plate_scale_y,
        nphotons,
        rng,
        background_to_subtract=None,
        image_threshold=0.0,
        plots=False,
        logger=None
):
    """Simulate photons mimicking a 2D image from FITS file.

    Parameters
    ----------
    infile : str
        Input file containing the FITS image to be simulated.
    diagonal_fov_arcsec : `~astropy.units.Quantity`
        Desired field of View (arcsec) corresponding to the diagonal
        of the FITS image.
    plate_scale_x : `~astropy.units.Quantity`
        Plate scale of the IFU in the X direction.
    plate_scale_y : `~astropy.units.Quantity`
        Plate scale of the IFU in the Y direction.
    nphotons : int
        Number of photons to be simulated.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    background_to_subtract : str or None
        If not None, this parameters indicates how to compute
        the background to be subtracted.
    image_threshold : float
        Data below this threshold is set to zero.
    plots : bool
        If True, plot intermediate results.
    logger : logging.Logger or None
        Logger for logging messages. If None, a default logger is used.

    Returns
    -------
    simulated_x_ifu : `~numpy.ndarray`
        Array of simulated photon X coordinates in the IFU.
    simulated_y_ify : `~numpy.ndarray`
        Array of simulated photon Y coordinates in the IFU.

    """

    if logger is None:
        logger = logging.getLogger(__name__)

    # read input FITS file
    logger.debug(f'Reading {infile=}')
    with fits.open(infile) as hdul:
        image2d_ini = hdul[0].data
    image2d_reference = image2d_ini.astype(float)
    naxis2, naxis1 = image2d_reference.shape
    npixels = naxis1 * naxis2

    # subtract background
    if background_to_subtract == 'mode':
        nbins = int(np.sqrt(npixels) + 0.5)
        h, bin_edges = np.histogram(image2d_reference.flatten(), bins=nbins)
        imax = np.argmax(h)
        skylevel = (bin_edges[imax] + bin_edges[imax+1]) / 2
        logger.debug(f'Subtracting {skylevel=} (image mode)')
    elif background_to_subtract == 'median':
        skylevel = np.median(image2d_reference.flatten())
        logger.debug(f'Subtracting {skylevel=} (image median)')
    elif background_to_subtract == 'none':
        skylevel = 0.0
        logger.debug('Skipping background subtraction')
    else:
        try:
            skylevel = float(background_to_subtract)
        except ValueError:
            skylevel = None   # avoid PyCharm warning (not aware of raise ValueError)
            raise_ValueError(f'Invalid {background_to_subtract=}')
        logger.debug(f"Subtracting {skylevel=} (user's value)")
    image2d_reference -= skylevel

    # impose image threshold
    logger.debug(f'Applying {image_threshold=}')
    image2d_reference[image2d_reference <= image_threshold] = 0
    if np.min(image2d_reference) < 0.0:
        raise_ValueError(f'{np.min(image2d_reference)=} must be >= 0.0')

    # flatten image to be simulated
    image1d = image2d_reference.flatten()
    # compute normalized cumulative area
    xpixel = 1 + np.arange(npixels)
    cumulative_area = np.concatenate((
        [0],
        np.cumsum((image1d[:-1] + image1d[1:]) / 2 * (xpixel[1:] - xpixel[:-1]))
    ))
    normalized_cumulative_area = cumulative_area / cumulative_area[-1]
    if plots:
        fig, ax = plt.subplots()
        ax.plot(xpixel, normalized_cumulative_area, '.')
        ax.set_xlabel(f'xpixel')
        ax.set_ylabel('Normalized cumulative area')
        ax.set_title(os.path.basename(infile))
        plt.tight_layout()
        plt.show()
    # invert normalized cumulative area using random uniform samples
    unisamples = rng.uniform(low=0, high=1, size=nphotons)
    simulated_pixel = np.interp(x=unisamples, xp=normalized_cumulative_area, fp=xpixel)
    # compute histogram of 1D data
    bins_pixel = 0.5 + np.arange(npixels + 1)
    int_simulated_pixel, bin_edges = np.histogram(simulated_pixel, bins=bins_pixel)
    # reshape 1D into 2D image
    image2d_simulated = int_simulated_pixel.reshape((naxis2, naxis1))
    # scale factors to insert simulated image in requested field of view
    plate_scale = diagonal_fov_arcsec / (np.sqrt(naxis1**2 + naxis2**2) * u.pix)
    factor_x = abs((plate_scale / plate_scale_x.to(u.arcsec / u.pix)).value)
    factor_y = abs((plate_scale / plate_scale_y.to(u.arcsec / u.pix)).value)
    # redistribute photons in each pixel of the simulated image using a
    # random distribution within the considered pixel
    jcenter = naxis1 / 2
    icenter = naxis2 / 2
    simulated_x_ifu = []
    simulated_y_ifu = []
    for i, j in np.ndindex(naxis2, naxis1):
        nphotons_in_pixel = image2d_simulated[i, j]
        if nphotons_in_pixel > 0:
            jmin = j - jcenter - 0.5
            jmax = j - jcenter + 0.5
            simulated_x_ifu += (rng.uniform(low=jmin, high=jmax, size=nphotons_in_pixel) * factor_x).tolist()
            imin = i - icenter - 0.5
            imax = i - icenter + 0.5
            simulated_y_ifu += (rng.uniform(low=imin, high=imax, size=nphotons_in_pixel) * factor_y).tolist()

    # return result
    return np.array(simulated_x_ifu), np.array(simulated_y_ifu)
