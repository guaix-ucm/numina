#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.units import Unit
import astropy.units as u
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
import numpy as np

from numina.array.distortion import fmap
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.distortion import compute_distortion, rectify2d


def update_image2d_rss_method1(
    islice,
    image2d_detector_method0,
    dict_ifu2detector,
    naxis1_detector,
    naxis1_ifu,
    wv_crpix1,
    wv_crval1,
    wv_cdelt1,
    image2d_rss_method1,
    debug=False,
):
    """Update the RSS image from the detector image.

    The function updates the following 2D array:
    - image2d_rss_method1,
    with the data of the slice 'islice' in the detector image.

    This function can be executed in parallel.

    Parameters
    ----------
    islice : int
        Slice number.
    image2d_detector_method0 : `~numpy.ndarray`
        2D array containing the detector image.
    dict_ifu2detector : dict
        A Python dictionary containing the 2D polynomials that allow
        to transform (X, Y, wavelength) coordinates in the IFU focal
        plane to (X, Y) coordinates in the detector.
    naxis1_detector : `~astropy.units.Quantity`
        Detector NAXIS1 (along the spectral direction).
    naxis1_ifu : `~astropy.units.Quantity`
        IFU NAXIS1 (along the slice).
    wv_crpix1 : `~astropy.units.Quantity`
        CRPIX1 value along the spectral direction.
    wv_crval1 : `~astropy.units.Quantity`
        CRVAL1 value along the spectral direction.
    wv_cdelt1 : `~astropy.units.Quantity`
        CDELT1 value along the spectral direction.
    image2d_rss_method1 : `~numpy.ndarray`
        2D array containing the RSS image. This array is
        updated by this function.
    debug : bool
        If True, show debugging information/plots.

    """

    slice_id = islice + 1

    # minimum and maximum pixel X coordinate defining the IFU focal plane
    min_x_ifu = 0.5 * u.pix
    max_x_ifu = naxis1_ifu + 0.5 * u.pix

    # determine upper and lower frontiers of each slice in the detector
    x_ifu_lower = np.repeat([min_x_ifu.value], naxis1_detector.value)
    x_ifu_upper = np.repeat([max_x_ifu.value], naxis1_detector.value)

    # wavelength values at each pixel in the spectral direction of the detector
    wavelength = wv_crval1 + ((np.arange(naxis1_detector.value) + 1) * u.pix - wv_crpix1) * wv_cdelt1

    # minimum and maximum wavelengths to be considered
    wmin = wv_crval1 + (0.5 * u.pix - wv_crpix1) * wv_cdelt1
    wmax = wmin + naxis1_detector * wv_cdelt1

    # use model to predict location in detector
    # important: reverse here X <-> Y
    wavelength_unit = Unit(dict_ifu2detector["wavelength-unit"])
    dumdict = dict_ifu2detector["contents"][islice]
    order = dumdict["order"]
    aij = np.array(dumdict["aij"])
    bij = np.array(dumdict["bij"])

    y_detector_lower_index, x_detector_lower_index = fmap(
        order=order, aij=aij, bij=bij, x=x_ifu_lower, y=wavelength.to(wavelength_unit).value
    )
    # subtract 1 to work with array indices
    x_detector_lower_index -= 1
    y_detector_lower_index -= 1

    y_detector_upper_index, x_detector_upper_index = fmap(
        order=order, aij=aij, bij=bij, x=x_ifu_upper, y=wavelength.to(wavelength_unit).value
    )
    # subtract 1 to work with array indices
    x_detector_upper_index -= 1
    y_detector_upper_index -= 1

    if debug:
        debugplot = 1
    else:
        debugplot = 0
    poly_lower_index, residuals = polfit_residuals(
        x=x_detector_lower_index,
        y=y_detector_lower_index,
        deg=order,
        xlabel="x_detector_lower_index",
        ylabel="y_detector_lower_index",
        title=f"slice_id #{slice_id}",
        debugplot=debugplot,
    )
    if debug:
        plt.tight_layout()
        plt.show()
    poly_upper_index, residuals = polfit_residuals(
        x=x_detector_upper_index,
        y=y_detector_upper_index,
        deg=order,
        xlabel="x_detector_upper_index",
        ylabel="y_detector_upper_index",
        title=f"slice_id #{slice_id}",
        debugplot=debugplot,
    )
    if debug:
        plt.tight_layout()
        plt.show()

    # full image containing only the slice data (zero elsewhere)
    image2d_detector_slice = np.zeros_like(image2d_detector_method0)
    xdum = np.arange(naxis1_detector.value)
    ypoly_lower_index = (poly_lower_index(xdum) + 0.5).astype(int)
    ypoly_upper_index = (poly_upper_index(xdum) + 0.5).astype(int)
    for j in range(naxis1_detector.value):
        i1 = ypoly_lower_index[j]
        i2 = ypoly_upper_index[j]
        image2d_detector_slice[i1 : (i2 + 1), j] = image2d_detector_method0[i1 : (i2 + 1), j]

    if debug:
        xmin = np.min(np.concatenate((x_detector_lower_index, x_detector_upper_index)))
        xmax = np.max(np.concatenate((x_detector_lower_index, x_detector_upper_index)))
        ymin = np.min(np.concatenate((y_detector_lower_index, y_detector_upper_index)))
        ymax = np.max(np.concatenate((y_detector_lower_index, y_detector_upper_index)))
        print(f"{xmin=}, {xmax=}, {ymin=}, {ymax=}")
        dy = ymax - ymin
        yminplot = ymin - dy / 5
        ymaxplot = ymax + dy / 5
        fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(6.4 * 2, 4.8 * 2))
        vmin, vmax = ZScaleInterval().get_limits(image2d_detector_method0)
        for iplot in range(2):
            ax = axarr[iplot]
            if iplot == 0:
                ax.imshow(image2d_detector_method0, vmin=vmin, vmax=vmax, aspect="auto")
            else:
                ax.imshow(image2d_detector_slice, vmin=vmin, vmax=vmax, aspect="auto")
            ax.plot(x_detector_lower_index, y_detector_lower_index, "C1--")
            ax.plot(x_detector_upper_index, y_detector_upper_index, "C1--")
            ax.plot(xdum, ypoly_lower_index, "w:")
            ax.plot(xdum, ypoly_upper_index, "w:")
            ax.set_ylim(yminplot, ymaxplot)
        plt.tight_layout()
        plt.show()

    # mathematical transformation for the considered slice
    i1_rss_index = islice * naxis1_ifu.value
    i2_rss_index = i1_rss_index + naxis1_ifu.value
    if debug:
        print(f"{i1_rss_index=}, {i2_rss_index=}, {i2_rss_index - i1_rss_index=}")
    # generate a grid to compute the 2D transformation
    nx_grid_rss = 20
    ny_grid_rss = 20
    # points along the slice in the IFU focal pline
    x_ifu_grid = np.tile(np.linspace(1, naxis1_ifu.value, num=ny_grid_rss), nx_grid_rss)
    wavelength_grid = np.repeat(np.linspace(wmin, wmax, num=nx_grid_rss), ny_grid_rss)
    # pixel in the RSS image
    wavelength_grid_pixel_rss = (wavelength_grid - wv_crval1) / wv_cdelt1 + wv_crpix1
    if debug:
        fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8))
        ax.plot(wavelength_grid_pixel_rss.value, x_ifu_grid + i1_rss_index, "r.")
        ax.set_xlabel("RSS pixel in wavelength direction")
        ax.set_ylabel("RSS pixel in spatial direction")
        plt.tight_layout()
        plt.show()
    # project the previous points in the detector
    y_hawaii_grid_index, x_hawaii_grid_index = fmap(
        order=order,
        aij=aij,
        bij=bij,
        x=x_ifu_grid,
        y=wavelength_grid.to(wavelength_unit).value,  # ignore PyCharm warning here
    )
    x_hawaii_grid_index -= 1
    y_hawaii_grid_index -= 1

    y_hawaii_grid_index, x_hawaii_grid_index = fmap(
        order=order,
        aij=aij,
        bij=bij,
        x=x_ifu_grid,
        y=wavelength_grid.to(wavelength_unit).value,  # ignore PyCharm warning here
    )
    x_hawaii_grid_index -= 1
    y_hawaii_grid_index -= 1
    if debug:
        fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8))
        ax.imshow(image2d_detector_slice, vmin=vmin, vmax=vmax, aspect="auto")  # ignore PyCharm warning here
        ax.plot(x_hawaii_grid_index, y_hawaii_grid_index, "r.")
        ax.set_ylim(yminplot, ymaxplot)  # ignore PyCharm warning here
        plt.tight_layout()
        plt.show()
    # compute distortion transformation
    aij_resample, bij_resample = compute_distortion(
        x_orig=x_hawaii_grid_index + 1,
        y_orig=y_hawaii_grid_index + 1,
        x_rect=wavelength_grid_pixel_rss.value,
        y_rect=x_ifu_grid,
        order=order,
        debugplot=0,
    )

    # rectify image
    image2d_slice_rss = rectify2d(
        image2d=image2d_detector_slice,
        aij=aij_resample,
        bij=bij_resample,
        resampling=2,  # 2: flux preserving interpolation
        naxis2out=naxis1_ifu.value,
    )
    if debug:
        fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8))
        ax.imshow(image2d_slice_rss, vmin=vmin, vmax=vmax, aspect="auto")
        plt.tight_layout()
        plt.show()

    # insert result in final image
    image2d_rss_method1[i1_rss_index:i2_rss_index, :] = image2d_slice_rss[:, :]
