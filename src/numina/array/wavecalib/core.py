#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


"""Wavelength calibration of a spectrum

Functions to collapse the spectra of a 2D image, read the list of
wavelengths of the arc lines, locate the line peaks and compute the
wavelength calibration of a spectrum. The script numina-wavecalib
(``numina.array.wavecalib.__main__``) uses them.
"""

import logging

from astropy.io import fits
from astropy.table import Table
import numpy as np
from scipy import ndimage

from .arccalibration import arccalibration
from .arccalibration import fit_list_of_wvfeatures
from ..display.pause_debugplot import pause_debugplot
from ..display.ximplotxy import ximplotxy
from ..stats import robust_std
from .peaks_spectrum import find_peaks_spectrum
from .peaks_spectrum import refine_peaks_spectrum


def collapsed_spectrum(fitsfile, ns1, ns2, method="mean", nwin_background=0, reverse=False, out_sp=None):
    """Compute a collapsed spectrum from a 2D image using scans in [ns1,ns2].

    Parameters
    ----------
    fitsfile : file object
        File name of FITS file containing the spectra to be calibrated.
    ns1 : int
        First scan (from 1 to NAXIS2).
    ns2 : int
        Last scan (from 1 to NAXIS2).
    method : string
        Indicates collapsing method. Possible values are "mean" or
        "median".
    nwin_background : int
        Window size for the computation of background using a median
        filtering with that window width. This background is computed
        and subtracted only if this parameter is > 0.
    reverse : bool
        If True, reserve wavelength direction prior to wavelength
        calibration.
    out_sp : string or None
        File name to save the selected spectrum in FITS format before
        performing the wavelength calibration.

    Returns
    -------
    sp : 1d numpy array
        Collapsed spectrum.

    """
    logger = logging.getLogger(__name__)

    # read FITS file
    with fits.open(fitsfile) as hdulist:
        image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape
    logger.info(f"Reading file: '{fitsfile}'")
    logger.info(f"NAXIS1: {naxis1}")
    logger.info(f"NAXIS2: {naxis2}")

    if 1 <= ns1 <= ns2 <= naxis2:
        # extract spectrum using the requested scans and collapsing along the scan direction
        logger.info(f"Collapsing scans from {ns1} to {ns2} using method '{method}'")
        if method == "mean":
            sp = np.mean(image2d[(ns1 - 1) : ns2], axis=0)
        elif method == "median":
            sp = np.median(image2d[(ns1 - 1) : ns2], axis=0)
        else:
            raise ValueError("Invalid method '" + str(method) + "'")

        # reverse spectrum if necessary
        if reverse:
            sp = sp[::-1]

        # fit and subtract background
        if nwin_background > 0:
            background = ndimage.median_filter(sp, size=nwin_background)
            sp -= background

        # save spectrum before wavelength calibration in external
        # FITS file
        if out_sp is not None:
            hdu = fits.PrimaryHDU(sp)
            hdu.writeto(out_sp, overwrite=True)
    else:
        logger.error("ERROR: Invalid ns1=" + str(ns1) + ", ns2=" + str(ns2) + " values")
        raise SystemExit()

    return sp


def read_wv_master_from_array(master_table, lines="brightest"):
    """read arc line wavelengths from numpy array

    Parameters
    ----------
    master_table : Numpy array
        Numpy array containing the wavelength database.
    lines : string
        Indicates which lines to read. For files with a single column
        or two columns this parameter is irrelevant. For files with
        three columns, lines='brightest' indicates that only the
        brightest lines are read, whereas lines='all' means that all
        the lines are considered.

    Returns
    -------
    wv_master : 1d numpy array
        Array with arc line wavelengths.

    """
    # protection
    if lines not in ["brightest", "all"]:
        raise ValueError("Unexpected lines=" + str(lines))

    # determine wavelengths according to the number of columns
    if master_table.ndim == 1:
        wv_master = master_table
    else:
        wv_master_all = master_table[:, 0]
        if master_table.shape[1] == 2:  # assume old format
            wv_master = np.copy(wv_master_all)
        elif master_table.shape[1] == 3:  # assume new format
            if lines == "brightest":
                wv_flag = master_table[:, 1]
                wv_master = wv_master_all[np.where(wv_flag == 1)]
            else:
                wv_master = np.copy(wv_master_all)
        else:
            raise ValueError("Lines_catalog file does not have the expected number of columns")

    return wv_master


def read_wv_master_file(wv_master_file, lines="brightest"):
    """read arc line wavelengths from external file.

    Parameters
    ----------
    wv_master_file : string
        File name of txt file containing the wavelength database.
    lines : string
        Indicates which lines to read. For files with a single column
        or two columns this parameter is irrelevant. For files with
        three columns, lines='brightest' indicates that only the
        brightest lines are read, whereas lines='all' means that all
        the lines are considered.

    Returns
    -------
    wv_master : 1d numpy array
        Array with arc line wavelengths.

    """
    logger = logging.getLogger(__name__)

    # protection
    if lines not in ["brightest", "all"]:
        raise ValueError("Unexpected lines=" + str(lines))

    # read table from txt file
    master_table = np.genfromtxt(wv_master_file)

    wv_master = read_wv_master_from_array(master_table, lines)

    logger.debug(f"Reading master table: {wv_master_file}")
    logger.debug(f"wv_master:\n{wv_master}")

    return wv_master


def find_fxpeaks(
    sp,
    times_sigma_threshold,
    minimum_threshold,
    nwinwidth_initial,
    nwinwidth_refined,
    npix_avoid_border,
    nbrightlines,
    sigma_gaussian_filtering,
    minimum_gaussian_filtering,
    plottitle=None,
    geometry=None,
    extra_plots=False,
):
    """Locate line peaks in array coordinates (from 0 to naxis1-1).

    Parameters
    ----------
    sp : 1d numpy array
        Spectrum to be wavelength calibrated.
    times_sigma_threshold : float
        Times robust sigma above the median to detect line peaks.
    minimum_threshold : float or None
            Minimum value of the threshold.
    nwinwidth_initial : int
        Width of the window where each peak must be found using
        the initial method (approximate)
    nwinwidth_refined : int
        Width of the window where each peak location will be
        refined.
    npix_avoid_border : int
            Number of pixels at the borders of the spectrum where peaks
            are not considered. If zero, the actual number will be
            given by nwinwidth_initial.
    nbrightlines : int or list of integers
        Maximum number of brightest lines to be employed in the
        wavelength calibration. If this value is 0, all the detected
        lines will be employed.
    sigma_gaussian_filtering : float
        Sigma of the gaussian filter to be applied to the spectrum in
        order to avoid problems with saturated lines. This filtering is
        skipped when this parameter is <= 0.
    minimum_gaussian_filtering : float
        Minimum pixel value to employ gaussian filtering. This value is
        employed only when sigma_gaussian_filtering is > 0.
    plottile : string
        Plot title.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the Qt backend geometry.
    extra_plots : bool
        If True, display the spectrum and the peaks found, and wait
        until the window is closed.

    Returns
    -------
    fxpeaks : 1d numpy array
        Refined location of peaks in array index scale, i.e, from 0
        to naxis1 - 1.
    sxpeaks : 1d numpy array
        Refined line peak widths.

    """
    logger = logging.getLogger(__name__)

    # spectrum dimension
    naxis1 = sp.shape[0]

    # apply gaussian filtering when requested
    if sigma_gaussian_filtering > 0:
        spf = ndimage.gaussian_filter(sp, sigma=sigma_gaussian_filtering)
        lpreserve = sp < minimum_gaussian_filtering
        spf[lpreserve] = sp[lpreserve]
    else:
        spf = np.copy(sp)

    # threshold to search for peaks
    q50 = np.percentile(spf, q=50)
    sigma_g = robust_std(spf)
    threshold = q50 + times_sigma_threshold * sigma_g
    logger.debug(f"median....: {q50}")
    logger.debug(f"robust std: {sigma_g}")
    logger.debug(f"threshold.: {threshold}")
    if minimum_threshold > threshold:
        threshold = minimum_threshold
    logger.debug(f"minimum threshold: {minimum_threshold}")
    logger.debug(f"final threshold..: {threshold}")

    # initial location of the peaks (integer values)
    ixpeaks = find_peaks_spectrum(spf, nwinwidth=nwinwidth_initial, threshold=threshold)

    # select a maximum number of brightest lines in each region
    if len(nbrightlines) == 1 and nbrightlines[0] == 0:
        pass
    else:
        logger.debug(f"nbrightlines = {nbrightlines}")
        logger.debug(f"ixpeaks in whole spectrum:\n{ixpeaks}")
        region_size = (naxis1 - 1) / len(nbrightlines)
        ixpeaks_filtered = np.array([], dtype=int)
        for iregion, nlines_in_region in enumerate(nbrightlines):
            if nlines_in_region > 0:
                imin = int(iregion * region_size)
                imax = int((iregion + 1) * region_size)
                if iregion > 0:
                    imin += 1
                ixpeaks_region = ixpeaks[np.logical_and(ixpeaks >= imin, ixpeaks <= imax)]
                if len(ixpeaks_region) > 0:
                    peak_fluxes = spf[ixpeaks_region]
                    spos = peak_fluxes.argsort()
                    ixpeaks_tmp = ixpeaks_region[spos[-nlines_in_region:]]
                    ixpeaks_tmp.sort()  # in-place sort
                    logger.debug(f"ixpeaks in region........:\n{ixpeaks_tmp}")
                    ixpeaks_filtered = np.concatenate((ixpeaks_filtered, ixpeaks_tmp))
        ixpeaks = ixpeaks_filtered
        logger.debug(f"ixpeaks filtered.........:\n{ixpeaks}")

    # remove peaks too close to any of the borders of the spectrum
    if npix_avoid_border > 0:
        lok_ini = ixpeaks >= npix_avoid_border
        lok_end = ixpeaks <= naxis1 - 1 - npix_avoid_border
        ixpeaks = ixpeaks[lok_ini * lok_end]

    # refined location of the peaks (float values)
    fxpeaks, sxpeaks = refine_peaks_spectrum(spf, ixpeaks, nwinwidth=nwinwidth_refined, method="gaussian")

    # print peak location and width of fitted lines
    logger.info(f"Number of lines found: {len(fxpeaks)}")
    lines_found_table = Table(names=["line_number", "channel", "width"], dtype=[int, float, float])
    lines_found_table["channel"].info.format = ".3f"
    lines_found_table["width"].info.format = ".3f"
    for i, (fx, sx) in enumerate(zip(fxpeaks, sxpeaks)):
        lines_found_table.add_row([i, fx + 1, sx])
    logger.info(lines_found_table)

    # display median spectrum and peaks if requested
    if extra_plots:
        xplot = np.arange(1, naxis1 + 1, dtype=float)
        ax = ximplotxy(xplot, sp, show=False, geometry=geometry, **{"label": "original spectrum"})
        ax.set_xlabel("pixel (from 1 to NAXIS1)")
        ax.set_ylabel("counts")
        if plottitle is not None:
            ax.set_title(plottitle)
        if sigma_gaussian_filtering > 0:
            ax.plot(xplot, spf, label="filtered spectrum")
        ymin = sp.min()
        ymax = sp.max()
        dy = ymax - ymin
        ymin -= dy / 20.0
        ymax += dy / 20.0
        ax.set_ylim([ymin, ymax])
        # display threshold
        ax.axhline(y=threshold, color="black", linestyle="dotted", label="detection threshold")
        # mark peak location
        ax.plot(ixpeaks + 1, spf[ixpeaks], "bo", label="initial location")
        ax.plot(fxpeaks + 1, spf[ixpeaks], "go", label="refined location")
        # legend
        ax.legend(title=f"{len(fxpeaks)} lines found", numpoints=1)
        # show plot
        pause_debugplot(debugplot=2, pltshow=True)

    return fxpeaks, sxpeaks


def wvcal_spectrum(
    sp,
    fxpeaks,
    poly_degree_wfit,
    wv_master,
    wv_ini_search=None,
    wv_end_search=None,
    wvmin_useful=None,
    wvmax_useful=None,
    geometry=None,
    debugplot=0,
):
    """Execute wavelength calibration of a spectrum using fixed line peaks.

    Parameters
    ----------
    sp : 1d numpy array
        Spectrum to be wavelength calibrated.
    fxpeaks : 1d numpy array
        Refined location of peaks in array index scale, i.e, from 0
        to naxis1 - 1. The wavelength calibration is performed using
        these line locations.
    poly_degree_wfit : int
        Degree for wavelength calibration polynomial.
    wv_master : 1d numpy array
        Array with arc line wavelengths.
    wv_ini_search : float or None
        Minimum expected wavelength in spectrum.
    wv_end_search : float or None
        Maximum expected wavelength in spectrum.
    wvmin_useful : float or None
        If not None, this value is used to clip detected lines below it.
    wvmax_useful : float or None
        If not None, this value is used to clip detected lines above it.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the Qt backend geometry.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------
    solution_wv : instance of SolutionArcCalibration
        Wavelength calibration solution.

    """
    logger = logging.getLogger(__name__)

    # check there are enough lines for fit
    if len(fxpeaks) <= poly_degree_wfit:
        logger.warning(
            "not enough lines to fit the spectrum, %d lines for a polynomial of degree %d",
            len(fxpeaks),
            poly_degree_wfit,
        )
        return None

    # spectrum dimension
    naxis1 = sp.shape[0]

    wv_master_range = wv_master[-1] - wv_master[0]
    delta_wv_master_range = 0.20 * wv_master_range
    if wv_ini_search is None:
        wv_ini_search = wv_master[0] - delta_wv_master_range
    if wv_end_search is None:
        wv_end_search = wv_master[-1] + delta_wv_master_range

    # use channels (pixels from 1 to naxis1)
    xchannel = fxpeaks + 1.0

    # wavelength calibration
    list_of_wvfeatures = arccalibration(
        wv_master=wv_master,
        xpos_arc=xchannel,
        naxis1_arc=naxis1,
        crpix1=1.0,
        wv_ini_search=wv_ini_search,
        wv_end_search=wv_end_search,
        wvmin_useful=wvmin_useful,
        wvmax_useful=wvmax_useful,
        error_xpos_arc=3,
        times_sigma_r=3.0,
        frac_triplets_for_sum=0.50,
        times_sigma_theil_sen=10.0,
        poly_degree_wfit=poly_degree_wfit,
        times_sigma_polfilt=10.0,
        times_sigma_cook=10.0,
        times_sigma_inclusion=10.0,
        geometry=geometry,
        debugplot=debugplot,
    )

    title = "Wavelength calibration"
    solution_wv = fit_list_of_wvfeatures(
        list_of_wvfeatures=list_of_wvfeatures,
        naxis1_arc=naxis1,
        crpix1=1.0,
        poly_degree_wfit=poly_degree_wfit,
        weighted=False,
        plot_title=title,
        geometry=geometry,
        debugplot=debugplot,
    )

    if abs(debugplot) % 10 != 0:
        # final plot with identified lines
        xplot = np.arange(1, naxis1 + 1, dtype=float)
        ax = ximplotxy(
            xplot,
            sp,
            title=title,
            show=False,
            xlabel="pixel (from 1 to NAXIS1)",
            ylabel="number of counts",
            geometry=geometry,
        )
        ymin = sp.min()
        ymax = sp.max()
        dy = ymax - ymin
        ymin -= dy / 20.0
        ymax += dy / 20.0
        ax.set_ylim([ymin, ymax])
        # plot wavelength of each identified line
        for feature in solution_wv.features:
            xpos = feature.xpos
            reference = feature.reference
            ax.text(xpos, sp[int(xpos + 0.5) - 1], str(reference), fontsize=8, horizontalalignment="center")
        # show plot
        logger.info("Plot with identified lines")
        pause_debugplot(12, pltshow=True)

    # return the wavelength calibration solution
    return solution_wv
