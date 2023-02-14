#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Compute the wavelength calibration of a particular spectrum."""

import argparse
from astropy.io import fits
import numpy as np
import os
from scipy import ndimage
import sys

from .arccalibration import arccalibration
from .arccalibration import fit_list_of_wvfeatures
from .arccalibration import refine_arccalibration
from ..display.pause_debugplot import pause_debugplot
from ..display.ximplotxy import ximplotxy
from ..stats import robust_std
from ...tools.arg_file_is_new import arg_file_is_new
from .peaks_spectrum import find_peaks_spectrum
from .peaks_spectrum import refine_peaks_spectrum

from ..display.pause_debugplot import DEBUGPLOT_CODES


def collapsed_spectrum(fitsfile, ns1, ns2,
                       method='mean', nwin_background=0,
                       reverse=False, out_sp=None, debugplot=0):
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
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------
    sp : 1d numpy array
        Collapsed spectrum.

    """

    # read FITS file
    with fits.open(fitsfile) as hdulist:
        image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape
    if abs(debugplot) >= 10:
        print('>>> Reading file:', fitsfile.name)
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)

    if 1 <= ns1 <= ns2 <= naxis2:
        # extract spectrum
        if method == "mean":
            sp = np.mean(image2d[(ns1 - 1):ns2], axis=0)
        elif method == "median":
            sp = np.median(image2d[(ns1 - 1):ns2], axis=0)
        else:
            raise ValueError("Invalid method '" + str(method) + "'")

        # reverse spectrum if necessary
        if reverse:
            sp = sp[::-1]

        # fit and subtract background
        if nwin_background > 0:
            background = ndimage.filters.median_filter(
                sp, size=nwin_background
            )
            sp -= background

        # save spectrum before wavelength calibration in external
        # FITS file
        if out_sp is not None:
            hdu = fits.PrimaryHDU(sp)
            hdu.writeto(out_sp, overwrite=True)
    else:
        raise ValueError("Invalid ns1=" + str(ns1) + ", ns2=" + str(ns2) +
                         " values")

    return sp


def read_wv_master_from_array(master_table, lines='brightest', debugplot=0):
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
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------
    wv_master : 1d numpy array
        Array with arc line wavelengths.

    """

    # protection
    if lines not in ['brightest', 'all']:
        raise ValueError('Unexpected lines=' + str(lines))

    # determine wavelengths according to the number of columns
    if master_table.ndim == 1:
        wv_master = master_table
    else:
        wv_master_all = master_table[:, 0]
        if master_table.shape[1] == 2:  # assume old format
            wv_master = np.copy(wv_master_all)
        elif master_table.shape[1] == 3:  # assume new format
            if lines == 'brightest':
                wv_flag = master_table[:, 1]
                wv_master = wv_master_all[np.where(wv_flag == 1)]
            else:
                wv_master = np.copy(wv_master_all)
        else:
            raise ValueError('Lines_catalog file does not have the '
                             'expected number of columns')

    if abs(debugplot) >= 10:
        print("Reading master table from numpy array")
        print("wv_master:\n", wv_master)

    return wv_master


def read_wv_master_file(wv_master_file, lines='brightest', debugplot=0):
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
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------
    wv_master : 1d numpy array
        Array with arc line wavelengths.

    """

    # protection
    if lines not in ['brightest', 'all']:
        raise ValueError('Unexpected lines=' + str(lines))

    # read table from txt file
    master_table = np.genfromtxt(wv_master_file)

    wv_master = read_wv_master_from_array(master_table, lines)

    if abs(debugplot) >= 10:
        print("Reading master table: " + wv_master_file)
        print("wv_master:\n", wv_master)

    return wv_master


def find_fxpeaks(sp,
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
                 debugplot=0):
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
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------
    fxpeaks : 1d numpy array
        Refined location of peaks in array index scale, i.e, from 0
        to naxis1 - 1.
    sxpeaks : 1d numpy array
        Line peak widths.

    """

    # spectrum dimension
    naxis1 = sp.shape[0]

    # apply gaussian filtering when requested
    if sigma_gaussian_filtering > 0:
        spf = ndimage.filters.gaussian_filter(
            sp,
            sigma=sigma_gaussian_filtering
        )
        lpreserve = sp < minimum_gaussian_filtering
        spf[lpreserve] = sp[lpreserve]
    else:
        spf = np.copy(sp)

    # threshold to search for peaks
    q50 = np.percentile(spf, q=50)
    sigma_g = robust_std(spf)
    threshold = q50 + times_sigma_threshold * sigma_g
    if abs(debugplot) >= 10:
        print("median....:", q50)
        print("robuts std:", sigma_g)
        print("threshold.:", threshold)
    if minimum_threshold > threshold:
        threshold = minimum_threshold
    if abs(debugplot) >= 10:
        print("minimum threshold:", minimum_threshold)
        print("final threshold..:", threshold)

    # initial location of the peaks (integer values)
    ixpeaks = find_peaks_spectrum(spf,
                                  nwinwidth=nwinwidth_initial,
                                  threshold=threshold)

    # select a maximum number of brightest lines in each region
    if len(nbrightlines) == 1 and nbrightlines[0] == 0:
        pass
    else:
        if abs(debugplot) >= 10:
            print('nbrightlines =', nbrightlines)
            print('ixpeaks in whole spectrum:\n', ixpeaks)
        region_size = (naxis1-1)/len(nbrightlines)
        ixpeaks_filtered = np.array([], dtype=int)
        for iregion, nlines_in_region in enumerate(nbrightlines):
            if nlines_in_region > 0:
                imin = int(iregion * region_size)
                imax = int((iregion + 1) * region_size)
                if iregion > 0:
                    imin += 1
                ixpeaks_region = \
                    ixpeaks[np.logical_and(ixpeaks >= imin, ixpeaks <= imax)]
                if len(ixpeaks_region) > 0:
                    peak_fluxes = spf[ixpeaks_region]
                    spos = peak_fluxes.argsort()
                    ixpeaks_tmp = ixpeaks_region[spos[-nlines_in_region:]]
                    ixpeaks_tmp.sort()  # in-place sort
                    if abs(debugplot) >= 10:
                        print('ixpeaks in region........:\n', ixpeaks_tmp)
                    ixpeaks_filtered = np.concatenate((ixpeaks_filtered,
                                                       ixpeaks_tmp))
        ixpeaks = ixpeaks_filtered
        if abs(debugplot) >= 10:
            print('ixpeaks filtered.........:\n', ixpeaks)

    # remove peaks too close to any of the borders of the spectrum
    if npix_avoid_border > 0:
        lok_ini = ixpeaks >= npix_avoid_border
        lok_end = ixpeaks <= naxis1 - 1 - npix_avoid_border
        ixpeaks = ixpeaks[lok_ini * lok_end]

    # refined location of the peaks (float values)
    fxpeaks, sxpeaks = refine_peaks_spectrum(spf, ixpeaks,
                                             nwinwidth=nwinwidth_refined,
                                             method="gaussian")

    # print peak location and width of fitted lines
    if abs(debugplot) >= 10:
        print(">>> Number of lines found:", len(fxpeaks))
        print("# line_number, channel, width")
        for i, (fx, sx) in enumerate(zip(fxpeaks, sxpeaks)):
            print(i, fx+1, sx)

    # display median spectrum and peaks
    if abs(debugplot) % 10 != 0:
        xplot = np.arange(1, naxis1 + 1, dtype=float)
        ax = ximplotxy(xplot, sp, show=False, geometry=geometry,
                       **{'label': 'original spectrum'})
        ax.set_xlabel('pixel (from 1 to NAXIS1)')
        ax.set_ylabel('counts')
        if plottitle is not None:
            ax.set_title(plottitle)
        if sigma_gaussian_filtering > 0:
            ax.plot(xplot, spf, label="filtered spectrum")
        ymin = sp.min()
        ymax = sp.max()
        dy = ymax - ymin
        ymin -= dy/20.
        ymax += dy/20.
        ax.set_ylim([ymin, ymax])
        # display threshold
        ax.axhline(y=threshold, color="black", linestyle="dotted",
                   label="detection threshold")
        # mark peak location
        ax.plot(ixpeaks + 1, spf[ixpeaks], 'bo', label="initial location")
        ax.plot(fxpeaks + 1, spf[ixpeaks], 'go', label="refined location")
        # legend
        ax.legend(numpoints=1)
        # show plot
        pause_debugplot(debugplot, pltshow=True)

    return fxpeaks, sxpeaks


def wvcal_spectrum(sp, fxpeaks, poly_degree_wfit, wv_master,
                   wv_ini_search=None, wv_end_search=None,
                   wvmin_useful=None, wvmax_useful=None,
                   geometry=None, debugplot=0):
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

    # check there are enough lines for fit
    if len(fxpeaks) <= poly_degree_wfit:
        print(">>> Warning: not enough lines to fit spectrum")
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
        debugplot=debugplot
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
        debugplot=debugplot
    )

    if abs(debugplot) % 10 != 0:
        # final plot with identified lines
        xplot = np.arange(1, naxis1 + 1, dtype=float)
        ax = ximplotxy(xplot, sp, title=title, show=False,
                       xlabel='pixel (from 1 to NAXIS1)',
                       ylabel='number of counts',
                       geometry=geometry)
        ymin = sp.min()
        ymax = sp.max()
        dy = ymax-ymin
        ymin -= dy/20.
        ymax += dy/20.
        ax.set_ylim([ymin, ymax])
        # plot wavelength of each identified line
        for feature in solution_wv.features:
            xpos = feature.xpos
            reference = feature.reference
            ax.text(xpos, sp[int(xpos+0.5)-1],
                    str(reference), fontsize=8,
                    horizontalalignment='center')
        # show plot
        print('Plot with identified lines')
        pause_debugplot(12, pltshow=True)

    # return the wavelength calibration solution
    return solution_wv


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser()
    # required parameters
    parser.add_argument("fitsfile",
                        help="FITS image containing the spectra",
                        type=argparse.FileType('rb'))
    parser.add_argument("--scans", required=True,
                        help="Tuple ns1[,ns2] (from 1 to NAXIS2)")
    parser.add_argument("--wv_master_file", required=True,
                        help="TXT file containing wavelengths")
    parser.add_argument("--degree", required=True,
                        help="Polynomial degree", type=int)
    # optional arguments
    parser.add_argument("--wvmin",
                        help="Minimum expected wavelength",
                        type=float)
    parser.add_argument("--wvmax",
                        help="Maximum expected wavelength",
                        type=float)
    parser.add_argument("--wvmin_useful",
                        help="Minimum useful wavelength",
                        type=float)
    parser.add_argument("--wvmax_useful",
                        help="Maximum useful wavelength",
                        type=float)
    parser.add_argument("--nwin_background",
                        help="window to compute background (0=none)"
                        " (default=0)",
                        default=0, type=int)
    parser.add_argument("--method",
                        help="collapsing method (default='mean', 'median')",
                        default='mean',
                        choices=['mean', 'median'])
    parser.add_argument("--times_sigma_threshold",
                        help="Threshold (times robust sigma to detect lines)"
                             " (default=10)",
                        default=10, type=float)
    parser.add_argument("--minimum_threshold",
                        help="Minimum threshold to detect lines"
                             " (default=0)",
                        default=0, type=float)
    parser.add_argument("--nwinwidth_initial",
                        help="Initial window width to detect lines"
                             " (default=7)",
                        default=7, type=int)
    parser.add_argument("--nwinwidth_refined",
                        help="Refined window width to detect lines"
                             " (default=5)",
                        default=5, type=int)
    parser.add_argument("--npix_avoid_border",
                        help="Number of pixels in the borders to be avoided"
                             " (default=6)",
                        default=6, type=int)
    parser.add_argument("--nbrightlines",
                        help="Tuple n1,[n2,[n3,...]] with maximum number of "
                             "brightest lines to be used [0=all] (default=0)",
                        default=0)
    parser.add_argument("--degree_refined",
                        help="Degree of the refined fit using faint lines "
                             "from wv_master_file (default=None, i.e. no "
                             "refinement)",
                        default=None, type=int)
    parser.add_argument("--sigma_gauss_filt",
                        help="Sigma (pixels) of gaussian filtering to avoid "
                             "saturared lines (default=0)",
                        default=0, type=float)
    parser.add_argument("--minimum_gauss_filt",
                        help="Minimum pixel value to use gaussian filtering "
                             "(default=0)",
                        default=0, type=float)
    parser.add_argument("--reverse",
                        help="Reverse wavelength direction",
                        action="store_true")
    parser.add_argument("--out_sp",
                        help="File name to save the selected spectrum in FITS "
                             "format before performing the wavelength "
                             "calibration (default=None)",
                        default=None,
                        type=argparse.FileType('wb'))
    parser.add_argument("--geometry",
                        help="tuple x,y,dx,dy (default 0,0,640,480)",
                        default="0,0,640,480")
    parser.add_argument("--pdffile",
                        help="Output PDF file name",
                        type=lambda x: arg_file_is_new(parser, x, mode='wb'))
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                        " (default=0)",
                        default=0, type=int,
                        choices=DEBUGPLOT_CODES)
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")
    args = parser.parse_args(args=args)

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    # ---

    # read pdffile
    if args.pdffile is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(args.pdffile.name)
        interactive_refinement = False
    else:
        pdf = None
        interactive_refinement = True

    # if refinement is going to be used, check that the corresponding
    # polynomial degree is at least as large as the initial degree
    if args.degree_refined is not None:
        if args.degree > args.degree_refined:
            raise ValueError("degree_refined must be >= degree")

    # scan range
    tmp_str = args.scans.split(",")
    if len(tmp_str) == 2:
        ns1 = int(tmp_str[0])
        ns2 = int(tmp_str[1])
    elif len(tmp_str) == 1:
        ns1 = int(tmp_str[0])
        ns2 = ns1
    else:
        raise ValueError("Invalid tuple for scan range")

    # geometry
    if args.geometry is None:
        geometry = None
    else:
        tmp_str = args.geometry.split(",")
        x_geom = int(tmp_str[0])
        y_geom = int(tmp_str[1])
        dx_geom = int(tmp_str[2])
        dy_geom = int(tmp_str[3])
        geometry = x_geom, y_geom, dx_geom, dy_geom

    # convert nbrightlines in a list of integers
    nbrightlines = [int(nlines) for nlines in args.nbrightlines.split(",")]

    # compute collapsed spectrum
    sp = collapsed_spectrum(
        fitsfile=args.fitsfile, ns1=ns1, ns2=ns2,
        method=args.method,
        nwin_background=args.nwin_background,
        reverse=args.reverse,
        debugplot=args.debugplot
    )

    # read arc line wavelengths from external file (all the lines)
    wv_master_all = read_wv_master_file(args.wv_master_file, lines='all')
    # read arc line wavelengths from external file (brightest lines only)
    wv_master = read_wv_master_file(args.wv_master_file, lines='brightest')

    # clip master arc line lists to expected wavelength range
    if args.wvmin is None:
        wvmin = -np.infty
    else:
        wvmin = args.wvmin
    if args.wvmax is None:
        wvmax = np.infty
    else:
        wvmax = args.wvmax

    lok1 = wvmin <= wv_master_all
    lok2 = wv_master_all <= wvmax
    lok = lok1 * lok2
    if abs(args.debugplot) >= 10:
        print("Number of lines in wv_master_all........: ", len(wv_master_all))
    wv_master_all = wv_master_all[lok]
    if abs(args.debugplot) >= 10:
        print("Number of lines in clipped wv_master_all: ", len(wv_master_all))
        print("clipped wv_master_all:\n", wv_master_all)

    lok1 = wvmin <= wv_master
    lok2 = wv_master <= wvmax
    lok = lok1 * lok2
    if abs(args.debugplot) >= 10:
        print("Number of lines in wv_master............: ", len(wv_master))
    wv_master = wv_master[lok]
    if abs(args.debugplot) >= 10:
        print("Number of lines in clipped wv_master....: ", len(wv_master))
        print("clipped wv_master....:\n", wv_master)

    # determine refined peak location in array coordinates, i.e.,
    # from 0 to (naxis - 1)
    plottitle = os.path.basename(args.fitsfile.name) + \
                ' [{}, {}:{}],  line list: {}'.format(
                    args.method, ns1, ns2,
                    os.path.basename(args.wv_master_file)
                )
    fxpeaks, sxpeaks = find_fxpeaks(
        sp=sp,
        times_sigma_threshold=args.times_sigma_threshold,
        minimum_threshold=args.minimum_threshold,
        nwinwidth_initial=args.nwinwidth_initial,
        nwinwidth_refined=args.nwinwidth_refined,
        npix_avoid_border=args.npix_avoid_border,
        nbrightlines=nbrightlines,
        sigma_gaussian_filtering=args.sigma_gauss_filt,
        minimum_gaussian_filtering=args.minimum_gauss_filt,
        plottitle=plottitle,
        geometry=geometry,
        debugplot=args.debugplot
    )

    # perform wavelength calibration
    solution_wv = wvcal_spectrum(
        sp=sp,
        fxpeaks=fxpeaks,
        poly_degree_wfit=args.degree,
        wv_master=wv_master,
        wv_ini_search=args.wvmin,
        wv_end_search=args.wvmax,
        wvmin_useful=args.wvmin_useful,
        wvmax_useful=args.wvmax_useful,
        geometry=geometry,
        debugplot=args.debugplot
    )

    # apply gaussian filtering
    if args.sigma_gauss_filt > 0:
        spf = ndimage.filters.gaussian_filter(
            sp,
            sigma=args.sigma_gauss_filt
        )
    else:
        spf = np.copy(sp)

    # save fitted spectrum
    if args.out_sp is not None:
        hdu = fits.PrimaryHDU(spf.astype(np.float32))
        hdu.writeto(args.out_sp, overwrite=True)

    # clip master arc line lists to useful wavelength range
    if args.wvmin_useful is None:
        wvmin = -np.infty
    else:
        wvmin = args.wvmin_useful
    if args.wvmax_useful is None:
        wvmax = np.infty
    else:
        wvmax = args.wvmax_useful

    lok1 = wvmin <= wv_master_all
    lok2 = wv_master_all <= wvmax
    lok = lok1 * lok2
    if abs(args.debugplot) >= 10:
        print("Number of lines in wv_master_all........: ", len(wv_master_all))
    wv_master_all = wv_master_all[lok]
    if abs(args.debugplot) >= 10:
        print("Number of lines in clipped wv_master_all: ", len(wv_master_all))
        print("clipped wv_master_all:\n", wv_master_all)

    # refine wavelength calibration when requested
    if args.degree_refined is not None:
        poly_refined, yres_summary = refine_arccalibration(
            sp=spf,
            poly_initial=np.polynomial.Polynomial(solution_wv.coeff),
            wv_master=wv_master_all,
            poldeg=args.degree_refined,
            ntimes_match_wv=1,
            plottitle=plottitle,
            interactive=interactive_refinement,
            geometry=geometry,
            pdf=pdf,
            debugplot=args.debugplot
        )

    if pdf is not None:
        pdf.close()
    else:
        try:
            input("\nPress RETURN to QUIT...")
        except SyntaxError:
            pass


if __name__ == "__main__":

    main()
