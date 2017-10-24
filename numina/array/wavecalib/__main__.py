#
# Copyright 2015-2017 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

"""Compute the wavelength calibration of a particular spectrum."""

from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import numpy as np
from scipy import ndimage

from .arccalibration import arccalibration
from .arccalibration import fit_list_of_wvfeatures
from ..display.pause_debugplot import pause_debugplot
from ..display.ximplot import ximplot
from ..stats import robust_std
from .peaks_spectrum import find_peaks_spectrum
from .peaks_spectrum import refine_peaks_spectrum


def collapsed_spectrum(filename, ns1, ns2,
                       method='mean', nwin_background=0,
                       reverse=False, out_sp=None, debugplot=0):
    """Compute a collapsed spectrum from a 2D image using scans in [ns1,ns2].

    Parameters
    ----------
    filename : string
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
    hdulist = fits.open(filename)
    image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape
    hdulist.close()
    if abs(debugplot) >= 10:
        print('>>> Reading file:', filename)
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
            hdu.writeto(out_sp, clobber=True)
    else:
        raise ValueError("Invalid ns1=" + str(ns1) + ", ns2=" + str(ns2) +
                         " values")

    return sp


def read_wv_master_file(wv_master_file, debugplot):
    """read arc line wavelengths from external file.

    Parameters
    ----------
    wv_master_file : string
        File name of txt file containing the wavelength database.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------
    wv_master : 1d numpy array
        Array with arc line wavelengths.

    """

    master_table = np.genfromtxt(wv_master_file)
    if master_table.ndim == 1:
        wv_master = master_table
    else:
        wv_master_all = master_table[:, 0]
        if master_table.shape[1] == 2:  # assume old format
            wv_master = np.copy(wv_master_all)
        elif master_table.shape[1] == 3:  # assume new format
            wv_flag = master_table[:, 1]
            wv_master = wv_master_all[np.where(wv_flag == 1)]
        else:
            raise ValueError('lines_catalog file does not have the '
                             'expected number of columns')

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
                 debugplot):
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

    # threshold to search for peaks
    q50 = np.percentile(sp, q=50)
    sigma_g = robust_std(sp)
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
    ixpeaks = find_peaks_spectrum(sp,
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
                    peak_fluxes = sp[ixpeaks_region]
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
    fxpeaks, sxpeaks = refine_peaks_spectrum(sp, ixpeaks,
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
        ax = ximplot(sp, plot_bbox=(1, naxis1),
                     show=False)
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
        ax.plot(ixpeaks + 1, sp[ixpeaks], 'bo', label="initial location")
        ax.plot(fxpeaks + 1, sp[ixpeaks], 'go', label="refined location")
        # legend
        ax.legend(numpoints=1)
        # show plot
        pause_debugplot(debugplot, pltshow=True)

    return fxpeaks, sxpeaks


def wvcal_spectrum(sp, fxpeaks, poly_degree_wfit, wv_master, debugplot):
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

    # use channels (pixels from 1 to naxis1)
    xchannel = fxpeaks + 1.0

    # wavelength calibration
    list_of_wvfeatures = arccalibration(
        wv_master=wv_master,
        xpos_arc=xchannel,
        naxis1_arc=naxis1,
        crpix1=1.0,
        wv_ini_search=wv_master[0] - delta_wv_master_range,
        wv_end_search=wv_master[-1] + delta_wv_master_range,
        error_xpos_arc=3,
        times_sigma_r=3.0,
        frac_triplets_for_sum=0.50,
        times_sigma_theil_sen=10.0,
        poly_degree_wfit=poly_degree_wfit,
        times_sigma_polfilt=10.0,
        times_sigma_cook=10.0,
        times_sigma_inclusion=10.0,
        debugplot=debugplot
    )

    title = "Wavelength calibration"
    solution_wv = fit_list_of_wvfeatures(
        list_of_wvfeatures=list_of_wvfeatures,
        naxis1_arc=naxis1,
        crpix1=1.0,
        poly_degree_wfit=poly_degree_wfit,
        weighted=False,
        debugplot=debugplot,
        plot_title=title
    )

    if abs(debugplot) % 10 != 0:
        # final plot with identified lines
        ax = ximplot(sp, title=title, show=False,
                     plot_bbox=(1, naxis1))
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
            ax.text(xpos, sp[int(xpos+0.5)-1] + dy/100,
                    str(reference), fontsize=8,
                    horizontalalignment='center')
        # show plot
        pause_debugplot(11, pltshow=True, tight_layout=False)

    # return the wavelength calibration solution
    return solution_wv


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='wavecalib')
    # required parameters
    parser.add_argument("filename",
                        help="FITS image containing the spectra")
    parser.add_argument("--scans", required=True,
                        help="Tuple ns1[,ns2] (from 1 to NAXIS2)")
    parser.add_argument("--wv_master_file", required=True,
                        help="TXT file containing wavelengths")
    parser.add_argument("--degree", required=True,
                        help="Polynomial degree", type=int)
    # optional arguments
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
    parser.add_argument("--reverse",
                        help="Reverse wavelength direction",
                        action="store_true")
    parser.add_argument("--out_sp",
                        help="File name to save the selected spectrum in FITS "
                             "format before performing the wavelength "
                             "calibration (default=None)",
                        default=None,
                        type=argparse.FileType('w'))
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                        " (default=0)",
                        default=0, type=int,
                        choices=[0, 1, 2, 10, 11, 12, 21, 22])
    args = parser.parse_args(args=args)

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

    # convert nbrightlines in a list of integers
    nbrightlines = [int(nlines) for nlines in args.nbrightlines.split(",")]

    # compute collapsed spectrum
    sp = collapsed_spectrum(
        filename=args.filename, ns1=ns1, ns2=ns2,
        method=args.method,
        nwin_background=args.nwin_background,
        reverse=args.reverse,
        out_sp=args.out_sp,
        debugplot=args.debugplot
    )

    # read arc line wavelengths from external file
    wv_master = read_wv_master_file(
        args.wv_master_file,
        debugplot=args.debugplot
    )

    # determine refined peak location in array coordinates, i.e.,
    # from 0 to (naxis - 1)
    fxpeaks, sxpeaks = find_fxpeaks(
        sp=sp,
        times_sigma_threshold=args.times_sigma_threshold,
        minimum_threshold=args.minimum_threshold,
        nwinwidth_initial=args.nwinwidth_initial,
        nwinwidth_refined=args.nwinwidth_refined,
        npix_avoid_border=args.npix_avoid_border,
        nbrightlines=nbrightlines,
        debugplot=args.debugplot
    )

    # perform wavelength calibration
    solution_wv = wvcal_spectrum(
        sp=sp,
        fxpeaks=fxpeaks,
        poly_degree_wfit=args.degree,
        wv_master=wv_master,
        debugplot=args.debugplot
    )

    try:
        input("\nPress RETURN to QUIT...")
    except SyntaxError:
        pass


if __name__ == "__main__":

    main()
