#
# Copyright 2015-2016 Universidad Complutense de Madrid
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


def wvcal_spectrum(filename, ns1, ns2,
                   poly_degree_wfit,
                   nwin_background,
                   times_sigma_threshold,
                   nbrightlines,
                   wv_master_file,
                   reverse,
                   out_sp,
                   debugplot):
    """Execute wavelength calibration of a spectrum.

    The initial image can be 2 dimensional, although in this case the
    image is supposed to have been rectified and the spectrum to be
    calibrated will be the average of the spectra in the range
    [ns1,ns2] (counting from 1 to NAXIS2).

    Parameters
    ----------
    filename : string
        File name of FITS file containing the spectra to be calibrated.
    ns1 : int
        First scan (from 1 to NAXIS2).
    ns2 : int
        Last scan (from 1 to NAXIS2).
    poly_degree_wfit : int
        Degree for wavelength calibration polynomial.
    nwin_background : int
        Window size for the computation of background using a median
        filtering with that window width. This background is computed
        and subtracted only if this parameter is > 0.
    times_sigma_threshold : float
        Times robust sigma above the median to detect line peaks.
    nbrightlines : int
        Maximum number of brightest lines to be employed in the
        wavelength calibration. If this value is 0, all the detected
        lines will be employed.
    wv_master_file : string
        File name of txt file containing the wavelength database.
    reverse : bool
        If True, reserve wavelength direction prior to wavelength
        calibration.
    out_sp : string or None
        File name to save the selected spectrum in FITS format before
        performing the wavelength calibration.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    Returns
    -------
    sp : numpy array (floats)
        1d array containing the selected spectrum before computing the
        wavelength calibration.
    solution_wv : instance of SolutionArcCalibration
        Wavelength calibration solution.

    """

    # read FITS file
    hdulist = fits.open(filename)
    image2d = hdulist[0].data
    naxis2, naxis1 = image2d.shape
    hdulist.close()
    if debugplot >= 10:
        print('>>> Reading file:', filename)
        print('>>> NAXIS1:', naxis1)
        print('>>> NAXIS2:', naxis2)

    if 1 <= ns1 <= ns2 <= naxis2:
        # extract spectrum
        sp_mean = np.mean(image2d[(ns1-1):ns2], axis=0)

        # reverse spectrum if necessary
        if reverse:
            sp_mean = sp_mean[::-1]

        # fit and subtract background
        if nwin_background > 0:
            background = ndimage.filters.median_filter(sp_mean,
                                                       size=nwin_background)
            sp_mean -= background

        # save spectrum before wavelength calibration in external
        # FITS file
        if out_sp is not None:
            hdu = fits.PrimaryHDU(sp_mean)
            hdu.writeto(out_sp, clobber=True)

        # initial location of the peaks (integer values)
        q50 = np.percentile(sp_mean, q=50)
        sigma_g = robust_std(sp_mean)
        threshold = q50 + times_sigma_threshold * sigma_g
        if debugplot >= 10:
            print("median....:", q50)
            print("robuts std:", sigma_g)
            print("threshold.:", threshold)
        nwinwidth_initial = 7
        ixpeaks = find_peaks_spectrum(sp_mean,
                                      nwinwidth=nwinwidth_initial,
                                      threshold=threshold)

        # select a maximum number of brightest lines in each region
        if len(nbrightlines) == 1 and nbrightlines[0] == 0:
            pass
        else:
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
                        peak_fluxes = sp_mean[ixpeaks_region]
                        spos = peak_fluxes.argsort()
                        ixpeaks_tmp = ixpeaks_region[spos[-nlines_in_region:]]
                        ixpeaks_tmp.sort()  # in-place sort
                        ixpeaks_filtered=np.concatenate((ixpeaks_filtered,
                                                         ixpeaks_tmp))
            ixpeaks = ixpeaks_filtered

        # check there are enough lines for fit
        if len(ixpeaks) <= poly_degree_wfit:
            print(">>> Warning: not enough lines to fit spectrum computed" +
                  " from scans [" + str(ns1) + "," + str(ns2), "]")
            sp_mean = np.zeros(naxis1)
            return sp_mean, None

        # refined location of the peaks (float values)
        nwinwidth_refined = 5
        fxpeaks, sxpeaks = refine_peaks_spectrum(sp_mean, ixpeaks,
                                                 nwinwidth=nwinwidth_refined,
                                                 method="gaussian")

        # print peak location and width of fitted lines
        if debugplot >= 10:
            print(">>> Number of lines found:", len(fxpeaks))
            print("# line_number, channel, width")
            for i, (fx, sx) in enumerate(zip(fxpeaks, sxpeaks)):
                print(i, fx+1, sx)

        # display median spectrum and peaks
        if debugplot % 10 != 0:
            title = filename
            ax = ximplot(sp_mean, title=title, plot_bbox=(1, naxis1),
                         show=False)
            ymin = sp_mean.min()
            ymax = sp_mean.max()
            dy = ymax - ymin
            ymin -= dy/20.
            ymax += dy/20.
            ax.set_ylim([ymin, ymax])
            # display threshold
            from numina.array.display.matplotlib_qt import plt
            plt.axhline(y=threshold, color="black", linestyle="dotted",
                        label="detection threshold")
            # mark peak location
            plt.plot(ixpeaks + 1,
                     sp_mean[ixpeaks], 'bo', label="initial location")
            plt.plot(fxpeaks + 1,
                     sp_mean[ixpeaks], 'go', label="refined location")
            # legend
            plt.legend(numpoints=1)
            # show plot
            plt.show(block=False)
            plt.pause(0.001)
            pause_debugplot(debugplot)

        # read arc line wavelengths from external file
        master_table = np.genfromtxt(wv_master_file)
        if master_table.ndim == 1:
            wv_master = master_table
        else:
            wv_master = master_table[:, 0]
            wv_master_flag = master_table[:, 1]
            iremove = np.where(wv_master_flag == -1)
            if len(iremove) > 0:
                wv_master = np.delete(wv_master, iremove)

        if debugplot >= 10:
            print("Reading master table: " + wv_master_file)
            print("wv_master:\n", wv_master)

        wv_master_range = wv_master[-1] - wv_master[0]
        delta_wv_master_range = 0.20 * wv_master_range

        # wavelength calibration
        xchannel = fxpeaks + 1.0
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

        title = filename + "[" + str(ns1) + ":" + str(ns2) + "]"
        title += "\n" + wv_master_file
        solution_wv = fit_list_of_wvfeatures(
            list_of_wvfeatures=list_of_wvfeatures,
            naxis1_arc=naxis1,
            crpix1=1.0,
            poly_degree_wfit=poly_degree_wfit,
            weighted=False,
            debugplot=debugplot,
            plot_title=title
        )

        if debugplot % 10 != 0:
            # final plot with identified lines
            ax = ximplot(sp_mean, title=title, show=False,
                         plot_bbox=(1, naxis1))
            ymin = sp_mean.min()
            ymax = sp_mean.max()
            dy = ymax-ymin
            ymin -= dy/20.
            ymax += dy/20.
            ax.set_ylim([ymin, ymax])
            # plot wavelength of each identified line
            for feature in solution_wv.features:
                xpos = feature.xpos
                reference = feature.reference
                ax.text(xpos, sp_mean[int(xpos+0.5)-1],
                        str(reference), fontsize=8,
                        horizontalalignment='center')
            # show plot
            from numina.array.display.matplotlib_qt import plt
            plt.show(block=False)
            plt.pause(0.001)
            pause_debugplot(11)

        # return the spectrum before the wavelength calibration and
        # the wavelength calibration solution
        return sp_mean, solution_wv

    else:
        raise ValueError("Invalid ns1=" + str(ns1) + ", ns2=" + str(ns2) +
                         " values")


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
    parser.add_argument("--times_sigma_threshold",
                        help="Threshold (times robust sigma to detect lines)"
                             " (default=10)",
                        default=10, type=float)
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

    wvcal_spectrum(filename=args.filename,
                   ns1=ns1, ns2=ns2,
                   poly_degree_wfit=args.degree,
                   nwin_background=args.nwin_background,
                   times_sigma_threshold=args.times_sigma_threshold,
                   nbrightlines=nbrightlines,
                   wv_master_file=args.wv_master_file,
                   reverse=args.reverse,
                   out_sp=args.out_sp,
                   debugplot=args.debugplot)

    try:
        input("\nPress RETURN to QUIT...")
    except SyntaxError:
        pass


if __name__ == "__main__":

    main()
