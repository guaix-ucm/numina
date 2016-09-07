#
# Copyright 2015 Universidad Complutense de Madrid
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
from .arccalibration import fit_list_of_dict
from .arccalibration import SolutionArcCalibration
from ..display.pause_debugplot import pause_debugplot
from .peaks_spectrum import find_peaks_spectrum
from .peaks_spectrum import refine_peaks_spectrum
from ..display.ximplot import ximplot

def wvcal_spectrum(filename, ns1, ns2,
                   nwin_background,
                   times_sigma_threshold,
                   wv_master_file,
                   reverse, debugplot,
                   poly_degree_wfit,
                   savesp):
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
    nwin_background : int
        Window size for the computation of background using a median
        filtering with that window width. This background is computed
        and subtracted only if this parameter is > 0.
    times_sigma_threshold : float
        Times robust sigma above the median to detect line peaks.
    wv_master_file : string
        File name of txt file containing the wavelength database.
    reverse : bool
        If True, reserve wavelength direction prior to wavelength
        calibration.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses
    poly_degree_wfit : int
        Degree for wavelength calibration polynomial.
    savesp : bool
        If True, save spectrum as xxx.fits.

    """

    # read FITS file
    hdulist = fits.open(filename)
    image_header = hdulist[0].header
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
            background = ndimage.filters.median_filter(sp_mean, size=81)
            sp_mean -= background

        # save spectrum in external FITS file
        if savesp:
            hdu = fits.PrimaryHDU(sp_mean)
            hdu.writeto("xxx.fits", clobber=True)

        # initial location of the peaks (integer values)
        q25, q50, q75 = np.percentile(sp_mean, q=[25.0, 50.0, 75.0])
        sigma_g = 0.7413 * (q75 - q25)  # robust standard deviation
        threshold = q50 + times_sigma_threshold * sigma_g
        if debugplot >= 10:
            print("median....:", q50)
            print("robuts std:", sigma_g)
            print("threshold.:", threshold)
        nwinwidth_initial = 7
        ixpeaks = find_peaks_spectrum(sp_mean,
                                      nwinwidth=nwinwidth_initial,
                                      threshold=threshold)
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
            ax = ximplot(sp_mean, title=title, plot_bbox=(1,naxis1),
                         show=False)
            ymin = sp_mean.min()
            ymax = sp_mean.max()
            dy = ymax - ymin
            ymin -= dy/20.
            ymax += dy/20.
            ax.set_ylim([ymin, ymax])
            # display threshold
            import matplotlib
            matplotlib.use('Qt4Agg')
            import matplotlib.pyplot as plt
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

        # wavelength calibration
        xchannel = fxpeaks + 1.0
        list_of_dict = arccalibration(
            wv_master=wv_master,
            xpos_arc=xchannel,
            naxis1_arc=naxis1,
            crpix1=1.0,
            wv_ini_search=wv_master[0] - 1000.0,
            wv_end_search=wv_master[-1] + 1000.0,
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

        title = filename + "[" + str(ns1) + ":" + str(ns2) + "]" + \
                     "\n" + wv_master_file
        coeff, crval1_linear, crmin1_linear, crmax1_linear, cdelt1_linear = \
            fit_list_of_dict(
                list_of_dict=list_of_dict,
                naxis1_arc=naxis1,
                crpix1=1.0,
                poly_degree_wfit=poly_degree_wfit,
                weighted=False,
                debugplot=12,
                plot_title=title
            )

        # note that the class SolutionArcCalibration only stores the
        # information in 'list_of_dict' corresponding to lines that
        # have been properly identified
        solution_wv = SolutionArcCalibration(
            list_of_dict=list_of_dict,
            coeff=coeff,
            crpix1_linear=1.0,
            crval1_linear=crval1_linear,
            crmin1_linear=crmin1_linear,
            crmax1_linear=crmax1_linear,
            cdelt1_linear=cdelt1_linear)

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
        for xpos, wv in zip(solution_wv.xpos, solution_wv.wv):
            ax.text(xpos, sp_mean[int(xpos+0.5)-1],
                    str(wv), fontsize=8,
                    horizontalalignment='center')
        # show plot
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        plt.show(block=False)
        plt.pause(0.001)
        pause_debugplot(11)
    else:
        raise ValueError("Invalid ns1=" + str(ns1) + ", ns2=" + str(ns2) +
                         " values")


if __name__ == "__main__":

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        help="FITS image containing the spectra")
    parser.add_argument("ns1",
                        help="First scan (from 1 to NAXIS2)")
    parser.add_argument("ns2",
                        help="Last scan (from 1 to NAXIS2)")
    parser.add_argument("wv_master_file",
                        help="TXT file containing wavelengths")
    parser.add_argument("nwin_background",
                        help="window to compute background (0=none)")
    parser.add_argument("times_sigma_threshold",
                        help="Threshold (times sigma_g to detect lines)")
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                        " (default=0)",
                        default=0)
    parser.add_argument("--reverse",
                        help="Reverse wavelength direction (yes/no)" +
                        " (default=no)",
                        default="no")
    parser.add_argument("--degree",
                        help="Polynomial degree (default=3)",
                        default=3)
    parser.add_argument("--savesp",
                        help="Save spectrum (yes/no)" +
                        " (default=no)",
                        default="no")
    args = parser.parse_args()

    ns1 = int(args.ns1)
    ns2 = int(args.ns2)
    nwin_background = int(args.nwin_background)
    times_sigma_threshold = float(args.times_sigma_threshold)
    debugplot = int(args.debugplot)
    reverse = (args.reverse == "yes")
    poly_degree_wfit=int(args.degree)
    savesp = (args.savesp == "yes")

    wvcal_spectrum(args.filename, ns1, ns2,
                   nwin_background,
                   times_sigma_threshold,
                   args.wv_master_file,
                   reverse, debugplot,
                   poly_degree_wfit,
                   savesp)

    try:
        input("\nPress RETURN to QUIT...")
    except SyntaxError:
        pass