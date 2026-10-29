#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compute the wavelength calibration of a particular spectrum.

This script computes the wavelength calibration of a particular spectrum
from a 2D image containing several spectra. The wavelength calibration
is performed using a list of known arc line wavelengths. The script
performs the following steps:
1. Read the 2D image from a FITS file.
2. Collapse the 2D image along the scan direction to obtain a 1D spectrum.
3. Locate the peaks in the 1D spectrum using a threshold based on the robust standard deviation
   of the spectrum.
4. Perform an initial wavelength calibration using the brightest lines from the provided list of known
   arc line wavelengths.
5. Optionally refine the wavelength calibration using fainter lines from the provided list of known
   arc line wavelengths.
"""

import argparse
import os
import warnings

from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from rich_argparse import RichHelpFormatter
from scipy import ndimage

from . import core
from .arccalibration import refine_arccalibration
from ...tools.arg_file_is_new import arg_file_is_new

from numina.tools.initialize_script_with_args import NuminaScriptDefinition

from ..display.pause_debugplot import DEBUGPLOT_CODES

# Functions moved to numina.array.wavecalib.core
_MOVED = ("collapsed_spectrum", "read_wv_master_from_array", "read_wv_master_file", "find_fxpeaks", "wvcal_spectrum")


def main(args=None):
    """Main function for the wavelength calibration script."""

    # parse command-line options
    parser = argparse.ArgumentParser(
        description="Compute the wavelength calibration of a particular spectrum.", formatter_class=RichHelpFormatter
    )
    # required parameters
    parser.add_argument("fitsfile", help="FITS image containing the spectra", type=str)
    parser.add_argument("--scans", required=True, help="Tuple ns1[,ns2] (from 1 to NAXIS2)")
    parser.add_argument("--wv_master_file", required=True, help="TXT file containing wavelengths")
    parser.add_argument("--degree", required=True, help="Polynomial degree", type=int)
    # optional arguments
    parser.add_argument("--wvmin", help="Minimum expected wavelength", type=float)
    parser.add_argument("--wvmax", help="Maximum expected wavelength", type=float)
    parser.add_argument("--wvmin_useful", help="Minimum useful wavelength", type=float)
    parser.add_argument("--wvmax_useful", help="Maximum useful wavelength", type=float)
    parser.add_argument(
        "--nwin_background", help="window to compute background (0=none)" " (default=0)", default=0, type=int
    )
    parser.add_argument(
        "--method", help="collapsing method (default='mean', 'median')", default="mean", choices=["mean", "median"]
    )
    parser.add_argument(
        "--times_sigma_threshold",
        help="Threshold (times robust sigma to detect lines)" " (default=10)",
        default=10,
        type=float,
    )
    parser.add_argument(
        "--minimum_threshold", help="Minimum threshold to detect lines" " (default=0)", default=0, type=float
    )
    parser.add_argument(
        "--nwinwidth_initial", help="Initial window width to detect lines" " (default=7)", default=7, type=int
    )
    parser.add_argument(
        "--nwinwidth_refined", help="Refined window width to detect lines" " (default=5)", default=5, type=int
    )
    parser.add_argument(
        "--npix_avoid_border", help="Number of pixels in the borders to be avoided" " (default=6)", default=6, type=int
    )
    parser.add_argument(
        "--nbrightlines",
        help="Tuple n1,[n2,[n3,...]] with maximum number of " "brightest lines to be used [0=all] (default=0)",
        default="0",
    )
    parser.add_argument(
        "--degree_refined",
        help="Degree of the refined fit using faint lines " "from wv_master_file (default=None, i.e. no " "refinement)",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--sigma_gauss_filt",
        help="Sigma (pixels) of gaussian filtering to avoid " "saturared lines (default=0)",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--minimum_gauss_filt",
        help="Minimum pixel value to use gaussian filtering " "(default=0)",
        default=0,
        type=float,
    )
    parser.add_argument("--reverse", help="Reverse wavelength direction", action="store_true")
    parser.add_argument(
        "--out_sp",
        help="File name to save the selected spectrum in FITS "
        "format before performing the wavelength "
        "calibration (default=None)",
        default=None,
        type=str,
    )
    parser.add_argument("--geometry", help="tuple x,y,dx,dy (default 0,0,640,480)", default="0,0,640,480")
    parser.add_argument("--pdffile", help="Output PDF file name", type=lambda x: arg_file_is_new(parser, x, mode="wb"))
    parser.add_argument(
        "--debugplot",
        help="Integer indicating plotting/debugging" + " (default=0)",
        default=0,
        type=int,
        choices=DEBUGPLOT_CODES,
    )
    parser.add_argument("--extra-plots", help="Display the spectrum and the peaks found in it", action="store_true")
    # Include default arguments for common actions, and initialize console and logging
    myscript = NuminaScriptDefinition(parser, argv=args)
    args = myscript.args  # noqa: F841
    logger = myscript.logger  # noqa: F841

    # ---

    # read pdffile
    if args.pdffile is not None:
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
    sp = core.collapsed_spectrum(
        fitsfile=args.fitsfile,
        ns1=ns1,
        ns2=ns2,
        method=args.method,
        nwin_background=args.nwin_background,
        reverse=args.reverse,
    )

    # read arc line wavelengths from external file (all the lines)
    wv_master_all = core.read_wv_master_file(args.wv_master_file, lines="all")
    # read arc line wavelengths from external file (brightest lines only)
    wv_master = core.read_wv_master_file(args.wv_master_file, lines="brightest")

    # clip master arc line lists to expected wavelength range
    if args.wvmin is None:
        wvmin = -np.inf
    else:
        wvmin = args.wvmin
    if args.wvmax is None:
        wvmax = np.inf
    else:
        wvmax = args.wvmax

    lok1 = wvmin <= wv_master_all
    lok2 = wv_master_all <= wvmax
    lok = lok1 * lok2
    lsize = len(str(len(wv_master_all)))
    logger.info(f"Number of lines in wv_master_all........: {len(wv_master_all):>{lsize}d}")
    wv_master_all = wv_master_all[lok]
    logger.info(f"Number of lines in clipped wv_master_all: {len(wv_master_all):>{lsize}d}")
    logger.debug(f"clipped wv_master_all:\n{wv_master_all}")

    lok1 = wvmin <= wv_master
    lok2 = wv_master <= wvmax
    lok = lok1 * lok2
    logger.info(f"Number of lines in wv_master............: {len(wv_master):>{lsize}d}")
    wv_master = wv_master[lok]
    logger.info(f"Number of lines in clipped wv_master....: {len(wv_master):>{lsize}d}")
    logger.debug(f"clipped wv_master....:\n{wv_master}")

    # determine refined peak location in array coordinates, i.e.,
    # from 0 to (naxis - 1)
    plottitle0 = " [{}, {}:{}],  line list: {}".format(args.method, ns1, ns2, os.path.basename(args.wv_master_file))
    plottitle = os.path.basename(args.fitsfile) + plottitle0
    fxpeaks, sxpeaks = core.find_fxpeaks(
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
        extra_plots=args.extra_plots,
    )

    # perform wavelength calibration
    solution_wv = core.wvcal_spectrum(
        sp=sp,
        fxpeaks=fxpeaks,
        poly_degree_wfit=args.degree,
        wv_master=wv_master,
        wv_ini_search=args.wvmin,
        wv_end_search=args.wvmax,
        wvmin_useful=args.wvmin_useful,
        wvmax_useful=args.wvmax_useful,
        geometry=geometry,
        debugplot=args.debugplot,
    )

    # apply gaussian filtering
    if args.sigma_gauss_filt > 0:
        spf = ndimage.gaussian_filter(sp, sigma=args.sigma_gauss_filt)
    else:
        spf = np.copy(sp)

    # save fitted spectrum
    if args.out_sp is not None:
        hdu = fits.PrimaryHDU(spf.astype(np.float32))
        hdu.writeto(args.out_sp, overwrite=True)

    # clip master arc line lists to useful wavelength range
    if args.wvmin_useful is None:
        wvmin = -np.inf
    else:
        wvmin = args.wvmin_useful
    if args.wvmax_useful is None:
        wvmax = np.inf
    else:
        wvmax = args.wvmax_useful

    lok1 = wvmin <= wv_master_all
    lok2 = wv_master_all <= wvmax
    lok = lok1 * lok2
    lsize = len(str(len(wv_master_all)))
    logger.debug(f"Number of lines in wv_master_all........: {len(wv_master_all):>{lsize}d}")
    wv_master_all = wv_master_all[lok]
    logger.debug(f"Number of lines in clipped wv_master_all: {len(wv_master_all):>{lsize}d}")
    logger.debug(f"clipped wv_master_all:\n{wv_master_all}")

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
            debugplot=args.debugplot,
        )

    if pdf is not None:
        pdf.close()

    # Display goodbye message and save console log if recording is enabled
    myscript.goodbye_message_and_save_console()


def __getattr__(name):
    if name in _MOVED:
        warnings.warn(
            f"numina.array.wavecalib.__main__.{name} is deprecated, it will be removed in 0.42, "
            f"use numina.array.wavecalib.core.{name}",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":

    main()
