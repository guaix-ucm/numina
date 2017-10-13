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

"""Supervised quality control of wavelength calibration"""

from __future__ import division
from __future__ import print_function

import argparse
import astropy.io.fits as fits
import numpy as np
import os

from numina.array.stats import summary
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.polfit_residuals import polfit_residuals
from .peaks_spectrum import find_peaks_spectrum
from .peaks_spectrum import refine_peaks_spectrum

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES


def match_wv_arrays(wv_master, wv_expected_all_peaks, delta_wv_max):
    """Verify expected wavelength for each line peak.

    Assign individual arc lines from wv_master to each expected
    wavelength when the latter is within the maximum allowed range.

    Parameters
    ----------
    wv_master : numpy array
        Array containing the master wavelengths.
    wv_expected_all_peaks : numpy array
        Array containing the expected wavelengths computed from the
        approximate polynomial calibration applied to the location of
        the line peaks.
    delta_wv_max : float
        Maximum distance to accept that the master wavelength
        corresponds to the expected wavelength.

    Returns
    -------
    wv_verified_all_peaks : numpy array
        Verified wavelengths from master list.

    """

    # initialize the output array to zero
    wv_verified_all_peaks = np.zeros_like(wv_expected_all_peaks)

    # initialize to True array to indicate that no peak has already
    # been verified (this flag avoids duplication)
    wv_unused = np.ones_like(wv_expected_all_peaks, dtype=bool)

    # since it is likely that len(wv_master) < len(wv_expected_all_peaks),
    # it is more convenient to execute the search in the following order
    for i in range(len(wv_master)):
        j = np.searchsorted(wv_expected_all_peaks, wv_master[i])
        if j == 0:
            if wv_unused[j]:
                delta_wv = abs(wv_master[i] - wv_expected_all_peaks[j])
                if delta_wv < delta_wv_max:
                    wv_verified_all_peaks[j] = wv_master[i]
                    wv_unused[j] = False
        elif j == len(wv_expected_all_peaks):
            if wv_unused[j-1]:
                delta_wv = abs(wv_master[i] - wv_expected_all_peaks[j-1])
                if delta_wv < delta_wv_max:
                    wv_verified_all_peaks[j-1] = wv_master[i]
                    wv_unused[j-1] = False
        else:
            delta_wv1 = abs(wv_master[i] - wv_expected_all_peaks[j-1])
            delta_wv2 = abs(wv_master[i] - wv_expected_all_peaks[j])
            if delta_wv1 < delta_wv2:
                if delta_wv1 < delta_wv_max:
                    if wv_unused[j-1]:
                        wv_verified_all_peaks[j-1] = wv_master[i]
                        wv_unused[j-1] = False
                    elif wv_unused[j]:
                        if delta_wv2 < delta_wv_max:
                            wv_verified_all_peaks[j] = wv_master[i]
                            wv_unused[j] = False
            else:
                if delta_wv2 < delta_wv_max:
                    if wv_unused[j]:
                        wv_verified_all_peaks[j] = wv_master[i]
                        wv_unused[j] = False
                    elif wv_unused[j-1]:
                        if delta_wv1 < delta_wv_max:
                            wv_verified_all_peaks[j-1] = wv_master[i]
                            wv_unused[j-1] = False

    return wv_verified_all_peaks


def fun_wv(xchannel, crpix1, crval1, cdelt1):
    """Compute wavelengths from channels.

    The wavelength calibration is provided through the usual parameters
    CRPIX1, CRVAL1 and CDELT1.

    Parameters
    ----------
    xchannel : numpy array
        Input channels where the wavelengths will be evaluated.
    crpix1: float
        CRPIX1 keyword.
    crval1: float
        CRVAL1 keyword.
    cdelt1: float
        CDELT1 keyword.

    Returns
    -------
    wv : numpy array
        Computed wavelengths

    """
    wv = crval1 + (xchannel - crpix1) * cdelt1
    return wv


def check_sp(sp, crpix1, crval1, cdelt1, wv_master,
             title, geometry, debugplot):
    """Process twilight image.

    Parameters
    ----------
    sp : numpy array
        Wavelength calibrated RSS FITS file name.
    crpix1: float
        CRPIX1 keyword.
    crval1: float
        CRVAL1 keyword.
    cdelt1: float
        CDELT1 keyword.
    wv_master: numpy array
        Array with the detailed list of expected arc lines.
    title : string
        Plot title.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the Qt backend geometry.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    """

    # protections
    if type(sp) is not np.ndarray:
        raise ValueError("sp must be a numpy.ndarray")
    elif sp.ndim != 1:
        raise ValueError("sp.ndim is not 1")

    # determine spectrum length
    naxis1 = sp.shape[0]

    # find initial line peaks
    nwinwidth_initial = 7
    ixpeaks = find_peaks_spectrum(sp, nwinwidth=nwinwidth_initial)

    # refine location of line peaks
    nwinwidth_refined = 5
    fxpeaks, sxpeaks = refine_peaks_spectrum(
        sp, ixpeaks,
        nwinwidth=nwinwidth_refined,
        method="gaussian"
    )

    ixpeaks_wv = fun_wv(ixpeaks + 1, crpix1, crval1, cdelt1)
    fxpeaks_wv = fun_wv(fxpeaks + 1, crpix1, crval1, cdelt1)

    # read list of expected arc lines
    if abs(debugplot) in (21, 22):
        print('wv_master:', wv_master)

    # match peaks with expected arc lines
    delta_wv_max = 2 * cdelt1
    wv_verified_all_peaks = match_wv_arrays(
        wv_master,
        fxpeaks_wv,
        delta_wv_max=delta_wv_max
    )
    lines_ok = np.where(wv_verified_all_peaks > 0)

    # compute residuals
    xresid = fxpeaks_wv[lines_ok]
    yresid = wv_verified_all_peaks[lines_ok] - fxpeaks_wv[lines_ok]
    ysummary = summary(yresid)

    # fit polynomial to residuals
    polyres, yresres = polfit_residuals(
        x=xresid,
        y=yresid,
        deg=1,
        use_r=True,
        debugplot=10
    )

    print('-' * 79)
    print(">>> Number of arc lines in master file:", len(wv_master))
    print(">>> Number of line peaks found........:", len(ixpeaks))
    print(">>> Number of identified lines........:", len(lines_ok[0]))
    list_wv_found = [str(round(wv, 4))
                     for wv in wv_verified_all_peaks if wv != 0]
    list_wv_master = [str(round(wv, 4)) for wv in wv_master]
    set1 = set(list_wv_master)
    set2 = set(list_wv_found)
    missing_wv = list(set1.symmetric_difference(set2))
    missing_wv.sort()
    print(">>> Unmatched lines...................:", missing_wv)

    # display results
    if abs(debugplot) % 10 != 0:
        from numina.array.display.matplotlib_qt import plt
        fig = plt.figure()
        if geometry is not None:
            x_geom, y_geom, dx_geom, dy_geom = geometry
            mngr = plt.get_current_fig_manager()
            mngr.window.setGeometry(x_geom, y_geom, dx_geom, dy_geom)

        # residuals
        ax2 = fig.add_subplot(2, 1, 1)
        ax2.plot(xresid, yresid, 'o')
        ax2.set_ylabel('Offset ' + r'($\AA$)')
        ax2.yaxis.label.set_size(10)
        ax2.set_title(title, **{'size': 10})
        xwv = fun_wv(np.arange(naxis1) + 1.0, crpix1, crval1, cdelt1)
        ax2.plot(xwv, polyres(xwv), '-')
        ax2.text(1, 0, 'CDELT1 (' + r'$\AA$' + '/pixel)=' + str(cdelt1),
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 transform=ax2.transAxes)
        ax2.text(0, 0, 'Wavelength ' + r'($\AA$) --->',
                 horizontalalignment='left',
                 verticalalignment='bottom',
                 transform=ax2.transAxes)
        ax2.text(0, 1, 'median=' + str(round(ysummary['median'], 4)) +
                 r' $\AA$',
                 horizontalalignment='left',
                 verticalalignment='top',
                 transform=ax2.transAxes)
        ax2.text(1, 1, 'robust_std=' + str(round(ysummary['robust_std'], 4)) +
                 r' $\AA$',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=ax2.transAxes)

        # median spectrum and peaks
        xmin = min(xwv)
        xmax = max(xwv)
        dx = xmax - xmin
        xmin -= dx / 80
        xmax += dx / 80
        ymin = min(sp)
        ymax = max(sp)
        dy = ymax - ymin
        ymin -= dy/20
        ymax += dy/20
        ax1 = fig.add_subplot(2, 1, 2, sharex=ax2)
        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([ymin, ymax])
        ax1.plot(xwv, sp)
        ax1.plot(ixpeaks_wv, sp[ixpeaks], 'o', label="initial location")
        ax1.plot(fxpeaks_wv, sp[ixpeaks], 'o', label="refined location")
        ax1.set_ylabel('Counts')
        ax1.yaxis.label.set_size(10)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')
        for i in range(len(ixpeaks)):
            # identified lines
            if wv_verified_all_peaks[i] > 0:
                ax1.text(fxpeaks_wv[i], sp[ixpeaks[i]],
                         wv_verified_all_peaks[i], fontsize=8,
                         horizontalalignment='center')
            # estimated wavelength from initial calibration
            estimated_wv = fun_wv(fxpeaks[i] + 1, crpix1, crval1, cdelt1)
            estimated_wv = str(round(estimated_wv, 4))
            ax1.text(fxpeaks_wv[i], 0,  # spmedian[ixpeaks[i]],
                     estimated_wv, fontsize=8, color='grey',
                     rotation='vertical',
                     horizontalalignment='center',
                     verticalalignment='top')
        if len(missing_wv) > 0:
            tmp = [float(wv) for wv in missing_wv]
            ax1.vlines(tmp, ymin=ymin, ymax=ymax,
                       colors='grey', linestyles='dotted',
                       label='missing lines')
        ax1.legend()
        pause_debugplot(debugplot, pltshow=True)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(prog='check_wlcalib')
    # positional parameters
    parser.add_argument("filename",
                        help="FITS image containing the spectra",
                        type=argparse.FileType('r'))
    parser.add_argument("--scans", required=True,
                        help="Tuple ns1[,ns2] (from 1 to NAXIS2)")
    parser.add_argument("--wv_master_file", required=True,
                        help="TXT file containing wavelengths",
                        type=argparse.FileType('r'))
    parser.add_argument("--geometry",
                        help="tuple x,y,dx,dy",
                        default="0,0,640,480")
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                             " (default=0)",
                        type=int, default=12,
                        choices=DEBUGPLOT_CODES)

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

    # read FITS file
    with fits.open(args.filename) as hdulist:
        image2d_header = hdulist[0].header
        image2d = hdulist[0].data
    if image2d.ndim == 1:
        naxis1 = image2d.shape[0]
        naxis2 = 1
    elif image2d.ndim == 2:
        naxis2, naxis1 = image2d.shape
    else:
        raise ValueError("Unexpected image dimensions!")
    crpix1 = image2d_header['crpix1']
    crval1 = image2d_header['crval1']
    cdelt1 = image2d_header['cdelt1']
    print('* Input file:', args.filename.name)
    print('>>> NAXIS1:', naxis1)
    print('>>> NAXIS2:', naxis2)
    print('>>> CRPIX1:', crpix1)
    print('>>> CRVAL1:', crval1)
    print('>>> CDELT1:', cdelt1)

    if 1 <= ns1 <= ns2 <= naxis2:
        if ns1 == ns2 == 1 and image2d.ndim == 1:
            spmedian = np.copy(image2d[:])
        else:
            # extract spectrum
            spmedian = np.median(image2d[(ns1-1):ns2], axis=0)
    else:
        raise ValueError("Invalid scan numbers")

    # read list of expected arc lines
    master_table = np.genfromtxt(args.wv_master_file)
    wv_master = master_table[:, 0]
    if abs(args.debugplot) in (21, 22):
        print('wv_master:', wv_master)

    # define plot title
    title = 'fitsfile: ' + os.path.basename(args.filename.name) + \
            ' [' + str(ns1) + ',' + str(ns2) + ']\n' + \
            'wv_master: ' + os.path.basename(args.wv_master_file.name)

    # check the wavelength calibration
    check_sp(sp=spmedian,
             crpix1=crpix1,
             crval1=crval1,
             cdelt1=cdelt1,
             wv_master=wv_master,
             title=title,
             geometry=geometry,
             debugplot=args.debugplot)


if __name__ == "__main__":

    main()
