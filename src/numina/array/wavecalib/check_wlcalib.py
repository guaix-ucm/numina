#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Supervised quality control of wavelength calibration"""

import argparse
import astropy.io.fits as fits
import decimal
import numpy as np
import os

from numina.array.stats import summary
from numina.array.display.iofunctions import readi
from numina.array.display.iofunctions import readf
from numina.array.display.iofunctions import readc
from numina.array.display.matplotlib_qt import set_window_geometry
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.polfit_residuals \
    import polfit_residuals_with_sigma_rejection
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


def check_wlcalib_sp(sp, crpix1, crval1, cdelt1, wv_master,
                     coeff_ini=None, naxis1_ini=None,
                     min_nlines_to_refine=0,
                     interactive=False,
                     threshold=0,
                     nwinwidth_initial=7,
                     nwinwidth_refined=5,
                     ntimes_match_wv=2,
                     poldeg_residuals=1,
                     times_sigma_reject=5,
                     use_r=False,
                     title=None,
                     remove_null_borders=True,
                     ylogscale=False,
                     geometry=None,
                     pdf=None,
                     debugplot=0):
    """Check wavelength calibration of the provided spectrum.

    Parameters
    ----------
    sp : numpy array
        Wavelength calibrated spectrum.
    crpix1: float
        CRPIX1 keyword.
    crval1: float
        CRVAL1 keyword.
    cdelt1: float
        CDELT1 keyword.
    wv_master: numpy array
        Array with the detailed list of expected arc lines.
    coeff_ini : array like
        Coefficients initially employed to obtain the wavelength
        calibration of the provided spectrum. When this coefficients
        are provided, this function computes a refined version of
        them, incorporating the corrections derived from the fit to
        the residuals.
    naxis1_ini : int
        NAXIS1 in original spectrum employed to fit the initial
        wavelength calibration.
    min_nlines_to_refine : int
        Minimum number of identified lines necessary to perform the
        wavelength calibration refinement. If zero, no minimum number
        is required.
    interactive : bool
        If True, the function allows the user to modify the residuals
        fit.
    threshold : float
        Minimum signal in the peaks.
    nwinwidth_initial : int
        Width of the window where each peak must be initially found.
    nwinwidth_refined : int
        Width of the window where each peak must be refined.
    ntimes_match_wv : float
        Times CDELT1 to match measured and expected wavelengths.
    poldeg_residuals : int
        Polynomial degree for fit to residuals.
    times_sigma_reject : float or None
        Number of times the standard deviation to reject points
        iteratively. If None, the fit does not reject any point.
    use_r : bool
        If True, additional statistical analysis is performed using R.
    title : string
        Plot title.
    remove_null_borders : bool
        If True, remove leading and trailing zeros in spectrum.
    ylogscale : bool
        If True, the spectrum is displayed in logarithmic units. Note
        that this is only employed for display purposes. The line peaks
        are found in the original spectrum.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the window geometry.
    pdf : PdfFile object or None
        If not None, output is sent to PDF file.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    coeff_refined : numpy array
        Refined version of the initial wavelength calibration
        coefficients. These coefficients are computed only when
        the input parameter 'coeff_ini' is not None.

    """

    # protections
    if type(sp) is not np.ndarray:
        raise ValueError("sp must be a numpy.ndarray")
    elif sp.ndim != 1:
        raise ValueError("sp.ndim is not 1")

    if coeff_ini is None and naxis1_ini is None:
        pass
    elif coeff_ini is not None and naxis1_ini is not None:
        pass
    else:
        raise ValueError("coeff_ini and naxis1_ini must be simultaneously "
                         "None of both different from None")

    # check that interactive use takes place when plotting
    if interactive:
        if abs(debugplot) % 10 == 0:
            raise ValueError("ERROR: interative use of this function is not "
                             "possible when debugplot=", debugplot)

   # interactive and pdf are incompatible
    if interactive:
        if pdf is not None:
            raise ValueError("ERROR: interactive use of this function is not "
                             "possible when pdf is not None")

    # display list of expected arc lines
    if abs(debugplot) in (21, 22):
        print('wv_master:', wv_master)

    # determine spectrum length
    naxis1 = sp.shape[0]

    # define default values in case no useful lines are identified
    fxpeaks = np.array([])
    ixpeaks_wv = np.array([])
    fxpeaks_wv = np.array([])
    wv_verified_all_peaks = np.array([])
    nlines_ok = 0
    xresid = np.array([], dtype=float)
    yresid = np.array([], dtype=float)
    reject = np.array([], dtype=bool)
    polyres = np.polynomial.Polynomial([0])
    poldeg_effective = 0
    ysummary = summary(np.array([]))
    local_ylogscale = ylogscale

    # find initial line peaks
    ixpeaks = find_peaks_spectrum(sp,
                                  nwinwidth=nwinwidth_initial,
                                  threshold=threshold)
    npeaks = len(ixpeaks)

    if npeaks > 0:
        # refine location of line peaks
        fxpeaks, sxpeaks = refine_peaks_spectrum(
            sp, ixpeaks,
            nwinwidth=nwinwidth_refined,
            method="gaussian"
        )
        ixpeaks_wv = fun_wv(ixpeaks + 1, crpix1, crval1, cdelt1)
        fxpeaks_wv = fun_wv(fxpeaks + 1, crpix1, crval1, cdelt1)

        # match peaks with expected arc lines
        delta_wv_max = ntimes_match_wv * cdelt1
        wv_verified_all_peaks = match_wv_arrays(
            wv_master,
            fxpeaks_wv,
            delta_wv_max=delta_wv_max
        )

    loop = True

    while loop:

        if npeaks > 0:

            lines_ok = np.where(wv_verified_all_peaks > 0)
            nlines_ok = len(lines_ok[0])

            # there are matched lines
            if nlines_ok > 0:
                # compute residuals
                xresid = fxpeaks_wv[lines_ok]
                yresid = wv_verified_all_peaks[lines_ok] - fxpeaks_wv[lines_ok]

                # determine effective polynomial degree
                if nlines_ok > poldeg_residuals:
                    poldeg_effective = poldeg_residuals
                else:
                    poldeg_effective = nlines_ok  - 1

                # fit polynomial to residuals
                polyres, yresres, reject = \
                    polfit_residuals_with_sigma_rejection(
                        x=xresid,
                        y=yresid,
                        deg=poldeg_effective,
                        times_sigma_reject=times_sigma_reject,
                        use_r=use_r,
                        debugplot=0
                    )
                ysummary = summary(yresres)
            else:
                polyres = np.polynomial.Polynomial([0.0])

        list_wv_found = [str(round(wv, 4))
                         for wv in wv_verified_all_peaks if wv != 0]
        list_wv_master = [str(round(wv, 4)) for wv in wv_master]
        set1 = set(list_wv_master)
        set2 = set(list_wv_found)
        missing_wv = list(set1.symmetric_difference(set2))
        missing_wv.sort()
        if abs(debugplot) >= 10:
            print('-' * 79)
            print(">>> Number of arc lines in master file:", len(wv_master))
        if abs(debugplot) in [21, 22]:
            print(">>> Unmatched lines...................:", missing_wv)
        elif abs(debugplot) >= 10:
            print(">>> Number of unmatched lines.........:", len(missing_wv))
        if abs(debugplot) >= 10:
            print(">>> Number of line peaks found........:", npeaks)
            print(">>> Number of identified lines........:", nlines_ok)
            print(">>> Number of unmatched lines.........:", len(missing_wv))
            print(">>> Polynomial degree in residuals fit:", poldeg_effective)
            print(">>> Polynomial fit to residuals.......:\n", polyres)

        # display results
        if (abs(debugplot) % 10 != 0) or (pdf is not None):
            from numina.array.display.matplotlib_qt import plt
            if pdf is not None:
                fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
            else:
                fig = plt.figure()
            set_window_geometry(geometry)

            # residuals
            ax2 = fig.add_subplot(2, 1, 1)
            if nlines_ok > 0:
                ymin = min(yresid)
                ymax = max(yresid)
                dy = ymax - ymin
                if dy > 0:
                    ymin -= dy/20
                    ymax += dy/20
                else:
                    ymin -= 0.5
                    ymax += 0.5
            else:
                ymin = -1.0
                ymax = 1.0
            ax2.set_ylim(ymin, ymax)
            if nlines_ok > 0:
                ax2.plot(xresid, yresid, 'o')
                ax2.plot(xresid[reject], yresid[reject], 'o', color='tab:gray')
            ax2.set_ylabel('Offset ' + r'($\AA$)')
            ax2.yaxis.label.set_size(10)
            if title is not None:
                ax2.set_title(title, **{'size': 12})
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
            ax2.text(0, 1, 'median=' +
                     str(round(ysummary['median'], 4)) + r' $\AA$',
                     horizontalalignment='left',
                     verticalalignment='top',
                     transform=ax2.transAxes)
            ax2.text(0.5, 1, 'npoints (total / used / removed)',
                     horizontalalignment='center',
                     verticalalignment='top',
                     transform=ax2.transAxes)
            ax2.text(0.5, 0.92,
                     str(ysummary['npoints']) + ' / ' +
                     str(ysummary['npoints'] - sum(reject)) + ' / ' +
                     str(sum(reject)),
                     horizontalalignment='center',
                     verticalalignment='top',
                     transform=ax2.transAxes)
            ax2.text(1, 1, 'robust_std=' +
                     str(round(ysummary['robust_std'], 4)) + r' $\AA$',
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=ax2.transAxes)

            # median spectrum and peaks
            # remove leading and trailing zeros in spectrum when requested
            if remove_null_borders:
                nonzero = np.nonzero(sp)[0]
                j1 = nonzero[0]
                j2 = nonzero[-1]
                xmin = xwv[j1]
                xmax = xwv[j2]
            else:
                xmin = min(xwv)
                xmax = max(xwv)
            dx = xmax - xmin
            if dx > 0:
                xmin -= dx / 80
                xmax += dx / 80
            else:
                xmin -= 0.5
                xmax += 0.5

            if local_ylogscale:
                spectrum = sp - sp.min() + 1.0
                spectrum = np.log10(spectrum)
                ymin = spectrum[ixpeaks].min()
            else:
                spectrum = sp.copy()
                ymin = min(spectrum)
            ymax = max(spectrum)
            dy = ymax - ymin
            if dy > 0:
                ymin -= dy/20
                ymax += dy/20
            else:
                ymin -= 0.5
                ymax += 0.5
            ax1 = fig.add_subplot(2, 1, 2, sharex=ax2)
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.plot(xwv, spectrum)
            if npeaks > 0:
                ax1.plot(ixpeaks_wv, spectrum[ixpeaks], 'o',
                         fillstyle='none', label="initial location")
                ax1.plot(fxpeaks_wv, spectrum[ixpeaks], 'o',
                         fillstyle='none', label="refined location")
                lok = wv_verified_all_peaks > 0
                ax1.plot(fxpeaks_wv[lok], spectrum[ixpeaks][lok], 'go',
                         label="valid line")
            if local_ylogscale:
                ax1.set_ylabel('~ log10(number of counts)')
            else:
                ax1.set_ylabel('number of counts')
            ax1.yaxis.label.set_size(10)
            ax1.xaxis.tick_top()
            ax1.xaxis.set_label_position('top')
            for i in range(len(ixpeaks)):
                # identified lines
                if wv_verified_all_peaks[i] > 0:
                    ax1.text(fxpeaks_wv[i], spectrum[ixpeaks[i]],
                             str(wv_verified_all_peaks[i]) +
                             '(' + str(i + 1) + ')',
                             fontsize=8,
                             horizontalalignment='center')
                else:
                    ax1.text(fxpeaks_wv[i], spectrum[ixpeaks[i]],
                             '(' + str(i + 1) + ')',
                             fontsize=8,
                             horizontalalignment='center')
                # estimated wavelength from initial calibration
                if npeaks > 0:
                    estimated_wv = fun_wv(fxpeaks[i] + 1,
                                          crpix1, crval1, cdelt1)
                    estimated_wv = str(round(estimated_wv, 4))
                    ax1.text(fxpeaks_wv[i], ymin,  # spmedian[ixpeaks[i]],
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
            if pdf is not None:
                pdf.savefig()
            else:
                if debugplot in [-22, -12, 12, 22]:
                    pause_debugplot(
                        debugplot=debugplot,
                        optional_prompt='Zoom/Unzoom or ' +
                                        'press RETURN to continue...',
                        pltshow=True
                    )
                else:
                    pause_debugplot(debugplot=debugplot, pltshow=True)

            # display results and request next action
            if interactive:
                print('Recalibration menu')
                print('------------------')
                print('[d] (d)elete all the identified lines')
                print('[r] (r)estart from begining')
                print('[a] (a)utomatic line inclusion')
                print('[l] toggle (l)ogarithmic scale on/off')
                print('[p] modify (p)olynomial degree')
                print('[o] (o)utput data with identified line peaks')
                print('[x] e(x)it without additional changes')
                print('[#] from 1 to ' + str(len(ixpeaks)) +
                      ' --> modify line #')
                ioption = readi('Option', default='x',
                                minval=1, maxval=len(ixpeaks),
                                allowed_single_chars='adloprx')
                if ioption == 'd':
                    wv_verified_all_peaks = np.zeros(npeaks)
                elif ioption == 'r':
                    delta_wv_max = ntimes_match_wv * cdelt1
                    wv_verified_all_peaks = match_wv_arrays(
                        wv_master,
                        fxpeaks_wv,
                        delta_wv_max=delta_wv_max
                    )
                elif ioption == 'a':
                    fxpeaks_wv_corrected = np.zeros_like(fxpeaks_wv)
                    for i in range(npeaks):
                        fxpeaks_wv_corrected[i] = fxpeaks_wv[i] + \
                            polyres(fxpeaks_wv[i])
                    delta_wv_max = ntimes_match_wv * cdelt1
                    wv_verified_all_peaks = match_wv_arrays(
                        wv_master,
                        fxpeaks_wv_corrected,
                        delta_wv_max=delta_wv_max
                    )
                elif ioption == 'l':
                    if local_ylogscale:
                        local_ylogscale = False
                    else:
                        local_ylogscale = True
                elif ioption == 'p':
                    poldeg_residuals = readi('New polynomial degree',
                                             minval=0)
                elif ioption == 'o':
                    for i in range(len(ixpeaks)):
                        # identified lines
                        if wv_verified_all_peaks[i] > 0:
                            print(wv_verified_all_peaks[i],
                                  spectrum[ixpeaks[i]])
                elif ioption == 'x':
                    loop = False
                else:
                    print(wv_master)
                    expected_value = fxpeaks_wv[ioption - 1] + \
                                     polyres(fxpeaks_wv[ioption - 1])
                    print(">>> Current expected wavelength: ", expected_value)
                    delta_wv_max = ntimes_match_wv * cdelt1
                    close_value = match_wv_arrays(
                        wv_master,
                        np.array([expected_value]),
                        delta_wv_max=delta_wv_max)
                    newvalue = readf('New value (0 to delete line)',
                                     default=close_value[0])
                    wv_verified_all_peaks[ioption - 1] = newvalue
            else:
                loop = False
        else:
            loop = False

    # refined wavelength calibration coefficients
    if coeff_ini is not None:
        npoints_total = len(xresid)
        npoints_removed = sum(reject)
        npoints_used = npoints_total - npoints_removed
        if abs(debugplot) >= 10:
            print('>>> Npoints (total / used / removed)..:',
                npoints_total, npoints_used, npoints_removed)
        if npoints_used < min_nlines_to_refine:
            print('Warning: number of lines insuficient to refine '
                  'wavelength calibration!')
            copc = 'n'
        else:
            if interactive:
                copc = readc('Refine wavelength calibration coefficients: '
                             '(y)es, (n)o', default='y', valid='yn')
            else:
                copc = 'y'
        if copc == 'y':
            coeff_refined = update_poly_wlcalib(
                    coeff_ini=coeff_ini,
                    coeff_residuals=polyres.coef,
                    naxis1_ini=naxis1_ini,
                    debugplot=0
                )
        else:
            coeff_refined = np.array(coeff_ini)
    else:
        coeff_refined = None

    if abs(debugplot) % 10 != 0:
        if coeff_refined is not None:
            for idum, fdum in \
                    enumerate(zip(coeff_ini, coeff_refined)):
                print(f">>> coef#{idum}:  ", end='')
                print(f"{decimal.Decimal(fdum[0]):+.8E}  -->  {decimal.Decimal(fdum[1]):+.8E}")

    return coeff_refined


def update_poly_wlcalib(coeff_ini, coeff_residuals, naxis1_ini, debugplot):
    """Update wavelength calibration polynomial using the residuals fit.

    The idea is to repeat the original fit using the information
    previously computed with the function check_wlcalib_sp() in this
    module.

    Parameters
    ----------
    coeff_ini : array like (floats)
        Coefficients corresponding to the initial wavelength
        calibration.
    coeff_residuals: array like (floats)
        Coefficients corresponding to the fit performed by the
        function check_wlcalib_sp() in this module.
    naxis1_ini : int
        NAXIS1 in original spectrum employed to fit the initial
        wavelength calibration.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    coeff_end : numpy array (floats)
        Updated coefficients.

    """

    # define initial wavelength calibration polynomial (use generic
    # code valid for lists of numpy.arrays)
    coeff = []
    for fdum in coeff_ini:
        coeff.append(fdum)
    poly_ini = np.polynomial.Polynomial(coeff)
    poldeg_wlcalib = len(coeff) - 1

    # return initial polynomial when there is no need to compute an
    # updated version
    if len(coeff_residuals) == 0:
        return poly_ini.coef
    else:
        if np.count_nonzero(poly_ini.coef) == 0:
            return poly_ini.coef

    # define polynomial corresponding to the residuals fit carried
    # out by check_wlcalib_sp()
    coeff = []
    for fdum in coeff_residuals:
        coeff.append(fdum)
    poly_residuals = np.polynomial.Polynomial(coeff)

    # define new points to be fitted
    xfit = np.zeros(naxis1_ini)
    yfit = np.zeros(naxis1_ini)
    for i in range(naxis1_ini):
        xfit[i] = float(i + 1)
        wv_tmp = poly_ini(xfit[i])
        yfit[i] = wv_tmp + poly_residuals(wv_tmp)

    # fit to get the updated polynomial
    if len(xfit) > poldeg_wlcalib:
        poldeg_effective = poldeg_wlcalib
    else:
        poldeg_effective = len(xfit) - 1
    poly_updated, ydum = polfit_residuals(
        x=xfit,
        y=yfit,
        deg=poldeg_effective,
        debugplot=debugplot
    )

    # return coefficients of updated polynomial
    return poly_updated.coef


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser()

    # positional parameters
    parser.add_argument("filename",
                        help="FITS image containing the spectra",
                        type=argparse.FileType('rb'))
    parser.add_argument("--scans", required=True,
                        help="Tuple ns1[,ns2] (from 1 to NAXIS2) to compute "
                             "median spectrum")
    parser.add_argument("--wv_master_file", required=True,
                        help="TXT file containing wavelengths",
                        type=argparse.FileType('rt'))

    # optional arguments
    parser.add_argument("--interactive",
                        help="Allows the user to modify de residuals fit",
                        action="store_true")
    parser.add_argument("--threshold",
                        help="Minimum signal in the line peaks (default=0)",
                        default=0, type=float)
    parser.add_argument("--nwinwidth_initial",
                        help="Window width where each peak must be found "
                             "(default=7)",
                        type=int, default=7)
    parser.add_argument("--nwinwidth_refined",
                        help="Window width where each peak must be refined "
                             "(default=5)",
                        type=int, default=5)
    parser.add_argument("--ntimes_match_wv",
                        help="Times CDELT1 to match measured and expected "
                             "wavelengths (default 2)",
                        default=2, type=float)
    parser.add_argument("--poldeg_residuals",
                        help="Polynomial degree for fit to residuals "
                             "(default 1)",
                        default=1, type=int)
    parser.add_argument("--times_sigma_reject",
                        help="Times the standard deviation to reject points "
                             "iteratively in the fit to residuals ("
                             "default=5)",
                        default=5, type=float)
    parser.add_argument("--use_r",
                        help="Perform additional statistical analysis with R",
                        action="store_true")
    parser.add_argument("--remove_null_borders",
                        help="Remove leading and trailing zeros in spectrum",
                        action="store_true")
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
    check_wlcalib_sp(sp=spmedian,
                     crpix1=crpix1,
                     crval1=crval1,
                     cdelt1=cdelt1,
                     wv_master=wv_master,
                     interactive=args.interactive,
                     threshold=args.threshold,
                     nwinwidth_initial=args.nwinwidth_initial,
                     nwinwidth_refined=args.nwinwidth_refined,
                     ntimes_match_wv=args.ntimes_match_wv,
                     poldeg_residuals=args.poldeg_residuals,
                     times_sigma_reject=args.times_sigma_reject,
                     use_r=args.use_r,
                     title=title,
                     remove_null_borders=args.remove_null_borders,
                     geometry=geometry,
                     debugplot=args.debugplot)


if __name__ == "__main__":

    main()
