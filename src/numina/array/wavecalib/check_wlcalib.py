#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Supervised quality control of wavelength calibration"""

import logging
import argparse
import astropy.io.fits as fits
import astropy.units as u
import decimal
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
from rich_argparse import RichHelpFormatter
import sys

from numina.array.stats import summary
from numina.array.display.iofunctions import readi
from numina.array.display.iofunctions import readf
from numina.array.display.iofunctions import readc
from numina.array.display.matplotlib_utils import new_figure, parse_geometry, set_window_geometry
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.polfit_residuals import polfit_residuals_with_sigma_rejection
from .peaks_spectrum import find_peaks_spectrum
from .peaks_spectrum import refine_peaks_spectrum
from .peaks_spectrum import odd_window_width
from . import core
from .arccalibration import _wv_unit_and_label
from numina.tools.arg_file_is_new import arg_file_is_new
from numina.tools.initialize_script_with_args import NuminaScriptDefinition

from numina.array.display.pause_debugplot import DEBUGPLOT_CODES

_logger = logging.getLogger(__name__)

# command-line options renamed from underscores to hyphens; the old names
# are still accepted (but not displayed in the help) for backward
# compatibility
_RENAMED_OPTIONS = {
    "--wv_master_file": "--wv-master-file",
    "--nwinwidth_initial": "--nwinwidth-initial",
    "--nwinwidth_refined": "--nwinwidth-refined",
    "--ntimes_match_wv": "--ntimes-match-wv",
    "--poldeg_residuals": "--poldeg-residuals",
    "--times_sigma_reject": "--times-sigma-reject",
    "--use_r": "--use-r",
    "--remove_null_borders": "--remove-null-borders",
}


def _translate_renamed_options(argv):
    """Replace old command-line option names by the new ones.

    Parameters
    ----------
    argv : list of str
        Command-line arguments (excluding the program name).

    Returns
    -------
    argv_new : list of str
        Command-line arguments with the old option names (with
        underscores) replaced by the new ones (with hyphens). Both
        '--option value' and '--option=value' are handled.

    """
    argv_new = []
    for arg in argv:
        option, sep, value = arg.partition("=")
        if option in _RENAMED_OPTIONS:
            arg = _RENAMED_OPTIONS[option] + sep + value
        argv_new.append(arg)
    return argv_new


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
            if wv_unused[j - 1]:
                delta_wv = abs(wv_master[i] - wv_expected_all_peaks[j - 1])
                if delta_wv < delta_wv_max:
                    wv_verified_all_peaks[j - 1] = wv_master[i]
                    wv_unused[j - 1] = False
        else:
            delta_wv1 = abs(wv_master[i] - wv_expected_all_peaks[j - 1])
            delta_wv2 = abs(wv_master[i] - wv_expected_all_peaks[j])
            if delta_wv1 < delta_wv2:
                if delta_wv1 < delta_wv_max:
                    if wv_unused[j - 1]:
                        wv_verified_all_peaks[j - 1] = wv_master[i]
                        wv_unused[j - 1] = False
                    elif wv_unused[j]:
                        if delta_wv2 < delta_wv_max:
                            wv_verified_all_peaks[j] = wv_master[i]
                            wv_unused[j] = False
            else:
                if delta_wv2 < delta_wv_max:
                    if wv_unused[j]:
                        wv_verified_all_peaks[j] = wv_master[i]
                        wv_unused[j] = False
                    elif wv_unused[j - 1]:
                        if delta_wv1 < delta_wv_max:
                            wv_verified_all_peaks[j - 1] = wv_master[i]
                            wv_unused[j - 1] = False

    return wv_verified_all_peaks


def _number_of_decimals(values, max_decimals=10):
    """Maximum number of decimals needed to represent the input values.

    For each value, the number of decimals is the smallest one that
    reproduces the value within a relative tolerance of 1e-12 (this
    avoids spurious decimals due to the floating point
    representation, e.g. after a unit conversion).

    Parameters
    ----------
    values : array_like
        Input values.
    max_decimals : int
        Maximum number of decimals to be returned.

    Returns
    -------
    ndecimals : int
        Maximum number of decimals (between 0 and max_decimals).

    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0
    for ndecimals in range(max_decimals + 1):
        if np.allclose(np.round(values, ndecimals), values, rtol=1e-12, atol=0):
            return ndecimals
    return max_decimals


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


def check_wlcalib_sp(
    sp,
    crpix1,
    crval1,
    cdelt1,
    wv_master,
    coeff_ini=None,
    naxis1_ini=None,
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
    debugplot=0,
    wv_unit="Angstrom",
):
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
        It must be an odd integer >= 3.
    nwinwidth_refined : int
        Width of the window where each peak must be refined.
        It must be an odd integer >= 3.
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
    wv_unit : str, astropy.units.UnitBase or None
        Wavelength unit employed in the plot labels. If None,
        Angstrom is assumed.

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
        raise ValueError("coeff_ini and naxis1_ini must be simultaneously " "None of both different from None")

    # check that interactive use takes place when plotting
    if interactive:
        if abs(debugplot) % 10 == 0:
            raise ValueError("ERROR: interative use of this function is not " "possible when debugplot=", debugplot)

    # interactive and pdf are incompatible
    if interactive:
        if pdf is not None:
            raise ValueError("ERROR: interactive use of this function is not " "possible when pdf is not None")

    # LaTeX representation of the wavelength unit for the plot labels
    wv_unit, wv_unit_label = _wv_unit_and_label(wv_unit)

    # number of decimals employed to display wavelengths in the plot
    # (the maximum number found in the list of arc lines)
    wv_decimals = _number_of_decimals(wv_master)

    # display list of expected arc lines
    if abs(debugplot) in (21, 22):
        print("wv_master:", wv_master)

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
    offset_median = np.nan
    local_ylogscale = ylogscale

    # find initial line peaks
    ixpeaks = find_peaks_spectrum(sp, nwinwidth=nwinwidth_initial, threshold=threshold)
    npeaks = len(ixpeaks)

    if npeaks > 0:
        # refine location of line peaks
        fxpeaks, sxpeaks = refine_peaks_spectrum(sp, ixpeaks, nwinwidth=nwinwidth_refined, method="gaussian")
        ixpeaks_wv = fun_wv(ixpeaks + 1, crpix1, crval1, cdelt1)
        fxpeaks_wv = fun_wv(fxpeaks + 1, crpix1, crval1, cdelt1)

        # match peaks with expected arc lines
        delta_wv_max = ntimes_match_wv * cdelt1
        wv_verified_all_peaks = match_wv_arrays(wv_master, fxpeaks_wv, delta_wv_max=delta_wv_max)

    # wavelength at the center of each pixel (pixels from 1 to NAXIS1)
    xwv = fun_wv(np.arange(naxis1) + 1.0, crpix1, crval1, cdelt1)

    # wavelength range displayed in the figure (without the plot margins),
    # removing leading and trailing zeros in spectrum when requested
    nonzero = np.nonzero(sp)[0]
    if remove_null_borders and len(nonzero) > 0:
        wvmin_plot = xwv[nonzero[0]]
        wvmax_plot = xwv[nonzero[-1]]
    else:
        wvmin_plot = min(xwv)
        wvmax_plot = max(xwv)

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
                    poldeg_effective = nlines_ok - 1

                # fit polynomial to residuals
                polyres, yresres, reject = polfit_residuals_with_sigma_rejection(
                    x=xresid,
                    y=yresid,
                    deg=poldeg_effective,
                    times_sigma_reject=times_sigma_reject,
                    use_r=use_r,
                    debugplot=0,
                )
                ysummary = summary(yresres)
                # median offset (expected - measured wavelength) of the
                # lines employed in the fit; note that ysummary describes
                # the residuals around the fitted polynomial instead
                offset_median = np.median(yresid[~reject]) if np.any(~reject) else np.nan
            else:
                polyres = np.polynomial.Polynomial([0.0])
                offset_median = np.nan

        list_wv_found = [str(round(wv, 4)) for wv in wv_verified_all_peaks if wv != 0]
        list_wv_master = [str(round(wv, 4)) for wv in wv_master]
        set1 = set(list_wv_master)
        set2 = set(list_wv_found)
        missing_wv = list(set1.symmetric_difference(set2))
        missing_wv.sort()
        if abs(debugplot) >= 10:
            print("-" * 79)
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
            print(
                f">>> Polynomial fit to residuals.......:\n(offset in {wv_unit} vs. x = wavelength in {wv_unit})\n",
                polyres,
            )
            # offset predicted by the fit at the borders of the displayed
            # wavelength range
            for clabel, wv_border in zip(["minimum", "maximum"], [wvmin_plot, wvmax_plot]):
                print(
                    f">>> Offset at {clabel} wavelength......:\n{polyres(wv_border):.6g} {wv_unit} "
                    f"(wavelength = {wv_border:.{wv_decimals}f} {wv_unit})"
                )

        # display results
        if (abs(debugplot) % 10 != 0) or (pdf is not None):
            if pdf is not None:
                # the figure is only saved in the PDF file
                fig = new_figure(interactive=False, figsize=(11.69, 8.27), dpi=100)
            else:
                fig = new_figure()
                set_window_geometry(geometry)

            # residuals
            ax2 = fig.add_subplot(2, 1, 1)
            if nlines_ok > 0:
                ymin = min(yresid)
                ymax = max(yresid)
                dy = ymax - ymin
                if dy > 0:
                    ymin -= dy / 20
                    ymax += dy / 20
                else:
                    ymin -= 0.5
                    ymax += 0.5
            else:
                ymin = -1.0
                ymax = 1.0
            ax2.set_ylim(ymin, ymax)
            if nlines_ok > 0:
                ax2.plot(xresid, yresid, "o")
                ax2.plot(xresid[reject], yresid[reject], "o", color="tab:gray")
            ax2.set_ylabel(f"Offset ({wv_unit_label})")
            ax2.yaxis.label.set_size(10)
            if title is not None:
                ax2.set_title(title, **{"size": 12})
            ax2.plot(xwv, polyres(xwv), "-")
            ax2.text(
                1,
                0,
                f"CDELT1 ({wv_unit_label}/pixel)=" + str(cdelt1),
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=ax2.transAxes,
            )
            ax2.text(
                0,
                0,
                f"Wavelength ({wv_unit_label}) --->",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=ax2.transAxes,
            )
            ax2.text(
                0,
                1,
                "median offset=" + str(round(offset_median, 4)) + f" {wv_unit_label}",
                horizontalalignment="left",
                verticalalignment="top",
                transform=ax2.transAxes,
            )
            ax2.text(
                0.5,
                1,
                "npoints (total / used / removed)",
                horizontalalignment="center",
                verticalalignment="top",
                transform=ax2.transAxes,
            )
            ax2.text(
                0.5,
                0.92,
                str(ysummary["npoints"]) + " / " + str(ysummary["npoints"] - sum(reject)) + " / " + str(sum(reject)),
                horizontalalignment="center",
                verticalalignment="top",
                transform=ax2.transAxes,
            )
            ax2.text(
                1,
                1,
                "robust_std=" + str(round(ysummary["robust_std"], 4)) + f" {wv_unit_label}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax2.transAxes,
            )

            # median spectrum and peaks
            xmin = wvmin_plot
            xmax = wvmax_plot
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
                ymin -= dy / 20
                ymax += dy / 20
            else:
                ymin -= 0.5
                ymax += 0.5
            ax1 = fig.add_subplot(2, 1, 2, sharex=ax2)
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.plot(xwv, spectrum)
            if npeaks > 0:
                ax1.plot(ixpeaks_wv, spectrum[ixpeaks], "o", fillstyle="none", label="initial location")
                ax1.plot(fxpeaks_wv, spectrum[ixpeaks], "o", fillstyle="none", label="refined location")
                lok = wv_verified_all_peaks > 0
                ax1.plot(fxpeaks_wv[lok], spectrum[ixpeaks][lok], "go", label="valid line")
            if local_ylogscale:
                ax1.set_ylabel("log10(counts - min(counts) + 1)")
            else:
                ax1.set_ylabel("number of counts")
            ax1.yaxis.label.set_size(10)
            ax1.xaxis.tick_top()
            ax1.xaxis.set_label_position("top")
            for i in range(len(ixpeaks)):
                # identified lines
                if wv_verified_all_peaks[i] > 0:
                    ax1.text(
                        fxpeaks_wv[i],
                        spectrum[ixpeaks[i]],
                        f"{wv_verified_all_peaks[i]:.{wv_decimals}f}({i + 1})",
                        fontsize=8,
                        horizontalalignment="center",
                    )
                else:
                    ax1.text(
                        fxpeaks_wv[i],
                        spectrum[ixpeaks[i]],
                        "(" + str(i + 1) + ")",
                        fontsize=8,
                        horizontalalignment="center",
                    )
                # estimated wavelength from initial calibration
                if npeaks > 0:
                    estimated_wv = fun_wv(fxpeaks[i] + 1, crpix1, crval1, cdelt1)
                    estimated_wv = f"{estimated_wv:.{wv_decimals}f}"
                    ax1.text(
                        fxpeaks_wv[i],
                        ymin,  # spmedian[ixpeaks[i]],
                        estimated_wv,
                        fontsize=8,
                        color="grey",
                        rotation="vertical",
                        horizontalalignment="center",
                        verticalalignment="top",
                    )
            if len(missing_wv) > 0:
                tmp = [float(wv) for wv in missing_wv]
                ax1.vlines(tmp, ymin=ymin, ymax=ymax, colors="grey", linestyles="dotted", label="missing lines")
            ax1.legend()
            # the layout engine (instead of a single call to
            # fig.tight_layout) applies tight_layout at every draw, so that
            # it is recomputed when the window gets its final size (when it
            # is shown) or is resized, and also when saving the PDF file
            fig.set_layout_engine("tight")
            if pdf is not None:
                pdf.savefig(fig)
            else:
                if debugplot in [-22, -12, 12, 22]:
                    pause_debugplot(
                        debugplot=debugplot,
                        optional_prompt="Zoom/Unzoom or " + "press RETURN to continue...",
                        pltshow=True,
                        tight_layout=False,
                    )
                else:
                    pause_debugplot(debugplot=debugplot, pltshow=True, tight_layout=False)

            # display results and request next action
            if interactive:
                print("Recalibration menu")
                print("------------------")
                print("(d) delete all the identified lines")
                print("(r) restart from begining")
                print("(a) automatic line inclusion")
                print("(l) toggle logarithmic scale on/off")
                print("(p) modify polynomial degree")
                print("(o) output data with identified line peaks")
                print("(x) exit without additional changes")
                print("(#) from 1 to " + str(len(ixpeaks)) + " --> modify line #")
                ioption = readi("Option", default="x", minval=1, maxval=len(ixpeaks), allowed_single_chars="adloprx")
                if ioption == "d":
                    wv_verified_all_peaks = np.zeros(npeaks)
                elif ioption == "r":
                    delta_wv_max = ntimes_match_wv * cdelt1
                    wv_verified_all_peaks = match_wv_arrays(wv_master, fxpeaks_wv, delta_wv_max=delta_wv_max)
                elif ioption == "a":
                    fxpeaks_wv_corrected = np.zeros_like(fxpeaks_wv)
                    for i in range(npeaks):
                        fxpeaks_wv_corrected[i] = fxpeaks_wv[i] + polyres(fxpeaks_wv[i])
                    delta_wv_max = ntimes_match_wv * cdelt1
                    wv_verified_all_peaks = match_wv_arrays(wv_master, fxpeaks_wv_corrected, delta_wv_max=delta_wv_max)
                elif ioption == "l":
                    if local_ylogscale:
                        local_ylogscale = False
                    else:
                        local_ylogscale = True
                elif ioption == "p":
                    poldeg_residuals = readi("New polynomial degree", minval=0)
                elif ioption == "o":
                    for i in range(len(ixpeaks)):
                        # identified lines
                        if wv_verified_all_peaks[i] > 0:
                            print(wv_verified_all_peaks[i], spectrum[ixpeaks[i]])
                elif ioption == "x":
                    loop = False
                else:
                    print(wv_master)
                    expected_value = fxpeaks_wv[ioption - 1] + polyres(fxpeaks_wv[ioption - 1])
                    print(">>> Current expected wavelength: ", expected_value)
                    delta_wv_max = ntimes_match_wv * cdelt1
                    close_value = match_wv_arrays(wv_master, np.array([expected_value]), delta_wv_max=delta_wv_max)
                    newvalue = readf("New value (0 to delete line)", default=close_value[0])
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
            print(">>> Npoints (total / used / removed)..:", npoints_total, npoints_used, npoints_removed)
        if npoints_used < min_nlines_to_refine:
            _logger.warning("number of lines insufficient to refine the wavelength calibration")
            copc = "n"
        else:
            if interactive:
                copc = readc("Refine wavelength calibration coefficients: " "(y)es, (n)o", default="y", valid="yn")
            else:
                copc = "y"
        if copc == "y":
            coeff_refined = update_poly_wlcalib(
                coeff_ini=coeff_ini, coeff_residuals=polyres.coef, naxis1_ini=naxis1_ini, debugplot=0
            )
        else:
            coeff_refined = np.array(coeff_ini)
    else:
        coeff_refined = None

    if abs(debugplot) % 10 != 0:
        if coeff_refined is not None:
            for idum, fdum in enumerate(zip(coeff_ini, coeff_refined)):
                print(f">>> coef#{idum}:  ", end="")
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
    poly_updated, ydum = polfit_residuals(x=xfit, y=yfit, deg=poldeg_effective, debugplot=debugplot)

    # return coefficients of updated polynomial
    return poly_updated.coef


def main(args=None):
    """Check the wavelength calibration of a spectrum in a FITS file.

    The spectrum, obtained by collapsing the requested scans of the
    FITS image, must be calibrated with a linear wavelength scale
    defined by the keywords CRPIX1, CRVAL1 and CDELT1 (and optionally
    CUNIT1). The line peaks are compared with the wavelengths of the
    provided arc line list.

    Parameters
    ----------
    args : list of str or None
        Command-line arguments (excluding the program name). If None,
        sys.argv[1:] is used.

    """

    # parse command-line options
    parser = argparse.ArgumentParser(
        description="Supervised quality control of the wavelength calibration of a spectrum.",
        formatter_class=RichHelpFormatter,
    )

    # required parameters
    parser.add_argument("filename", help="FITS image containing the spectra", type=str)
    parser.add_argument("--scans", required=True, help="Tuple ns1[,ns2] (from 1 to NAXIS2)")
    parser.add_argument("--wv-master-file", required=True, help="ECSV/ASCII file containing wavelengths", type=str)

    # optional arguments
    parser.add_argument(
        "--method",
        help="collapsing method to compute the average spectrum (default='median', 'mean')",
        default="median",
        choices=["mean", "median"],
    )
    parser.add_argument("--interactive", help="Allows the user to modify the residuals fit", action="store_true")
    parser.add_argument("--threshold", help="Minimum signal in the line peaks (default=0)", default=0, type=float)
    parser.add_argument(
        "--nwinwidth-initial",
        help="Window width where each peak must be found (odd integer >= 3, default=7)",
        type=odd_window_width,
        default=7,
    )
    parser.add_argument(
        "--nwinwidth-refined",
        help="Window width where each peak must be refined (odd integer >= 3, default=5)",
        type=odd_window_width,
        default=5,
    )
    parser.add_argument(
        "--ntimes-match-wv",
        help="Times CDELT1 to match measured and expected wavelengths (default=2)",
        default=2,
        type=float,
    )
    parser.add_argument(
        "--poldeg-residuals", help="Polynomial degree for fit to residuals (default=1)", default=1, type=int
    )
    parser.add_argument(
        "--times-sigma-reject",
        help="Times the standard deviation to reject points iteratively in the fit to residuals (default=5)",
        default=5,
        type=float,
    )
    parser.add_argument("--use-r", help="Perform additional statistical analysis with R", action="store_true")
    parser.add_argument(
        "--remove-null-borders", help="Remove leading and trailing zeros in spectrum", action="store_true"
    )
    parser.add_argument(
        "--geometry",
        help="Window geometry: tuple x,y,dx,dy (default 0,0,640,480)",
        default="0,0,640,480",
        type=parse_geometry,
    )
    parser.add_argument("--pdffile", help="Output PDF file name", type=lambda x: arg_file_is_new(parser, x, mode="wb"))
    parser.add_argument(
        "--debugplot",
        help="Integer indicating plotting/debugging (default=12)",
        type=int,
        default=12,
        choices=DEBUGPLOT_CODES,
    )

    # accept the old option names (with underscores)
    if args is None:
        args = sys.argv[1:]
    args = _translate_renamed_options(args)

    # Include default arguments for common actions, and initialize console and logging
    myscript = NuminaScriptDefinition(parser, argv=args)
    args = myscript.args
    logger = myscript.logger

    # interactive use and PDF output are incompatible
    if args.pdffile is not None:
        # the file handle is only employed to check that the file is new
        pdffilename = args.pdffile.name
        args.pdffile.close()
        if args.interactive:
            os.remove(pdffilename)
            parser.error("--interactive and --pdffile cannot be used simultaneously")
    else:
        pdffilename = None

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

    # collapsed spectrum (1D or 2D images are valid)
    sp = core.collapsed_spectrum(fitsfile=args.filename, ns1=ns1, ns2=ns2, method=args.method, debugplot=0)

    # linear wavelength calibration (FITS convention: pixels from 1 to NAXIS1)
    header = fits.getheader(args.filename)
    wcs_values = {}
    for keyword in ["CRPIX1", "CRVAL1", "CDELT1"]:
        if keyword not in header:
            raise ValueError(f"Keyword {keyword} not found in '{args.filename}'")
        wcs_values[keyword] = header[keyword]
        logger.info(f"{keyword}: {wcs_values[keyword]}")
    crpix1, crval1, cdelt1 = wcs_values["CRPIX1"], wcs_values["CRVAL1"], wcs_values["CDELT1"]

    # read list of expected arc lines (all the lines)
    wv_master, wv_unit = core.read_wv_master_file(args.wv_master_file, lines="all", return_unit=True)

    # when CUNIT1 is defined, express the arc line wavelengths in that unit
    cunit1 = header.get("CUNIT1")
    if cunit1 is None:
        logger.info(f"CUNIT1 not found: assuming the unit of the line list ({wv_unit})")
    else:
        logger.info(f"CUNIT1: {cunit1}")
        try:
            header_unit = u.Unit(cunit1, format="fits")
        except ValueError:
            header_unit = u.Unit(cunit1)
        if header_unit != wv_unit:
            try:
                wv_master = (wv_master * wv_unit).to_value(header_unit)
            except u.UnitConversionError:
                raise ValueError(
                    f"Wavelength unit of the line list ({wv_unit}) cannot be converted to CUNIT1 ({cunit1})"
                ) from None
            logger.info(f"Line list wavelengths converted from {wv_unit} to {header_unit}")
            wv_unit = header_unit
    logger.debug(f"wv_master:\n{wv_master}")

    # define plot title
    title = (
        f"fitsfile: {os.path.basename(args.filename)} [{ns1},{ns2}]\n"
        f"wv_master: {os.path.basename(args.wv_master_file)}"
    )

    # check the wavelength calibration
    pdf = PdfPages(pdffilename) if pdffilename is not None else None
    try:
        check_wlcalib_sp(
            sp=sp,
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
            geometry=args.geometry,
            pdf=pdf,
            debugplot=args.debugplot,
            wv_unit=wv_unit,
        )
    finally:
        if pdf is not None:
            pdf.close()

    # Display goodbye message and save console log if recording is enabled
    myscript.goodbye_message_and_save_console()


if __name__ == "__main__":

    main()
