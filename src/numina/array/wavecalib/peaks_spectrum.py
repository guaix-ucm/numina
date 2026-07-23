#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import numpy as np
from numpy.polynomial import Polynomial

from ..display.matplotlib_qt import set_window_geometry
from ..display.pause_debugplot import pause_debugplot


def find_highest_peaks_spectrum(
    sx, nmaxpeaks, nclean_around_peak, nwinwidth, threshold=0, nborder_to_ignore=0, debugplot=0, jupyter=False
):
    """Find a fixed number of highest peaks in 1D array.

    Peaks are found iteratively. Every time a new peak is found, the area
    around this new peak is cleaned (set to zero) in order to avoid finding
    the same peak again or finding a peak that is too close to it.
    The area to be cleaned is defined by the parameter nclean_around_peak.
    The width of the window where each peak must be found is defined by
    the parameter nwinwidth.

    The borders of the input array can be ignored by setting the parameter
    nborder_to_ignore to a positive value. In this case, the first and last
    nborder_to_ignore pixels of the input array will be set to zero before
    starting the peak search. This is useful when the borders of the array
    contain artifacts that could be mistaken for peaks.

    Parameters
    ----------
    sx : 1d numpy array, floats
        Input array.
    nmaxpeaks : int
        Maximum number of peaks to find.
    nclean_around_peak : int
        Number of pixels to clean around each peak.
    nwinwidth : int
        Width of the window where each peak must be found.
    threshold : float
        Minimum signal in the peaks.
    nborder_to_ignore : int
        Number of pixels to ignore at the borders of the input array.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses
    jupyter : bool
        If True, the plots will be displayed in a Jupyter notebook.

    Returns
    -------
    ixpeaks : 1d numpy array, int
        Peak locations, in array coordinates (integers).
        The peaks are sorted in ascending order of their array coordinates.
    """
    # check input parameters
    if len(sx) < nwinwidth:
        raise ValueError("Input array length is smaller than nwinwidth")
    if nmaxpeaks < 1:
        raise ValueError("nmaxpeaks must be at least 1")
    if nclean_around_peak < 0:
        raise ValueError("nclean_around_peak must be non-negative")
    if nwinwidth < 1:
        raise ValueError("nwinwidth must be at least 1")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if len(sx) < nmaxpeaks:
        raise ValueError("Input array length is smaller than nmaxpeaks")
    if nborder_to_ignore < 0:
        raise ValueError("nborder_to_ignore must be non-negative")
    if nborder_to_ignore * 2 >= len(sx):
        raise ValueError("nborder_to_ignore is too large for the input array length")

    # list to store the peak locations
    list_ixpeaks = []
    # make a copy of the input array to avoid modifying it
    sx_copy = np.copy(sx)
    # set to zero the first and last nborder_to_ignore pixels of the input array
    if nborder_to_ignore > 0:
        sx_copy[:nborder_to_ignore] = 0
        sx_copy[-nborder_to_ignore:] = 0
    # set the effective window width to the initial value
    nwinwidth_effective = nwinwidth
    for i in range(nmaxpeaks):
        loop = True
        while loop:
            # detect peaks in 1d array
            ixpeaks = find_peaks_spectrum(sx=sx_copy, nwinwidth=nwinwidth_effective, threshold=threshold, debugplot=0)
            # if no peaks are found, reduce the window width and try again
            if len(ixpeaks) < 1:
                nwinwidth_effective -= 2
                # if the window width is too small, take the maximum value as the peak
                if nwinwidth_effective < 1:
                    ixpeaks = [np.argmax(sx_copy)]
                    loop = False
            else:
                loop = False
        # peak values
        peak_values = sx_copy[ixpeaks]
        # find the index of the highest peak
        highest_peak_index = np.argmax(peak_values)
        # get the location of the highest peak
        highest_peak_location = ixpeaks[highest_peak_index]
        # add the highest peak location to the list of peaks
        list_ixpeaks.append(highest_peak_location)
        # clean the area around the highest peak
        start_clean = max(0, highest_peak_location - nclean_around_peak)
        end_clean = min(len(sx_copy), highest_peak_location + nclean_around_peak + 1)
        sx_copy[start_clean:end_clean] = 0

        if debugplot % 10 != 0:
            from numina.array.display.matplotlib_qt import plt

            fig, ax = plt.subplots()
            xdum = np.arange(len(sx_copy))
            ax.plot(xdum, sx, "-", color="gray")
            ax.plot(xdum, sx_copy, "C0.")
            ax.plot(list_ixpeaks, sx[list_ixpeaks], "go")
            ax.plot([highest_peak_location], sx[highest_peak_location], "ro")
            ax.set_xlabel("array index along Y axis")
            ax.set_ylabel("Signal")
            ax.set_title(f"Peak #{i+1}/{nmaxpeaks}, nwinwidth_eff={nwinwidth_effective}, nclean={nclean_around_peak}")
            if not jupyter:
                plt.show(block=False)
            plt.pause(0.001)
            pause_debugplot(debugplot)

    return np.array(np.sort(list_ixpeaks))


def find_peaks_spectrum(sx, nwinwidth, threshold=0, debugplot=0):
    """Find peaks in 1D array.

    The algorithm imposes that the signal at both sides of the peak
    decreases monotonically. Peaks are found within a window of width
    nwinwidth.

    Parameters
    ----------
    sx : 1d numpy array, floats
        Input array.
    nwinwidth : int
        Width of the window where each peak must be found.
    threshold : float
        Minimum signal in the peaks.
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
    ixpeaks : 1d numpy array, int
        Peak locations, in array coordinates (integers).

    """

    if not isinstance(sx, np.ndarray):
        raise ValueError("sx=" + str(sx) + " must be a numpy.ndarray")
    elif sx.ndim != 1:
        raise ValueError("sx.ndim=" + str(sx.ndim) + " must be 1")

    sx_shape = sx.shape
    nmed = nwinwidth // 2

    if debugplot >= 10:
        print("find_peaks_spectrum> sx shape......:", sx_shape)
        print("find_peaks_spectrum> nwinwidth.....:", nwinwidth)
        print("find_peaks_spectrum> nmed..........:", nmed)
        print("find_peaks_spectrum> data_threshold:", threshold)
        print("find_peaks_spectrum> the first and last", nmed, "pixels will be ignored")

    xpeaks = []  # list to store the peaks

    if sx_shape[0] < nwinwidth:
        print("find_peaks_spectrum> sx shape......:", sx_shape)
        print("find_peaks_spectrum> nwinwidth.....:", nwinwidth)
        raise ValueError("sx.shape < nwinwidth")

    i = nmed
    while i < sx_shape[0] - nmed:
        if sx[i] > threshold:
            peak_ok = True
            j = 0
            loop = True
            while loop:
                if sx[i - nmed + j] > sx[i - nmed + j + 1]:
                    peak_ok = False
                j += 1
                loop = (j < nmed) and peak_ok
            if peak_ok:
                j = nmed + 1
                loop = True
                while loop:
                    if sx[i - nmed + j - 1] < sx[i - nmed + j]:
                        peak_ok = False
                    j += 1
                    loop = (j < nwinwidth) and peak_ok
            if peak_ok:
                xpeaks.append(i)
                i += nwinwidth - 1
            else:
                i += 1
        else:
            i += 1

    ixpeaks = np.array(xpeaks)

    if debugplot >= 10:
        print("find_peaks_spectrum> number of peaks found:", len(ixpeaks))
        print(ixpeaks)

    return ixpeaks


def refine_peaks_spectrum(sx, ixpeaks, nwinwidth, method=None, geometry=None, debugplot=0):
    """Refine line peaks in spectrum.

    Parameters
    ----------
    sx : 1d numpy array, floats
        Input array.
    ixpeaks : 1d numpy array, int
        Initial peak locations, in array coordinates (integers).
        These values can be the output from the function
        find_peaks_spectrum().
    nwinwidth : int
        Width of the window where each peak must be refined.
    method : string
        "poly2" : fit to a 2nd order polynomial
        "gaussian" : fit to a Gaussian
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the window geometry.
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
    fxpeaks : 1d numpy array, float
        Refined peak locations, in array coordinates.
    sxpeaks : 1d numpy array, float
        When fitting Gaussians, this array stores the fitted line
        widths (sigma). Otherwise, this array returns zeros.

    """

    nmed = nwinwidth // 2

    xfpeaks = np.zeros(len(ixpeaks))
    sfpeaks = np.zeros(len(ixpeaks))

    for iline in range(len(ixpeaks)):
        jmax = ixpeaks[iline]
        x_fit = np.arange(-nmed, nmed + 1, dtype=float)
        # prevent possible problem when fitting a line too near to any
        # of the borders of the spectrum
        j1 = jmax - nmed
        j2 = jmax + nmed + 1
        if j1 < 0:
            j1 = 0
            j2 = 2 * nmed + 1
            if j2 >= len(sx):
                raise ValueError("Unexpected j2=" + str(j2) + " value when len(sx)=" + str(len(sx)))
        if j2 >= len(sx):
            j2 = len(sx)
            j1 = j2 - (2 * nmed + 1)
            if j1 < 0:
                raise ValueError("Unexpected j1=" + str(j1) + " value when len(sx)=" + str(len(sx)))
        # it is important to create a copy in the next instruction in
        # order to avoid modifying the original array when normalizing
        # the data to be fitted
        y_fit = np.copy(sx[j1:j2].astype(float))
        sx_peak_flux = y_fit.max()
        if sx_peak_flux != 0:
            y_fit /= sx_peak_flux  # normalize to maximum value

        if method == "gaussian":
            # check that there are no negative or null values
            if y_fit.min() <= 0:
                if debugplot >= 10:
                    print("WARNING: negative or null value encountered" + " in refine_peaks_spectrum with gaussian.")
                    print("         Using poly2 method instead.")
                final_method = "poly2"
            else:
                final_method = "gaussian"
        else:
            final_method = method

        if final_method == "poly2":
            poly_funct = Polynomial.fit(x_fit, y_fit, 2)
            poly_funct = Polynomial.cast(poly_funct)
            coef = poly_funct.coef
            if len(coef) == 3:
                if coef[2] != 0:
                    refined_peak = -coef[1] / (2.0 * coef[2]) + jmax
                else:
                    refined_peak = 0.0 + jmax
            else:
                refined_peak = 0.0 + jmax
        elif final_method == "gaussian":
            poly_funct = Polynomial.fit(x_fit, np.log(y_fit), 2)
            poly_funct = Polynomial.cast(poly_funct)
            coef = poly_funct.coef
            if len(coef) == 3:
                if coef[2] != 0:
                    refined_peak = -coef[1] / (2.0 * coef[2]) + jmax
                else:
                    refined_peak = 0.0 + jmax
                if coef[2] >= 0:
                    sfpeaks[iline] = None
                else:
                    sfpeaks[iline] = np.sqrt(-1 / (2.0 * coef[2]))
            else:
                refined_peak = 0.0 + jmax
                sfpeaks[iline] = None
        else:
            raise ValueError("Invalid method=" + str(final_method) + " value")

        xfpeaks[iline] = refined_peak

        if debugplot % 10 != 0:
            from numina.array.display.matplotlib_qt import plt

            fig = plt.figure()
            set_window_geometry(geometry)
            ax = fig.add_subplot(111)
            xmin = x_fit.min() - 1
            xmax = x_fit.max() + 1
            ymin = 0
            ymax = y_fit.max() * 1.10
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("index around initial integer peak")
            ax.set_ylabel("Normalized number of counts")
            ax.set_title("Fit to line at array index " + str(jmax) + "\n(method=" + final_method + ")")
            plt.plot(x_fit, y_fit, "bo")
            x_plot = np.linspace(start=-nmed, stop=nmed, num=1000, dtype=float)
            if final_method == "poly2":
                y_plot = poly_funct(x_plot)
            elif final_method == "gaussian":
                amp = np.exp(coef[0] - coef[1] * coef[1] / (4 * coef[2]))
                x0 = -coef[1] / (2.0 * coef[2])
                sigma = np.sqrt(-1 / (2.0 * coef[2]))
                y_plot = amp * np.exp(-((x_plot - x0) ** 2) / (2 * sigma**2))
            else:
                raise ValueError("Invalid method=" + str(final_method) + " value")
            ax.plot(x_plot, y_plot, color="red")
            print("Refined peak location:", refined_peak)
            plt.show(block=False)
            plt.pause(0.001)
            pause_debugplot(debugplot)

    return xfpeaks, sfpeaks
