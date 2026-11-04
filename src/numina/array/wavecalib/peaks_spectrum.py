#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import logging

import warnings

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import OptimizeWarning, curve_fit, minimize_scalar
from scipy.special import erf, erfc

from ..display.matplotlib_qt import set_window_geometry
from ..display.pause_debugplot import pause_debugplot

_logger = logging.getLogger(__name__)


def odd_window_width(nwinwidth):
    """Check that a window width is an odd integer >= 3.

    The window used to find or refine a peak at pixel i is
    [i - nwinwidth // 2, i + nwinwidth // 2], which contains exactly
    nwinwidth pixels only when nwinwidth is odd.

    This function can also be used as the ``type`` of an argparse
    argument, since argparse reports the ValueError as a usage error.

    Parameters
    ----------
    nwinwidth : int or str
        Window width.

    Returns
    -------
    int
        Validated window width.

    Raises
    ------
    ValueError
        If nwinwidth is not an odd integer >= 3.
    """
    value = int(nwinwidth)
    if value < 3 or value % 2 == 0:
        raise ValueError(f"nwinwidth={nwinwidth} must be an odd integer >= 3")
    return value


def fwhm_nonparametric(x, y, debugplot=0):
    """
    Non-parametric FWHM: distance between the half-maximum crossings
    closest to the peak, using linear interpolation between samples.

    The spectrum must be continuum-subtracted and contain a single line
    in the given range.

    If the profile does not fall below half maximum on either side
    within the supplied range, the function returns None for the
    corresponding half-maximum crossing.

    Parameters
    ----------
    x, y : array_like
        Continuum-subtracted spectrum.
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
    fwhm : float or None
        Full width at half maximum, in the units of x. None if either
        half-maximum crossing is not found.
    x_left, x_right : float or None
        Positions of the half-maximum crossings. None if the profile does
        not fall below half maximum on the corresponding side.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ipk = np.argmax(y)  # index of the peak sample
    half = y[ipk] / 2.0  # half-maximum level

    # Left crossing: walk away from the peak until the flux drops to <= half
    i = ipk
    while i > 0 and y[i] > half:
        i -= 1
    if y[i] > half:
        if abs(debugplot) >= 10:
            _logger.warning("The line does not drop below half maximum on the left side; widen the range.")
        x_left = None
    else:
        # Linear interpolation between samples i and i + 1
        x_left = x[i] + (half - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])

    # Right crossing: same procedure towards increasing x
    j = ipk
    while j < len(y) - 1 and y[j] > half:
        j += 1
    if y[j] > half:
        if abs(debugplot) >= 10:
            _logger.warning("The line does not drop below half maximum on the right side; widen the range.")
        x_right = None
    else:
        # Linear interpolation between samples j - 1 and j
        x_right = x[j - 1] + (half - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1])

    if x_left is None or x_right is None:
        fwhm_np = None
    else:
        fwhm_np = x_right - x_left
    return fwhm_np, x_left, x_right


def line_moments(x, y, window=None, debugplot=0):
    """
    Flux-weighted moments of the line profile, treating the continuum-
    subtracted flux as a probability density:

        m0     = ∫ f dx                         (integrated flux)
        mu     = ∫ x f dx / m0                  (centroid)
        sigma² = ∫ (x - mu)² f dx / m0
        gamma1 = ∫ (x - mu)³ f dx / (m0 sigma³) (skewness)
        gamma2 = ∫ (x - mu)⁴ f dx / (m0 sigma⁴) - 3  (excess kurtosis)


    When the variance is zero or negative, the standard deviation, skewness and
    excess kurtosis are set to NaN.

    Parameters
    ----------
    x, y : array_like
        Spectrum; y must be continuum-subtracted.
    window : (float, float) or None, optional
        Integration interval in x. If None, the whole range is used.
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
    dict
        Keys: 'flux', 'centroid', 'sigma', 'skewness', 'kurtosis_excess'.

    Raises
    ------
    ValueError
        If the integrated flux is not positive (moments are undefined).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)

    # Restrict the integration to the requested window
    if window is not None:
        m = (x >= window[0]) & (x <= window[1])
        x, y = x[m], y[m]

    # Zeroth moment: integrated flux
    m0 = np.trapezoid(y, x)
    if m0 <= 0:
        raise ValueError(
            "Integrated flux is not positive; the moments are not defined. Check the window and the continuum."
        )

    # First moment: flux-weighted centroid
    mu = np.trapezoid(x * y, x) / m0

    # Second central moment: standard deviation of the profile
    var = np.trapezoid((x - mu) ** 2 * y, x) / m0
    if var < 0:
        if abs(debugplot) >= 10:
            _logger.warning("Negative variance encountered in line_moments; setting sigma to NaN.")
        sigma = np.nan
    else:
        sigma = np.sqrt(var)

    # Standardized third and fourth central moments
    if sigma > 0:
        skew = np.trapezoid((x - mu) ** 3 * y, x) / (m0 * sigma**3)
        kurt = np.trapezoid((x - mu) ** 4 * y, x) / (m0 * sigma**4) - 3.0  # excess
    else:
        skew = np.nan
        kurt = np.nan

    return dict(flux=m0, centroid=mu, sigma=sigma, skewness=skew, kurtosis_excess=kurt)


def skewnorm_profile(x, amp, xi, omega, alpha):
    """Skew-normal line profile with free amplitude.

    The profile is

        y(x) = amp * exp(-z²/2) * [1 + erf(alpha z / sqrt(2))],  z = (x - xi) / omega

    which is proportional to the skew-normal probability density
    (2/omega) phi(z) Phi(alpha z). For alpha = 0 it reduces to a Gaussian
    with amplitude amp, centre xi and standard deviation omega.

    Parameters
    ----------
    x : array_like
        Abscissae where the profile is evaluated.
    amp : float
        Amplitude (value of the Gaussian factor at x = xi).
    xi : float
        Location parameter. It is not the position of the maximum
        unless alpha = 0.
    omega : float
        Scale parameter (must be positive).
    alpha : float
        Shape parameter. Positive values give a tail towards
        increasing x.

    Returns
    -------
    numpy.ndarray
        Profile evaluated at x.
    """
    z = (np.asarray(x, dtype=float) - xi) / omega
    return amp * np.exp(-0.5 * z**2) * (1.0 + erf(alpha * z / np.sqrt(2.0)))


def fit_skewnorm(x, y, amp0, xi0, omega0, alpha_max=20.0):
    """Fit a skew-normal profile starting from a Gaussian solution.

    The fit starts at alpha = 0, where the skew-normal profile coincides
    with the Gaussian defined by amp0, xi0 and omega0.

    Parameters
    ----------
    x, y : numpy.ndarray
        Data to fit. y must be continuum-subtracted.
    amp0, xi0, omega0 : float
        Amplitude, centre and standard deviation of the initial
        Gaussian fit.
    alpha_max : float, optional
        Bound on the absolute value of the shape parameter. For large
        |alpha| the profile tends to a half-normal and the parameters
        become strongly degenerate.

    Returns
    -------
    dict
        Keys:
        'amp', 'xi', 'omega', 'alpha' : fitted parameters;
        'amp_err', 'xi_err', 'omega_err', 'alpha_err' : their
        uncertainties from the covariance matrix (inf if it cannot
        be estimated);
        'mode' : position of the maximum of the profile;
        'mean', 'std', 'skewness', 'kurtosis_excess' : mean, standard deviation,
        skewness and kurtosis excess of the skew-normal distribution.
        Positions are in the same units and origin as x.

    Raises
    ------
    RuntimeError
        If the least-squares fit does not converge.

    Notes
    -----
    Near alpha = 0, small changes in alpha and xi are degenerate to first
    order, so alpha (and xi) are poorly constrained for nearly symmetric
    lines, while the mode, the standard deviation and the skewness are
    well determined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xrange = x.max() - x.min()
    # xi may lie outside the data range when |alpha| is large, since it
    # is not the position of the maximum
    lower = [0.0, x.min() - xrange, 1e-3 * omega0, -alpha_max]
    upper = [np.inf, x.max() + xrange, 10 * xrange, alpha_max]
    p0 = np.clip([amp0, xi0, omega0, 0.0], lower, upper)

    with warnings.catch_warnings():
        # OptimizeWarning is issued when the covariance cannot be
        # estimated; the uncertainties are then returned as inf
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(skewnorm_profile, x, y, p0=p0, bounds=(lower, upper))
    amp, xi, omega, alpha = popt
    perr = np.sqrt(np.diag(pcov))

    # Moments of the skew-normal distribution
    delta = alpha / np.sqrt(1.0 + alpha**2)
    b = delta * np.sqrt(2.0 / np.pi)
    mean = xi + omega * b
    std = omega * np.sqrt(1.0 - b**2)
    skewness = (4.0 - np.pi) / 2.0 * b**3 / (1.0 - b**2) ** 1.5

    # The mode has no closed form. The profile is log-concave and the
    # mode lies in [xi - omega, xi + omega] (|m0(alpha)| < sqrt(2/pi) < 1)
    res = minimize_scalar(
        lambda t: -skewnorm_profile(t, amp, xi, omega, alpha),
        bounds=(xi - omega, xi + omega),
        method="bounded",
        options={"xatol": 1e-8 * omega},
    )

    # Kurtosis excess of the skew-normal distribution
    kurtosis_excess = 2 * (np.pi - 3) * b**4 / (1.0 - b**2) ** 2

    return dict(
        amp=amp,
        xi=xi,
        omega=omega,
        alpha=alpha,
        amp_err=perr[0],
        xi_err=perr[1],
        omega_err=perr[2],
        alpha_err=perr[3],
        mode=res.x,
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis_excess=kurtosis_excess,
    )


def gaussian_pixel_integrated(x, flux, mu, sigma):
    """Gaussian profile integrated over pixels of unit width.

    The value at each pixel is the integral of a Gaussian with total flux
    `flux`, centre `mu` and standard deviation `sigma` over the pixel
    [x - 0.5, x + 0.5] (pixel centres at the integer array coordinates):

        m(x) = flux / 2 * [erf((x + 0.5 - mu) / (sqrt(2) sigma))
                           - erf((x - 0.5 - mu) / (sqrt(2) sigma))]

    so that the sum over all the pixels is `flux`.

    Parameters
    ----------
    x : array_like
        Pixel centres.
    flux : float
        Integrated flux of the Gaussian (in the units of the pixel values
        times pixels).
    mu : float
        Centre of the Gaussian.
    sigma : float
        Standard deviation of the Gaussian (must be positive).

    Returns
    -------
    numpy.ndarray
        Integral of the Gaussian over each pixel.
    """
    s = np.sqrt(2.0) * sigma
    a = (np.asarray(x, dtype=float) - 0.5 - mu) / s
    b = (np.asarray(x, dtype=float) + 0.5 - mu) / s
    # erf(b) - erf(a) loses precision in the tails, where both arguments
    # have the same sign and a large absolute value (erf -> +1 or -1); use
    # the equivalent expressions erfc(a) - erfc(b) (right tail) and
    # erfc(-b) - erfc(-a) (left tail) there
    return 0.5 * flux * np.where(a > 0, erfc(a) - erfc(b), np.where(b < 0, erfc(-b) - erfc(-a), erf(b) - erf(a)))


def fit_gaussian_pixel_integrated(x, y, flux0, mu0, sigma0):
    """Fit a pixel-integrated Gaussian, preserving the flux in each pixel.

    The model value at each pixel is the integral of the Gaussian over the
    pixel (see gaussian_pixel_integrated), so that the fit matches the
    signal of each pixel with the flux that the Gaussian puts into it.

    Parameters
    ----------
    x, y : numpy.ndarray
        Pixel centres and pixel values to fit (continuum-subtracted).
    flux0, mu0, sigma0 : float
        Initial values of the integrated flux, centre and standard
        deviation (e.g. from a Gaussian fitted to the pixel values).

    Returns
    -------
    dict
        Keys 'flux', 'mean' and 'sigma' (fitted parameters), 'amp' (peak
        value of the Gaussian, flux / (sigma sqrt(2 pi))), and
        'flux_err', 'mean_err' and 'sigma_err' (uncertainties from the
        covariance matrix, inf if it cannot be estimated).

    Raises
    ------
    RuntimeError
        If the least-squares fit does not converge.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # the centre is kept within the fitted pixels
    lower = [0.0, x.min() - 0.5, 1e-3 * sigma0]
    upper = [np.inf, x.max() + 0.5, 10 * (x.max() - x.min() + 1)]
    p0 = np.clip([flux0, mu0, sigma0], lower, upper)

    with warnings.catch_warnings():
        # OptimizeWarning is issued when the covariance cannot be
        # estimated (e.g. 3 pixels for 3 parameters)
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(gaussian_pixel_integrated, x, y, p0=p0, bounds=(lower, upper))
    flux, mu, sigma = popt
    perr = np.sqrt(np.diag(pcov))

    return dict(
        flux=flux,
        mean=mu,
        sigma=sigma,
        amp=flux / (sigma * np.sqrt(2.0 * np.pi)),
        flux_err=perr[0],
        mean_err=perr[1],
        sigma_err=perr[2],
    )


def zero_monotonic_flanks(signal, peak_idx, inplace=False):
    """Set to zero the peak and its monotonically decreasing flanks.

    Starting at ``peak_idx``, move outwards in both directions, setting
    values to zero while the signal does not increase. Stop as soon as
    the signal starts to increase again. Consecutive equal values (a
    plateau around the peak or ties in the flanks) are also set to zero.
    The cleaning also stops at NaN values.

    Parameters
    ----------
    signal : numpy.ndarray
        1D input array.
    peak_idx : int
        Index of the detected peak.
    inplace : bool, optional
        If True, modify ``signal`` directly. Otherwise work on a copy.

    Returns
    -------
    numpy.ndarray
        Array with the peak and its monotonic flanks set to zero.
    """
    result = signal if inplace else signal.copy()
    n = len(result)

    # Reference values before zeroing anything
    ref_right = signal[peak_idx]
    ref_left = signal[peak_idx]

    result[peak_idx] = 0.0

    # Move to the right. Equal values are included (<=): otherwise the
    # remaining pixels of a plateau, or of a tie in a flank, would be
    # detected as new peaks
    i = peak_idx + 1
    while i < n and signal[i] <= ref_right:
        ref_right = signal[i]
        result[i] = 0.0
        i += 1

    # Move to the left
    i = peak_idx - 1
    while i >= 0 and signal[i] <= ref_left:
        ref_left = signal[i]
        result[i] = 0.0
        i -= 1

    return result


def find_highest_peaks_spectrum(
    sx,
    nmaxpeaks,
    nclean_around_peak,
    nwinwidth,
    threshold=0,
    nborder_to_ignore=0,
    title=None,
    debugplot=0,
    jupyter=False,
):
    """Find the highest peaks in 1D array (at most nmaxpeaks).

    Peaks are found iteratively. Every time a new peak is found, the area
    around this new peak is cleaned (set to zero) in order to avoid finding
    the same peak again or finding a peak that is too close to it.
    The area to be cleaned is defined by the parameter nclean_around_peak.
    The width of the window where each peak must be found is defined by
    the parameter nwinwidth. If no peak is found, the window width is
    reduced in steps of 2 down to 3; if there is still no peak, the
    maximum remaining value is taken as the peak, provided it is above
    threshold. Otherwise the search stops, and fewer than nmaxpeaks
    peaks are returned.

    The borders of the input array can be ignored by setting the parameter
    nborder_to_ignore to a positive value. In this case, the first and last
    nborder_to_ignore pixels of the input array will be set to zero before
    starting the peak search. This is useful when the borders of the array
    contain artifacts that could be mistaken for peaks.

    Parameters
    ----------
    sx : numpy.ndarray
        Input 1D array of floats.
    nmaxpeaks : int
        Maximum number of peaks to find.
    nclean_around_peak : int
        Number of pixels to clean around each peak. If this value is zero,
        only the peak pixel will be set to zero. If this value is positive,
        the pixels in the range [peak-nclean_around_peak, peak+nclean_around_peak]
        will be set to zero. If this value is -1, all the neighboring pixels with
        monotonically decreasing signal from the peak will be set to zero. If this
        value is less than -1, an error will be raised. The cleaning is done in order
        to avoid finding the same peak again or finding a peak that is too close to it.
    nwinwidth : int
        Width of the window where each peak must be found. It must be
        an odd integer >= 3.
    threshold : float
        Minimum signal in the peaks. It must be non-negative: the cleaned
        pixels (and the ignored borders) are set to zero, so with a
        negative threshold they could be detected as peaks.
    nborder_to_ignore : int
        Number of pixels to ignore at the borders of the input array.
    title: str or None
        Title for the debug plots. The provided string will be prepended
        to the default title. If None, the default title will be used.
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
    ixpeaks : numpy.ndarray
        1D array with the peak locations, in array coordinates (integers).
        The peaks are sorted in ascending order of their array coordinates.
        It may contain fewer than nmaxpeaks elements.
    """
    # check input parameters
    if len(sx) < nwinwidth:
        raise ValueError("Input array length is smaller than nwinwidth")
    if nmaxpeaks < 1:
        raise ValueError("nmaxpeaks must be at least 1")
    if nclean_around_peak < -1:
        raise ValueError("nclean_around_peak must be -1 or non-negative")
    odd_window_width(nwinwidth)
    # cleaned pixels are set to zero, which must not be above threshold
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
    for i in range(nmaxpeaks):
        # each peak is searched starting with the requested window width
        nwinwidth_effective = nwinwidth
        ixpeaks = find_peaks_spectrum(sx=sx_copy, nwinwidth=nwinwidth_effective, threshold=threshold, debugplot=0)
        # if no peaks are found, reduce the window width (minimum value: 3)
        while len(ixpeaks) < 1 and nwinwidth_effective - 2 >= 3:
            nwinwidth_effective -= 2
            if abs(debugplot) >= 10:
                _logger.info(
                    "find_highest_peaks_spectrum> No peaks found, "
                    f"reducing nwinwidth_effective to {nwinwidth_effective}"
                )
            ixpeaks = find_peaks_spectrum(sx=sx_copy, nwinwidth=nwinwidth_effective, threshold=threshold, debugplot=0)
        if len(ixpeaks) < 1:
            # no peak found even with the smallest window: take the maximum
            # remaining value, provided it is above the threshold
            imax = np.argmax(sx_copy)
            if not sx_copy[imax] > threshold:
                if abs(debugplot) >= 10:
                    _logger.warning(f"find_highest_peaks_spectrum> only {i} peaks found above threshold={threshold}")
                break
            if abs(debugplot) >= 10:
                _logger.warning("find_highest_peaks_spectrum> no peaks found, taking the maximum value as the peak")
            ixpeaks = [imax]
        # peak values
        peak_values = sx_copy[ixpeaks]
        if abs(debugplot) >= 10:
            _logger.info(f"find_highest_peaks_spectrum> peak_values={peak_values}")
        # find the index of the highest peak
        highest_peak_index = np.argmax(peak_values)
        # get the location of the highest peak
        highest_peak_location = ixpeaks[highest_peak_index]
        # add the highest peak location to the list of peaks
        list_ixpeaks.append(highest_peak_location)
        # clean the area around the highest peak
        if nclean_around_peak == -1:
            sx_copy = zero_monotonic_flanks(sx_copy, highest_peak_location, inplace=True)
        else:
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
            default_title = f"Peak #{i+1}/{nmaxpeaks}, nwinwidth_eff={nwinwidth_effective}, nclean={nclean_around_peak}"
            if title is not None:
                title_ = title + "\n" + default_title
            else:
                title_ = default_title
            ax.set_title(title_)
            if not jupyter:
                plt.show(block=False)
            plt.pause(0.001)
            pause_debugplot(debugplot, pltshow=True, tight_layout=True)

    return np.sort(np.array(list_ixpeaks, dtype=int))


def find_peaks_spectrum(sx, nwinwidth, threshold=0, debugplot=0):
    """Find peaks in 1D array.

    The algorithm imposes that the signal at both sides of the peak
    decreases monotonically. Peaks are found within a window of width
    nwinwidth.

    A run of consecutive equal values (a plateau, e.g. a saturated line)
    is treated as a single point. A run [p1, p2] (p1 == p2 when there is
    no plateau) is a peak when:

    - sx[p1] > threshold;
    - the signal increases strictly when entering the run
      (sx[p1 - 1] < sx[p1]) and decreases strictly when leaving it
      (sx[p2 + 1] < sx[p2]);
    - the signal is non-decreasing in [p1 - nwinwidth // 2, p1] and
      non-increasing in [p2, p2 + nwinwidth // 2] (equal consecutive
      values are allowed in the flanks).

    The returned location of a plateau is (p1 + p2) // 2. After a peak
    is found, the search resumes at p2 + nwinwidth - 1. A window that
    contains NaN values is never accepted, since every comparison with
    NaN is False.

    Parameters
    ----------
    sx : numpy.ndarray
        Input 1D array of floats.
    nwinwidth : int
        Width of the window where each peak must be found. It must be
        an odd integer >= 3. The window around pixel i is
        [i - nwinwidth // 2, i + nwinwidth // 2], so peaks closer than
        nwinwidth // 2 pixels to the borders are not detected.
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
    ixpeaks : numpy.ndarray
        1D array with the peak locations, in array coordinates (integers).

    """

    if not isinstance(sx, np.ndarray):
        raise TypeError(f"sx must be a numpy.ndarray, not {type(sx).__name__}")
    elif sx.ndim != 1:
        raise ValueError(f"{sx.ndim=} must be 1")

    sx_shape = sx.shape
    odd_window_width(nwinwidth)

    nmed = nwinwidth // 2

    if abs(debugplot) >= 10:
        _logger.info(f"find_peaks_spectrum> sx shape......: {sx_shape}")
        _logger.info(f"find_peaks_spectrum> nwinwidth.....: {nwinwidth}")
        _logger.info(f"find_peaks_spectrum> nmed..........: {nmed}")
        _logger.info(f"find_peaks_spectrum> data_threshold: {threshold}")
        _logger.info(f"find_peaks_spectrum> the first and last {nmed} pixels will be ignored")

    xpeaks = []  # list to store the peaks

    if sx_shape[0] < nwinwidth:
        raise ValueError(f"sx.shape[0]={sx_shape[0]} < nwinwidth={nwinwidth}, sx.shape={sx_shape}")

    npix = sx_shape[0]
    i = nmed
    while i < npix - nmed:
        # Only the first pixel of a run of equal values can start a peak,
        # and the signal must increase strictly when entering the run.
        # The conditions are written so that a comparison involving NaN
        # (always False) rejects the candidate.
        if sx[i] > threshold and sx[i - 1] < sx[i]:
            # last pixel of the run of values equal to sx[i]
            p2 = i
            while p2 + 1 < npix and sx[p2 + 1] == sx[i]:
                p2 += 1
            # the right flank [p2, p2 + nmed] must lie within the array
            peak_ok = p2 + nmed < npix
            if peak_ok:
                # strict decrease when leaving the run, non-decreasing
                # left flank and non-increasing right flank (pairwise
                # comparisons instead of np.diff, which would wrap around
                # for unsigned integer arrays)
                left = sx[i - nmed : i + 1]
                right = sx[p2 : p2 + nmed + 1]
                peak_ok = sx[p2 + 1] < sx[p2] and np.all(left[:-1] <= left[1:]) and np.all(right[:-1] >= right[1:])
            if peak_ok:
                ipeak = (i + p2) // 2
                if p2 > i and abs(debugplot) >= 10:
                    _logger.warning(
                        f"find_peaks_spectrum> plateau of {p2 - i + 1} pixels in [{i}, {p2}] "
                        f"(possibly saturated), peak located at {ipeak}"
                    )
                xpeaks.append(ipeak)
                i = p2 + nwinwidth - 1
            else:
                # no pixel inside the run can start a peak (no strict
                # increase), so continue after the run
                i = p2 + 1
        else:
            i += 1

    ixpeaks = np.array(xpeaks, dtype=int)

    if abs(debugplot) >= 10:
        _logger.info(f"find_peaks_spectrum> number of peaks found: {len(ixpeaks)}")
        _logger.info(f"find_peaks_spectrum> peaks: {ixpeaks}")

    return ixpeaks


def _parabola_vertex(coef, x_fit):
    """Abscissa of the maximum of a fitted parabola, if acceptable.

    Parameters
    ----------
    coef : numpy.ndarray
        Coefficients (c0, c1, c2) of the parabola c0 + c1 x + c2 x².
    x_fit : numpy.ndarray
        Abscissae of the fitted points, in ascending order.

    Returns
    -------
    float or None
        Abscissa of the vertex, or None when the parabola is not concave
        (c2 >= 0, so the vertex is not a maximum) or when the vertex lies
        outside the range of x_fit.
    """
    if len(coef) != 3 or not coef[2] < 0:
        return None
    xvertex = -coef[1] / (2.0 * coef[2])
    if not x_fit[0] <= xvertex <= x_fit[-1]:
        return None
    return xvertex


def refine_peaks_spectrum(
    sx,
    ixpeaks,
    nwinwidth,
    method,
    geometry=None,
    title=None,
    debugplot=0,
    ax_external=None,
    preserve_flux=False,
    return_dict_result=False,
):
    """Refine line peaks in spectrum.

    This function refines the locations of line peaks in a 1D spectrum by fitting
    a model to each peak. The refined peak locations are returned as a 1D array of floats.

    For some particular methods (e.g., "gaussian+moments" and "skewnorm"),
    the function only works when the number of peaks is 1, and
    the input array sx must contain a single continuum-subtracted line.
    In these cases, the final fit is performed on the whole array sx,
    and the function returns a dictionary with the relevant results
    depending on the method.

    Parameters
    ----------
    sx : numpy.ndarray
        Input 1D array of floats.
    ixpeaks : numpy.ndarray
        1D array with the initial peak locations, in array coordinates (integers).
        These values can be the output from the function
        find_peaks_spectrum().
    nwinwidth : int
        Width of the window where each peak must be refined, i.e. number
        of pixels used in the fit. It must be an odd integer >= 3.
    method : str
        Fitting method (required):

        - "poly2" : fit to a 2nd order polynomial
        - "gaussian" : fit to a Gaussian
        - "gaussian+moments" : fit to a Gaussian and compute moments. This is only
          available when the number of peaks is 1, since the moments and the
          non-parametric FWHM are computed using the whole array sx, which must
          therefore contain a single continuum-subtracted line.
        - "skewnorm" : fit to a Gaussian (as in "gaussian"), used as the initial
          guess of a least-squares fit of a skew-normal profile (see
          skewnorm_profile) to the whole array sx. As for "gaussian+moments",
          this is only available when the number of peaks is 1, sx must contain
          a single continuum-subtracted line, and it must have at least 5 pixels.
    geometry : tuple (4 integers) or None
        x, y, dx, dy values employed to set the window geometry.
    title : str or None
        Title for the debug plots. The provided string will be prepended
        to the default title. If None, the default title will be used.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses
    ax_external : list of matplotlib.axes.Axes or None
        If not None, the plots will be drawn on these axes objects.
        This is useful when the function is called from another function
        that already has a figure and axes objects, and wants to draw
        the plots on those axes objects instead of creating a new figure.
        This option must only be used when the number of peaks is equal to
        the number of axes objects in ax_external. Otherwise, an error
        will be raised. The value of debugplot will only determine the
        behavior of the debug messages, and the plots will always be drawn
        on the provided axes objects.
    preserve_flux : bool
        Only available with method="gaussian+moments". If True, the Gaussian
        fitted to log(y) is used as the initial guess of a least-squares fit,
        within the same window of nwinwidth pixels, of a Gaussian integrated
        over each pixel (see gaussian_pixel_integrated). The fit then preserves
        the flux: the integral of the Gaussian over each pixel matches the
        signal of the pixel. fxpeaks and sxpeaks contain the centre and the
        (intrinsic) sigma of this refined fit. The moments are computed as
        without preserve_flux. If the refined fit cannot be performed, the
        result of the initial Gaussian fit is returned (with fit_ok=False in
        dict_result).
    return_dict_result : bool
        Important: this option is only available when the number of peaks is 1.
        If True, the function returns all the computed values (fxpeaks, sxpeaks
        and a dictionary with additional results). This is relevant only when method
        is "poly2" or "gaussian" (by default these methods return only two values:
        fxpeaks and sxpeaks). For the remaining methods, the function always returns
        a third value (a dictionary with the relevant results depending on the method).

    Returns
    -------
    fxpeaks : numpy.ndarray
        1D array of floats with the refined peak locations, in array coordinates.
        With method="skewnorm", this is the mode of the fitted skew-normal
        profile (not its location parameter xi).
    sxpeaks : numpy.ndarray
        1D array of floats. When fitting Gaussians, this array stores the fitted line
        widths (sigma). With method="skewnorm", it stores the standard deviation of
        the fitted skew-normal distribution. The poly2 method does not provide a
        width, so the value is NaN when poly2 is used (also when it replaces
        another method that cannot be applied). If when fitting a Gaussian the
        fitted width is not defined, or the maximum of the fitted Gaussian lies
        outside the fitting window, the corresponding value is also set to NaN.
    dict_result : dict
        Extra dictionary with the results of the fit, for the last (and only)
        peak. It is returned only when method is "gaussian+moments" or
        "skewnorm", or when return_dict_result=True. It always contains the keys:

        - 'method' : requested method;
        - 'final_method' : method actually used ("poly2" when the requested
          Gaussian-based method cannot be applied, e.g. because of negative or
          null values in the fitting window);
        - 'fit_ok' : True when the requested method could be applied and
          produced a valid result. When False, fxpeaks and sxpeaks contain the
          result of the method that could be applied (or jmax and NaN), and
          the method-specific keys below may be missing.

        The remaining keys depend on final_method:

        - "poly2" : 'coef', coefficients (c0, c1, c2) of the parabola fitted
          to the data normalized to the maximum of the fitting window, in
          pixel coordinates relative to the initial peak location;
        - "gaussian" : 'amp' (in the units of sx), 'mean' and 'sigma' (in
          pixels) of the fitted Gaussian;
        - "gaussian+moments" : 'flux' (integrated flux, in the units of sx
          times pixels), 'centroid', 'sigma', 'skewness' and 'kurtosis_excess'
          (see line_moments), and the non-parametric FWHM and its left and
          right crossings, with the keys 'fwhm_np', 'fwhm_np_left' and
          'fwhm_np_right' (see fwhm_nonparametric). Positions and widths are
          in pixels. These keys are missing when the integrated flux is not
          positive. With preserve_flux=True, the dictionary also contains the
          parameters of the flux-preserving Gaussian fit, with the prefix
          'gaussian_' to distinguish them from the moments: 'gaussian_flux'
          (in the units of sx times pixels), 'gaussian_amp' (in the units of
          sx), 'gaussian_mean' (array coordinates), 'gaussian_sigma' (pixels)
          and their uncertainties 'gaussian_flux_err', 'gaussian_mean_err'
          and 'gaussian_sigma_err'. These keys are missing when that fit
          cannot be performed;
        - "skewnorm" : the keys returned by fit_skewnorm, with the positions
          'xi', 'mode' and 'mean' in array coordinates and 'amp' and 'amp_err'
          in the units of sx. They are missing when the skew-normal fit cannot
          be performed; fxpeaks and sxpeaks then contain the result of the
          initial Gaussian fit.

    """

    odd_window_width(nwinwidth)
    nmed = nwinwidth // 2

    npeaks = len(ixpeaks)
    if npeaks < 1:
        raise ValueError("No peaks to refine, ixpeaks is empty")

    if ax_external is not None:
        ax_external = np.ravel(ax_external)  # flatten the array of axes objects
        if len(ax_external) != npeaks:
            raise ValueError(
                "The number of axes objects in ax_external must be equal to the number of peaks. "
                f"len(ax_external)={len(ax_external)}, npeaks={npeaks}"
            )

    if return_dict_result and npeaks > 1:
        raise ValueError("return_dict_result=True is only available when npeaks=1")
    if method not in ["poly2", "gaussian", "gaussian+moments", "skewnorm"]:
        raise ValueError("Invalid method=" + str(method) + " value")
    if method in ["gaussian+moments", "skewnorm"] and npeaks > 1:
        raise ValueError(f"method={method} is only available when npeaks=1")
    # the skew-normal profile has 4 free parameters: at least one more
    # pixel is required to leave one degree of freedom in the fit
    if method == "skewnorm" and len(sx) < 5:
        raise ValueError(f"method=skewnorm requires at least 5 pixels in sx: len(sx)={len(sx)}")
    if preserve_flux and method != "gaussian+moments":
        raise ValueError(f"preserve_flux=True is only available with method=gaussian+moments, not with method={method}")
    if method in ["gaussian+moments", "skewnorm"]:
        return_dict_result = True  # always return the dictionary with the results

    xfpeaks = np.zeros(npeaks, dtype=float)
    sfpeaks = np.zeros(npeaks, dtype=float)

    for iline in range(npeaks):
        jmax = ixpeaks[iline]
        # skew-normal fit (method="skewnorm") and pixel-integrated Gaussian
        # fit (method="gaussian" with preserve_flux=True), in the coordinates
        # relative to jmax and in units of the normalized data
        skn = None
        gpi = None
        # prevent possible problem when fitting a line too near to any
        # of the borders of the spectrum (j2 is an exclusive index)
        j1 = jmax - nmed
        j2 = jmax + nmed + 1
        if j1 < 0:
            j1 = 0
            j2 = 2 * nmed + 1
            if j2 > len(sx):
                raise ValueError("Unexpected j2=" + str(j2) + " value when len(sx)=" + str(len(sx)))
        if j2 > len(sx):
            j2 = len(sx)
            j1 = j2 - (2 * nmed + 1)
            if j1 < 0:
                raise ValueError("Unexpected j1=" + str(j1) + " value when len(sx)=" + str(len(sx)))
        # abscissae of the fitted pixels relative to jmax; the window may be
        # off-centre when the peak is close to the borders of the spectrum
        x_fit = np.arange(j1, j2, dtype=float) - jmax
        # astype() returns a copy, so the original array is not modified
        # when normalizing the data to be fitted
        y_fit = sx[j1:j2].astype(float)
        # the whole spectrum is only needed for a single peak: to compute
        # the moments (gaussian+moments), to fit the skew-normal profile
        # (skewnorm) and to plot the original data
        if npeaks == 1:
            y_full = sx.astype(float)
            x_full = np.arange(len(sx), dtype=float)
        else:
            x_full = None
            y_full = None
        sx_peak_flux = y_fit.max()
        # normalize only by a positive maximum: dividing by a negative value
        # would flip the profile and hide the check for non-positive values
        if sx_peak_flux > 0:
            y_fit /= sx_peak_flux  # normalize to maximum value
            if npeaks == 1:
                y_full /= sx_peak_flux  # normalize to maximum value

        if method in ["gaussian", "gaussian+moments", "skewnorm"]:
            # check that there are no negative or null values
            if y_fit.min() <= 0:
                if abs(debugplot) >= 10:
                    _logger.warning(
                        f"Negative or null value encountered in refine_peaks_spectrum with {method} "
                        f"around {jmax}; using poly2 method instead."
                    )
                final_method = "poly2"
            else:
                final_method = method
        else:
            final_method = method

        # initialize the results dictionary with the method and final_method keys
        dict_result = dict()
        dict_result["method"] = method
        dict_result["final_method"] = final_method
        dict_result["fit_ok"] = False  # will be set to True if the fit is successful

        # refine the peak location using the selected method
        if final_method == "poly2":
            poly_funct = Polynomial.fit(x_fit, y_fit, 2)
            poly_funct = Polynomial.cast(poly_funct)
            coef = poly_funct.coef
            # use jmax when the fitted parabola has no maximum within the window
            xvertex = _parabola_vertex(coef, x_fit)
            refined_peak = jmax + (xvertex if xvertex is not None else 0.0)
            # poly2 does not provide a line width
            sfpeaks[iline] = np.nan
            if xvertex is None and abs(debugplot) >= 10:
                _logger.warning(f"No maximum within the fitting window around {jmax} (poly2); using {jmax}")
            dict_result["coef"] = coef
            # the result is valid only if the original method was poly2 and
            # the parabola has a maximum within the window; otherwise the fit
            # was only used to provide a reasonable peak location when the
            # requested method cannot be applied (e.g. when the data contains
            # negative or null values)
            if method == "poly2" and xvertex is not None:
                dict_result["fit_ok"] = True
        elif final_method in ["gaussian", "gaussian+moments", "skewnorm"]:
            # Gaussian fit (parabola fitted to log(y)): final result for
            # "gaussian" and "gaussian+moments", initial guess for "skewnorm"
            poly_funct = Polynomial.fit(x_fit, np.log(y_fit), 2)
            poly_funct = Polynomial.cast(poly_funct)
            coef = poly_funct.coef
            if len(coef) == 3:
                # use jmax when the fitted parabola has no maximum within the window
                xvertex = _parabola_vertex(coef, x_fit)
                refined_peak = jmax + (xvertex if xvertex is not None else 0.0)
                if xvertex is None and abs(debugplot) >= 10:
                    _logger.warning(f"No maximum within the fitting window around {jmax} (gaussian); using {jmax}")
                if coef[2] >= 0:
                    if abs(debugplot) >= 10:
                        _logger.warning(
                            f"Non-negative quadratic coefficient encountered in refine_peaks_spectrum with gaussian "
                            f"around {jmax}; setting fitted width to NaN."
                        )
                    sfpeaks[iline] = np.nan
                elif xvertex is None:
                    # concave parabola, but its maximum is outside the fitting
                    # window: the fitted Gaussian does not describe the line
                    sfpeaks[iline] = np.nan
                else:
                    sfpeaks[iline] = np.sqrt(-1 / (2.0 * coef[2]))
                if final_method == "gaussian":
                    # the Gaussian is only valid when the parabola is concave and
                    # its maximum lies within the window (otherwise sfpeaks is NaN)
                    if not np.isnan(sfpeaks[iline]):
                        # y_fit is normalized to sx_peak_flux (> 0, since all the
                        # values in the window are positive)
                        dict_result["amp"] = np.exp(coef[0] - coef[1] * coef[1] / (4 * coef[2])) * sx_peak_flux
                        dict_result["mean"] = refined_peak
                        dict_result["sigma"] = sfpeaks[iline]
                        dict_result["fit_ok"] = True
                elif final_method == "gaussian+moments":
                    if preserve_flux:
                        # refine the Gaussian fitted to log(y) with a Gaussian integrated
                        # over each pixel of the window (only when the initial Gaussian
                        # is valid, i.e. sfpeaks is not NaN)
                        if np.isnan(sfpeaks[iline]):
                            if abs(debugplot) >= 10:
                                _logger.warning(
                                    f"No valid initial Gaussian fit around {jmax}; "
                                    "flux-preserving Gaussian fit not performed."
                                )
                        else:
                            amp0 = np.exp(coef[0] - coef[1] * coef[1] / (4 * coef[2]))
                            flux0 = amp0 * sfpeaks[iline] * np.sqrt(2.0 * np.pi)
                            try:
                                gpi = fit_gaussian_pixel_integrated(x_fit, y_fit, flux0, xvertex, sfpeaks[iline])
                            except RuntimeError as e:
                                if abs(debugplot) >= 10:
                                    _logger.warning(
                                        f"Flux-preserving Gaussian fit around {jmax} did not converge ({e}); "
                                        "using the initial Gaussian fit."
                                    )
                        if gpi is not None:
                            refined_peak = gpi["mean"] + jmax
                            sfpeaks[iline] = gpi["sigma"]
                            # prefix "gaussian_" to avoid overwriting the moments 'flux'
                            # and 'sigma'; position in array coordinates and fluxes in
                            # the units of sx
                            dict_result["gaussian_mean"] = refined_peak
                            dict_result["gaussian_mean_err"] = gpi["mean_err"]
                            dict_result["gaussian_sigma"] = gpi["sigma"]
                            dict_result["gaussian_sigma_err"] = gpi["sigma_err"]
                            for key in ["flux", "amp", "flux_err"]:
                                dict_result["gaussian_" + key] = gpi[key] * sx_peak_flux
                    # compute the moments of the line; sx is assumed to contain
                    # a single continuum-subtracted line
                    try:
                        moments = line_moments(x_full, y_full, debugplot=debugplot)
                    except ValueError as e:
                        # non-positive integrated flux: the moments are not defined
                        if abs(debugplot) >= 10:
                            _logger.warning(f"{e} Moments not computed.")
                        moments = None
                    if moments is not None:
                        dict_result.update(moments)
                        # y_full is normalized to the peak flux: express the
                        # integrated flux in the units of sx
                        if sx_peak_flux > 0:
                            dict_result["flux"] *= sx_peak_flux
                        # compute the non-parametric FWHM and its left and right crossings
                        fwhm_np, fwhm_np_left, fwhm_np_right = fwhm_nonparametric(x_full, y_full, debugplot=debugplot)
                        # add the non-parametric FWHM and its left and right crossings to the moments dictionary
                        dict_result["fwhm_np"] = fwhm_np
                        dict_result["fwhm_np_left"] = fwhm_np_left
                        dict_result["fwhm_np_right"] = fwhm_np_right
                        # with preserve_flux=True, the flux-preserving fit must also succeed
                        dict_result["fit_ok"] = gpi is not None if preserve_flux else True
                elif final_method == "skewnorm":
                    # fit a skew-normal profile to the whole spectrum, using the Gaussian
                    # fit as the initial guess. The skew-normal profile has 4 free
                    # parameters: amplitude, location (xi), scale (omega) and shape (alpha).
                    if np.isnan(sfpeaks[iline]):
                        if abs(debugplot) >= 10:
                            _logger.warning(
                                f"No valid initial Gaussian fit around {jmax}; skew-normal fit not performed."
                            )
                    else:
                        amp0 = np.exp(coef[0] - coef[1] * coef[1] / (4 * coef[2]))
                        try:
                            # fit to the whole spectrum, which is assumed to
                            # contain a single continuum-subtracted line
                            skn = fit_skewnorm(x_full - jmax, y_full, amp0, xvertex, sfpeaks[iline])
                        except RuntimeError as e:
                            if abs(debugplot) >= 10:
                                _logger.warning(
                                    f"Skew-normal fit around {jmax} did not converge ({e}); "
                                    "skew-normal fit not performed."
                                )
                    if skn is not None:
                        refined_peak = skn["mode"] + jmax
                        sfpeaks[iline] = skn["std"]
                        # positions in array coordinates and amplitude in the units of sx
                        flux_scale = sx_peak_flux if sx_peak_flux > 0 else 1.0
                        dict_result.update(skn)
                        for key in ["xi", "mode", "mean"]:
                            dict_result[key] = skn[key] + jmax
                        for key in ["amp", "amp_err"]:
                            dict_result[key] = skn[key] * flux_scale
                        dict_result["fit_ok"] = True
                else:
                    # implementation safety check
                    raise ValueError("Method=" + str(final_method) + " is not yet implemented")
            else:
                # do not refine the peak location when the fitted parabola has no maximum within the window
                refined_peak = 0.0 + jmax
                sfpeaks[iline] = np.nan
        else:
            # this should never happen, since the final_method is checked above
            raise ValueError("Invalid method=" + str(final_method) + " value")

        xfpeaks[iline] = refined_peak

        if ax_external is None:
            debugplot_effective = debugplot
        else:
            debugplot_effective = 1
        if debugplot_effective % 10 != 0:
            if ax_external is None:
                from numina.array.display.matplotlib_qt import plt

                fig = plt.figure()
                if geometry is not None:
                    set_window_geometry(geometry)
                ax = fig.add_subplot(111)
            else:
                ax = ax_external[iline]

            ax.set_xlabel("index around initial integer peak")
            ax.set_ylabel("Normalized number of counts")
            default_title = "Fit to line at array index " + str(jmax) + " (method=" + final_method + ")"
            if title is not None:
                title_ = title + "\n" + default_title
            else:
                title_ = default_title
            ax.set_title(title_)
            if npeaks == 1:
                ax.plot(x_full - jmax, y_full, "+", color="gray", markersize=10, label="Original data")
            ax.plot(x_fit, y_fit, "bo", label="Data to fit")
            xmin, xmax = ax.get_xlim()
            x_plot = np.linspace(start=xmin, stop=xmax, num=1000, dtype=float)
            if final_method == "poly2":
                y_plot = poly_funct(x_plot)
                ax.plot(x_plot, y_plot, color="red", label="Fitted curve")
            elif final_method == "skewnorm":
                if skn is not None:
                    # overplot initial Gaussian fit (in orange) and fitted skew-normal profile (in red)
                    amp0 = np.exp(coef[0] - coef[1] * coef[1] / (4 * coef[2]))
                    x0 = -coef[1] / (2.0 * coef[2])
                    sigma = np.sqrt(-1 / (2.0 * coef[2]))
                    y_plot = amp0 * np.exp(-((x_plot - x0) ** 2) / (2 * sigma**2))
                    ax.plot(x_plot, y_plot, linestyle="--", color="orange", label="Initial Gaussian fit")
                    y_plot = skewnorm_profile(x_plot, skn["amp"], skn["xi"], skn["omega"], skn["alpha"])
                    ax.plot(x_plot, y_plot, color="red", label=f"Fitted skew-normal (alpha={skn['alpha']:.2f})")
                    ax.axvline(x=skn["xi"], color="purple", linestyle=":", label="xi")
                # refined_peak is always defined (it falls back to the
                # Gaussian fit or to jmax)
                ax.axvline(x=refined_peak - jmax, color="green", linestyle="--", label="Refined peak")
            elif final_method in ["gaussian", "gaussian+moments"]:
                # The Gaussian is only defined when the parabola fitted to
                # log(y) is concave (coef[2] < 0); otherwise amp, x0 and
                # sigma would be undefined (division by zero or sqrt of a
                # negative number)
                if gpi is not None:
                    # initial Gaussian fit, pixel-integrated model and refined
                    # (intrinsic) Gaussian
                    amp0 = np.exp(coef[0] - coef[1] * coef[1] / (4 * coef[2]))
                    x00 = -coef[1] / (2.0 * coef[2])
                    sigma0 = np.sqrt(-1 / (2.0 * coef[2]))
                    y_plot = amp0 * np.exp(-((x_plot - x00) ** 2) / (2 * sigma0**2))
                    ax.plot(x_plot, y_plot, color="gray", linestyle=":", label="Initial Gaussian fit")
                    ax.stairs(
                        gaussian_pixel_integrated(x_fit, gpi["flux"], gpi["mean"], gpi["sigma"]),
                        np.append(x_fit - 0.5, x_fit[-1] + 0.5),
                        color="magenta",
                        label="Pixel-integrated model",
                    )
                    amp, x0, sigma = gpi["amp"], gpi["mean"], gpi["sigma"]
                    fwhm = 2.355 * sigma
                    y_plot = amp * np.exp(-((x_plot - x0) ** 2) / (2 * sigma**2))
                    ax.plot(x_plot, y_plot, color="red", label="Fitted curve (flux preserved)")
                    ax.axvline(x=x0 - fwhm / 2, color="orange", linestyle="--", label=f"Fitted FWHM ({fwhm:.2f})")
                    ax.axvline(x=x0 + fwhm / 2, color="orange", linestyle="--")
                elif len(coef) == 3 and coef[2] < 0:
                    amp = np.exp(coef[0] - coef[1] * coef[1] / (4 * coef[2]))
                    x0 = -coef[1] / (2.0 * coef[2])
                    sigma = np.sqrt(-1 / (2.0 * coef[2]))
                    fwhm = 2.355 * sigma
                    y_plot = amp * np.exp(-((x_plot - x0) ** 2) / (2 * sigma**2))
                    ax.plot(x_plot, y_plot, color="red", label="Fitted curve")
                    ax.axvline(x=x0 - fwhm / 2, color="orange", linestyle="--", label=f"Fitted FWHM ({fwhm:.2f})")
                    ax.axvline(x=x0 + fwhm / 2, color="orange", linestyle="--")
                # refined_peak is always defined (it falls back to jmax)
                ax.axvline(x=refined_peak - jmax, color="green", linestyle="--", label="Refined peak")
                if "fwhm_np" in dict_result and dict_result["fwhm_np"] is not None:
                    label_fwhm_np = f"non-parametric FWHM ({dict_result['fwhm_np']:.2f})"
                else:
                    label_fwhm_np = "non-parametric FWHM (undefined)"
                if "fwhm_np_left" in dict_result and dict_result["fwhm_np_left"] is not None:
                    label_fwhm_np_displayed = True
                    ax.axvline(x=dict_result["fwhm_np_left"] - jmax, color="cyan", linestyle="--", label=label_fwhm_np)
                else:
                    label_fwhm_np_displayed = False
                if "fwhm_np_right" in dict_result and dict_result["fwhm_np_right"] is not None:
                    if label_fwhm_np_displayed:
                        label_fwhm_np = None  # avoid duplicate legend entry
                    ax.axvline(x=dict_result["fwhm_np_right"] - jmax, color="cyan", linestyle="--", label=label_fwhm_np)
            else:
                raise ValueError("Invalid method=" + str(final_method) + " value")
            ax.legend()
            ymin, ymax = ax.get_ylim()
            if ymin > 0:
                ymin = 0
            ax.set_ylim(ymin, ymax)
            if ax_external is None:
                if abs(debugplot) >= 10:
                    _logger.info(f"Refined peak location: {refined_peak}")
                plt.show(block=False)
                plt.pause(0.001)
                pause_debugplot(debugplot, pltshow=True, tight_layout=True, fig=fig)

    if method in ["skewnorm", "gaussian+moments"]:
        return xfpeaks, sfpeaks, dict_result
    else:
        if return_dict_result:
            return xfpeaks, sfpeaks, dict_result
        else:
            return xfpeaks, sfpeaks
