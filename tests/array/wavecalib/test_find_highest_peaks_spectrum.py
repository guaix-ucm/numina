"""Tests of find_highest_peaks_spectrum and zero_monotonic_flanks"""

import numpy as np
import pytest

from numina.array.wavecalib.peaks_spectrum import find_highest_peaks_spectrum, zero_monotonic_flanks


def gaussian_lines(npix, centers, amplitudes, sigma=2.0):
    """Sum of Gaussian lines evaluated at integer pixels"""
    x = np.arange(npix, dtype=float)
    sp = np.zeros(npix)
    for center, amplitude in zip(centers, amplitudes):
        sp += amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return sp


# --- zero_monotonic_flanks ------------------------------------------------------


def test_zero_monotonic_flanks_stops_where_signal_rises():
    signal = np.array([5, 1, 2, 3, 9, 4, 2, 3, 0], dtype=float)
    result = zero_monotonic_flanks(signal, 4)
    np.testing.assert_array_equal(result, [5, 0, 0, 0, 0, 0, 0, 3, 0])


def test_zero_monotonic_flanks_includes_plateau_and_ties():
    # plateau around the peak and a tie (2, 2) in the right flank
    signal = np.array([1, 0, 3, 7, 7, 7, 4, 2, 2, 1, 5], dtype=float)
    result = zero_monotonic_flanks(signal, 4)
    np.testing.assert_array_equal(result, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])


def test_zero_monotonic_flanks_stops_at_nan():
    signal = np.array([0, 1, np.nan, 3, 9, 4, 1], dtype=float)
    result = zero_monotonic_flanks(signal, 4)
    np.testing.assert_array_equal(result, [0, 1, np.nan, 0, 0, 0, 0])


@pytest.mark.parametrize("inplace", [False, True])
def test_zero_monotonic_flanks_inplace(inplace):
    signal = np.array([0, 1, 5, 1, 0], dtype=float)
    original = signal.copy()
    result = zero_monotonic_flanks(signal, 2, inplace=inplace)
    np.testing.assert_array_equal(result, np.zeros(5))
    if inplace:
        assert result is signal
    else:
        np.testing.assert_array_equal(signal, original)


# --- find_highest_peaks_spectrum ------------------------------------------------


def test_highest_peaks_sorted_by_location():
    sx = gaussian_lines(100, [15, 40, 70, 85], [30, 100, 60, 10])
    ixpeaks = find_highest_peaks_spectrum(sx, nmaxpeaks=3, nclean_around_peak=-1, nwinwidth=5)
    np.testing.assert_array_equal(ixpeaks, [15, 40, 70])
    assert ixpeaks.dtype.kind == "i"


@pytest.mark.parametrize("nclean_around_peak", [-1, 6])
def test_more_peaks_requested_than_available(nclean_around_peak):
    # Previously the reduced window width was kept for the next peak and
    # reached nwinwidth=1, which raised ValueError; then the maximum of
    # the cleaned array (0, at index 0) was returned as a peak. The
    # threshold excludes the Gaussian tails left beyond nclean_around_peak
    sx = gaussian_lines(60, [20], [100])
    ixpeaks = find_highest_peaks_spectrum(
        sx, nmaxpeaks=3, nclean_around_peak=nclean_around_peak, nwinwidth=5, threshold=1
    )
    np.testing.assert_array_equal(ixpeaks, [20])


def test_no_peak_above_threshold_returns_empty_int_array():
    sx = gaussian_lines(60, [20], [10])
    ixpeaks = find_highest_peaks_spectrum(sx, nmaxpeaks=2, nclean_around_peak=-1, nwinwidth=5, threshold=50)
    assert ixpeaks.size == 0
    assert ixpeaks.dtype.kind == "i"


def test_window_is_reduced_when_no_peak_is_found():
    # The line at 20 has a bump in its flank, so it is not found with
    # nwinwidth=7: the line at 45 is found first, and the window is then
    # reduced to find the line at 20
    sx = np.zeros(60)
    sx[16:25] = [1, 2, 4, 8, 10, 8, 4, 5, 2]
    sx += gaussian_lines(60, [45], [6], sigma=1.0)
    ixpeaks = find_highest_peaks_spectrum(sx, nmaxpeaks=2, nclean_around_peak=-1, nwinwidth=7)
    np.testing.assert_array_equal(ixpeaks, [20, 45])


def test_maximum_taken_when_no_window_works():
    # A peak at the last pixel cannot be found with any window (the
    # borders are excluded), so the maximum value is taken
    sx = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 10], dtype=float)
    ixpeaks = find_highest_peaks_spectrum(sx, nmaxpeaks=1, nclean_around_peak=0, nwinwidth=5)
    np.testing.assert_array_equal(ixpeaks, [9])


@pytest.mark.parametrize("nwinwidth", [3, 5, 7])
def test_saturated_line_gives_a_single_peak(nwinwidth):
    # With strict comparisons in zero_monotonic_flanks only the centre of
    # the plateau was cleaned, and its remains were found as new peaks
    x = np.arange(60, dtype=float)
    sx = np.minimum(100 * np.exp(-0.5 * ((x - 20) / 3) ** 2), 60) + 40 * np.exp(-0.5 * ((x - 45) / 2) ** 2)
    ixpeaks = find_highest_peaks_spectrum(sx, nmaxpeaks=2, nclean_around_peak=-1, nwinwidth=nwinwidth)
    np.testing.assert_array_equal(ixpeaks, [20, 45])


def test_nborder_to_ignore():
    sx = gaussian_lines(60, [3, 30], [100, 50])
    ixpeaks = find_highest_peaks_spectrum(sx, nmaxpeaks=1, nclean_around_peak=-1, nwinwidth=5, nborder_to_ignore=8)
    np.testing.assert_array_equal(ixpeaks, [30])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(nmaxpeaks=0), "nmaxpeaks"),
        (dict(nclean_around_peak=-2), "nclean_around_peak"),
        (dict(nwinwidth=4), "odd integer"),
        (dict(threshold=-1), "threshold"),
        (dict(nborder_to_ignore=-1), "nborder_to_ignore"),
        (dict(nborder_to_ignore=30), "nborder_to_ignore"),
    ],
)
def test_invalid_arguments(kwargs, match):
    args = dict(nmaxpeaks=1, nclean_around_peak=-1, nwinwidth=5)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        find_highest_peaks_spectrum(np.zeros(60), **args)
