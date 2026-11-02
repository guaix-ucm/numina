"""Tests of find_peaks_spectrum

The tests marked as xfail(strict=True) describe the desired behaviour in
cases that the current implementation handles incorrectly. When the
function is fixed, they will start to pass and pytest will report them
as XPASS (failure), so the marker must then be removed.
"""

import numpy as np
import pytest

from numina.array.wavecalib.peaks_spectrum import find_peaks_spectrum


def gaussian_lines(npix, centers, amplitude=100.0, sigma=2.0):
    """Sum of Gaussian lines evaluated at integer pixels"""
    x = np.arange(npix, dtype=float)
    sp = np.zeros(npix)
    for center in centers:
        sp += amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    return sp


def reference_find_peaks(sx, nwinwidth, threshold=0):
    """Straightforward reimplementation of the documented algorithm.

    A pixel i is a peak when sx[i] > threshold, the signal is
    non-decreasing in [i - nmed, i] and non-increasing in [i, i + nmed],
    with nmed = nwinwidth // 2. After a peak is found, the search
    resumes at i + nwinwidth - 1.
    """
    nmed = nwinwidth // 2
    peaks = []
    i = nmed
    while i < len(sx) - nmed:
        left = sx[i - nmed : i + 1]
        right = sx[i : i + nmed + 1]
        if sx[i] > threshold and np.all(np.diff(left) >= 0) and np.all(np.diff(right) <= 0):
            peaks.append(i)
            i += nwinwidth - 1
        else:
            i += 1
    return np.array(peaks, dtype=int)


# --- input validation ---------------------------------------------------------


def test_list_input_raises_typeerror():
    with pytest.raises(TypeError):
        find_peaks_spectrum([0.0, 1.0, 0.0], nwinwidth=3)


def test_2d_input_raises_valueerror():
    with pytest.raises(ValueError, match="ndim"):
        find_peaks_spectrum(np.zeros((5, 5)), nwinwidth=3)


@pytest.mark.parametrize("nwinwidth", [-1, 0, 1, 2, 4, 6])
def test_invalid_nwinwidth_raises_valueerror(nwinwidth):
    with pytest.raises(ValueError, match="odd integer"):
        find_peaks_spectrum(np.zeros(10), nwinwidth=nwinwidth)


def test_spectrum_shorter_than_window_raises_valueerror():
    with pytest.raises(ValueError, match="nwinwidth"):
        find_peaks_spectrum(np.zeros(4), nwinwidth=5)


# --- output type and empty results -------------------------------------------


@pytest.mark.parametrize(
    "sx",
    [
        np.zeros(20),
        -gaussian_lines(40, [20]),
        np.arange(20, dtype=float),
        np.arange(20, dtype=float)[::-1].copy(),
    ],
    ids=["zeros", "absorption", "increasing", "decreasing"],
)
def test_no_peaks_returns_empty_int_array(sx):
    ixpeaks = find_peaks_spectrum(sx, nwinwidth=5)
    assert isinstance(ixpeaks, np.ndarray)
    assert ixpeaks.dtype.kind == "i"
    assert ixpeaks.size == 0


def test_input_is_not_modified():
    sx = gaussian_lines(50, [20, 35])
    sx_orig = sx.copy()
    find_peaks_spectrum(sx, nwinwidth=5, threshold=10)
    np.testing.assert_array_equal(sx, sx_orig)


def test_integer_input():
    sx = np.array([0, 1, 3, 7, 3, 1, 0, 0, 2, 9, 2, 0, 0])
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=5), [3, 9])


# --- basic detection ----------------------------------------------------------


@pytest.mark.parametrize("nwinwidth", [3, 5, 7, 9])
def test_single_gaussian_line(nwinwidth):
    sx = gaussian_lines(60, [30.3])
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=nwinwidth), [30])


def test_several_lines_sorted():
    centers = [12, 30, 47, 80]
    sx = gaussian_lines(100, centers)
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=7), centers)


def test_noise_bump_rejected_by_wide_window():
    # The flank of the line at 10 contains a bump at 13 (signal rises again)
    sx = np.array([0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 10, 8, 4, 5, 2, 1, 0, 0, 0, 0], dtype=float)
    # with nwinwidth=5 the window [8, 12] is monotonic on both sides
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=5), [10])
    # with nwinwidth=7 the window [7, 13] includes the rise 4 -> 5
    assert find_peaks_spectrum(sx, nwinwidth=7).size == 0


def test_window_allows_equal_neighbours():
    # Non-strict monotonicity: a repeated value in the flank is accepted
    sx = np.array([0, 1, 1, 5, 2, 2, 0, 0], dtype=float)
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=5), [3])


# --- threshold ----------------------------------------------------------------


def test_threshold_is_strict():
    sx = gaussian_lines(40, [20], amplitude=50.0)
    peak_value = sx[20]
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=5, threshold=peak_value - 1e-9), [20])
    assert find_peaks_spectrum(sx, nwinwidth=5, threshold=peak_value).size == 0


def test_threshold_selects_lines():
    x = np.arange(80, dtype=float)
    sx = 100 * np.exp(-0.5 * ((x - 20) / 2) ** 2) + 10 * np.exp(-0.5 * ((x - 50) / 2) ** 2)
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=5, threshold=0), [20, 50])
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=5, threshold=50), [20])


def test_negative_threshold():
    # Emission line on a negative background
    sx = gaussian_lines(40, [20]) - 200.0
    assert find_peaks_spectrum(sx, nwinwidth=5).size == 0
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=5, threshold=-150), [20])


# --- borders ------------------------------------------------------------------


@pytest.mark.parametrize("nwinwidth", [3, 5, 7])
def test_peaks_closer_than_nmed_to_borders_are_ignored(nwinwidth):
    nmed = nwinwidth // 2
    npix = 30
    # peaks exactly at nmed and npix - 1 - nmed are detected
    sx = np.zeros(npix)
    sx[nmed] = sx[npix - 1 - nmed] = 10.0
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=nwinwidth), [nmed, npix - 1 - nmed])
    # one pixel closer to the borders they are not
    sx = np.zeros(npix)
    sx[nmed - 1] = sx[npix - nmed] = 10.0
    assert find_peaks_spectrum(sx, nwinwidth=nwinwidth).size == 0


# --- window width conventions --------------------------------------------------


def test_minimum_separation_between_peaks():
    # With strictly monotonic flanks two peaks must be at least
    # 2 * nmed pixels apart to both satisfy the criterion
    sx = gaussian_lines(60, [20, 24], sigma=0.7)
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=5), [20, 24])
    assert find_peaks_spectrum(sx, nwinwidth=7).size < 2


# --- comparison with a reference implementation ---------------------------------


@pytest.mark.parametrize("nwinwidth", [3, 5, 7, 11])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_spectra_match_reference(nwinwidth, seed):
    # Random spectra (no ties) with lines and noise
    rng = np.random.default_rng(seed)
    npix = 500
    centers = rng.uniform(10, npix - 10, size=25)
    sx = gaussian_lines(npix, centers, amplitude=50.0, sigma=1.5) + rng.normal(scale=2.0, size=npix)
    threshold = 5.0
    np.testing.assert_array_equal(
        find_peaks_spectrum(sx, nwinwidth=nwinwidth, threshold=threshold),
        reference_find_peaks(sx, nwinwidth, threshold),
    )


# --- known defects (desired behaviour) -------------------------------------------


@pytest.mark.xfail(strict=True, reason="a constant signal above threshold is reported as several peaks")
def test_constant_signal_has_no_peaks():
    assert find_peaks_spectrum(np.ones(20), nwinwidth=5).size == 0


@pytest.mark.xfail(strict=True, reason="a flat-topped line is reported as several peaks")
def test_plateau_gives_a_single_peak():
    sx = np.array([0, 0, 1, 5, 5, 5, 5, 1, 0, 0], dtype=float)
    ixpeaks = find_peaks_spectrum(sx, nwinwidth=3)
    assert ixpeaks.size == 1
    assert 3 <= ixpeaks[0] <= 6


@pytest.mark.xfail(strict=True, reason="comparisons with NaN are False, so the neighbours of a NaN become peaks")
def test_nan_at_line_centre_does_not_create_peaks():
    sx = np.array([0, 1, 2, np.nan, 2, 1, 0, 0], dtype=float)
    assert find_peaks_spectrum(sx, nwinwidth=3).size == 0


@pytest.mark.xfail(strict=True, reason="a NaN inside the window does not invalidate the peak")
def test_nan_in_window_invalidates_peak():
    sx = np.array([0, 1, np.nan, 5, 4, 3, 0, 0], dtype=float)
    assert find_peaks_spectrum(sx, nwinwidth=5).size == 0
