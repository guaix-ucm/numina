"""Tests of find_peaks_spectrum"""

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
    """Reimplementation of the documented algorithm based on runs.

    The spectrum is split into runs [p1, p2] of consecutive equal values
    (p1 == p2 for isolated values). A run is a peak when sx[p1] > threshold,
    the signal increases strictly before the run and decreases strictly
    after it, and it is non-decreasing in [p1 - nmed, p1] and
    non-increasing in [p2, p2 + nmed], with nmed = nwinwidth // 2. The
    peak is located at (p1 + p2) // 2, and runs starting before
    p2 + nwinwidth - 1 are not considered after a peak.
    """
    nmed = nwinwidth // 2
    npix = len(sx)
    # first and last pixel of each run of equal values
    starts = np.flatnonzero(np.r_[True, sx[1:] != sx[:-1]])
    ends = np.r_[starts[1:] - 1, npix - 1]
    peaks = []
    next_allowed = nmed
    for p1, p2 in zip(starts, ends):
        if p1 < next_allowed or p1 >= npix - nmed or p2 + nmed >= npix:
            continue
        if not (sx[p1] > threshold and sx[p1 - 1] < sx[p1] and sx[p2 + 1] < sx[p2]):
            continue
        left_ok = all(sx[k] <= sx[k + 1] for k in range(p1 - nmed, p1))
        right_ok = all(sx[k] >= sx[k + 1] for k in range(p2, p2 + nmed))
        if left_ok and right_ok:
            peaks.append((p1 + p2) // 2)
            next_allowed = p2 + nwinwidth - 1
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


def test_unsigned_integer_input():
    # The left flank 6 -> 3 -> 8 is not monotonic. A check based on
    # np.diff would accept it for unsigned integers (3 - 6 wraps around)
    values = [0, 6, 3, 8, 4, 1, 0]
    assert find_peaks_spectrum(np.array(values, dtype=float), nwinwidth=5).size == 0
    assert find_peaks_spectrum(np.array(values, dtype=np.uint16), nwinwidth=5).size == 0


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
def test_random_spectra_with_ties_match_reference(nwinwidth, seed):
    # Integer spectra with saturated lines: many runs of equal values
    rng = np.random.default_rng(seed)
    npix = 500
    centers = rng.uniform(10, npix - 10, size=25)
    sx = np.round(gaussian_lines(npix, centers, amplitude=30.0, sigma=2.0) + rng.normal(scale=1.0, size=npix))
    sx = np.minimum(sx, 20.0)
    threshold = 3.0
    np.testing.assert_array_equal(
        find_peaks_spectrum(sx, nwinwidth=nwinwidth, threshold=threshold),
        reference_find_peaks(sx, nwinwidth, threshold),
    )


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


# --- runs of equal values (plateaus) ------------------------------------------


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0, 1, 4, 7, 7, 4, 1, 0, 0], [3]),
        ([0, 1, 4, 7, 7, 7, 4, 1, 0, 0], [4]),
        ([0, 1, 4, 7, 7, 7, 7, 4, 1, 0, 0], [4]),
        ([0, 1, 1, 3, 7, 3, 3, 1, 0, 0], [4]),
        ([0, 1, 3, 3, 7, 4, 1, 0, 0, 0], [4]),
        ([0, 1, 5, 3, 3, 5, 1, 0, 0], [2, 5]),
    ],
    ids=["2px-top", "3px-plateau", "4px-plateau", "flank-ties", "shoulder", "flat-valley"],
)
def test_runs_of_equal_values(values, expected):
    # Same results as scipy.signal.find_peaks for nwinwidth=3
    np.testing.assert_array_equal(find_peaks_spectrum(np.array(values, dtype=float), nwinwidth=3), expected)


@pytest.mark.parametrize("dtype", [float, int])
def test_constant_signal_has_no_peaks(dtype):
    assert find_peaks_spectrum(np.ones(20, dtype=dtype), nwinwidth=5).size == 0


def test_saturated_line_gives_a_single_peak_at_its_centre():
    sx = np.minimum(gaussian_lines(60, [30.0], amplitude=100.0, sigma=3.0), 60.0)
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=7), [30])


def test_plateau_flank_beyond_border_is_ignored():
    # The right flank [p2, p2 + nmed] of the plateau [6, 8] does not fit
    sx = np.array([0, 0, 0, 1, 2, 3, 5, 5, 5, 4], dtype=float)
    assert find_peaks_spectrum(sx, nwinwidth=5).size == 0


def test_search_resumes_after_plateau():
    # Plateau [2, 4] -> peak at 3; the search resumes at 4 + 3 - 1 = 6
    sx = np.array([0, 1, 5, 5, 5, 1, 3, 1, 0], dtype=float)
    np.testing.assert_array_equal(find_peaks_spectrum(sx, nwinwidth=3), [3, 6])


def test_plateau_warning(capsys):
    sx = np.array([0, 1, 4, 7, 7, 7, 7, 4, 1, 0, 0], dtype=float)
    find_peaks_spectrum(sx, nwinwidth=3, debugplot=0)
    assert "plateau" not in capsys.readouterr().out
    find_peaks_spectrum(sx, nwinwidth=3, debugplot=10)
    assert "WARNING: plateau of 4 pixels in [3, 6]" in capsys.readouterr().out


# --- NaN values -------------------------------------------------------------------


def test_nan_at_line_centre_does_not_create_peaks():
    sx = np.array([0, 1, 2, np.nan, 2, 1, 0, 0], dtype=float)
    assert find_peaks_spectrum(sx, nwinwidth=3).size == 0


@pytest.mark.parametrize("inan", [1, 2, 4, 5])
def test_nan_in_window_invalidates_peak(inan):
    sx = np.array([0, 1, 2, 5, 4, 3, 0, 0], dtype=float)
    sx[inan] = np.nan
    assert find_peaks_spectrum(sx, nwinwidth=5).size == 0
