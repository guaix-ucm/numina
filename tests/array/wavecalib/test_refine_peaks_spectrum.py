"""Tests of refine_peaks_spectrum"""

import logging

import numpy as np
import pytest

from numina.array.wavecalib.peaks_spectrum import _parabola_vertex, refine_peaks_spectrum


def gaussian_line(npix, center, amplitude=100.0, sigma=2.0):
    """Gaussian line evaluated at integer pixels"""
    x = np.arange(npix, dtype=float)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


# --- _parabola_vertex -------------------------------------------------------------


X_FIT = np.arange(-3, 4, dtype=float)


@pytest.mark.parametrize(
    "coef, expected",
    [
        ([1.0, 0.4, -1.0], 0.2),  # concave, vertex inside the window
        ([1.0, 6.0, -1.0], 3.0),  # vertex exactly at the right edge
        ([1.0, -6.0, -1.0], -3.0),  # vertex exactly at the left edge
        ([1.0, 6.2, -1.0], None),  # vertex outside the window
        ([1.0, 0.4, 1.0], None),  # convex: the vertex is a minimum
        ([1.0, 0.4, 0.0], None),  # straight line
        ([1.0, 0.4], None),  # not a parabola
        ([1.0, 0.4, np.nan], None),
    ],
)
def test_parabola_vertex(coef, expected):
    result = _parabola_vertex(np.array(coef), X_FIT)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# --- regular lines --------------------------------------------------------------


@pytest.mark.parametrize("method", ["gaussian", "gaussian+moments"])
def test_gaussian_line(method):
    sx = gaussian_line(60, 30.3)
    result = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method=method)
    assert result[0][0] == pytest.approx(30.3, abs=1e-6)
    assert result[1][0] == pytest.approx(2.0, rel=1e-6)


def test_poly2_parabolic_line():
    x = np.arange(60, dtype=float)
    sx = 100.0 - (x - 30.3) ** 2
    xfpeaks, sfpeaks = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method="poly2")
    assert xfpeaks[0] == pytest.approx(30.3, abs=1e-6)
    # poly2 does not provide a width
    assert np.isnan(sfpeaks[0])


# --- no maximum within the fitting window ----------------------------------------


@pytest.mark.parametrize("method", ["poly2", "gaussian"])
def test_convex_data_returns_jmax(method):
    # Previously the vertex of the fitted parabola (a minimum, at 31.7)
    # was returned as the peak
    x = np.arange(60, dtype=float)
    sx = 10.0 + 0.5 * (x - 31.7) ** 2
    xfpeaks, sfpeaks = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method=method)
    assert xfpeaks[0] == 30.0
    if method == "gaussian":
        assert np.isnan(sfpeaks[0])


@pytest.mark.parametrize("method", ["poly2", "gaussian"])
def test_vertex_outside_window_returns_jmax(method):
    # Nearly flat concave profile with its maximum far from the window:
    # previously 200 (poly2) or 190.5 (gaussian) was returned for jmax=30
    x = np.arange(60, dtype=float)
    sx = 100.0 - 1e-4 * (x - 200.0) ** 2
    xfpeaks, sfpeaks = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method=method)
    assert xfpeaks[0] == 30.0
    if method == "gaussian":
        # the parabola fitted to log(y) is concave, but its maximum is far
        # outside the window: the width is not meaningful
        assert np.isnan(sfpeaks[0])


@pytest.mark.parametrize("debugplot", [10, -10])
def test_no_maximum_warning(caplog, debugplot):
    x = np.arange(60, dtype=float)
    sx = 10.0 + 0.5 * (x - 31.7) ** 2
    logger_name = "numina.array.wavecalib.peaks_spectrum"
    with caplog.at_level(logging.INFO, logger=logger_name):
        refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method="poly2", debugplot=0)
    assert not caplog.records
    with caplog.at_level(logging.INFO, logger=logger_name):
        refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method="poly2", debugplot=debugplot)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "No maximum within the fitting window around 30" in warnings[0].getMessage()


# --- normalization ----------------------------------------------------------------


def test_negative_spectrum_gaussian_falls_back_to_poly2():
    # All values in the window are negative. Previously the data were
    # divided by the (negative) maximum, which flipped the profile and
    # passed the check for non-positive values
    x = np.arange(60, dtype=float)
    sx = -50.0 - (x - 30.3) ** 2
    xfpeaks, sfpeaks = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method="gaussian")
    assert xfpeaks[0] == pytest.approx(30.3, abs=1e-6)
    # poly2 does not provide a width
    assert np.isnan(sfpeaks[0])


def test_negative_spectrum_gaussian_moments_not_computed():
    # Previously meaningless moments were computed on the flipped profile
    sx = -5.0 + 3.0 * np.exp(-0.5 * ((np.arange(60) - 20) / 2.0) ** 2)
    _, _, dict_result = refine_peaks_spectrum(sx, np.array([20]), nwinwidth=7, method="gaussian+moments")
    assert dict_result["method"] == "gaussian+moments"
    assert dict_result["final_method"] == "poly2"
    assert not dict_result["fit_ok"]
    assert "flux" not in dict_result


@pytest.mark.parametrize("method", ["poly2", "gaussian"])
def test_input_is_not_modified(method):
    sx = gaussian_line(60, 30.3)
    sx_orig = sx.copy()
    refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method=method)
    np.testing.assert_array_equal(sx, sx_orig)


# --- gaussian+moments ---------------------------------------------------------------


def test_gaussian_moments_values():
    # Single line on a cutout: flux in the units of sx (amplitude * sigma * sqrt(2 pi))
    amplitude, center, sigma = 250.0, 30.3, 2.0
    sx = gaussian_line(60, center, amplitude=amplitude, sigma=sigma)
    _, _, dict_moments = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method="gaussian+moments")
    assert dict_moments["flux"] == pytest.approx(amplitude * sigma * np.sqrt(2 * np.pi), rel=1e-6)
    assert dict_moments["centroid"] == pytest.approx(center, abs=1e-6)
    assert dict_moments["sigma"] == pytest.approx(sigma, rel=1e-3)
    assert dict_moments["skewness"] == pytest.approx(0.0, abs=1e-6)
    assert dict_moments["kurtosis_excess"] == pytest.approx(0.0, abs=1e-2)
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert dict_moments["fwhm_np"] == pytest.approx(fwhm, rel=2e-2)
    assert dict_moments["fwhm_np_left"] < center < dict_moments["fwhm_np_right"]


def test_gaussian_moments_flux_independent_of_peak_window():
    # The flux must not depend on the maximum within the fitting window
    # (the windows of jmax=28 and jmax=32 do not include the line centre)
    sx = gaussian_line(60, 30.0, amplitude=100.0, sigma=2.0)
    flux = [
        refine_peaks_spectrum(sx, np.array([jmax]), nwinwidth=3, method="gaussian+moments")[2]["flux"]
        for jmax in [28, 30, 32]
    ]
    assert flux == pytest.approx([flux[0]] * 3)


@pytest.mark.parametrize("debugplot", [0, 10])
def test_gaussian_moments_non_positive_flux(caplog, debugplot):
    # Positive values around the line but negative integrated flux: previously
    # the ValueError raised by line_moments was not caught
    sx = gaussian_line(60, 30.0, amplitude=10.0, sigma=1.0) - 1.0
    sx[29:32] += 1.0
    with caplog.at_level(logging.WARNING, logger="numina.array.wavecalib.peaks_spectrum"):
        xfpeaks, _, dict_result = refine_peaks_spectrum(
            sx, np.array([30]), nwinwidth=3, method="gaussian+moments", debugplot=debugplot
        )
    assert dict_result["final_method"] == "gaussian+moments"
    assert not dict_result["fit_ok"]
    assert "flux" not in dict_result
    assert xfpeaks[0] == pytest.approx(30.0, abs=1e-6)
    assert any("not positive" in r.getMessage() for r in caplog.records) == (debugplot == 10)


# --- arguments ----------------------------------------------------------------------


def test_method_is_required():
    with pytest.raises(TypeError, match="method"):
        refine_peaks_spectrum(gaussian_line(60, 30.0), np.array([30]), nwinwidth=7)


@pytest.mark.parametrize("method", [None, "gauss", "poly3"])
def test_invalid_method(method):
    with pytest.raises(ValueError, match="Invalid method"):
        refine_peaks_spectrum(gaussian_line(60, 30.0), np.array([30]), nwinwidth=7, method=method)


# --- return_dict_result -----------------------------------------------------------


def test_geometry_keeps_its_position():
    # return_dict_result is the last argument, so geometry can still be
    # passed by position without changing the number of returned values
    sx = gaussian_line(60, 30.3)
    result = refine_peaks_spectrum(sx, np.array([30]), 7, "gaussian", (0, 0, 640, 480))
    assert len(result) == 2


def test_return_dict_result_requires_a_single_peak():
    sx = gaussian_line(60, 20.0) + gaussian_line(60, 40.0)
    with pytest.raises(ValueError, match="return_dict_result"):
        refine_peaks_spectrum(sx, np.array([20, 40]), nwinwidth=7, method="gaussian", return_dict_result=True)


def test_return_dict_result_gaussian():
    amplitude, center, sigma = 250.0, 30.3, 2.0
    sx = gaussian_line(60, center, amplitude=amplitude, sigma=sigma)
    xfpeaks, sfpeaks, dict_result = refine_peaks_spectrum(
        sx, np.array([30]), nwinwidth=7, method="gaussian", return_dict_result=True
    )
    assert dict_result["method"] == dict_result["final_method"] == "gaussian"
    assert dict_result["fit_ok"]
    assert dict_result["amp"] == pytest.approx(amplitude, rel=1e-6)
    assert dict_result["mean"] == pytest.approx(xfpeaks[0])
    assert dict_result["sigma"] == pytest.approx(sfpeaks[0])


def test_return_dict_result_gaussian_without_valid_maximum():
    # Convex data: no valid Gaussian, so fit_ok is False and the Gaussian
    # parameters are not computed
    x = np.arange(60, dtype=float)
    sx = 10.0 + 0.5 * (x - 31.7) ** 2
    xfpeaks, sfpeaks, dict_result = refine_peaks_spectrum(
        sx, np.array([30]), nwinwidth=7, method="gaussian", return_dict_result=True
    )
    assert not dict_result["fit_ok"]
    assert "amp" not in dict_result
    assert xfpeaks[0] == 30.0
    assert np.isnan(sfpeaks[0])


def test_return_dict_result_poly2():
    x = np.arange(60, dtype=float)
    sx = 100.0 - (x - 30.3) ** 2
    _, _, dict_result = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method="poly2", return_dict_result=True)
    assert dict_result["fit_ok"]
    # coefficients of the parabola fitted to the data normalized to the
    # maximum of the window, relative to jmax
    c0, c1, c2 = dict_result["coef"]
    assert -c1 / (2 * c2) == pytest.approx(0.3, abs=1e-6)


@pytest.mark.parametrize("method", ["poly2", "gaussian"])
def test_return_dict_result_fallback_or_invalid_vertex(method):
    # poly2 with its vertex outside the window, or gaussian replaced by
    # poly2 because of null values in the window: fit_ok is False
    x = np.arange(60, dtype=float)
    if method == "poly2":
        sx = 100.0 - 1e-4 * (x - 200.0) ** 2
    else:
        sx = 100.0 - (x - 30.3) ** 2
        sx[31] = 0.0
    _, _, dict_result = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method=method, return_dict_result=True)
    assert dict_result["final_method"] == "poly2"
    assert not dict_result["fit_ok"]
