"""Tests of the flux-preserving Gaussian fit (preserve_flux=True) in refine_peaks_spectrum"""

import logging
from math import erfc, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import quad

from numina.array.wavecalib.peaks_spectrum import (
    fit_gaussian_pixel_integrated,
    fit_skewnorm,
    gaussian_pixel_integrated,
    refine_peaks_spectrum,
    skewnorm_pixel_integrated,
    skewnorm_profile,
)

MOMENT_KEYS = ["flux", "centroid", "sigma", "skewness", "kurtosis_excess", "fwhm_np", "fwhm_np_left", "fwhm_np_right"]


def integrated_line(npix, flux, mu, sigma):
    """Gaussian line integrated over each pixel [i - 0.5, i + 0.5]"""
    return gaussian_pixel_integrated(np.arange(npix, dtype=float), flux, mu, sigma)


def refine(sx, jmax, nwinwidth=9, **kwargs):
    """refine_peaks_spectrum with gaussian+moments and preserve_flux=True"""
    return refine_peaks_spectrum(
        sx, np.array([jmax]), nwinwidth=nwinwidth, method="gaussian+moments", preserve_flux=True, **kwargs
    )


# --- gaussian_pixel_integrated and fit_gaussian_pixel_integrated -----------------------


def test_pixel_integrated_matches_numerical_integration():
    flux, mu, sigma = 7.0, 3.3, 0.8
    x = np.arange(8, dtype=float)
    amp = flux / (sigma * np.sqrt(2 * np.pi))
    expected = [quad(lambda t: amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2), xi - 0.5, xi + 0.5)[0] for xi in x]
    np.testing.assert_allclose(gaussian_pixel_integrated(x, flux, mu, sigma), expected, rtol=1e-10)


def test_pixel_integrated_sums_to_flux():
    np.testing.assert_allclose(integrated_line(200, 123.4, 100.3, 2.5).sum(), 123.4, rtol=1e-12)


def test_pixel_integrated_far_tails_are_accurate():
    # erf(b) - erf(a) would be exactly 0 in both tails; erfc keeps the precision
    value = gaussian_pixel_integrated(np.array([12.0, -12.0]), 1.0, 0.0, 1.0)
    expected = 0.5 * (erfc(11.5 / sqrt(2)) - erfc(12.5 / sqrt(2)))
    assert value[0] > 0
    np.testing.assert_allclose(value, [expected, expected], rtol=1e-10)


def test_fit_gaussian_pixel_integrated():
    x = np.arange(-4, 5, dtype=float)
    y = gaussian_pixel_integrated(x, 10.0, 0.3, 0.7)
    res = fit_gaussian_pixel_integrated(x, y, flux0=8.0, mu0=0.0, sigma0=1.0)
    assert res["flux"] == pytest.approx(10.0, rel=1e-6)
    assert res["mean"] == pytest.approx(0.3, abs=1e-6)
    assert res["sigma"] == pytest.approx(0.7, rel=1e-6)
    assert res["amp"] == pytest.approx(10.0 / (0.7 * np.sqrt(2 * np.pi)), rel=1e-6)


# --- refine_peaks_spectrum with gaussian+moments and preserve_flux=True ----------------


@pytest.mark.parametrize("sigma", [0.6, 1.0, 2.0])
def test_preserve_flux_recovers_intrinsic_gaussian(sigma):
    flux, mu = 500.0, 30.3
    sx = integrated_line(60, flux, mu, sigma)
    jmax = int(np.argmax(sx))
    xfpeaks, sfpeaks, dict_result = refine(sx, jmax)
    assert dict_result["fit_ok"]
    assert xfpeaks[0] == pytest.approx(mu, abs=1e-6)
    assert sfpeaks[0] == pytest.approx(sigma, rel=1e-6)
    assert dict_result["gaussian_mean"] == pytest.approx(xfpeaks[0])
    assert dict_result["gaussian_sigma"] == pytest.approx(sfpeaks[0])
    assert dict_result["gaussian_flux"] == pytest.approx(flux, rel=1e-6)
    assert dict_result["gaussian_amp"] == pytest.approx(flux / (sigma * np.sqrt(2 * np.pi)), rel=1e-6)
    # the Gaussian fitted to the pixel values (without preserve_flux) is
    # broader (sigma^2 + ~1/12)
    _, sfpeaks_sampled, _ = refine_peaks_spectrum(sx, np.array([jmax]), nwinwidth=9, method="gaussian+moments")
    assert sfpeaks_sampled[0] > sigma


def test_preserve_flux_keeps_the_moments():
    # the moments are computed as without preserve_flux, and the keys of the
    # flux-preserving fit do not overwrite them
    sx = integrated_line(60, 500.0, 30.3, 1.0)
    _, _, with_pf = refine(sx, 30)
    _, _, without_pf = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=9, method="gaussian+moments")
    for key in MOMENT_KEYS:
        assert with_pf[key] == without_pf[key]
    assert not any(key.startswith("gaussian_") for key in without_pf)


def test_preserve_flux_matches_pixel_values():
    # the integral of the fitted Gaussian over each pixel of the window
    # reproduces the signal of the pixel
    sx = integrated_line(60, 200.0, 25.6, 0.9)
    jmax = 26
    _, _, dict_result = refine(sx, jmax, nwinwidth=7)
    x_window = np.arange(jmax - 3, jmax + 4, dtype=float)
    model = gaussian_pixel_integrated(
        x_window, dict_result["gaussian_flux"], dict_result["gaussian_mean"], dict_result["gaussian_sigma"]
    )
    np.testing.assert_allclose(model, sx[jmax - 3 : jmax + 4], rtol=1e-6)


def test_preserve_flux_with_noise():
    rng = np.random.default_rng(3)
    flux, mu, sigma = 1000.0, 30.3, 0.8
    sx = integrated_line(60, flux, mu, sigma) + 1.0 + rng.normal(scale=1.0, size=60)
    xfpeaks, sfpeaks, dict_result = refine(sx, 30)
    assert xfpeaks[0] == pytest.approx(mu, abs=0.02)
    assert sfpeaks[0] == pytest.approx(sigma, rel=0.05)
    assert dict_result["gaussian_flux"] == pytest.approx(flux, rel=0.05)
    assert np.isfinite(dict_result["gaussian_sigma_err"])


@pytest.mark.parametrize("method", ["poly2", "gaussian"])
def test_preserve_flux_only_with_gaussian_moments_or_skewnorm(method):
    sx = integrated_line(60, 500.0, 30.3, 1.0)
    with pytest.raises(ValueError, match="preserve_flux=True is only available with method=gaussian\\+moments or"):
        refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method=method, preserve_flux=True)


def test_preserve_flux_single_peak_only():
    sx = integrated_line(100, 300.0, 20.3, 0.8) + integrated_line(100, 150.0, 70.8, 1.2)
    with pytest.raises(ValueError, match="only available when npeaks=1"):
        refine_peaks_spectrum(sx, np.array([20, 71]), nwinwidth=9, method="gaussian+moments", preserve_flux=True)


def test_preserve_flux_without_valid_initial_gaussian():
    # Gaussian whose maximum lies far outside the window: no valid initial
    # Gaussian, so the flux-preserving fit is not performed (the moments are)
    x = np.arange(60, dtype=float)
    sx = 100.0 * np.exp(-0.5 * ((x - 45.0) / 8.0) ** 2) + 1.0
    xfpeaks, sfpeaks, dict_result = refine(sx, 20, nwinwidth=7)
    assert not dict_result["fit_ok"]
    assert "gaussian_flux" not in dict_result
    assert "flux" in dict_result
    assert xfpeaks[0] == 20.0
    assert np.isnan(sfpeaks[0])


def test_preserve_flux_fallback_to_poly2(caplog):
    # null value in the window: poly2 is used, preserve_flux has no effect
    sx = integrated_line(60, 500.0, 30.3, 1.0)
    sx[32] = 0.0
    with caplog.at_level(logging.WARNING, logger="numina.array.wavecalib.peaks_spectrum"):
        _, sfpeaks, dict_result = refine(sx, 30, nwinwidth=7, debugplot=10)
    assert dict_result["final_method"] == "poly2"
    assert not dict_result["fit_ok"]
    assert "gaussian_flux" not in dict_result
    assert np.isnan(sfpeaks[0])
    assert any("using poly2" in r.getMessage() for r in caplog.records)


def test_gaussian_method_unchanged():
    # the plain gaussian method is not affected by the new fit
    sx = integrated_line(60, 500.0, 30.3, 1.0)
    xfpeaks, sfpeaks, dict_result = refine_peaks_spectrum(
        sx, np.array([30]), nwinwidth=7, method="gaussian", return_dict_result=True
    )
    assert sorted(dict_result) == ["amp", "final_method", "fit_ok", "mean", "method", "sigma"]


def test_preserve_flux_external_axes():
    sx = integrated_line(60, 500.0, 30.3, 0.8)
    fig, ax = plt.subplots()
    refine(sx, 30, nwinwidth=7, ax_external=ax)
    labels = [artist.get_label() for artist in ax.get_children()]
    assert "Pixel-integrated model" in labels
    assert "Fitted curve (flux preserved)" in labels
    plt.close(fig)


# --- skewnorm_pixel_integrated and method="skewnorm" with preserve_flux=True -----------


def integrated_skewnorm_line(npix, flux, xi, omega, alpha):
    """Skew-normal line integrated over each pixel [i - 0.5, i + 0.5]"""
    return skewnorm_pixel_integrated(np.arange(npix, dtype=float), flux, xi, omega, alpha)


@pytest.mark.parametrize("alpha", [-4.0, 0.0, 3.0])
def test_skewnorm_pixel_integrated_matches_numerical_integration(alpha):
    flux, xi, omega = 7.0, 3.3, 0.8
    amp = flux / (omega * np.sqrt(2 * np.pi))
    x = np.arange(8, dtype=float)
    expected = [quad(lambda t: skewnorm_profile(t, amp, xi, omega, alpha), xi_ - 0.5, xi_ + 0.5)[0] for xi_ in x]
    np.testing.assert_allclose(skewnorm_pixel_integrated(x, flux, xi, omega, alpha), expected, rtol=1e-9, atol=1e-15)


def test_skewnorm_pixel_integrated_matches_scipy():
    from scipy.stats import skewnorm

    x = np.arange(-10, 30, dtype=float)
    xi, omega, alpha = 5.3, 2.0, 4.0
    expected = skewnorm.cdf(x + 0.5, alpha, loc=xi, scale=omega) - skewnorm.cdf(x - 0.5, alpha, loc=xi, scale=omega)
    np.testing.assert_allclose(skewnorm_pixel_integrated(x, 1.0, xi, omega, alpha), expected, rtol=1e-8, atol=1e-15)


def test_skewnorm_pixel_integrated_sums_to_flux():
    np.testing.assert_allclose(integrated_skewnorm_line(200, 123.4, 100.3, 2.5, 3.0).sum(), 123.4, rtol=1e-12)


def test_fit_skewnorm_pixel_integrated():
    x = np.arange(-10, 11, dtype=float)
    y = skewnorm_pixel_integrated(x, 10.0, 0.3, 0.8, 3.0)
    res = fit_skewnorm(x, y, amp0=4.0, xi0=0.5, omega0=0.6, pixel_integrated=True)
    assert res["flux"] == pytest.approx(10.0, rel=1e-6)
    assert res["xi"] == pytest.approx(0.3, abs=1e-5)
    assert res["omega"] == pytest.approx(0.8, rel=1e-5)
    assert res["alpha"] == pytest.approx(3.0, rel=1e-4)
    assert res["amp"] == pytest.approx(10.0 / (0.8 * np.sqrt(2 * np.pi)), rel=1e-5)


def test_fit_skewnorm_without_pixel_integration_has_no_flux():
    x = np.arange(-10, 11, dtype=float)
    res = fit_skewnorm(x, skewnorm_profile(x, 1.0, 0.3, 2.0, 2.0), amp0=1.0, xi0=0.0, omega0=2.0)
    assert "flux" not in res


@pytest.mark.parametrize("omega", [0.7, 1.0, 2.0])
def test_skewnorm_preserve_flux_recovers_intrinsic_profile(omega):
    flux, xi, alpha = 500.0, 30.3, 3.0
    sx = integrated_skewnorm_line(60, flux, xi, omega, alpha)
    jmax = int(np.argmax(sx))
    xfpeaks, sfpeaks, dict_result = refine_peaks_spectrum(
        sx, np.array([jmax]), nwinwidth=7, method="skewnorm", preserve_flux=True
    )
    assert dict_result["fit_ok"]
    assert dict_result["flux"] == pytest.approx(flux, rel=1e-5)
    assert dict_result["xi"] == pytest.approx(xi, abs=1e-4)
    assert dict_result["omega"] == pytest.approx(omega, rel=1e-4)
    assert dict_result["alpha"] == pytest.approx(alpha, rel=1e-3)
    # fxpeaks and sxpeaks: mode and standard deviation of the intrinsic profile
    delta = alpha / np.sqrt(1 + alpha**2)
    assert sfpeaks[0] == pytest.approx(omega * np.sqrt(1 - 2 * delta**2 / np.pi), rel=1e-4)
    x = np.linspace(xi - 2 * omega, xi + 2 * omega, 400001)
    mode = x[np.argmax(skewnorm_profile(x, 1.0, xi, omega, alpha))]
    assert xfpeaks[0] == pytest.approx(mode, abs=1e-3)
    assert dict_result["mode"] == pytest.approx(xfpeaks[0])
    if omega >= 1.0:
        # the fit to the pixel values (without preserve_flux) gives a broader
        # profile; for narrower lines that fit is too degraded to compare
        _, sfpeaks_sampled, _ = refine_peaks_spectrum(sx, np.array([jmax]), nwinwidth=7, method="skewnorm")
        assert sfpeaks_sampled[0] > sfpeaks[0]


def test_skewnorm_preserve_flux_with_noise():
    rng = np.random.default_rng(5)
    flux, xi, omega, alpha = 1000.0, 30.3, 1.5, 3.0
    sx = integrated_skewnorm_line(60, flux, xi, omega, alpha) + rng.normal(scale=1.0, size=60)
    # with nwinwidth=7 the window of the initial Gaussian fit would include a
    # negative noise pixel on the steep side of the line (poly2 would be used);
    # the flux-preserving fit uses the whole spectrum in any case
    _, _, dict_result = refine_peaks_spectrum(
        sx, np.array([int(np.argmax(sx))]), nwinwidth=5, method="skewnorm", preserve_flux=True
    )
    assert dict_result["fit_ok"]
    assert dict_result["flux"] == pytest.approx(flux, rel=0.02)
    assert dict_result["omega"] == pytest.approx(omega, rel=0.1)
    assert np.isfinite(dict_result["flux_err"])


def test_skewnorm_without_preserve_flux_unchanged():
    sx = integrated_skewnorm_line(60, 500.0, 30.3, 1.0, 3.0)
    jmax = int(np.argmax(sx))
    _, _, dict_result = refine_peaks_spectrum(sx, np.array([jmax]), nwinwidth=7, method="skewnorm")
    assert dict_result["fit_ok"]
    assert "flux" not in dict_result


def test_skewnorm_preserve_flux_without_valid_initial_gaussian():
    # convex data: no initial Gaussian, so no skew-normal fit at all
    x = np.arange(60, dtype=float)
    sx = 10.0 + 0.5 * (x - 31.7) ** 2
    xfpeaks, sfpeaks, dict_result = refine_peaks_spectrum(
        sx, np.array([30]), nwinwidth=7, method="skewnorm", preserve_flux=True
    )
    assert not dict_result["fit_ok"]
    assert "flux" not in dict_result
    assert xfpeaks[0] == 30.0
    assert np.isnan(sfpeaks[0])


def test_skewnorm_preserve_flux_external_axes():
    sx = integrated_skewnorm_line(60, 500.0, 30.3, 1.0, 3.0)
    jmax = int(np.argmax(sx))
    fig, ax = plt.subplots()
    refine_peaks_spectrum(sx, np.array([jmax]), nwinwidth=7, method="skewnorm", preserve_flux=True, ax_external=ax)
    labels = [artist.get_label() for artist in ax.get_children()]
    assert "Pixel-integrated model" in labels
    assert any(label.startswith("Fitted skew-normal, flux preserved") for label in labels)
    plt.close(fig)


def test_skewnorm_preserve_flux_narrow_line_local_minimum():
    # omega = 0.7 pixels: the fit to the pixel values is a poor starting point
    # (omega ~ 0.2, alpha < 0) and a single fit from it converges to a local
    # minimum with alpha ~ 0; the multi-start recovers the true profile
    sx = integrated_skewnorm_line(60, 500.0, 30.3, 0.7, 3.0)
    _, _, dict_result = refine_peaks_spectrum(
        sx, np.array([int(np.argmax(sx))]), nwinwidth=7, method="skewnorm", preserve_flux=True
    )
    assert dict_result["alpha"] == pytest.approx(3.0, rel=1e-3)
    assert dict_result["omega"] == pytest.approx(0.7, rel=1e-4)
