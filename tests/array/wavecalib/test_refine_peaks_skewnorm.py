"""Tests of the skew-normal fit (method="skewnorm") in refine_peaks_spectrum"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numina.array.wavecalib.peaks_spectrum import fit_skewnorm, refine_peaks_spectrum, skewnorm_profile


def true_mode(amp, xi, omega, alpha):
    """Mode of a skew-normal profile evaluated on a very fine grid"""
    x = np.linspace(xi - 2 * omega, xi + 2 * omega, 400001)
    return x[np.argmax(skewnorm_profile(x, amp, xi, omega, alpha))]


def true_std(omega, alpha):
    """Standard deviation of the skew-normal distribution"""
    delta = alpha / np.sqrt(1 + alpha**2)
    return omega * np.sqrt(1 - 2 * delta**2 / np.pi)


# --- skewnorm_profile and fit_skewnorm -------------------------------------------


def test_profile_reduces_to_gaussian_for_alpha_zero():
    x = np.linspace(-10, 10, 101)
    np.testing.assert_allclose(skewnorm_profile(x, 3.0, 0.5, 2.0, 0.0), 3.0 * np.exp(-0.5 * ((x - 0.5) / 2.0) ** 2))


def test_profile_is_proportional_to_scipy_skewnorm():
    from scipy.stats import skewnorm

    x = np.linspace(-10, 10, 101)
    xi, omega, alpha = 0.5, 2.0, 3.0
    # skewnorm.pdf = 2/omega * phi(z) * Phi(alpha z) = amp-free profile / (omega * sqrt(2 pi))
    expected = skewnorm.pdf(x, alpha, loc=xi, scale=omega) * omega * np.sqrt(2 * np.pi)
    np.testing.assert_allclose(skewnorm_profile(x, 1.0, xi, omega, alpha), expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("alpha", [-5.0, -1.0, 0.5, 3.0, 10.0])
def test_fit_skewnorm_moments_match_scipy(alpha):
    from scipy.stats import skewnorm

    xi, omega = 0.3, 2.0
    x = np.linspace(-15, 15, 301)
    res = fit_skewnorm(x, skewnorm_profile(x, 1.0, xi, omega, alpha), amp0=1.0, xi0=xi, omega0=omega)
    mean, var, skew, kurt = skewnorm.stats(alpha, loc=xi, scale=omega, moments="mvsk")
    assert res["mean"] == pytest.approx(mean, abs=1e-6)
    assert res["std"] == pytest.approx(np.sqrt(var), rel=1e-6)
    assert res["skewness"] == pytest.approx(skew, abs=1e-6)
    assert res["kurtosis_excess"] == pytest.approx(kurt, abs=1e-6)


def test_fit_skewnorm_symmetric_data():
    # Near alpha = 0, alpha and xi are degenerate to first order: alpha is
    # only loosely constrained, while mode, std and skewness are not
    x = np.arange(-10, 11, dtype=float)
    y = skewnorm_profile(x, 2.0, 0.4, 2.0, 0.0)
    res = fit_skewnorm(x, y, amp0=1.8, xi0=0.0, omega0=2.2)
    assert res["alpha"] == pytest.approx(0.0, abs=1e-2)
    assert res["mode"] == pytest.approx(0.4, abs=1e-4)
    assert res["std"] == pytest.approx(2.0, rel=1e-4)
    assert res["skewness"] == pytest.approx(0.0, abs=1e-6)


# --- refine_peaks_spectrum with method="skewnorm" ----------------------------------


@pytest.mark.parametrize("alpha", [-4.0, 0.0, 3.0])
@pytest.mark.parametrize("nwinwidth", [3, 7])
def test_skewnorm_line(alpha, nwinwidth):
    # The fit uses the whole spectrum, so even a narrow window (used only
    # for the initial Gaussian fit) recovers the asymmetry of the line
    amp, xi, omega = 1000.0, 30.3, 3.0
    sx = skewnorm_profile(np.arange(60, dtype=float), amp, xi, omega, alpha)
    jmax = np.argmax(sx)
    xfpeaks, sfpeaks, dict_skewnorm = refine_peaks_spectrum(
        sx, np.array([jmax]), nwinwidth=nwinwidth, method="skewnorm"
    )
    assert dict_skewnorm["fit_ok"]
    assert dict_skewnorm["final_method"] == "skewnorm"
    assert xfpeaks[0] == pytest.approx(true_mode(amp, xi, omega, alpha), abs=1e-3)
    assert dict_skewnorm["mode"] == pytest.approx(xfpeaks[0])
    assert sfpeaks[0] == pytest.approx(true_std(omega, alpha), rel=1e-3)
    assert dict_skewnorm["std"] == pytest.approx(sfpeaks[0])
    assert dict_skewnorm["omega"] == pytest.approx(omega, rel=1e-3)
    assert dict_skewnorm["alpha"] == pytest.approx(alpha, abs=1e-2)
    assert dict_skewnorm["amp"] == pytest.approx(amp, rel=1e-3)
    if alpha != 0.0:
        # xi is degenerate with alpha only near alpha = 0
        assert dict_skewnorm["xi"] == pytest.approx(xi, abs=1e-3)


def test_skewnorm_line_with_noise():
    rng = np.random.default_rng(1)
    amp, xi, omega, alpha = 1000.0, 30.3, 3.0, 4.0
    sx = skewnorm_profile(np.arange(60, dtype=float), amp, xi, omega, alpha) + rng.normal(scale=5.0, size=60)
    jmax = np.argmax(sx)
    xfpeaks, sfpeaks, dict_skewnorm = refine_peaks_spectrum(sx, np.array([jmax]), nwinwidth=7, method="skewnorm")
    assert xfpeaks[0] == pytest.approx(true_mode(amp, xi, omega, alpha), abs=0.05)
    assert sfpeaks[0] == pytest.approx(true_std(omega, alpha), rel=0.02)
    assert dict_skewnorm["alpha"] > 2.0
    assert dict_skewnorm["skewness"] > 0


def test_skewnorm_more_than_one_peak_raises():
    sx = skewnorm_profile(np.arange(100, dtype=float), 1.0, 30.0, 3.0, 2.0)
    sx += skewnorm_profile(np.arange(100, dtype=float), 1.0, 70.0, 3.0, 2.0)
    with pytest.raises(ValueError, match="only available when npeaks=1"):
        refine_peaks_spectrum(sx, np.array([30, 70]), nwinwidth=7, method="skewnorm")


def test_skewnorm_too_short_spectrum_raises():
    sx = np.array([0.1, 1.0, 3.0, 0.5])
    with pytest.raises(ValueError, match="at least 5 pixels"):
        refine_peaks_spectrum(sx, np.array([2]), nwinwidth=3, method="skewnorm")


def test_skewnorm_minimum_length():
    sx = skewnorm_profile(np.arange(5, dtype=float), 1.0, 1.8, 1.0, 1.0)
    _, _, dict_skewnorm = refine_peaks_spectrum(sx, np.array([np.argmax(sx)]), nwinwidth=3, method="skewnorm")
    assert dict_skewnorm["fit_ok"]


@pytest.mark.parametrize("debugplot", [0, 10])
def test_skewnorm_without_initial_gaussian_fit(caplog, debugplot):
    # A null value in the fitting window prevents the initial Gaussian fit,
    # so poly2 is used and the skew-normal fit is not performed
    x = np.arange(60, dtype=float)
    sx = 100.0 - (x - 30.3) ** 2
    sx[31] = 0.0
    with caplog.at_level(logging.WARNING, logger="numina.array.wavecalib.peaks_spectrum"):
        xfpeaks, sfpeaks, dict_skewnorm = refine_peaks_spectrum(
            sx, np.array([30]), nwinwidth=5, method="skewnorm", debugplot=debugplot
        )
    assert dict_skewnorm["final_method"] == "poly2"
    assert not dict_skewnorm["fit_ok"]
    assert "alpha" not in dict_skewnorm
    assert np.isnan(sfpeaks[0])
    assert any("using poly2" in r.getMessage() for r in caplog.records) == (debugplot == 10)


def test_skewnorm_without_valid_gaussian_maximum():
    # Convex data around jmax: no valid initial Gaussian, jmax is returned
    x = np.arange(60, dtype=float)
    sx = 10.0 + 0.5 * (x - 31.7) ** 2
    xfpeaks, sfpeaks, dict_skewnorm = refine_peaks_spectrum(sx, np.array([30]), nwinwidth=7, method="skewnorm")
    assert dict_skewnorm["final_method"] == "skewnorm"
    assert not dict_skewnorm["fit_ok"]
    assert "alpha" not in dict_skewnorm
    assert xfpeaks[0] == 30.0
    assert np.isnan(sfpeaks[0])


def test_skewnorm_does_not_change_other_methods():
    sx = skewnorm_profile(np.arange(60, dtype=float), 1000.0, 30.3, 3.0, 3.0)
    result = refine_peaks_spectrum(sx, np.array([np.argmax(sx)]), nwinwidth=7, method="gaussian")
    assert len(result) == 2


def test_skewnorm_external_axes():
    sx = skewnorm_profile(np.arange(60, dtype=float), 1.0, 30.0, 3.0, 2.0)
    fig, ax = plt.subplots()
    refine_peaks_spectrum(sx, np.array([np.argmax(sx)]), nwinwidth=7, method="skewnorm", ax_external=ax)
    labels = [line.get_label() for line in ax.get_lines()]
    assert "Refined peak" in labels
    assert any(label.startswith("Fitted skew-normal") for label in labels)
    plt.close(fig)
