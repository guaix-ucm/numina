"""Tests of the Strehl ratio (fwhm_diffraction) computed by refine_peaks_spectrum"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numina.array.wavecalib.peaks_spectrum import (
    fit_gaussian_pixel_integrated,
    gaussian_pixel_integrated,
    refine_peaks_spectrum,
)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
FWHM_DIFFRACTION = 3.0  # pixels
SIGMA_DIFFRACTION = FWHM_DIFFRACTION * FWHM_TO_SIGMA


def integrated_line(npix, flux, mu, sigma):
    """Gaussian line integrated over each pixel [i - 0.5, i + 0.5]"""
    return gaussian_pixel_integrated(np.arange(npix, dtype=float), flux, mu, sigma)


def strehl(sx, jmax, nwinwidth=5, **kwargs):
    """Dictionary of results of gaussian+moments with preserve_flux and fwhm_diffraction"""
    _, _, dict_result = refine_peaks_spectrum(
        sx,
        np.array([jmax]),
        nwinwidth=nwinwidth,
        method="gaussian+moments",
        preserve_flux=True,
        fwhm_diffraction=FWHM_DIFFRACTION,
        **kwargs,
    )
    return dict_result


# --- values -------------------------------------------------------------------------


@pytest.mark.parametrize("mu", [60.0, 60.25, 60.5])
def test_diffraction_limited_profile_has_unit_strehl(mu):
    # independent of the position of the line centre within the pixel
    sx = integrated_line(120, 1000.0, mu, SIGMA_DIFFRACTION)
    dict_result = strehl(sx, int(round(mu)))
    assert dict_result["fit_ok"]
    assert dict_result["strehl"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("factor", [1.5, 2.0, 4.0])
def test_broadened_gaussian(factor):
    # a Gaussian broadened by a factor has a Strehl ratio 1 / factor
    sx = integrated_line(120, 1000.0, 60.3, factor * SIGMA_DIFFRACTION)
    nwinwidth = 2 * int(factor * 2) + 1  # window covering the core
    assert strehl(sx, 60, nwinwidth=nwinwidth)["strehl"] == pytest.approx(1.0 / factor, rel=1e-5)


@pytest.mark.parametrize("core_fraction", [0.9, 0.7, 0.5])
def test_core_and_halo(core_fraction):
    # diffraction-limited core plus a broad halo (sigma 5 times larger) with
    # the same centre: the true Strehl ratio is f + (1 - f) / 5. The estimate
    # is close to it because the total flux of the moments includes the
    # halo, while the ratio of widths of the Gaussian fit of the core
    # (which ignores the halo) is much larger
    f = core_fraction
    sx = integrated_line(120, 1000.0 * f, 60.3, SIGMA_DIFFRACTION)
    sx += integrated_line(120, 1000.0 * (1 - f), 60.3, 5 * SIGMA_DIFFRACTION)
    dict_result = strehl(sx, 60)
    true_strehl = f + (1 - f) / 5
    assert dict_result["strehl"] == pytest.approx(true_strehl, abs=0.02)
    assert dict_result["strehl"] < SIGMA_DIFFRACTION / dict_result["gaussian_sigma"] - 0.05


def test_strehl_with_noise():
    rng = np.random.default_rng(7)
    sx = integrated_line(120, 1000.0, 60.3, 1.2 * SIGMA_DIFFRACTION) + rng.normal(scale=0.5, size=120)
    dict_result = strehl(sx, 60)
    assert dict_result["strehl"] == pytest.approx(1 / 1.2, rel=0.03)
    assert np.isfinite(dict_result["gaussian_amp_err"])


def test_gaussian_amp_err_matches_propagation():
    rng = np.random.default_rng(1)
    x = np.arange(-4, 5, dtype=float)
    y = gaussian_pixel_integrated(x, 10.0, 0.3, 0.7) + rng.normal(scale=0.05, size=x.size)
    res = fit_gaussian_pixel_integrated(x, y, flux0=8.0, mu0=0.0, sigma0=1.0)
    # amp = flux / (sigma sqrt(2 pi)): relative errors add in quadrature when
    # the correlation between flux and sigma is neglected; the propagated
    # error (with the covariance) must be finite and of the same order
    rel = np.hypot(res["flux_err"] / res["flux"], res["sigma_err"] / res["sigma"])
    assert 0 < res["amp_err"] <= 1.5 * rel * res["amp"]


# --- availability -----------------------------------------------------------------------


def test_no_strehl_without_fwhm_diffraction():
    sx = integrated_line(120, 1000.0, 60.3, SIGMA_DIFFRACTION)
    _, _, dict_result = refine_peaks_spectrum(
        sx, np.array([60]), nwinwidth=5, method="gaussian+moments", preserve_flux=True
    )
    assert "strehl" not in dict_result


def test_no_strehl_when_flux_preserving_fit_fails():
    # maximum of the line far from the window: no valid initial Gaussian
    x = np.arange(60, dtype=float)
    sx = 100.0 * np.exp(-0.5 * ((x - 45.0) / 8.0) ** 2) + 1.0
    dict_result = strehl(sx, 20, nwinwidth=7)
    assert not dict_result["fit_ok"]
    assert "strehl" not in dict_result
    assert "flux" in dict_result


@pytest.mark.parametrize(
    "method, preserve_flux",
    [("gaussian+moments", False), ("gaussian", False), ("poly2", False), ("skewnorm", True)],
)
def test_fwhm_diffraction_requires_gaussian_moments_and_preserve_flux(method, preserve_flux):
    sx = integrated_line(120, 1000.0, 60.3, SIGMA_DIFFRACTION)
    with pytest.raises(ValueError, match="fwhm_diffraction"):
        refine_peaks_spectrum(
            sx, np.array([60]), nwinwidth=5, method=method, preserve_flux=preserve_flux, fwhm_diffraction=3.0
        )


@pytest.mark.parametrize("fwhm", [0.0, -1.0, np.nan])
def test_fwhm_diffraction_must_be_positive(fwhm):
    sx = integrated_line(120, 1000.0, 60.3, SIGMA_DIFFRACTION)
    with pytest.raises(ValueError, match="must be positive"):
        refine_peaks_spectrum(
            sx, np.array([60]), nwinwidth=5, method="gaussian+moments", preserve_flux=True, fwhm_diffraction=fwhm
        )


def test_strehl_in_plot_title():
    sx = integrated_line(120, 1000.0, 60.3, 2.0 * SIGMA_DIFFRACTION)
    fig, ax = plt.subplots()
    dict_result = strehl(sx, 60, nwinwidth=9, ax_external=ax)
    assert f"Strehl={dict_result['strehl']:.3f}" in ax.get_title()
    plt.close(fig)
