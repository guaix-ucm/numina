"""Tests of refine_peaks_spectrum for peaks close to the borders of the spectrum"""

import numpy as np
import pytest

from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum


def profile(x, center, method):
    """Line profile that the given method reproduces exactly"""
    if method == "poly2":
        return 100.0 - (x - center) ** 2
    return 100.0 * np.exp(-0.5 * ((x - center) / 2.0) ** 2)


@pytest.mark.parametrize("method", ["poly2", "gaussian"])
@pytest.mark.parametrize("center", [1.3, 0.2, 47.6, 48.9])
def test_peak_close_to_border(method, center):
    # The fitting window [j1, j2) is shifted to fit inside the spectrum,
    # so it is not centred on jmax
    x = np.arange(50, dtype=float)
    sx = profile(x, center, method)
    ixpeaks = np.array([np.argmax(sx)])
    xfpeaks, _ = refine_peaks_spectrum(sx, ixpeaks, nwinwidth=7, method=method)
    assert xfpeaks[0] == pytest.approx(center, abs=1e-6)


@pytest.mark.parametrize("ixpeak", [1, 5])
def test_spectrum_as_long_as_window(ixpeak):
    # A spectrum with exactly 2 * nmed + 1 pixels must be accepted
    x = np.arange(7, dtype=float)
    center = ixpeak + 0.3
    sx = profile(x, center, "gaussian")
    xfpeaks, sfpeaks = refine_peaks_spectrum(sx, np.array([ixpeak]), nwinwidth=7, method="gaussian")
    assert xfpeaks[0] == pytest.approx(center, abs=1e-6)
    assert sfpeaks[0] == pytest.approx(2.0, rel=1e-6)


def test_spectrum_shorter_than_window():
    sx = profile(np.arange(6, dtype=float), 2.0, "gaussian")
    with pytest.raises(ValueError, match="Unexpected"):
        refine_peaks_spectrum(sx, np.array([2]), nwinwidth=7, method="gaussian")


@pytest.mark.parametrize("nwinwidth", [1, 2, 4, 6])
def test_invalid_nwinwidth_raises_valueerror(nwinwidth):
    sx = profile(np.arange(50, dtype=float), 20.3, "gaussian")
    with pytest.raises(ValueError, match="odd integer"):
        refine_peaks_spectrum(sx, np.array([20]), nwinwidth=nwinwidth, method="gaussian")
