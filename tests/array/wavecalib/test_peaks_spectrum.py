"""Refinement of the peaks of a spectrum"""

import numpy as np
import pytest

from numina.array.wavecalib.peaks_spectrum import refine_peaks_spectrum


def line_spectrum(center=50.3, sigma=2.0):
    """Spectrum with one Gaussian line on a constant background"""
    x = np.arange(100)
    return 10.0 + 1000.0 * np.exp(-0.5 * ((x - center) / sigma) ** 2)


@pytest.mark.parametrize("method", ["poly2", "gaussian"])
def test_refine_no_peaks(method):
    # a spectrum without lines, as a fiber without peaks in an arc
    fxpeaks, sxpeaks = refine_peaks_spectrum(line_spectrum(), np.array([], dtype=int), 5, method=method)
    assert fxpeaks.shape == (0,)
    assert sxpeaks.shape == (0,)


def test_refine_gaussian():
    fxpeaks, sxpeaks = refine_peaks_spectrum(line_spectrum(), np.array([50]), 5, method="gaussian")
    assert np.allclose(fxpeaks, [50.3], atol=0.05)
    assert sxpeaks.shape == (1,)


@pytest.mark.parametrize("ixpeaks", [[], [30, 50]])
def test_refine_moments_one_peak(ixpeaks):
    with pytest.raises(ValueError, match="only available when npeaks=1"):
        refine_peaks_spectrum(line_spectrum(), np.array(ixpeaks, dtype=int), 5, method="gaussian+moments")


def test_refine_invalid_method():
    with pytest.raises(ValueError, match="Invalid method"):
        refine_peaks_spectrum(line_spectrum(), np.array([], dtype=int), 5, method="spline")
