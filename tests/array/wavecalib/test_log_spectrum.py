"""Tests for numina.array.wavecalib.log_spectrum"""

import warnings

import numpy as np
import pytest

from numina.array.wavecalib.log_spectrum import log_spectrum


@pytest.mark.parametrize(
    "sp, expected",
    [
        ([2.0, 11.0, 101.0], [0.0, 1.0, 2.0]),
        ([0.0, 9.0, 99.0], [0.0, 1.0, 2.0]),
        ([-10.0, -1.0, 89.0], [0.0, 1.0, 2.0]),
        ([2, 11, 101], [0.0, 1.0, 2.0]),  # integer input
    ],
)
def test_percentile_zero(sp, expected):
    # percentile=0: log10(sp - min(sp) + 1)
    splog, label = log_spectrum(np.array(sp), percentile=0)
    assert np.allclose(splog, expected)
    assert label == "log10(counts - min(counts) + 1)"


def test_default_label():
    _, label = log_spectrum(np.arange(10.0))
    assert label == "log10(counts - P1 + 1)"


def test_negative_outliers():
    # a few strongly negative pixels must not compress the dynamic range
    rng = np.random.default_rng(1234)
    sp = rng.uniform(0, 10, 1000)
    sp[500] = 1.0e5
    sp[[10, 200, 700]] = -1.0e6
    splog, _ = log_spectrum(sp)
    spfloor = np.percentile(sp, 1)
    assert np.all(splog[[10, 200, 700]] == 0)
    assert splog.min() == 0
    assert np.isclose(splog.max(), np.log10(1.0e5 - spfloor + 1))
    # result without the outliers is not modified
    sp_no_outliers = np.delete(sp, [10, 200, 700])
    splog_no_outliers = np.delete(splog, [10, 200, 700])
    above = sp_no_outliers >= spfloor
    assert np.allclose(splog_no_outliers[above], np.log10(sp_no_outliers[above] - spfloor + 1))


@pytest.mark.parametrize("sp", [[-5.0, -3.0, -2.0], [0.0, 0.0, 0.0], [-1.0e6, 1.0, 2.0]])
def test_no_warnings_and_non_negative(sp):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        splog, _ = log_spectrum(np.array(sp))
    assert np.all(np.isfinite(splog))
    assert np.all(splog >= 0)


def test_non_finite_values():
    sp = np.array([np.nan, 0.0, 9.0, np.inf, 99.0])
    splog, _ = log_spectrum(sp, percentile=0)
    assert np.isnan(splog[0]) and np.isnan(splog[3])
    assert np.allclose(splog[[1, 2, 4]], [0.0, 1.0, 2.0])


def test_all_non_finite():
    splog, _ = log_spectrum(np.array([np.nan, np.nan]))
    assert np.all(np.isnan(splog))


@pytest.mark.parametrize("percentile", [-1, 101])
def test_invalid_percentile(percentile):
    with pytest.raises(ValueError, match="percentile must be in"):
        log_spectrum(np.arange(10.0), percentile=percentile)
