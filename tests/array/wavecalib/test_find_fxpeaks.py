"""The plots of find_fxpeaks are requested with extra_plots, not with the level of the logger"""

import logging

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import pytest

import numina.array.wavecalib.__main__ as wavecalib_main
from numina.array.wavecalib.__main__ import find_fxpeaks

CENTERS = [200, 500, 800]


def arc_spectrum():
    """Spectrum with three lines on a noisy background"""
    rng = np.random.default_rng(1)
    x = np.arange(1000)
    sp = rng.normal(100.0, 5.0, x.size)
    for center in CENTERS:
        sp += 5000 * np.exp(-0.5 * ((x - center) / 2.0) ** 2)
    return sp


def call_find_fxpeaks(**kwargs):
    return find_fxpeaks(
        arc_spectrum(),
        times_sigma_threshold=10,
        minimum_threshold=0,
        nwinwidth_initial=7,
        nwinwidth_refined=5,
        npix_avoid_border=6,
        nbrightlines=[0],
        sigma_gaussian_filtering=0,
        minimum_gaussian_filtering=0,
        **kwargs,
    )


@pytest.fixture
def pauses(monkeypatch):
    """Record the calls to pause_debugplot instead of showing the figures"""
    calls = []
    monkeypatch.setattr(wavecalib_main, "pause_debugplot", lambda **kwargs: calls.append(kwargs))
    yield calls
    plt.close("all")


def test_no_plots_with_logger_debug(caplog, pauses):
    # numina run leaves the logger numina at DEBUG
    caplog.set_level(logging.DEBUG, logger="numina")
    fxpeaks, _ = call_find_fxpeaks()
    assert np.allclose(fxpeaks, CENTERS, atol=0.1)
    assert pauses == []
    assert plt.get_fignums() == []


def test_extra_plots(pauses):
    fxpeaks, _ = call_find_fxpeaks(extra_plots=True, plottitle="arc")
    assert np.allclose(fxpeaks, CENTERS, atol=0.1)
    assert pauses == [{"debugplot": 2, "pltshow": True}]
    assert plt.gcf().axes[0].get_title() == "arc"


class StopAfterPeaks(Exception):
    pass


@pytest.mark.parametrize("options, extra_plots", [([], False), (["--extra-plots"], True)])
def test_main_extra_plots(tmp_path, monkeypatch, options, extra_plots):
    fitsfile = tmp_path / "arc.fits"
    fits.writeto(fitsfile, np.tile(arc_spectrum(), (3, 1)))
    wv_master_file = tmp_path / "lines.txt"
    wv_master_file.write_text("4000.0\n5000.0\n6000.0\n")

    received = {}

    def fake_find_fxpeaks(**kwargs):
        received.update(kwargs)
        raise StopAfterPeaks

    monkeypatch.setattr(wavecalib_main, "find_fxpeaks", fake_find_fxpeaks)
    args = [str(fitsfile), "--scans", "2", "--wv_master_file", str(wv_master_file), "--degree", "1"]
    # without --nbrightlines, main fails: its default is the integer 0
    args += ["--nbrightlines", "0"]
    with pytest.raises(StopAfterPeaks):
        wavecalib_main.main(args + options)
    assert received["extra_plots"] is extra_plots
