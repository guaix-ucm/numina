"""The figures saved in PDF files are created without pyplot"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backends.backend_pdf import PdfPages
from numpy.polynomial import Polynomial

from numina.array.wavecalib.arccalibration import refine_arccalibration
from numina.array.wavecalib.check_wlcalib import check_wlcalib_sp
from numina.array.wavecalib.crosscorrelation import periodic_corr1d

WV_MASTER = np.linspace(4050, 4950, 15)


@pytest.fixture(autouse=True)
def no_pyplot_figures():
    """The test does not create pyplot figures"""
    before = plt.get_fignums()
    yield
    assert plt.get_fignums() == before


@pytest.fixture
def arc():
    """Arc spectrum with lines in WV_MASTER, CRVAL1=4000, CDELT1=1"""
    wv = 4000.0 + np.arange(1000)
    sp = np.full_like(wv, 10.0)
    for wv_line in WV_MASTER:
        sp += 1000 * np.exp(-0.5 * ((wv - wv_line) / 1.5) ** 2)
    return sp


def test_refine_arccalibration_pdf(tmp_path, arc):
    with PdfPages(tmp_path / "refine.pdf") as pdf:
        poly, summary = refine_arccalibration(
            arc, Polynomial([4000.5, 1.0]), WV_MASTER, poldeg=2, plottitle="arc", pdf=pdf
        )
        assert pdf.get_pagecount() == 1
    np.testing.assert_allclose(poly.coef[:2], [3999.0, 1.0], atol=1e-3)
    assert summary["npoints"] == len(WV_MASTER)


def test_check_wlcalib_sp_pdf(tmp_path, arc):
    with PdfPages(tmp_path / "check.pdf") as pdf:
        check_wlcalib_sp(arc, crpix1=1.0, crval1=4000.0, cdelt1=1.0, wv_master=WV_MASTER, pdf=pdf)
        assert pdf.get_pagecount() == 1


def test_periodic_corr1d_pdf(tmp_path, arc):
    with PdfPages(tmp_path / "corr.pdf") as pdf:
        offset, fpeak = periodic_corr1d(arc, np.roll(arc, 3), norm_spectra=True, plottitle="corr", pdf=pdf)
        # spectra, normalized spectra and correlation
        assert pdf.get_pagecount() == 3
    assert offset == pytest.approx(3.0)
