"""The figures saved in files are created without pyplot"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from numina.array.display.matplotlib_qt import new_figure
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.ximplotxy import ximplotxy
from numina.array.display.ximshow import ximshow, ximshow_file


@pytest.fixture(autouse=True)
def no_pyplot_figures():
    """The test does not create pyplot figures"""
    before = plt.get_fignums()
    yield
    assert plt.get_fignums() == before


def test_new_figure():
    fig = new_figure(interactive=False, figsize=(3, 2))
    assert isinstance(fig, Figure)
    assert fig.canvas.manager is None
    assert tuple(fig.get_size_inches()) == (3, 2)


def test_ximplotxy_fig():
    fig = Figure()
    ax = ximplotxy([1, 2, 3], [4, 5, 6], title="plot", show=True, fig=fig)
    assert ax.figure is fig
    assert ax.get_title() == "plot"


def test_ximshow_fig():
    fig = Figure()
    ax = ximshow(np.arange(100.0).reshape(10, 10), cbar_orientation="vertical", fig=fig)
    assert ax.figure is fig
    # image and color bar
    assert len(fig.axes) == 2


@pytest.mark.parametrize("output", ["pdf", "png"])
def test_ximshow_file(tmp_path, output):
    fitsfile = tmp_path / "image.fits"
    fits.PrimaryHDU(np.arange(100.0).reshape(10, 10)).writeto(fitsfile)
    filename = tmp_path / f"image.{output}"
    if output == "pdf":
        with PdfPages(filename) as pdf:
            ximshow_file(str(fitsfile), pdf=pdf)
    else:
        ximshow_file(str(fitsfile), png=filename)
    assert filename.stat().st_size > 0


def test_polfit_residuals_fig(tmp_path):
    x = np.arange(10.0)
    pdffile = tmp_path / "fit.pdf"
    fig = Figure()
    poly, yres = polfit_residuals(x, 2 * x + 1, deg=1, title="fit", show=True, debugplot=1, pdffile=pdffile, fig=fig)
    np.testing.assert_allclose(poly.coef, [1, 2])
    # data and residuals
    assert len(fig.axes) == 2
    assert fig.axes[1].get_title() == "fit"
    assert pdffile.stat().st_size > 0
