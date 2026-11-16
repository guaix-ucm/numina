"""Tests for the wavelength labels in the figure of check_wlcalib_sp"""

import re

import numpy as np
import pytest

from numina.array.wavecalib.check_wlcalib import _number_of_decimals, check_wlcalib_sp

# line list with different number of decimals (maximum: 4)
WV_MASTER = np.array([4050.0, 4158.591, 4200.6745, 4345.17, 4545.0519, 4764.865, 4879.9, 4965.08])


@pytest.mark.parametrize(
    "values, expected",
    [
        ([4050.0, 4158.591, 4200.6745], 4),
        ([4050.0, 5000.0], 0),
        ([4050.5, 5000.25], 2),
        (np.array([6965.4307, 7067.218]) / 10, 5),  # after Angstrom -> nm conversion
        ([1 / 3], 10),  # limited to max_decimals
        ([], 0),
        ([np.nan, 4050.25], 2),
    ],
)
def test_number_of_decimals(values, expected):
    assert _number_of_decimals(values) == expected


def test_max_decimals():
    assert _number_of_decimals([1 / 3], max_decimals=3) == 3


class FakePdf:
    """Store the figure instead of saving it in a PDF file"""

    def __init__(self):
        self.figures = []

    def savefig(self, fig):
        fig.canvas.draw()
        self.figures.append(fig)


def test_labels_same_number_of_decimals():
    # arc spectrum with CRPIX1=1, CRVAL1=4000, CDELT1=1
    wv = 4000.0 + np.arange(1000)
    sp = np.full_like(wv, 10.0)
    for wv_line in WV_MASTER:
        sp += 1000 * np.exp(-0.5 * ((wv - wv_line) / 1.5) ** 2)
    pdf = FakePdf()
    check_wlcalib_sp(sp, crpix1=1.0, crval1=4000.0, cdelt1=1.0, wv_master=WV_MASTER, pdf=pdf)
    ax_spectrum = pdf.figures[0].axes[1]
    labels = [text.get_text() for text in ax_spectrum.texts]
    # identified lines: 'wavelength(number)'; estimated wavelengths: 'wavelength'
    wavelengths = [re.sub(r"\(\d+\)$", "", label) for label in labels]
    assert len(wavelengths) == 2 * len(WV_MASTER)
    for wavelength in wavelengths:
        assert re.fullmatch(r"\d+\.\d{4}", wavelength), wavelength
    identified = sorted(float(label.split("(")[0]) for label in labels if label.endswith(")"))
    assert np.allclose(identified, WV_MASTER)
