"""Warnings logged and details in the errors of the array functions"""

import logging

import numpy as np
import pytest

from numina.array.crmasks.all_valid_numbers import all_valid_numbers
from numina.array.wavecalib.core import wvcal_spectrum
from numina.array.wavecalib.resample import resample_image2d_flux, shiftx_image2d_flux
from numina.frame.combine import _inspect_method


def test_not_enough_lines_is_logged(caplog, capsys):
    caplog.set_level(logging.WARNING)
    assert wvcal_spectrum(np.zeros(100), np.array([10.0, 20.0]), 3, np.array([1.0, 2.0])) is None
    assert "not enough lines to fit the spectrum, 2 lines for a polynomial of degree 3" in caplog.text
    assert capsys.readouterr().out == ""


def test_details_in_the_errors():
    with pytest.raises(ValueError, match=r"image2d_orig.shape=\(2, 2, 2\)"):
        shiftx_image2d_flux(np.zeros((2, 2, 2)), 1.0)
    with pytest.raises(ValueError, match=r"image2d_orig.shape=\(2, 2, 2\)"):
        resample_image2d_flux(np.zeros((2, 2, 2)), 10, 1.0, 1.0, 1.0, [0.0, 1.0])
    with pytest.raises(ValueError, match=r"length 3, \[1, 2\] has length 2"):
        all_valid_numbers([1, 2], fixed_length=3)
    with pytest.raises(KeyError, match="invalid method other, valid methods are"):
        _inspect_method("other")
