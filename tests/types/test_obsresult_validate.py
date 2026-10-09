"""Validation of the observation result: the raw images and the validator of the mode"""

import astropy.io.fits as fits
import pytest

import numina.core.config as cfg
import numina.types.obsresult as obstype
from numina.core import ObservationResult
from numina.core.pipeline import ObservingMode
from numina.exceptions import ValidationError
from numina.types.dataframe import DataFrame


def create_obsres(values):
    obsres = ObservationResult(instrument="TESTVAL", mode="mode1")
    for value in values:
        hdu = fits.PrimaryHDU()
        hdu.header["VALUE"] = value
        obsres.frames.append(DataFrame(frame=fits.HDUList([hdu])))
    return obsres


@pytest.fixture
def mode(monkeypatch):
    mode = ObservingMode("TESTVAL")
    mode.key = "mode1"
    mode.rawimage = "RAWTYPE"
    monkeypatch.setattr(obstype, "_obtain_mode", lambda instrument, mode_key: mode)
    return mode


@pytest.fixture
def checker(monkeypatch):
    calls = []

    def check(hdulist, astype=None, level=None):
        calls.append(astype)
        if hdulist[0].header["VALUE"] < 0:
            raise ValueError("VALUE must be positive")
        return True

    monkeypatch.setitem(cfg.check._loaders, "TESTVAL", check)
    return calls


def test_raw_frames_valid(mode, checker):
    obstype.ObservationResultType().validate(create_obsres([1, 2]))
    assert checker == ["RAWTYPE", "RAWTYPE"]


def test_raw_frames_invalid(mode, checker):
    with pytest.raises(ValidationError, match="frame 1: ValueError: VALUE must be positive"):
        obstype.ObservationResultType().validate(create_obsres([1, -2]))


def test_mode_validator(mode, checker):
    def validator(mod, obj):
        raise ValidationError("not enough images")

    mode.validator = validator
    with pytest.raises(ValidationError, match="not enough images"):
        obstype.ObservationResultType().validate(create_obsres([1]))


def test_raw_frames_without_checker(mode):
    """Without a checker for the instrument, the images are not checked"""
    assert "TESTVAL" not in cfg.check
    assert obstype.ObservationResultType().validate(create_obsres([-1])) is True
