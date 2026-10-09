"""Models of the atmosphere"""

import math

import numpy
import pytest

from numina.instrument.simulation.atmosphere import (
    SeeingSizeModel,
    generate_gaussian_profile,
    generate_lorentz_profile,
    generate_moffat_profile,
)

WL0 = 500e-9  # m
R0 = 0.1  # m


def test_seeing_at_zenith():
    model = SeeingSizeModel(WL0, R0)
    assert model.r0(WL0, 0.0) == pytest.approx(R0)
    # about 1 arcsec
    assert math.degrees(model.fwhm(WL0, 0.0)) * 3600 == pytest.approx(0.98 * WL0 / R0 * 206264.8, rel=1e-6)


def test_seeing_wavelength_and_zenith_distance():
    model = SeeingSizeModel(WL0, R0)
    # r0 ~ wl**(6/5), so the FWHM ~ wl**(-1/5)
    assert model.r0(2 * WL0, 0.0) == pytest.approx(R0 * 2**1.2)
    assert model.fwhm(2 * WL0, 0.0) / model.fwhm(WL0, 0.0) == pytest.approx(2**-0.2)
    # r0 ~ cos(zd)**(3/5), at airmass 2 the seeing is 2**(3/5) worse
    zd = math.radians(60)
    assert model.fwhm(WL0, zd) / model.fwhm(WL0, 0.0) == pytest.approx(2**0.6)


@pytest.mark.parametrize(
    "profile",
    [generate_gaussian_profile(1.0), generate_moffat_profile(1.0, 3.0), generate_lorentz_profile(1.0)],
)
def test_profiles_normalized(profile):
    x = numpy.linspace(-200, 200, 2001)
    xx, yy = numpy.meshgrid(x, x)
    step = x[1] - x[0]
    total = profile(xx, yy).sum() * step**2
    # the wings of the Moffat profile with alpha=1.5 decrease slowly
    assert total == pytest.approx(1.0, abs=0.01)
