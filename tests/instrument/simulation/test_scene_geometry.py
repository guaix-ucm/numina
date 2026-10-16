"""Geometry of the photons of a scene block in the IFU"""

import logging

import astropy.units as u
import numpy as np
import pytest
from astropy.wcs import WCS

from numina.instrument.simulation.ifu.generate_geometry_for_scene_block import generate_geometry_for_scene_block


def wcs_cube():
    """WCS of a data cube of 64x64 pixels of 0.1 arcsec, in microns"""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN", "WAVE"]
    wcs.wcs.cunit = ["deg", "deg", "m"]
    wcs.wcs.crpix = [32.5, 32.5, 1.0]
    wcs.wcs.crval = [0.0, 0.0, 1.5e-6]
    wcs.wcs.cd = np.diag([-0.1 / 3600, 0.1 / 3600, 1e-10])
    return wcs


@pytest.mark.parametrize("airmass", [1.0, 1.5])
def test_point_like_with_differential_refraction(airmass):
    """With airmass > 1, the photons are displaced by the differential refraction"""
    nphotons = 100
    rng = np.random.default_rng(1234)
    simulated_wave = rng.uniform(1.4, 1.6, size=nphotons) * u.micron
    x, y = generate_geometry_for_scene_block(
        scene_fname="scene.yaml",
        scene_block={"geometry": {"type": "point-like"}},
        nphotons=nphotons,
        apply_seeing=False,
        seeing_fwhm_arcsec=0.0 * u.arcsec,
        seeing_psf="gaussian",
        airmass=airmass,
        parallactic_angle=0 * u.deg,
        reference_wave_vacuum_differential_refraction=1.5 * u.micron,
        simulated_wave=simulated_wave,
        instrument_pa=0 * u.deg,
        wcs3d=wcs_cube(),
        min_x_ifu=0.5 * u.pix,
        max_x_ifu=64.5 * u.pix,
        min_y_ifu=0.5 * u.pix,
        max_y_ifu=64.5 * u.pix,
        nslices=32,
        rng=rng,
        logger=logging.getLogger(__name__),
        plots=False,
    )
    assert len(x) == len(y) == nphotons
    spread = np.ptp(np.hypot(x.value, y.value))
    if airmass == 1.0:
        assert spread == pytest.approx(0.0)
    else:
        assert spread > 0.0
