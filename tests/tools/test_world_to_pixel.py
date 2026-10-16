"""numina-world_to_pixel"""

import astropy.io.fits as fits
import numpy as np
import pytest
from astropy.wcs import WCS

from numina.tools.world_to_pixel import world_to_pixel


@pytest.mark.parametrize("verbose", [False, True])
def test_world_to_pixel(tmp_path, capsys, verbose):
    """The reference pixel of the WCS, without coordinates"""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    wcs.wcs.crpix = [10.0, 20.0]
    wcs.wcs.crval = [30.0, 40.0]
    wcs.wcs.cdelt = [-1e-4, 1e-4]
    filename = tmp_path / "image.fits"
    fits.writeto(filename, np.zeros((40, 20)), header=wcs.to_header())

    world_to_pixel(filename, sky=None, wave=None, extnum=0, verbose=verbose)
    out = capsys.readouterr().out
    assert [float(v) for v in out.splitlines()[-1].split()] == pytest.approx([10.0, 20.0])
    if verbose:
        # the description of the file by hdul.info()
        assert "PrimaryHDU" in out
