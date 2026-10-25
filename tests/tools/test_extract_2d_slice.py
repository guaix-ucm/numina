"""The PNG file of extract_slice is saved without pyplot"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from numina.tools.extract_2d_slice_from_3d_cube import extract_slice


def test_extract_slice_png(tmp_path):
    cube = tmp_path / "cube.fits"
    fits.PrimaryHDU(np.ones((5, 4, 3), dtype=np.float32)).writeto(cube)
    png = tmp_path / "slice.png"
    output = tmp_path / "slice.fits"
    before = plt.get_fignums()
    extract_slice(
        str(cube),
        extnum=0,
        extname=None,
        axis=3,
        i1=1,
        i2=5,
        method="sum",
        wavecal="none",
        transpose=False,
        vmin=None,
        vmax=None,
        noplot=False,
        output=str(output),
        png=str(png),
    )
    assert png.stat().st_size > 0
    np.testing.assert_allclose(fits.getdata(output), 5.0)
    assert plt.get_fignums() == before
