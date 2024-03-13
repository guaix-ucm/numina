import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..broadsp_gaussian_linearwv import apply_gaussian_broadening_linearwv


def test_apply_constant_sigma_in_pixels():

    rng = np.random.default_rng(1234)

    # generate fake spectrum
    naxis1 = 1000
    flux = np.zeros(naxis1)
    border = 10
    nlines = 15
    ipix = rng.uniform(low=border, high=naxis1-border, size=nlines).astype(int)
    for i in (ipix):
        flux[i] = rng.uniform(low=0.2, high=1.0, size=1)[0]
    flux = gaussian_filter1d(flux, 3)
    flux = 1 - flux

    # compute the effect of a gaussian broadining with a kernel of fixed
    # size (in pixels)
    method1 = gaussian_filter1d(
        input=flux,
        sigma=5.0,
        truncate=4.0,
        mode='constant'
    )
    method2 = apply_gaussian_broadening_linearwv(
        flux=flux,
        sigma_pix=np.ones_like(flux) * 5.0,
        tsigma=4.0
    )

    # compare results from both methods (should be identical)
    assert np.allclose(method1, method2)
