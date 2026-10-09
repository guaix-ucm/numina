#
# Copyright 2016-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Models of the atmosphere: emission, extinction, refraction and seeing"""

import math
from astropy.modeling.functional_models import Gaussian2D, Moffat2D


class AtmosphereModel:
    """Model of the atmosphere.

    Parameters
    ----------
    twilight, nightsky, extinction : callable
        Spectrum of the twilight, spectrum of the night sky, and
        extinction, as functions of the wavelength.
    seeing : object
        Seeing model, as :class:`SeeingSizeModel` or :class:`ConstSeeing`.
    refraction : object
        Model with a method 'refraction(z, wl, ref)'.
    """

    def __init__(self, twilight, nightsky, seeing, extinction, refraction):
        self.tw_interp = twilight
        self.ng_interp = nightsky
        self.ext_interp = extinction
        self.seeing = seeing
        self.refraction_model = refraction

    def twilight_spectrum(self, wl_in):
        """Twilight spectrum"""
        return self.tw_interp(wl_in)

    def night_spectrum(self, wl_in):
        """Night spectrum"""
        return self.ng_interp(wl_in)

    def extinction(self, wl_in):
        """Night extinction"""
        return self.ext_interp(wl_in)

    def refraction(self, z, wl, ref):
        """Atmospheric refraction at the zenith distance `z` and wavelength `wl`, relative to `ref`"""
        return self.refraction_model.refraction(z, wl, ref)


class SeeingSizeModel:
    """Seeing that depends on the wavelength and the zenith distance.

    The model of Kolmogorov turbulence: the Fried parameter is
    ``r0 * (wl / wl0)**(6/5) * cos(zd)**(3/5)``, and the FWHM of the
    seeing, in radians, is ``0.98 * wl / r0``.

    Parameters
    ----------
    wl : float
        Reference wavelength, in the units of `r0`.
    r0 : float
        Fried parameter at the reference wavelength and the zenith.
    """

    def __init__(self, wl, r0):
        self._r0 = r0
        self._wl0 = wl

    def fwhm(self, wl, zd):
        """FWHM of the seeing, in radians, at the wavelength `wl` and zenith distance `zd` (radians)"""
        return 0.98 * wl / self.r0(wl, zd)

    def r0(self, wl, zd):
        """Fried parameter at the wavelength `wl` and zenith distance `zd` (radians)"""
        return self._r0 * (wl / self._wl0) ** 1.2 * math.cos(zd) ** 0.6

    def profile(self, fwhm):
        """Normalized Gaussian profile with the given FWHM"""
        return generate_gaussian_profile(fwhm)


class ConstSeeing:
    """Seeing with a constant FWHM"""

    def __init__(self, seeing):
        self._s = seeing

    def fwhm(self, wl, zd):
        """FWHM of the seeing, the same for all wavelengths and zenith distances"""
        return self._s

    def profile(self, fwhm):
        """Normalized Gaussian profile with the given FWHM"""
        return generate_gaussian_profile(fwhm)


def generate_gaussian_profile(seeing_fwhm):
    """Generate a normalized Gaussian profile from its FWHM"""
    FWHM_G = 2 * math.sqrt(2 * math.log(2))
    sigma = seeing_fwhm / FWHM_G
    amplitude = 1.0 / (2 * math.pi * sigma * sigma)
    seeing_model = Gaussian2D(amplitude=amplitude, x_mean=0.0, y_mean=0.0, x_stddev=sigma, y_stddev=sigma)
    return seeing_model


def generate_moffat_profile(seeing_fwhm, alpha):
    """Generate a normalized Moffat profile from its FWHM and alpha"""

    scale = 2 * math.sqrt(2 ** (1.0 / alpha) - 1)
    gamma = seeing_fwhm / scale
    amplitude = 1.0 / math.pi * (alpha - 1) / gamma**2
    seeing_model = Moffat2D(amplitude=amplitude, x_0=0.0, y_0=0.0, gamma=gamma, alpha=alpha)
    return seeing_model


def generate_lorentz_profile(seeing_fwhm):
    """Generate a normalized Moffat profile with alpha=1.5 from its FWHM.

    It is used as an approximation of a Lorentzian profile, the Moffat
    profile with alpha=1, that cannot be normalized in two dimensions.
    """

    return generate_moffat_profile(seeing_fwhm, alpha=1.5)
