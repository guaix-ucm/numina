#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Auxiliary class for linear wavelength calibration"""

import astropy.units as u


class LinearWaveCal(object):
    """Class to store a linear wavelength calibration.

    The parameters are stored making use of astropy units.

    Parameters
    ----------
    crpix1_wavecal : `~astropy.units.Quantity`
        CRPIX1 value of the wavelength calibrated spectrum.
    crval1_wavecal : `~astropy.units.Quantigy`
        CRVAL1 value of the wavelength calibrated spectrum.
    cdelt1_wavecal : `~astropy.units.Quantigy`
        CDELT1 value of the wavelength calibrated spectrum.
    naxis1_wavecal : `~astropy.units.Quantity`
        NAXIS1 value of the wavelength calibrated spectrum.
    default_wavelength_unit : `~astropy.units.core.Unit
        Default wavelength unit to be employed in CRVAL1 and CDELT1.

    Attributes
    ----------
    crpix1_wavecal : `~astropy.units.Quantity`
        CRPIX1 value of the wavelength calibrated spectrum.
    crval1_wavecal : `~astropy.units.Quantigy`
        CRVAL1 value of the wavelength calibrated spectrum.
    cdelt1_wavecal : `~astropy.units.Quantigy`
        CDELT1 value of the wavelength calibrated spectrum.
    naxis1_wavecal : `~astropy.units.Quantity`
        NAXIS1 value of the wavelength calibrated spectrum.
    default_wavelength_unit : `~astropy.units.core.Unit`
        Default wavelength unit to be employed in CRVAL1 and CDELT1.
    wmin : `~astropy.units.Quantity`
        Minimum wavelength, computed at pixel 0.5.
    wmax : `~astropy.units.Quantity`
        Maximum wavelength, computed at pixel naxis1_wavecal + 0.5.

    Methods
    -------
    wave_at_pixel(pixel):
        Compute wavelength(s) at the pixel coordinate(s).
    pixel_at_wave(wavelength)
        Compute pixel coordinate(s) at the wavelength(s).

    """

    def __init__(self, crpix1_wavecal, crval1_wavecal, cdelt1_wavecal, naxis1_wavecal, default_wavelength_unit):
        # check units
        if not crpix1_wavecal.unit.is_equivalent(u.pix):
            raise ValueError(f"Unexpected crpix1_wavecal unit: {crpix1_wavecal.unit}")
        if not crval1_wavecal.unit.is_equivalent(u.m):
            raise ValueError(f"Unexpected crval1_wavecal unit: {crval1_wavecal.unit}")
        if not cdelt1_wavecal.unit.is_equivalent(u.m / u.pix):
            raise ValueError(f"Unexpected cdelt1_wavecal unit: {cdelt1_wavecal.unit}")
        if not naxis1_wavecal.unit.is_equivalent(u.pix):
            raise ValueError(f"Unexpected naxis1_wavecal unit: {naxis1_wavecal.unit}")
        if not default_wavelength_unit.is_equivalent(u.m):
            raise ValueError(f"Unexpected default_wavelength_unit: {default_wavelength_unit}")

        # define attributes
        self.crpix1_wavecal = crpix1_wavecal
        self.crval1_wavecal = crval1_wavecal.to(default_wavelength_unit)
        self.cdelt1_wavecal = cdelt1_wavecal.to(default_wavelength_unit / u.pix)
        self.naxis1_wavecal = naxis1_wavecal
        self.wmin = self.wave_at_pixel(0.5 * u.pix)
        self.wmax = self.wave_at_pixel(naxis1_wavecal + 0.5 * u.pix)
        self.default_wavelength_unit = default_wavelength_unit

    def __str__(self):
        output = "<LinearWaveCal instance>\n"
        output += f"crpix1_wavecal: {self.crpix1_wavecal}\n"
        output += f"crval1_wavecal: {self.crval1_wavecal}\n"
        output += f"cdelt1_wavecal: {self.cdelt1_wavecal}\n"
        output += f"naxis1_wavecal: {self.naxis1_wavecal}\n"
        output += f"wmin..........: {self.wmin}\n"
        output += f"wmax..........: {self.wmax}\n"
        output += f"default_wavelength_unit: {self.default_wavelength_unit}"
        return output

    def __repr__(self):
        output = f"LinearWaveCal(\n"
        output += f"    crpix1_wavecal={self.crpix1_wavecal.value} * {self.crpix1_wavecal.unit.__repr__()},\n"
        output += f"    crval1_wavecal={self.crval1_wavecal.value} * {self.crval1_wavecal.unit.__repr__()},\n"
        output += f"    cdelt1_wavecal={self.cdelt1_wavecal.value} * {self.cdelt1_wavecal.unit.__repr__()},\n"
        output += f"    naxis1_wavecal={self.naxis1_wavecal.value} * {self.naxis1_wavecal.unit.__repr__()},\n"
        output += f"    default_wavelength_unit={self.default_wavelength_unit.__repr__()}\n"
        output += ")"
        return output

    def __eq__(self, other):
        if isinstance(other, LinearWaveCal):
            return self.__dict__ == other.__dict__
        else:
            return False

    def wave_at_pixel(self, pixel):
        """Compute wavelength(s) at the pixel coordinate(s).

        Parameters
        ----------
        pixel : `~astropy.units.Quantity`
            A single number or a numpy array with pixel coordinates.
            The units used serve to decide the criterion used to
            indicate the coordinates: u.pix for the FITS system
            (which starts at 1) and u.dimensionless_unscaled to
            indicate that the positions correspond to indices of
            an array (which starts at 0).

        Returns
        -------
        wave : `~astropy.units.Quantity`
            Wavelength computed at the considered pixel value(s).

        """

        if not isinstance(pixel, u.Quantity):
            raise ValueError(f"Object 'pixel': {pixel} is not a Quantity instance")

        if pixel.unit == u.pix:
            fitspixel = pixel
        elif pixel.unit == u.dimensionless_unscaled:
            fitspixel = (pixel.value + 1) * u.pix
        else:
            raise ValueError(f"Unexpected 'pixel' units: {pixel.unit}")

        wave = self.crval1_wavecal + (fitspixel - self.crpix1_wavecal) * self.cdelt1_wavecal

        return wave

    def pixel_at_wave(self, wave, return_units):
        """Compute pixel coordinate(s) at the wavelength(s).

        Parameters
        ----------
        wave : `~astropy.units.Quantity`
            A single number or a numpy array with wavelengths.
        return_units : `astropy.units.core.Unit`
            The return units serve to decide the criterion used to
            indicate the pixel coordinates: u.pix for the FITS system
            (which starts at 1) and u.dimensionless_unscaled to
            indicate that the positions correspond to indices of
            an array (which starts at 0).

        Returns
        -------
        pixel : `~astropy.units.Quantity`
            Pixel coordinates computed at the considered wavelength(s).

        """

        if not isinstance(wave, u.Quantity):
            raise ValueError(f"Object 'wave': {wave} is not a Quantity instance")
        if return_units not in [u.pix, u.dimensionless_unscaled]:
            raise ValueError(f"Unexpected return_units: {return_units}")

        waveunit = self.default_wavelength_unit
        fitspixel = (wave.to(waveunit) - self.crval1_wavecal) / self.cdelt1_wavecal + self.crpix1_wavecal

        if return_units == u.pix:
            return fitspixel
        else:
            return (fitspixel.value - 1) * u.dimensionless_unscaled
