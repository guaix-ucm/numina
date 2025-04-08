#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.io import fits
from astropy.units import Unit
import numpy as np
import os

from .raise_valueerror import raise_ValueError


def load_atmosphere_transmission_curve(atmosphere_transmission, wmin, wmax, wv_cunit1, faux_dict, verbose):
    """Load atmosphere transmission curve.

    Parameters
    ----------
    atmosphere_transmission : str
        String indicating whether the atmosphere transmission of
        the atmosphere is applied or not. Two possible values are:
        - 'default': use default curve defined in 'faux_dict'
        - 'none': do not apply atmosphere transmission
    wmin : `~astropy.units.Quantity`
        Minimum wavelength to be considered.
    wmax : `~astropy.units.Quantity`
        Maximum wavelength to be considered.
    wv_cunit1 : `~astropy.units.core.Unit`
        Default wavelength unit to be employed in the wavelength scale.
    faux_dict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector
    verbose : bool
        If True, display additional information.

    Returns
    -------
    wave_transmission : `~astropy.units.Quantity`
        Wavelength column of the tabulated transmission curve.
    curve_transmission : `~astropy.units.Quantity`
        Transmission values for the wavelengths given in
        'wave_transmission'.

    """

    if atmosphere_transmission == "default":
        infile = faux_dict['skycalc']
        if verbose:
            print(f'\nLoading atmosphere transmission curve {os.path.basename(infile)}')
        with fits.open(infile) as hdul:
            skycalc_header = hdul[1].header
            skycalc_table = hdul[1].data
        if skycalc_header['TTYPE1'] != 'lam':
            raise_ValueError(f"Unexpected TTYPE1: {skycalc_header['TTYPE1']}")
        cwave_unit = skycalc_header['TUNIT1']
        wave_transmission = skycalc_table['lam'] * Unit(cwave_unit)
        curve_transmission = skycalc_table['trans']
        if wmin < np.min(wave_transmission) or wmax > np.max(wave_transmission):
            print(f'{wmin=} (simulated photons)')
            print(f'{wmax=} (simulated photons)')
            print(f'{np.min(wave_transmission.to(wv_cunit1))=} (transmission curve)')
            print(f'{np.max(wave_transmission.to(wv_cunit1))=} (transmission curve)')
            raise_ValueError('Wavelength range covered by the tabulated transmission curve is insufficient')
    elif atmosphere_transmission == "none":
        wave_transmission = None
        curve_transmission = None
        if verbose:
            print('Skipping application of the atmosphere transmission')
    else:
        wave_transmission = None   # avoid PyCharm warning (not aware of raise ValueError)
        curve_transmission = None  # avoid PyCharm warning (not aware of raise ValueError)
        raise_ValueError(f'Unexpected {atmosphere_transmission=}')

    return wave_transmission, curve_transmission
