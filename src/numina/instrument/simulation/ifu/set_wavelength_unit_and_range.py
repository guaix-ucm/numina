#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.units import Unit
import pprint

from numina.tools.ctext import ctext

from .raise_valueerror import raise_ValueError

pp = pprint.PrettyPrinter(indent=1, sort_dicts=False)


def set_wavelength_unit_and_range(scene_fname, scene_block, wmin, wmax, verbose):
    """Set the wavelength unit and range for a scene block.

    Parameters
    ----------
    scene_fname : str
        YAML scene file name.
    scene_block : dict
        Dictonary storing the scene block.
    wmin : `~astropy.units.Quantity`
        Minimum wavelength covered by the detector.
    wmax : `~astropy.units.Quantity`
        Maximum wavelength covered by the detector.
    verbose : bool
       If True, display additional information.

    Returns
    -------
    wave_unit : `~astropy.units.core.Unit`
        Wavelength unit to be used in the scene block.
    wave_min : `~astropy.units.Quantity`
        Minimum wavelength to be used in the scene block.
    wave_max : `~astropy.units.Quantity`
        Maximum wavelength to be used in the scene block.

    """

    expected_keys_in_spectrum = {'type'}

    spectrum_keys = set(scene_block['spectrum'].keys())
    if not expected_keys_in_spectrum.issubset(spectrum_keys):
        print(ctext(f'ERROR while processing: {scene_fname}', fg='red'))
        print(ctext('expected keys..: ', fg='blue') + f'{expected_keys_in_spectrum}')
        print(ctext('keys found.....: ', fg='blue') + f'{spectrum_keys}')
        list_unexpected_keys = list(spectrum_keys.difference(expected_keys_in_spectrum))
        if len(list_unexpected_keys) > 0:
            print(ctext('unexpected keys: ', fg='red') + f'{list_unexpected_keys}')
        list_missing_keys = list(expected_keys_in_spectrum.difference(spectrum_keys))
        if len(list_missing_keys) > 0:
            print(ctext('missing keys:..: ', fg='red') + f'{list_missing_keys}')
        pp.pprint(scene_block)
        raise_ValueError(f'Invalid format in file: {scene_fname}')
    if 'wave_unit' in scene_block['spectrum']:
        wave_unit = scene_block['spectrum']['wave_unit']
    else:
        wave_unit = wmin.unit
        if verbose:
            print(ctext(f'Assuming wave_unit: {wave_unit}', faint=True))
    if 'wave_min' in scene_block['spectrum']:
        wave_min = float(scene_block['spectrum']['wave_min'])
    else:
        if verbose:
            print(ctext('Assuming wave_min: null', faint=True))
        wave_min = None
    if wave_min is None:
        wave_min = wmin.to(wave_unit)
    else:
        wave_min *= Unit(wave_unit)
    if 'wave_max' in scene_block['spectrum']:
        wave_max = float(scene_block['spectrum']['wave_max'])
    else:
        if verbose:
            print(ctext('Assuming wave_max: null', faint=True))
        wave_max = None
    if wave_max is None:
        wave_max = wmax.to(wave_unit)
    else:
        wave_max = wave_max * Unit(wave_unit)

    if verbose:
        print(ctext(f'{wave_min=}', faint=True))
        print(ctext(f'{wave_max=}', faint=True))
        print(ctext(f'{wave_unit=}', faint=True))

    return wave_unit, wave_min, wave_max
