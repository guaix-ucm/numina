#
# Copyright 2024-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
import logging

from astropy.units import Unit
import pprint

from .raise_valueerror import raise_ValueError

pp = pprint.PrettyPrinter(indent=1, sort_dicts=False)


def set_wavelength_unit_and_range(scene_fname, scene_block, wmin, wmax, logger=None):
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
    logger : logging.Logger or None, optional
        Logger for logging messages. If None, a default logger will be used.

    Returns
    -------
    wave_unit : `~astropy.units.core.Unit`
        Wavelength unit to be used in the scene block.
    wave_min : `~astropy.units.Quantity`
        Minimum wavelength to be used in the scene block.
    wave_max : `~astropy.units.Quantity`
        Maximum wavelength to be used in the scene block.

    """

    if logger is None:
        logger = logging.getLogger(__name__)

    expected_keys_in_spectrum = {'type'}

    spectrum_keys = set(scene_block['spectrum'].keys())
    if not expected_keys_in_spectrum.issubset(spectrum_keys):
        print(f'[red]ERROR while processing: {scene_fname}[/red]')
        print(f'[blue]expected keys..: {expected_keys_in_spectrum}[/blue]')
        print(f'[blue]keys found.....: {spectrum_keys}[/blue]')
        list_unexpected_keys = list(spectrum_keys.difference(expected_keys_in_spectrum))
        if len(list_unexpected_keys) > 0:
            print(f'[red]unexpected keys: {list_unexpected_keys}[/red]')
        list_missing_keys = list(expected_keys_in_spectrum.difference(spectrum_keys))
        if len(list_missing_keys) > 0:
            print(f'[red]missing keys: {list_missing_keys}[/red]')
        pp.pprint(scene_block)
        raise_ValueError(f'Invalid format in file: {scene_fname}')
    if 'wave_unit' in scene_block['spectrum']:
        wave_unit = scene_block['spectrum']['wave_unit']
    else:
        wave_unit = wmin.unit
        logger.debug(f'[faint]Assuming wave_unit: {wave_unit}[/faint]')
    if 'wave_min' in scene_block['spectrum']:
        wave_min = float(scene_block['spectrum']['wave_min'])
    else:
        logger.debug(f'[faint]Assuming wave_min: null[/faint]')
        wave_min = None
    if wave_min is None:
        wave_min = wmin.to(wave_unit)
    else:
        wave_min *= Unit(wave_unit)
    if 'wave_max' in scene_block['spectrum']:
        wave_max = float(scene_block['spectrum']['wave_max'])
    else:
        logger.debug(f'[faint]Assuming wave_max: null[/faint]')
        wave_max = None
    if wave_max is None:
        wave_max = wmax.to(wave_unit)
    else:
        wave_max = wave_max * Unit(wave_unit)

    logger.debug(f'[faint]{wave_min=}[/faint]')
    logger.debug(f'[faint]{wave_max=}[/faint]')
    logger.debug(f'[faint]{wave_unit=}[/faint]')

    return wave_unit, wave_min, wave_max
