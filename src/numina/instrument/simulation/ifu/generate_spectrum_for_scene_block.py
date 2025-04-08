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
import astropy.units as u
import numpy as np
import os

from numina.tools.ctext import ctext

from .fapply_atmosphere_transmission import fapply_atmosphere_transmission
from .simulate_constant_photlam import simulate_constant_photlam
from .simulate_delta_lines import simulate_delta_lines
from .simulate_spectrum import simulate_spectrum
from .raise_valueerror import raise_ValueError


def generate_spectrum_for_scene_blok(scene_fname, scene_block, faux_dict, wave_unit,
                                     wave_min, wave_max, nphotons, wavelength_sampling,
                                     apply_atmosphere_transmission, wave_transmission, curve_transmission,
                                     rng, naxis1_detector,
                                     verbose, plots):
    """Generate photons for the scene block.

    Parameters
    ----------
    scene_fname : str
        YAML scene file name.
    scene_block : dict
        Dictonary storing a scene block.
    faux_dict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector
    wave_unit : `~astropy.units.core.Unit`
        Wavelength unit to be used in the scene block.
    wave_min : `~astropy.units.Quantity`
        Minimum wavelength to be used in the scene block.
    wave_max : `~astropy.units.Quantity`
        Maximum wavelength to be used in the scene block.
    nphotons : int
        Number of photons to be generated in the scene block.
    wavelength_sampling : str
        Method to sample the wavelength values. Two options are valid:
        - 'random': the wavelengt of each photon is randomly determined
          using the spectrum shape as the density probability function.
        - 'fixed': the wavelength of each photon is exactly determined
          using the spectrum shape as the density probability function.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    apply_atmosphere_transmission : bool
        If True, apply atmosphere transmission to simulated photons.
    wave_transmission : `~astropy.units.Quantity`
        Wavelength column of the tabulated transmission curve.
    curve_transmission : `~astropy.units.Quantity`
        Transmission values for the wavelengths given in
        'wave_transmission'.
    naxis1_detector : `~astropy.units.Quantity`
        Detector NAXIS1, dispersion direction.
    verbose : bool
        If True, display additional information.
    plots : bool
        If True, plot intermediate results.

    Returns
    -------
    simulated_wave : '~numpy.ndarray'
        Array containing `nphotons` simulated photons with the
        spectrum requested in the scene block.

    """

    spectrum_type = scene_block['spectrum']['type']
    if spectrum_type == 'delta-lines':
        mandatory_keys = ['filename', 'wave_column', 'flux_column']
        for key in mandatory_keys:
            if key not in scene_block['spectrum']:
                raise_ValueError(f"Expected key '{key}' not found!")
        filename = scene_block['spectrum']['filename']
        wave_column = scene_block['spectrum']['wave_column'] - 1
        flux_column = scene_block['spectrum']['flux_column'] - 1
        if filename[0] == '@':
            # retrieve file name from dictionary of auxiliary
            # file names for the considered instrument
            filename = faux_dict[filename[1:]]
        catlines = np.genfromtxt(filename)
        line_wave = catlines[:, wave_column] * Unit(wave_unit)
        if not np.all(np.diff(line_wave.value) > 0):
            raise_ValueError(f'Wavelength array {line_wave=} is not sorted!')
        line_flux = catlines[:, flux_column]
        simulated_wave = simulate_delta_lines(
            line_wave=line_wave,
            line_flux=line_flux,
            nphotons=nphotons,
            wavelength_sampling=wavelength_sampling,
            rng=rng,
            wmin=wave_min,
            wmax=wave_max,
            plots=plots,
            plot_title=filename
        )
    elif spectrum_type == 'skycalc-radiance':
        faux_skycalc = faux_dict['skycalc']
        with fits.open(faux_skycalc) as hdul:
            skycalc_table = hdul[1].data
        if wave_unit != Unit('nm'):
            print(ctext(f'Ignoring wave_unit: {wave_unit} (assuming {u.nm})', faint=True))
        wave = skycalc_table['lam'] * u.nm
        if not np.all(np.diff(wave.value) > 0):
            raise_ValueError(f'Wavelength array {wave=} is not sorted!')
        flux = skycalc_table['flux']
        flux_type = 'photlam'
        simulated_wave = simulate_spectrum(
            wave=wave,
            flux=flux,
            flux_type=flux_type,
            nphotons=nphotons,
            wavelength_sampling=wavelength_sampling,
            rng=rng,
            wmin=wave_min,
            wmax=wave_max,
            convolve_sigma_km_s=0 * u.km / u.s,
            nbins_histo=naxis1_detector.value,
            plots=plots,
            plot_title=os.path.basename(faux_skycalc),
            verbose=verbose
        )
    elif spectrum_type == 'tabulated-spectrum':
        mandatory_keys = ['filename', 'wave_column', 'flux_column', 'flux_type']
        for key in mandatory_keys:
            if key not in scene_block['spectrum']:
                raise_ValueError(f"Expected key '{key}' not found!")
        filename = scene_block['spectrum']['filename']
        wave_column = scene_block['spectrum']['wave_column'] - 1
        flux_column = scene_block['spectrum']['flux_column'] - 1
        flux_type = scene_block['spectrum']['flux_type']
        if 'redshift' in scene_block['spectrum']:
            redshift = float(scene_block['spectrum']['redshift'])
        else:
            if verbose:
                print(ctext('Assuming redshift: 0', faint=True))
            redshift = 0.0
        if 'convolve_sigma_km_s' in scene_block['spectrum']:
            convolve_sigma_km_s = float(scene_block['spectrum']['convolve_sigma_km_s'])
        else:
            if verbose:
                print(ctext('Assuming convolve_sigma_km_s: 0', faint=True))
            convolve_sigma_km_s = 0.0
        convolve_sigma_km_s *= u.km / u.s
        # read data
        table_data = np.genfromtxt(filename)
        wave = table_data[:, wave_column] * (1 + redshift) * Unit(wave_unit)
        if not np.all(np.diff(wave.value) > 0):
            raise_ValueError(f'Wavelength array {wave=} is not sorted!')
        flux = table_data[:, flux_column]
        simulated_wave = simulate_spectrum(
            wave=wave,
            flux=flux,
            flux_type=flux_type,
            nphotons=nphotons,
            wavelength_sampling=wavelength_sampling,
            rng=rng,
            wmin=wave_min,
            wmax=wave_max,
            convolve_sigma_km_s=convolve_sigma_km_s,
            nbins_histo=naxis1_detector.value,
            plots=plots,
            plot_title=os.path.basename(filename),
            verbose=verbose
        )
    elif spectrum_type == 'constant-flux':
        simulated_wave = simulate_constant_photlam(
            wmin=wave_min,
            wmax=wave_max,
            nphotons=nphotons,
            wavelength_sampling=wavelength_sampling,
            rng=rng
        )
    else:
        simulated_wave = None  # avoid PyCharm warning (not aware of raise ValueError)
        raise_ValueError(f'Unexpected {spectrum_type=} in file {scene_fname}')

    # apply atmosphere transmission
    if apply_atmosphere_transmission:
        fapply_atmosphere_transmission(
            simulated_wave=simulated_wave,
            wave_transmission=wave_transmission,
            curve_transmission=curve_transmission,
            rng=rng,
            verbose=verbose,
            plots=plots
        )

    return simulated_wave
