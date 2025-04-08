#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib.pyplot as plt
import numpy as np

from numina.tools.ctext import ctext
from numina.tools.compute_adr_wavelength import compute_adr_wavelength

from .raise_valueerror import raise_ValueError
from .simulate_image2d_from_fitsfile import simulate_image2d_from_fitsfile


def generate_geometry_for_scene_block(
        scene_fname, scene_block, nphotons,
        apply_seeing, seeing_fwhm_arcsec, seeing_psf,
        airmass, parallactic_angle,
        reference_wave_vacuum_differential_refraction, simulated_wave,
        instrument_pa,
        wcs3d,
        min_x_ifu, max_x_ifu, min_y_ifu, max_y_ifu,
        rng,
        verbose, plots
):
    """Distribute photons in the IFU focal plane for the scene block.

    Parameters
    ----------
    scene_fname : str
        YAML scene file name.
    scene_block : dict
        Dictonary storing a scene block.
    nphotons : int
        Number of photons to be generated in the scene block.
    apply_seeing : bool
        If True, apply seeing to simulated photons.
    seeing_fwhm_arcsec : `~astropy.units.Quantity`
        Seeing FWHM.
    seeing_psf : str
        Seeing PSF.
    airmass : float
        Airmass.
    parallactic_angle : `~astropy.units.Quantity`
        Parallactic angle. This number must be within the range
        [-90,+90] deg.
    reference_wave_vacuum_differential_refraction : `~astropy.units.Quantity`
        Reference wavelength to compute the differential refraction
        correction. This wavelength corresponds to a correction of
        zero.
    simulated_wave : `~astropy.units.Quantity`
        Array containing `nphotons` simulated photons with the
        spectrum requested in the scene block. These values are
        required to compute the differential refraction correction.
    instrument_pa : `~astropy.units.Quantity`
        Instrument position angle.
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    min_x_ifu : `~astropy.units.Quantity`
        Minimum pixel X coordinate defining the IFU focal plane.
    max_x_ifu : `~astropy.units.Quantity`
        Maximum pixel X coordinate defining the IFU focal plane.
    min_y_ifu : `~astropy.units.Quantity`
        Minimum pixel Y coordinate defining the IFU focal plane.
    max_y_ifu : `~astropy.units.Quantity`
        Maximum pixel Y coordinate defining the IFU focal plane.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    verbose : bool
        If True, display additional information.
    plots : bool
        If True, plot intermediate results.

    Returns
    -------
    simulated_x_ifu_corrected : `~astropy.units.Quantity`
        Array containing the X coordinate of the 'nphotons' photons
        in the focal plane of the IFU.
    simulated_y_ifu_corrected : `~astropy.units.Quantity`
        Array containing the X coordinate of the 'nphotons' photons
        in the focal plane of the IFU.

    """

    if len(simulated_wave) != nphotons:
        raise ValueError(f'Unexpected {len(simulated_wave)=} != {nphotons=}')

    factor_fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))

    # plate scale
    plate_scale_x, plate_scale_y, plate_scale_wave = proj_plane_pixel_scales(wcs3d)
    plate_scale_x *= u.deg / u.pix
    plate_scale_y *= u.deg / u.pix

    if verbose:
        print(f'{wcs3d.wcs.cd=}')
        print(f'{plate_scale_x=}')
        print(f'{plate_scale_y=}')

    # define geometry type for scene block
    geometry_type = scene_block['geometry']['type']

    # simulate photons following the selected geometry
    if geometry_type == 'flatfield':
        simulated_x_ifu = rng.uniform(low=min_x_ifu.value, high=max_x_ifu.value, size=nphotons)
        simulated_y_ifu = rng.uniform(low=min_y_ifu.value, high=max_y_ifu.value, size=nphotons)
    elif geometry_type in ['gaussian', 'point-like', 'from-FITS-image']:
        if 'ra_deg' in scene_block['geometry']:
            ra_deg = scene_block['geometry']['ra_deg']
        else:
            if verbose:
                print(ctext('Assuming ra_deg: 0', faint=True))
            ra_deg = 0.0
        ra_deg *= u.deg
        if 'dec_deg' in scene_block['geometry']:
            dec_deg = scene_block['geometry']['dec_deg']
        else:
            if verbose:
                print(ctext('Assuming dec_deg: 0', faint=True))
            dec_deg = 0.0
        dec_deg *= u.deg
        if 'delta_ra_arcsec' in scene_block['geometry']:
            delta_ra_arcsec = scene_block['geometry']['delta_ra_arcsec']
        else:
            if verbose:
                print(ctext('Assuming delta_ra_deg: 0', faint=True))
            delta_ra_arcsec = 0.0
        delta_ra_arcsec *= u.arcsec
        if 'delta_dec_arcsec' in scene_block['geometry']:
            delta_dec_arcsec = scene_block['geometry']['delta_dec_arcsec']
        else:
            if verbose:
                print(ctext('Assuming delta_dec_deg: 0', faint=True))
            delta_dec_arcsec = 0.0
        delta_dec_arcsec *= u.arcsec
        x_center, y_center = wcs3d.celestial.world_to_pixel(
            SkyCoord(ra=ra_deg + delta_ra_arcsec.to(u.deg), dec=dec_deg + delta_dec_arcsec.to(u.deg))
        )
        # the previous pixel coordinates are assumed to be 0 at the center
        # of the first pixel in each dimension
        x_center += 1
        y_center += 1
        if geometry_type == 'point-like':
            simulated_x_ifu = np.repeat(x_center, nphotons)
            simulated_y_ifu = np.repeat(y_center, nphotons)
        elif geometry_type == 'gaussian':
            mandatory_keys = ['fwhm_ra_arcsec']
            for key in mandatory_keys:
                if key not in scene_block['geometry']:
                    raise_ValueError(f"Expected key '{key}' not found!")
            fwhm_ra_arcsec = scene_block['geometry']['fwhm_ra_arcsec'] * u.arcsec
            if 'fwhm_dec_arcsec' in scene_block['geometry']:
                fwhm_dec_arcsec = scene_block['geometry']['fwhm_dec_arcsec'] * u.arcsec
            else:
                fwhm_dec_arcsec = fwhm_ra_arcsec
                if verbose:
                    print(ctext(f'Assuming {fwhm_dec_arcsec=}', faint=True))
            if 'position_angle_deg' in scene_block['geometry']:
                position_angle_deg = scene_block['geometry']['position_angle_deg'] * u.deg
            else:
                position_angle_deg = 0.0 * u.deg
                if verbose:
                    print(ctext(f'Assuming {position_angle_deg=}', faint=True))
            # covariance matrix for the multivariate normal
            std_x = fwhm_ra_arcsec * factor_fwhm_to_sigma / plate_scale_x.to(u.arcsec / u.pix)
            std_y = fwhm_dec_arcsec * factor_fwhm_to_sigma / plate_scale_y.to(u.arcsec / u.pix)
            rotation_matrix = np.array(  # note the sign to rotate N -> E -> S -> W
                [
                    [np.cos(position_angle_deg), np.sin(position_angle_deg)],
                    [-np.sin(position_angle_deg), np.cos(position_angle_deg)]
                ]
            )
            covariance = np.diag([std_x.value ** 2, std_y.value ** 2])
            rotated_covariance = np.dot(rotation_matrix.T, np.dot(covariance, rotation_matrix))
            # simulate X, Y values
            simulated_xy_ifu = rng.multivariate_normal(
                mean=[x_center, y_center],
                cov=rotated_covariance,
                size=nphotons
            )
            # compensate for instrument position angle
            simulated_x_ifu = \
                simulated_xy_ifu[:, 0] * np.cos(instrument_pa).value + \
                simulated_xy_ifu[:, 1] * np.sin(instrument_pa).value
            simulated_y_ifu = \
                -simulated_xy_ifu[:, 0] * np.sin(instrument_pa).value + \
                simulated_xy_ifu[:, 1] * np.cos(instrument_pa).value
        elif geometry_type == 'from-FITS-image':
            mandatory_keys = ['filename', 'diagonal_fov_arcsec', 'background_to_subtract']
            for key in mandatory_keys:
                if key not in scene_block['geometry']:
                    raise_ValueError(f"Expected key '{key}' not found!")
            # read reference FITS file
            infile = scene_block['geometry']['filename']
            diagonal_fov_arcsec = scene_block['geometry']['diagonal_fov_arcsec'] * u.arcsec
            background_to_subtract = scene_block['geometry']['background_to_subtract']
            # generate simulated locations in the IFU
            simulated_x_ifu_0, simulated_y_ifu_0 = simulate_image2d_from_fitsfile(
                infile=infile,
                diagonal_fov_arcsec=diagonal_fov_arcsec,
                plate_scale_x=plate_scale_x,
                plate_scale_y=plate_scale_y,
                nphotons=nphotons,
                rng=rng,
                background_to_subtract=background_to_subtract,
                plots=plots,
                verbose=verbose
            )
            # compensate for instrument rotation angle
            simulated_x_ifu = \
                simulated_x_ifu_0 * np.cos(instrument_pa).value + \
                simulated_y_ifu_0 * np.sin(instrument_pa).value
            simulated_y_ifu = \
                -simulated_x_ifu_0 * np.sin(instrument_pa).value + \
                simulated_y_ifu_0 * np.cos(instrument_pa).value
            # shift image center
            simulated_x_ifu += x_center
            simulated_y_ifu += y_center
        else:
            simulated_x_ifu = None  # avoid PyCharm warning (not aware of raise ValueError)
            simulated_y_ifu = None  # avoid PyCharm warning (not aware of raise ValueError)
            raise_ValueError(f'Unexpected {geometry_type=}')
    else:
        simulated_x_ifu = None  # avoid PyCharm warning (not aware of raise ValueError)
        simulated_y_ifu = None  # avoid PyCharm warning (not aware of raise ValueError)
        raise_ValueError(f'Unexpected {geometry_type=} in file {scene_fname}')

    # apply seeing
    if apply_seeing:
        if seeing_psf == "gaussian":
            if verbose:
                print(f'Applying Gaussian PSF with {seeing_fwhm_arcsec=}')
            std_x = seeing_fwhm_arcsec * factor_fwhm_to_sigma / plate_scale_x.to(u.arcsec / u.pix)
            simulated_x_ifu += rng.normal(loc=0, scale=abs(std_x.value), size=nphotons)
            std_y = seeing_fwhm_arcsec * factor_fwhm_to_sigma / plate_scale_y.to(u.arcsec / u.pix)
            simulated_y_ifu += rng.normal(loc=0, scale=abs(std_y.value), size=nphotons)
        else:
            raise_ValueError(f'Unexpected {seeing_psf=}')

    # apply differential refraction (as a function of wavelength)
    if (geometry_type != 'flatfield') and (airmass > 1.0):
        if verbose:
            print('Applying differential refraction correction as a function of wavelength')
        differential_refraction = compute_adr_wavelength(
            airmass=airmass,
            reference_wave_vacuum=reference_wave_vacuum_differential_refraction,
            wave_vacuum=simulated_wave,
            debug=verbose
        )
        # compute RA and DEC of each simulated photon
        simulated_coor = wcs3d.celestial.pixel_to_world(
            simulated_x_ifu - 1.0,
            simulated_y_ifu - 1.0
        )
        simulated_ra = simulated_coor.ra
        simulated_dec = simulated_coor.dec
        # apply differential refraction correction (first declination
        # and then right ascension; see Eq. (39) and (40), pp. 71-72,
        # in Textbook on Spherical Astronomy, Smart, 1977).
        simulated_dec += differential_refraction.to(u.deg) * np.cos(parallactic_angle)
        simulated_ra += differential_refraction.to(u.deg) * np.sin(parallactic_angle) / np.cos(simulated_dec)
        # recompute X, Y coordinates in the IFU focal plane
        simulated_x_ifu_corrected, simulated_y_ifu_corrected = wcs3d.celestial.world_to_pixel(
            SkyCoord(ra=simulated_ra, dec=simulated_dec)
        )
        simulated_x_ifu_corrected += 1
        simulated_y_ifu_corrected += 1
        if plots:
            iok = np.argwhere(simulated_wave > 0 * u.m)
            # differential refraction (as a function of wavelength)
            # computed at the center of the IFU
            wmin_simulated = np.min(simulated_wave[iok])
            wmax_simulated = np.max(simulated_wave[iok])
            nsample_wavelengths = 20
            sample_wavelengths = np.linspace(wmin_simulated, wmax_simulated, nsample_wavelengths)
            differential_refraction_center_ifu = compute_adr_wavelength(
                airmass=airmass,
                reference_wave_vacuum=reference_wave_vacuum_differential_refraction,
                wave_vacuum=sample_wavelengths
            )
            x_center_ifu, y_center_ifu = wcs3d.celestial.wcs.crpix
            simulated_coor = wcs3d.celestial.pixel_to_world(
                x_center_ifu - 1.0,
                y_center_ifu - 1.0
            )
            ra_center_ifu = simulated_coor.ra
            dec_center_ifu = simulated_coor.dec
            ra_center_ifu = np.repeat(ra_center_ifu, nsample_wavelengths)
            dec_center_ifu = np.repeat(dec_center_ifu, nsample_wavelengths)
            dec_center_ifu += differential_refraction_center_ifu.to(u.deg) * np.cos(parallactic_angle)
            ra_center_ifu += differential_refraction_center_ifu.to(u.deg) * np.sin(parallactic_angle) / np.cos(dec_center_ifu)
            x_center_ifu_corrected, y_center_ifu_corrected = wcs3d.celestial.world_to_pixel(
                SkyCoord(ra=ra_center_ifu, dec=dec_center_ifu)
            )
            x_center_ifu_corrected += 1
            y_center_ifu_corrected += 1
            delta_x_center_ifu = x_center_ifu_corrected - x_center_ifu
            delta_y_center_ifu = y_center_ifu_corrected - y_center_ifu
            # differential refraction (as a function of wavelength)
            # computed for each useful simulated photon
            delta_simulated_x_ifu = simulated_x_ifu_corrected - simulated_x_ifu
            delta_simulated_y_ifu = simulated_y_ifu_corrected - simulated_y_ifu
            fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(6.4, 6.4))
            for iplot in range(2):
                ax = axarr[iplot]
                if iplot == 0:
                    ax.plot(simulated_wave[iok], delta_simulated_x_ifu[iok], ',', label='simulated photons')
                    ax.plot(sample_wavelengths, delta_x_center_ifu, 'r+', label='IFU center')
                    ax.set_ylabel(r'$\Delta$simulated_x_ifu (pixel)')
                else:
                    ax.plot(simulated_wave[iok], delta_simulated_y_ifu[iok], ',', label='simulated photons')
                    ax.plot(sample_wavelengths, delta_y_center_ifu, 'r+', label='IFU center')
                    ax.set_ylabel(r'$\Delta$simulated_y_ifu (pixel)')
                ax.set_xlabel(f'Wavelength ({simulated_wave.unit})')
                ax.axhline(0, linestyle='--', color='gray')
                ax.axvline(reference_wave_vacuum_differential_refraction.value, linestyle=':', color='C1')
                ax.set_title(f'airmass: {airmass}, parallactic angle: {parallactic_angle}')
                ax.legend()
            plt.tight_layout()
            plt.show()
    else:
        simulated_x_ifu_corrected = simulated_x_ifu
        simulated_y_ifu_corrected = simulated_y_ifu

    # add units
    simulated_x_ifu_corrected *= u.pix
    simulated_y_ifu_corrected *= u.pix

    return simulated_x_ifu_corrected, simulated_y_ifu_corrected
