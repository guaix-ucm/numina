#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Resample a 3D cube in the wavelength axis (NAXIS3).
"""

import argparse
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import numpy as np
import sys

from .ctext import ctext

from numina.instrument.simulation.ifu.define_3d_wcs import header3d_after_merging_wcs2d_celestial_and_wcs1d_spectral


def resample_wave_3d_cube(hdu3d_image, crval3out, cdelt3out, naxis3out, verbose=False):
    """Resample a 3D cube to a new wavelength sampling.

    The celestial WCS is preserved, and the spectral WCS is modified
    in order to make use of the new wavelength sampling.

    Parameters
    ----------
    hdu3d_image : `astropy.io.fits.ImageHDU`
        HDU instance with the 3D image to be resampled.
    crval3out : `astropy.units.Quantity`
        Minimum wavelength for the output image.
    cdelt3out : `astropy.units.Quantity`
        Wavelength step for the output image.
    naxis3out : int
        Number of slices in the output image.
    verbose : bool, optional
        If True, display intermediate information.

    Returns
    -------
    resampled_hdu : `astropy.io.fits.ImageHDU`
        Resampled HDU instance with the 3D image.
    """
    # protections
    if not isinstance(hdu3d_image, fits.ImageHDU) and not isinstance(hdu3d_image, fits.PrimaryHDU):
        raise ValueError("Input HDU must be an ImageHDU or PrimaryHDU.")
    if crval3out is None:
        raise ValueError("crval3out must be specified.")
    if cdelt3out is None:
        raise ValueError("cdelt3out must be specified.")
    if naxis3out is None:
        raise ValueError("naxis3out must be specified.")
    if hdu3d_image.data.ndim != 3:
        raise ValueError("Input HDU must be a 3D cube.")

    # get shape of the input 3D cube
    naxis3, naxis2, naxis1 = hdu3d_image.data.shape

    # initial pixel borders in the spectral axis
    old_wcs1d_spectral = WCS(hdu3d_image.header).spectral
    old_wl_borders = old_wcs1d_spectral.pixel_to_world(np.arange(naxis3+1)-0.5)
    # modify slightly the first and last values to avoid numerical issues
    deltawave = old_wl_borders[1] - old_wl_borders[0]
    old_wl_borders[0] = old_wl_borders[0] - deltawave/1E6
    deltawave = old_wl_borders[-1] - old_wl_borders[-2]
    old_wl_borders[-1] = old_wl_borders[-1] + deltawave/1E6

    # final pixel borders in the spectral axis
    new_wl_borders = crval3out + cdelt3out * (np.arange(naxis3out + 1) - 0.5) * u.pix

    resample_needed = True
    if naxis3 == naxis3out:
        # if the old and new wavelength borders are the same, just copy the data
        if np.all(np.allclose(old_wl_borders, new_wl_borders)):
            resampled_data = hdu3d_image.data.astype(np.float32)
            resample_needed = False
            if verbose:
                print("Old and new wavelength borders are the same.\n"
                      "-> Copying original data without spectral resampling.")

    if resample_needed:
        if verbose:
            print("Spectral resampling of the original 3D cube...", end=' ')
        # resample the 3D cube (see wavecal.py in teareduce for reference)
        resampled_data = np.zeros((naxis3out, naxis2, naxis1))
        for i in range(naxis1):
            for j in range(naxis2):
                # resample each spectrum independently
                data_spectrum = hdu3d_image.data[:, j, i]
                accum_flux = np.zeros(naxis3 + 1)
                accum_flux[1:] = np.cumsum(data_spectrum)
                flux_borders = np.interp(
                    x=new_wl_borders.value,
                    xp=old_wl_borders.value,
                    fp=accum_flux,
                    left=np.nan, 
                    right=np.nan
                )
                resampled_data[:, j, i] = flux_borders[1:] - flux_borders[:-1]
        if verbose:
            print("resampling completed!")

    # create new HDU with resampled data
    resampled_hdu = fits.PrimaryHDU(data=resampled_data.astype(np.float32))
    header_spectral_resampled = fits.Header()
    header_spectral_resampled['NAXIS'] = 1
    header_spectral_resampled['NAXIS1'] = naxis3out
    header_spectral_resampled['CRPIX1'] = 1.0
    header_spectral_resampled['CDELT1'] = cdelt3out.to(u.m/u.pix).value
    header_spectral_resampled['CRVAL1'] = crval3out.to(u.m).value
    header_spectral_resampled['CUNIT1'] = 'm'
    header_spectral_resampled['CTYPE1'] = 'WAVE'
    wcs1d_spectral_resampled = WCS(header_spectral_resampled)
    header_resampled = header3d_after_merging_wcs2d_celestial_and_wcs1d_spectral(
        wcs2d_celestial=WCS(hdu3d_image.header).celestial,
        wcs1d_spectral=wcs1d_spectral_resampled
    )
    resampled_hdu.header.update(header_resampled)
    
    return resampled_hdu


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Resample a 3D cube in the wavelength axis (NAXIS3)."
    )
    parser.add_argument("input_file", type=str, 
                        help="Input FITS file with the 3D cube.")
    parser.add_argument("output_file", type=str, 
                        help="Output FITS file with the resampled 3D cube.")
    parser.add_argument("--crval3out", type=float, 
                        help="Minimum wavelength for the output image (in meters).")
    parser.add_argument("--cdelt3out", type=float, 
                        help="Wavelength step for the output image (in meters).")
    parser.add_argument("--naxis3out", type=int, 
                        help="Number of slices in the output image.")
    parser.add_argument("--extname", type=str, 
                        help="Extension name of the input HDU (default: 'PRIMARY').", 
                        default='PRIMARY')
    parser.add_argument("--verbose",
                        help="Display intermediate information",
                        action="store_true")
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(ctext(f'{arg}: {value}', faint=True))

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    input_file = args.input_file
    output_file = args.output_file
    crval3out = args.crval3out
    if crval3out is not None:
        crval3out = crval3out * u.m
    cdelt3out = args.cdelt3out
    if cdelt3out is not None:
        cdelt3out = cdelt3out * u.m / u.pix
    naxis3out = args.naxis3out
    extname = args.extname
    verbose = args.verbose

    with fits.open(input_file) as hdul:
        if extname not in hdul:
            raise ValueError(f"Extension '{extname}' not found in {input_file}.")
        hdu3d_image = hdul[extname].copy()
        if verbose:
            print(f"{hdu3d_image.header['NAXIS1']=}")
            print(f"{hdu3d_image.header['NAXIS2']=}")
            print(f"{hdu3d_image.header['NAXIS3']=}")
    
    if crval3out is None or cdelt3out is None:
        wcs1d_spectral = WCS(hdu3d_image.header).spectral
        wave = wcs1d_spectral.pixel_to_world(np.arange(hdu3d_image.data.shape[0]))
        if crval3out is None:
            crval3out = wave[0]
            if verbose:
                print(f"Assuming {crval3out=}.")
        if cdelt3out is None:
            cdelt3out = (wave[1] - wave[0]) / u.pix
            if verbose:
                print(f"Assuming {cdelt3out=}.")

    if naxis3out is None:
        naxis3out = hdu3d_image.data.shape[0]
        if verbose:
            print(f"Assuming {naxis3out=}.")

    resampled_hdu = resample_wave_3d_cube(
        hdu3d_image=hdu3d_image, 
        crval3out=crval3out, 
        cdelt3out=cdelt3out, 
        naxis3out=naxis3out,
        verbose=verbose
    )

    if verbose:
        print(f'Saving: {output_file}')
    resampled_hdu.writeto(output_file, overwrite=True)


if __name__ == "__main__":
    main()
