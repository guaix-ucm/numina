#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Resample a 3D cube in the wavelength axis (NAXIS3)."""

import sys

import argparse
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import logging
import numpy as np
from pathlib import Path
from rich_argparse import RichHelpFormatter

from numina.instrument.simulation.ifu.define_3d_wcs import header3d_after_merging_wcs2d_celestial_and_wcs1d_spectral
from numina.tools.initialize_script_with_args import include_default_arguments_for_common_actions
from numina.tools.initialize_script_with_args import initialize_script_with_args
from numina.tools.initialize_script_with_args import goodbye_message_and_save_console

from numina._version import __version__

from .add_script_info_to_fits_history import add_script_info_to_fits_history


def resample_wave_3d_cube(hdu3d_image, crval3out, cdelt3out, naxis3out):
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

    Returns
    -------
    resampled_hdu : `astropy.io.fits.ImageHDU`
        Resampled HDU instance with the 3D image.
    """
    logger = logging.getLogger(__name__)

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

    # create a copy of the header to avoid modifying the original
    header3d_copy = hdu3d_image.header.copy()
    # remove keywords that may cause issues
    for key in ["OBSGEO-X", "OBSGEO-Y", "OBSGEO-Z", "OBSGEO-L", "OBSGEO-B", "OBSGEO-H"]:
        header3d_copy.remove(key, ignore_missing=True)

    # initial pixel borders in the spectral axis
    old_wcs1d_spectral = WCS(header3d_copy).spectral
    old_wl_borders = old_wcs1d_spectral.pixel_to_world(np.arange(naxis3 + 1) - 0.5)
    # modify slightly the first and last values to avoid numerical issues
    deltawave = old_wl_borders[1] - old_wl_borders[0]
    old_wl_borders[0] = old_wl_borders[0] - deltawave / 1e6
    deltawave = old_wl_borders[-1] - old_wl_borders[-2]
    old_wl_borders[-1] = old_wl_borders[-1] + deltawave / 1e6

    # final pixel borders in the spectral axis
    new_wl_borders = crval3out + cdelt3out * (np.arange(naxis3out + 1) - 0.5) * u.pix

    resample_needed = True
    if naxis3 == naxis3out:
        # if the old and new wavelength borders are the same, just copy the data
        if np.all(np.allclose(old_wl_borders, new_wl_borders)):
            resampled_data = hdu3d_image.data.astype(np.float32)
            resample_needed = False
            logger.info(
                "Old and new wavelength borders are the same.\n" "-> Copying original data without spectral resampling."
            )

    if resample_needed:
        logger.info("Spectral resampling of the original 3D cube")
        logger.debug(f"Original wavelength borders:\n{old_wl_borders}")
        logger.debug(f"New wavelength borders:\n{new_wl_borders}")
        # resample the 3D cube (see wavecal.py in teareduce for reference)
        resampled_data = np.zeros((naxis3out, naxis2, naxis1), dtype=np.float32)
        logger.info(f"{np.isnan(hdu3d_image.data).sum()} NaN values in the original data.")
        for i in range(naxis1):
            for j in range(naxis2):
                # resample each spectrum independently
                data_spectrum = hdu3d_image.data[:, j, i].astype(np.float32)
                accum_flux = np.zeros(naxis3 + 1, dtype=np.float32)
                # the cumulative flux is computed as the cumulative sum of the original spectrum,
                # with NaN values replaced by 0
                accum_flux[1:] = np.nancumsum(data_spectrum)
                flux_borders = np.interp(
                    x=new_wl_borders.value, xp=old_wl_borders.value, fp=accum_flux, left=np.nan, right=np.nan
                )
                resampled_data[:, j, i] = flux_borders[1:] - flux_borders[:-1]
        logger.info(f"{np.isnan(resampled_data).sum()} NaN values in the resampled data.")

    # create new HDU with resampled data
    resampled_hdu = fits.PrimaryHDU(data=resampled_data.astype(np.float32))
    header_spectral_resampled = fits.Header()
    header_spectral_resampled["NAXIS"] = 1
    header_spectral_resampled["NAXIS1"] = naxis3out
    header_spectral_resampled["CRPIX1"] = 1.0
    header_spectral_resampled["CDELT1"] = cdelt3out.to(u.m / u.pix).value
    header_spectral_resampled["CRVAL1"] = crval3out.to(u.m).value
    header_spectral_resampled["CUNIT1"] = "m"
    header_spectral_resampled["CTYPE1"] = "WAVE"
    wcs1d_spectral_resampled = WCS(header_spectral_resampled)
    header_resampled = header3d_after_merging_wcs2d_celestial_and_wcs1d_spectral(
        wcs2d_celestial=WCS(header3d_copy).celestial, wcs1d_spectral=wcs1d_spectral_resampled
    )
    resampled_hdu.header.update(header_resampled)

    return resampled_hdu


def main(args=None):
    """Main function."""

    parser = argparse.ArgumentParser(
        description="Resample a 3D cube in the wavelength axis (NAXIS3).", formatter_class=RichHelpFormatter
    )
    parser.add_argument("input", type=str, help="Input FITS file with the 3D cube.")
    parser.add_argument("output", type=str, help="Output FITS file with the resampled 3D cube.")
    parser.add_argument("--crval3out", type=float, help="Minimum wavelength for the output image (in meters).")
    parser.add_argument("--cdelt3out", type=float, help="Wavelength step for the output image (in meters).")
    parser.add_argument("--naxis3out", type=int, help="Number of slices in the output image.")
    parser.add_argument(
        "--extname", type=str, help="Extension name of the input HDU (default: 'PRIMARY').", default="PRIMARY"
    )
    include_default_arguments_for_common_actions(parser)
    args = parser.parse_args(args)

    # Initialize the script with the provided arguments
    console, logger, datetime_ini = initialize_script_with_args(sys.argv, parser, args, __name__, __version__)

    input_file = args.input
    output_file = args.output
    if args.crval3out is None and args.cdelt3out is None and args.naxis3out is None:
        raise ValueError("At least one of --crval3out, --cdelt3out, or --naxis3out must be specified.")
    crval3out = args.crval3out
    if crval3out is not None:
        crval3out = crval3out * u.m
    cdelt3out = args.cdelt3out
    if cdelt3out is not None:
        cdelt3out = cdelt3out * u.m / u.pix
    naxis3out = args.naxis3out
    extname = args.extname

    with fits.open(input_file) as hdul:
        if extname not in hdul:
            raise ValueError(f"Extension '{extname}' not found in {input_file}.")
        hdu3d_image = hdul[extname].copy()
        logger.info(f"Loaded {hdu3d_image.header['NAXIS1']=}")
        logger.info(f"Loaded {hdu3d_image.header['NAXIS2']=}")
        logger.info(f"Loaded {hdu3d_image.header['NAXIS3']=}")

    if crval3out is None or cdelt3out is None:
        wcs1d_spectral = WCS(hdu3d_image.header).spectral
        wave = wcs1d_spectral.pixel_to_world(np.arange(hdu3d_image.data.shape[0]))
        if crval3out is None:
            crval3out = wave[0]
            logger.info(f"Assuming {crval3out=}.")
        if cdelt3out is None:
            cdelt3out = (wave[1] - wave[0]) / u.pix
            logger.info(f"Assuming {cdelt3out=}.")

    if naxis3out is None:
        naxis3out = hdu3d_image.data.shape[0]
        logger.info(f"Assuming {naxis3out=}.")

    resampled_hdu = resample_wave_3d_cube(
        hdu3d_image=hdu3d_image,
        crval3out=crval3out,
        cdelt3out=cdelt3out,
        naxis3out=naxis3out,
    )

    add_script_info_to_fits_history(resampled_hdu.header, args, parser)
    resampled_hdu.writeto(Path(args.output_dir) / output_file, overwrite=True)

    # Display goodbye message and save console log if recording is enabled
    goodbye_message_and_save_console(logger, console, datetime_ini, args.record, args.output_dir)


if __name__ == "__main__":
    main()
