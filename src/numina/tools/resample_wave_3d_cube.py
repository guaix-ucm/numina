#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Resample a 3D cube in the wavelength axis (NAXIS3)."""

import argparse
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import logging
import numpy as np
from pathlib import Path
from rich_argparse import RichHelpFormatter
from scipy import ndimage
import sys

from numina.instrument.simulation.ifu.define_3d_wcs import header3d_after_merging_wcs2d_celestial_and_wcs1d_spectral

from .add_script_info_to_fits_history import add_script_info_to_fits_history
from .hdul_utils import get_wcs_from_hdu
from .initialize_script_with_args import NuminaScriptDefinition


def resample_wave_3d_cube(hdu3d_image, wcskey, crval3out, cdelt3out, naxis3out, connected_zeros_to_nan=False):
    """Resample a 3D cube to a new wavelength sampling.

    The celestial WCS is preserved, and the spectral WCS is modified
    in order to make use of the new wavelength sampling.

    Parameters
    ----------
    hdu3d_image : `astropy.io.fits.ImageHDU`
        HDU instance with the 3D image to be resampled.
    wcskey : str
        WCS key to use when multiple WCS are present in the FITS header.
    crval3out : `astropy.units.Quantity`
        Minimum wavelength for the output image.
    cdelt3out : `astropy.units.Quantity`
        Wavelength step for the output image.
    naxis3out : int
        Number of slices in the output image.
    connected_zeros_to_nan : bool
        If True, convert connected zeros in the input image to NaN
        prior to resampling.

    Returns
    -------
    resampled_hdu : `astropy.io.fits.ImageHDU`
        Resampled HDU instance with the 3D image.
    """
    logger = logging.getLogger(__name__)

    # protections
    if not isinstance(hdu3d_image, fits.ImageHDU) and not isinstance(hdu3d_image, fits.PrimaryHDU):
        logger.error(f"Input HDU type: {type(hdu3d_image)}. It must be an ImageHDU or PrimaryHDU.")
        sys.exit(1)
    if crval3out is None:
        logger.error("crval3out must be specified.")
        sys.exit(1)
    if cdelt3out is None:
        logger.error("cdelt3out must be specified.")
        sys.exit(1)
    if naxis3out is None:
        logger.error("naxis3out must be specified.")
        sys.exit(1)
    if hdu3d_image.data.ndim != 3:
        logger.error(f"Input HDU has {hdu3d_image.data.ndim} dimensions. It must be a 3D cube.")
        sys.exit(1)

    # get shape of the input 3D cube
    naxis3, naxis2, naxis1 = hdu3d_image.data.shape

    # create a copy of the header to avoid modifying the original
    header3d_copy = hdu3d_image.header.copy()

    # initial pixel borders in the spectral axis
    old_wcs1d_spectral = get_wcs_from_hdu(hdu3d_image, wcskey=wcskey).spectral
    old_wl_borders = old_wcs1d_spectral.pixel_to_world(np.arange(naxis3 + 1) - 0.5)
    # modify slightly the first and last values to avoid numerical issues
    deltawave = old_wl_borders[1] - old_wl_borders[0]
    old_wl_borders[0] = old_wl_borders[0] - deltawave / 1e6
    deltawave = old_wl_borders[-1] - old_wl_borders[-2]
    old_wl_borders[-1] = old_wl_borders[-1] + deltawave / 1e6

    # final pixel borders in the spectral axis
    new_wl_borders = crval3out + cdelt3out * (np.arange(naxis3out + 1) - 0.5) * u.pix

    # copy the input data to avoid modifying the original
    input_data = hdu3d_image.data.astype(np.float32).copy()
    if connected_zeros_to_nan:
        # Convert connected zeros to NaN
        mask_zeros = input_data == 0
        if np.any(mask_zeros):
            # 3D neighbourhood: 26 neighbours (faces, edges, corners)
            kernel = np.ones((3, 3, 3), dtype=int)
            kernel[1, 1, 1] = 0  # exclude the center pixel itself
            num_zeros_neighbors = ndimage.convolve(mask_zeros.astype(int), kernel, mode="constant", cval=0)
            # Identify connected zeros: pixels that are zero and have at least one zero neighbor
            mask_connected_zeros = mask_zeros & (num_zeros_neighbors > 0)
            logger.info(f"Number of pixels in the input cube: {naxis1} x {naxis2} x {naxis3} = {mask_zeros.size}")
            ldum = len(str(mask_zeros.size))
            logger.info(f"Found     : {np.sum(mask_zeros):>{ldum}d} zeros in the input cube")
            logger.info(
                f"Converting: {np.sum(mask_connected_zeros):>{ldum}d} connected zeros to NaN in the input cube."
            )
            if np.any(mask_connected_zeros):
                input_data[mask_connected_zeros] = np.nan

    resample_needed = True
    if naxis3 == naxis3out:
        # if the old and new wavelength borders are the same, just copy the data
        if np.all(np.allclose(old_wl_borders, new_wl_borders)):
            resampled_data = input_data
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
        logger.info(f"Number of pixels in the output cube: {naxis1} x {naxis2} x {naxis3out} = {resampled_data.size}")
        ldum = len(str(resampled_data.size))
        logger.info(f"{np.isnan(input_data).sum():>{ldum}d} NaN values in the original {input_data.shape} array")
        # resample each spectrum independently
        for i in range(naxis1):
            for j in range(naxis2):
                # get the original spectrum for this pixel
                data_spectrum = input_data[:, j, i].astype(np.float32)
                # the cumulative flux is computed as the cumulative sum of the original spectrum,
                # with NaN values replaced by 0
                accum_data = np.zeros(naxis3 + 1, dtype=np.float32)
                accum_data[1:] = np.nancumsum(data_spectrum)
                # the cumulative flux is interpolated to the new wavelength borders, with the flux
                # outside the original wavelength range set to NaN (note that intermediate NaN values
                # are replaced by 0 in the cumulative sum, so they do not affect the interpolation,
                # but their location is lost, so we need to check later how NaN values are propagated
                # in the resampled data)
                flux_borders = np.interp(
                    x=new_wl_borders.value, xp=old_wl_borders.value, fp=accum_data, left=np.nan, right=np.nan
                )
                resampled_data[:, j, i] = flux_borders[1:] - flux_borders[:-1]
                # repeat the same work for the NaN spectrum, where 1 is invalid data and 0 is valid data;
                # this is necessary because the previous data interpolation is not propagating NaN values
                # correctly, and we need to set the resampled data to NaN if any of the original data was NaN
                mask_nan = np.isnan(data_spectrum)  # mask of NaN values in the original spectrum
                # if any of the original data was NaN, compute a resampled NaN spectrum and set the
                # affected pixels in the resampled data to NaN
                if np.any(mask_nan):
                    # compute the NaN spectrum, where 1 is invalid (NaN) data and 0 is valid data
                    nan_spectrum = np.zeros(naxis3, dtype=np.float32)
                    nan_spectrum[mask_nan] = 1.0
                    # compute the cumulative NaN spectrum
                    accum_nan = np.zeros(naxis3 + 1, dtype=np.float32)
                    accum_nan[1:] = np.cumsum(nan_spectrum)
                    # the cumulative NaN spectrum is interpolated to the new wavelength borders
                    nan_borders = np.interp(
                        x=new_wl_borders.value, xp=old_wl_borders.value, fp=accum_nan, left=np.nan, right=np.nan
                    )
                    nan_resampled = nan_borders[1:] - nan_borders[:-1]
                    # if the resampled NaN spectrum is not zero, set the resampled data to NaN
                    if np.any(nan_resampled > 0):
                        resampled_data[:, j, i][nan_resampled > 0] = np.nan
        logger.info(
            f"{np.isnan(resampled_data).sum():>{ldum}d} NaN values in the resampled {resampled_data.shape} array"
        )

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
    # Include default arguments for common actions, and initialize console and logging
    myscript = NuminaScriptDefinition(parser)
    args = myscript.args
    logger = myscript.logger

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
    myscript.goodbye_message_and_save_console()


if __name__ == "__main__":
    main()
