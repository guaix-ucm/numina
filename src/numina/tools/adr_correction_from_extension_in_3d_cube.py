#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Apply ADR correction using offsets from FITS extension"""
import argparse

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from datetime import datetime
import numpy.ma as ma
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_interp, reproject_adaptive, reproject_exact
import sys

from .ctext import ctext

REPROJECT_METHODS = ['interp', 'adaptive', 'exact']


def apply_adr_correction_from_extension_in_3d_cube(
        inputfile,
        extname_adr,
        extname_mask,
        outputfile,
        reproject_method,
        verbose=False,
):
    """Apply ADR correction using offsets from FITS extension.

    Parameters
    ----------
    inputfile : str, file-like or `pathlib.Path`
        Input FITS filename.
    extname_adr : str
        FITS extension name with ADR correction data.
    extname_mask : str
        FITS extension name with mask. If 'None', use np.nan in
        image data.
    outputfile : str, file-like or `pathlib.Path`
        Output FITS filename.
    reproject_method : str
        Reprojection method. See 'REPROJECT_METHODS' above.
    verbose : bool
        If True, display additional information.
    """
    with fits.open(inputfile) as hdul:
        if verbose:
            print(hdul.info())
        primary_header = hdul[0].header
        if primary_header["NAXIS"] != 3:
            raise ValueError("Expected NAXIS=3 not found in PRIMARY HDU")
        # read 3D cube
        data3d = hdul[0].data.copy()
        # read ADR data
        if extname_adr not in hdul:
            raise ValueError(f"Extension '{extname_adr}' not found in FITS file.")
        else:
            if verbose:
                print(f'Reading ADR correction from {extname_adr} extension')
            table_adrcross = hdul[extname_adr].data.copy()
        # read mask or generate one from np.nan
        if extname_mask is None:
            hdu3d_mask = None
        else:
            if extname_mask not in hdul:
                raise ValueError(f"Extension '{extname_mask}' not found in FITS file.")
            else:
                if verbose:
                    print(f'Reading mask from {extname_mask} extension')
                hdu3d_mask = hdul[extname_mask]
                bitpix_mask = hdu3d_mask.header['BITPIX']
                if bitpix_mask != 8:
                    raise ValueError(f"BITPIX (mask): {bitpix_mask} is not 8")
                if data3d.shape != hdu3d_mask.data.shape:
                    raise ValueError(f"Shape of PRIMARY and {extname_mask} are different")
        # generate mask from np.nan when necessary
        if hdu3d_mask is None:
            if verbose:
                print("Generating mask from np.nan in PRIMARY HDU")
            mask3d = np.isnan(data3d).astype(np.uint8)
        else:
            mask3d = hdu3d_mask.data.copy()
        num_masked_pixels_in_data3d = mask3d.sum()
        if verbose:
            print(f"Number of masked pixels in the input 3D array: {num_masked_pixels_in_data3d}")

    # generate 3D WCS object
    wcs3d = WCS(primary_header)
    naxis1, naxis2, naxis3 = wcs3d.pixel_shape

    # ADR at the center of the field of view
    delta_x_center_ifu = table_adrcross['Delta_x']
    delta_y_center_ifu = table_adrcross['Delta_y']
    x_center_ifu, y_center_ifu = wcs3d.celestial.wcs.crpix  # FITS convention
    center_ifu_coord = wcs3d.celestial.pixel_to_world(
        x_center_ifu - 1.0,  # Python convention
        y_center_ifu - 1.0  # Python convention
    )
    if verbose:
        print(f'Center IFU coord: {center_ifu_coord}')
    x_center_ifu_corrected = x_center_ifu + delta_x_center_ifu
    y_center_ifu_corrected = y_center_ifu + delta_y_center_ifu

    # optimal celestial WCS for corrected cube
    wcs2d_blue = wcs3d.celestial.deepcopy()
    wcs2d_blue.wcs.crpix = np.array([x_center_ifu_corrected[0], y_center_ifu_corrected[0]])
    if verbose:
        print(f"\nwcs2d_blue:\n{wcs2d_blue}")
    wcs2d_red = wcs3d.celestial.deepcopy()
    wcs2d_red.wcs.crpix = np.array([x_center_ifu_corrected[-1], y_center_ifu_corrected[-1]])
    if verbose:
        print(f"\nwcs2d_red:\n{wcs2d_red}")
    wcs_mosaic2d, shape_mosaic2d = find_optimal_celestial_wcs(
        input_data=[
            ((naxis2, naxis1), wcs2d_blue),
            ((naxis2, naxis1), wcs2d_red)
        ],
        auto_rotate=False
    )
    if verbose:
        print(f"\nwcs_mosaic2d:\n{wcs_mosaic2d}")
        print(f"shape_mosaic2d: {shape_mosaic2d}")

    # initialize corrected 3D cube (using a masked array!)
    naxis3_corrected3d = naxis3
    naxis2_corrected3d, naxis1_corrected3d = shape_mosaic2d
    if verbose:
        print("NAXIS1, NAXIS2, NAXIS3 of corrected 3D cube: "
              f"{naxis1_corrected3d}, {naxis2_corrected3d}, {naxis3_corrected3d}")
    shape_corrected3d = (naxis3_corrected3d, naxis2_corrected3d, naxis1_corrected3d)
    array3d_corrected = ma.array(np.zeros(shape_corrected3d),
                                 mask=np.full(shape_corrected3d, fill_value=False))

    # compute corrected 3D cube slice by slice
    wcs2d = wcs3d.celestial.deepcopy()
    if verbose:
        print(f"Reprojection method: {reproject_method} (please wait...)")
    time_ini = datetime.now()
    """
    # alternative code for debugging purposes
    range1 = range(50)
    range2 = range(naxis3-50, naxis3)
    combined_range = list(x for r in (range1, range2) for x in r)
    for k in combined_range:
    """
    for k in range(naxis3):
        wcs2d.wcs.crpix = np.array([x_center_ifu_corrected[k], y_center_ifu_corrected[k]])
        data2d = data3d[k, :, :]
        hdu2d_data = fits.PrimaryHDU(data2d)
        hdu2d_data.header.extend(wcs2d.to_header(), update=True)
        mask2d = mask3d[k, :, :]
        hdu2d_mask = fits.PrimaryHDU(mask2d)
        hdu2d_mask.header.extend(wcs2d.to_header(), update=True)
        if reproject_method == "interp":
            data2d_resampled, footprint_data2d_resampled = reproject_interp(
                input_data=hdu2d_data,
                output_projection=wcs_mosaic2d,
                shape_out=shape_mosaic2d,
            )
            if num_masked_pixels_in_data3d > 0:
                mask2d_resampled, _ = reproject_interp(
                    input_data=hdu2d_mask,
                    output_projection=wcs_mosaic2d,
                    shape_out=shape_mosaic2d,
                )
            else:
                mask2d_resampled = None
        elif reproject_method == "adaptive":
            data2d_resampled, footprint_data2d_resampled = reproject_adaptive(
                input_data=hdu2d_data,
                output_projection=wcs_mosaic2d,
                shape_out=shape_mosaic2d,
                conserve_flux=True,
                kernel='Gaussian'
            )
            if num_masked_pixels_in_data3d > 0:
                mask2d_resampled, _ = reproject_adaptive(
                    input_data=hdu2d_mask,
                    output_projection=wcs_mosaic2d,
                    shape_out=shape_mosaic2d,
                    conserve_flux=True,
                    kernel='Gaussian'
                )
            else:
                mask2d_resampled = None
        elif reproject_method == "exact":
            data2d_resampled, footprint_data2d_resampled = reproject_exact(
                input_data=hdu2d_data,
                output_projection=wcs_mosaic2d,
                shape_out=shape_mosaic2d,
            )
            if num_masked_pixels_in_data3d > 0:
                mask2d_resampled, _ = reproject_exact(
                    input_data=hdu2d_mask,
                    output_projection=wcs_mosaic2d,
                    shape_out=shape_mosaic2d,
                )
            else:
                mask2d_resampled = None
        else:
            raise ValueError(f"Reprojection method '{reproject_method}' not recognized.")
        array3d_corrected[k, :, :] = data2d_resampled
        if mask2d_resampled is None:
            # use only the footprint
            array3d_corrected[k, :, :].mask = (footprint_data2d_resampled == 0)
        else:
            # merge corrected mask with footprint
            array3d_corrected[k, :, :].mask = np.logical_or(
                np.logical_or(mask2d_resampled, mask2d_resampled==np.nan),
                (footprint_data2d_resampled == 0)
            )

    time_end = datetime.now()

    # 3D footprint
    footprint3d_corrected = np.ones_like(array3d_corrected)
    footprint3d_corrected[np.isnan(array3d_corrected)] = 0

    if verbose:
        print(f"Reprojection finished! (elapsed time: {(time_end - time_ini).total_seconds()} seconds)")
        print("\nFlux check:")
        flux1 = np.sum(data3d)
        flux2 = np.nansum(array3d_corrected)
        print(f"- total counts in original  3D cube: {flux1}")
        print(f"- total counts in corrected 3D cube: {flux2}")
        print(f"- ratio original/corrected.........: {flux1/flux2}")
        print(f"\nFootprint coverage (fraction): {np.sum(footprint3d_corrected) / footprint3d_corrected.size}")

    # generate single 3D WCS combining the celestial and spectral axes
    header3d_corrected = wcs_mosaic2d.to_header()
    header_spectral = wcs3d.spectral.to_header()
    header3d_corrected['WCSAXES'] = 3
    for item in ['CRPIX', 'CDELT', 'CUNIT', 'CTYPE', 'CRVAL']:
        # insert {item}3 after {item}2 to preserve the order in the header
        header3d_corrected.insert(
            f'{item}2',
            (f'{item}3', header_spectral[f'{item}1'], header_spectral.comments[f'{item}1']),
            after=True)
    if verbose:
        print("\nheader3d_corrected:")
        for line in header3d_corrected.cards:
            print(line)

    # save result
    hdu = fits.PrimaryHDU(array3d_corrected.data.astype(np.float32))
    hdu.header.update(header3d_corrected)
    hdu_mask = fits.ImageHDU(data=array3d_corrected.mask.astype(np.uint8))
    hdu_mask.header['EXTNAME'] = 'MASK'
    hdu_mask.header.update(header3d_corrected)
    hdul = fits.HDUList([hdu, hdu_mask])
    if verbose:
        print(f"\nSaving file: {outputfile}")
    hdul.writeto(outputfile, overwrite=True)


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(description="Compare ADR extensions in 3D cube")
    parser.add_argument("input", help="Input 3D FITS file", type=str)
    parser.add_argument("--extname_adr", help="Name of the extension with ADR correction",
                        type=str, default=None)
    parser.add_argument("--extname_mask",
                        help="Name of the extension with mask in input 3D cube. "
                        "Default 'None': use np.nan in image",
                        default=None, type=str)
    parser.add_argument("--output", help="Output filename",
                        type=str, default=None)
    parser.add_argument('--reproject_method',
                        help='Reprojection method (interp, adaptive, exact)',
                        type=str, choices=REPROJECT_METHODS, default='adaptive')
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

    if args.extname_adr is None:
        raise ValueError("You must specify an extension name with --extname_adr")
    else:
        extname_adr = args.extname_adr.upper()

    extname_mask = args.extname_mask
    if extname_mask is not None:
        if extname_mask.lower() in ['none', 'nan']:
            extname_mask = None
        else:
            extname_mask = extname_mask.upper()

    if args.output is None:
        raise ValueError("You must specify an output filename with --output")

    apply_adr_correction_from_extension_in_3d_cube(
        inputfile=args.input,
        extname_adr=extname_adr,
        extname_mask=extname_mask,
        outputfile=args.output,
        reproject_method=args.reproject_method,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
