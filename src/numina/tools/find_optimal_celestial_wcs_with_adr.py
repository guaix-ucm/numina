#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compute optimal celestial wcs considering ADR.
"""
import argparse
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from reproject.mosaicking import find_optimal_celestial_wcs
import sys

from .ctext import ctext


def find_optimal_celestial_wcs_with_adr(
        input_list,
        extname_adr,
        verbose=False,
):
    """Compute optimal celestial wcs considering ADR.

    Parameters
    ----------
    input_list : str
        Input text file with the name of the FITS files.
    extname_adr : str
        Extension name to where ADR distortion is stored.
    verbose : bool
        If True, display additional information.
    """

    # read input file with list of 3D images to be combined
    with open(input_list) as f:
        file_content = f.read().splitlines()

    # generate input for reproject.mosaicking.find_optimal_celestial_wcs
    list_of_inputs = []
    for fname in file_content:
        if len(fname) > 0:
            if fname[0] not in ['#']:
                print(f'\n* Reading: {fname}')
                with fits.open(fname) as hdul:
                    header3d = hdul[0].header
                    wcs3d = WCS(header3d)
                    if extname_adr is None:
                        wcs2d = wcs3d.celestial
                        list_of_inputs.append(((header3d['NAXIS2'], header3d['NAXIS1']), wcs2d))
                    else:
                        if extname_adr not in hdul:
                            raise ValueError(f"Expected '{extname_adr}' extension not found")
                        if verbose:
                            print(f'Reading ADR correction from {extname_adr} extension')
                        table_adrcross = hdul[extname_adr].data.copy()
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
                        list_of_inputs.append(((header3d['NAXIS2'], header3d['NAXIS1']), wcs2d_blue))
                        list_of_inputs.append(((header3d['NAXIS2'], header3d['NAXIS1']), wcs2d_red))

    wcs_mosaic2d, shape_mosaic2d = find_optimal_celestial_wcs(list_of_inputs)

    return wcs_mosaic2d, shape_mosaic2d


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list", help="TXT file with list of 3D images to be combined", type=str)
    parser.add_argument("--extname_adr",
                        help="Extension name to where ADR distortion is stored. Defaults to None (no ADR distortion)",
                        type=str)
    parser.add_argument("--output_celestial_2d_wcs", help="filename for output celestial 2D WCS", type=str)
    parser.add_argument("--verbose", help="Display intermediate information", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")

    args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(ctext(f'{arg}: {value}', faint=True))

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    input_list = args.input_list
    extname_adr = args.extname_adr
    if extname_adr is not None:
        extname_adr = extname_adr.upper()
    verbose = args.verbose
    output_celestial_2d_wcs = args.output_celestial_2d_wcs

    wcs_mosaic2d, shape_mosaic2d = find_optimal_celestial_wcs_with_adr(
        input_list=input_list,
        extname_adr=extname_adr,
        verbose=extname_adr
    )

    if verbose:
        print(f'\n{wcs_mosaic2d=}')
        print(f'\n{shape_mosaic2d=}')

    if output_celestial_2d_wcs is not None:
        header_2d_wcs = wcs_mosaic2d.to_header()
        if extname_adr is not None:
            header_2d_wcs['EXTN_ADR'] = extname_adr
        else:
            header_2d_wcs['EXTN_ADR'] = '<None>'
        hdu = fits.PrimaryHDU(np.zeros(shape_mosaic2d, dtype=np.uint8), header=header_2d_wcs)
        hdu.writeto(output_celestial_2d_wcs, overwrite=True)


if __name__ == "__main__":
    find_optimal_celestial_wcs_with_adr()
