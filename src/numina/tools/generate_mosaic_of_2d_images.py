#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Generate a 2D mosaic from individual 2D images."""

import argparse
from astropy.io import fits
import numpy as np
from reproject import reproject_interp, reproject_adaptive, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
import sys

LIST_OF_METHODS = ['interp', 'adaptive', 'exact']


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list",
                        help="TXT file with list of 2D images to be combined")
    parser.add_argument('output_filename',
                        help='filename of output FITS image')
    parser.add_argument('--method',
                        help='Reprojection method (interp, adaptive, exact)',
                        type=str, choices=LIST_OF_METHODS, default='adaptive')
    parser.add_argument('--extnum',
                        help='Extension number in input files (note that ' +
                             'first extension is 1 = default value)',
                        default=1, type=int)
    parser.add_argument("--verbose",
                        help="Display intermediate information",
                        action="store_true")
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args)

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    input_list = args.input_list
    output_filename = args.output_filename
    method = args.method
    verbose = args.verbose

    # first extension is number 1 for the user
    extnum = args.extnum - 1

    # read input file with list of 2D images to be combined
    with open(input_list) as f:
        file_content = f.read().splitlines()

    # list of HDU
    list_of_hdu = []
    for fname in file_content:
        if len(fname) > 0:
            if fname[0] not in ['#']:
                print(f'Reading: {fname}')
                with fits.open(fname) as hdul:
                    list_of_hdu.append(hdul[extnum].copy())
                    if verbose:
                        print(f"{hdul[extnum].header['NAXIS1']=}")
                        print(f"{hdul[extnum].header['NAXIS2']=}")

    # compute optimal WCS for combined image
    wcs_mosaic, shape_mosaic = find_optimal_celestial_wcs(list_of_hdu)
    if verbose:
        print(f'{wcs_mosaic=}')
        print(f'{shape_mosaic=}')

    # generate mosaic

    if method == 'interp':
        mosaic, footprint_mosaic = reproject_and_coadd(
            list_of_hdu,
            wcs_mosaic,
            shape_out=shape_mosaic,
            reproject_function=reproject_interp,
        )
    elif method == 'adaptive':
        mosaic, footprint_mosaic = reproject_and_coadd(
            list_of_hdu,
            wcs_mosaic,
            shape_out=shape_mosaic,
            reproject_function=reproject_adaptive,
            conserve_flux=True,
            kernel='Gaussian'
        )
    elif method == 'exact':
        mosaic, footprint_mosaic = reproject_and_coadd(
            list_of_hdu,
            wcs_mosaic,
            shape_out=shape_mosaic,
            reproject_function=reproject_exact,
        )
    else:
        raise ValueError(f'Unexpected {method=}')

    # save result
    hdu = fits.PrimaryHDU(mosaic.astype(np.float32))
    hdu.header.extend(wcs_mosaic.to_header(), update=True)
    hdu_footprint = fits.ImageHDU(data=footprint_mosaic)
    hdu_footprint.header['EXTNAME'] = 'FOOTPRINT'
    hdul = fits.HDUList([hdu, hdu_footprint])
    if verbose:
        print(f'Saving: {output_filename}')
    hdul.writeto(output_filename, overwrite='yes')


if __name__ == "__main__":

    main()
