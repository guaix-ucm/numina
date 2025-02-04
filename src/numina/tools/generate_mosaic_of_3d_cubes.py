#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Generate a 3D mosaic from individual 3D cubes.

The combination is performed preserving the spectral axis (NAXIS3).
"""

import argparse
from astropy.io import fits
from astropy.wcs import WCS, WCSSUB_CELESTIAL, WCSSUB_SPECTRAL
import numpy as np
from reproject import reproject_interp, reproject_adaptive, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
import sys

from numina.array.array_size_32bits import array_size_8bits, array_size_32bits

LIST_OF_METHODS = ['interp', 'adaptive', 'exact']


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list",
                        help="TXT file with list of 3D images to be combined")
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

    # read input file with list of 3D images to be combined
    with open(input_list) as f:
        file_content = f.read().splitlines()

    # list of HDU
    list_of_hdu3d = []
    for fname in file_content:
        if len(fname) > 0:
            if fname[0] not in ['#']:
                print(f'Reading: {fname}')
                with fits.open(fname) as hdul:
                    list_of_hdu3d.append(hdul[extnum].copy())
                    if verbose:
                        print(f"{hdul[extnum].header['NAXIS1']=}")
                        print(f"{hdul[extnum].header['NAXIS2']=}")
                        print(f"{hdul[extnum].header['NAXIS3']=}")

    wcs3d = WCS(list_of_hdu3d[0].header)
    if verbose:
        print(f'\n{wcs3d.sub([WCSSUB_CELESTIAL])=}')
        print(f'\n{wcs3d.sub([WCSSUB_SPECTRAL])=}')

    # check the wavelength sampling
    wcs1d_spectral_ini = None
    for i, hdu in enumerate(list_of_hdu3d):
        if i == 0:
            wcs1d_spectral_ini = WCS(hdu.header).sub([WCSSUB_SPECTRAL])
        else:
            wcs1d_spectral = WCS(hdu.header).sub([WCSSUB_SPECTRAL])
            if wcs1d_spectral.__str__() != wcs1d_spectral_ini.__str__():
                print(f'{list_of_hdu3d[i]}')
                raise ValueError('ERROR: spectral sampling is different!')

    # compute optimal 2D WCS for combined image
    list_of_hdu2d = []
    for hdu in list_of_hdu3d:
        header3d = hdu.header
        data3d = hdu.data
        wcs2d = WCS(header3d).sub([WCSSUB_CELESTIAL])
        data2d = np.sum(data3d, axis=0)
        hdu2d = fits.PrimaryHDU(data2d)
        hdu2d.header.extend(wcs2d.to_header(), update=True)
        list_of_hdu2d.append(hdu2d)

    wcs_mosaic2d, shape_mosaic2d = find_optimal_celestial_wcs(list_of_hdu2d)
    if verbose:
        print(f'\n{wcs_mosaic2d=}')
        print(f'\n{shape_mosaic2d=}')

    # generate 3D mosaic
    wcs3d = WCS(list_of_hdu3d[0].header)
    naxis3_mosaic3d = wcs3d.pixel_shape[-1]
    naxis2_mosaic3d, naxis1_mosaic3d = shape_mosaic2d
    mosaic3d_cube_by_cube = np.zeros((naxis3_mosaic3d, naxis2_mosaic3d, naxis1_mosaic3d))
    footprint3d = np.zeros((naxis3_mosaic3d, naxis2_mosaic3d, naxis1_mosaic3d))
    if verbose:
        print(f'\nNAXIS1, NAXIS2, NAXIS3 of 3D mosaic: {naxis1_mosaic3d}, {naxis2_mosaic3d}, {naxis3_mosaic3d}')
        size1 = array_size_32bits(mosaic3d_cube_by_cube)
        size2 = array_size_8bits(footprint3d)
        print(f'Combined image will require {size1 + size2:.2f}')

    nimages = len(list_of_hdu3d)
    for i in range(nimages):
        data_ini3d = list_of_hdu3d[i].data
        wcs_ini3d = WCS(list_of_hdu3d[i].header)
        wcs_ini2d = wcs_ini3d.sub([WCSSUB_CELESTIAL])
        if method == 'interp':
            temp3d, footprint_temp3d = reproject_interp(
                (data_ini3d, wcs_ini2d),
                wcs_mosaic2d,
                shape_out=shape_mosaic2d
            )
        elif method == 'adaptive':
            temp3d, footprint_temp3d = reproject_adaptive(
                (data_ini3d, wcs_ini2d),
                wcs_mosaic2d,
                shape_out=shape_mosaic2d,
                conserve_flux=True,
                kernel='Gaussian'
            )
        elif method == 'exact':
            temp3d, footprint_temp3d = reproject_exact(
                (data_ini3d, wcs_ini2d),
                wcs_mosaic2d,
                shape_out=shape_mosaic2d
            )
        else:
            raise ValueError(f'Unexpected {method=}')
        valid_region = footprint_temp3d > 0
        mosaic3d_cube_by_cube[valid_region] += temp3d[valid_region]
        footprint3d += footprint_temp3d

    valid_region = footprint3d > 0
    mosaic3d_cube_by_cube[valid_region] /= footprint3d[valid_region]

    # generate resulting 3D WCS object
    wcs_mosaic3d = WCS(naxis=3)
    wcs_mosaic3d.wcs.crpix = [wcs_mosaic2d.wcs.crpix[0], wcs_mosaic2d.wcs.crpix[1], wcs1d_spectral_ini.wcs.crpix[0]]
    wcs_mosaic3d.wcs.cdelt = [wcs_mosaic2d.wcs.cdelt[0], wcs_mosaic2d.wcs.cdelt[1], wcs1d_spectral_ini.wcs.cdelt[0]]
    wcs_mosaic3d.wcs.crval = [wcs_mosaic2d.wcs.crval[0], wcs_mosaic2d.wcs.crval[1], wcs1d_spectral_ini.wcs.crval[0]]
    wcs_mosaic3d.wcs.ctype = [wcs_mosaic2d.wcs.ctype[0], wcs_mosaic2d.wcs.ctype[1], wcs1d_spectral_ini.wcs.ctype[0]]
    # include the appropriate values of the PC matrix
    wcs_mosaic3d.wcs.pc = np.eye(3)
    wcs_mosaic3d.wcs.pc[0:2, 0:2] = wcs_mosaic2d.wcs.pc
    wcs_mosaic3d.wcs.pc[2, 2] = wcs1d_spectral_ini.wcs.pc[0, 0]
    if verbose:
        print(f'\n{wcs_mosaic3d=}')

    # save result
    hdu = fits.PrimaryHDU(mosaic3d_cube_by_cube.astype(np.float32))
    hdu.header.extend(wcs_mosaic3d.to_header(), update=True)
    hdu_footprint = fits.ImageHDU(data=footprint3d.astype(np.uint8))
    hdu_footprint.header['EXTNAME'] = 'FOOTPRINT'
    hdu_footprint.header.extend(wcs_mosaic3d.to_header(), update=True)
    hdul = fits.HDUList([hdu, hdu_footprint])
    if verbose:
        print(f'Saving: {output_filename}')
    hdul.writeto(output_filename, overwrite='yes')


if __name__ == "__main__":

    main()
