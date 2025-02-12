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
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import numpy as np
import numpy.ma as ma
from reproject import reproject_interp, reproject_adaptive, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs
import sys

REPROJECT_METHODS = ['interp', 'adaptive', 'exact']
COMBINATION_FUNCTIONS = ['mean', 'median', 'sum', 'std','sigmaclip_mean', 'sigmaclip_median', 'sigmaclip_stddev']


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list",
                        help="TXT file with list of 2D images to be combined")
    parser.add_argument('output_filename',
                        help='filename of output FITS image')
    parser.add_argument('--reproject_method',
                        help='Reprojection method (interp, adaptive, exact)',
                        type=str, choices=REPROJECT_METHODS, default='adaptive')
    parser.add_argument('--extname_image',
                        help='Extension name for image in input files. Default value: PRIMARY',
                        default='PRIMARY', type=str)
    parser.add_argument('--extname_mask',
                        help='Extension name for mask in input files. Default None=use np.nan',
                        default=None, type=str)
    parser.add_argument("--combination_function", help='Combination function. Default: mean',
                        type=str, default='mean', choices=COMBINATION_FUNCTIONS)
    parser.add_argument("--output_3D_stack",
                        help="filename for stacked 3D array. Default None",
                        default=None, type=str)
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
    reproject_method = args.reproject_method
    combination_function = args.combination_function
    output_3d_stack = args.output_3D_stack
    verbose = args.verbose

    # define extensions for image and mask
    extname_image = args.extname_image
    extname_mask = args.extname_mask
    if extname_mask is not None:
        if extname_mask.lower() == 'none':
            extname_mask = None

    # read input file with list of 2D images to be combined
    with open(input_list) as f:
        file_content = f.read().splitlines()

    # list of HDU
    list_of_hdu2d_images = []
    list_of_hdu2d_masks = []
    for fname in file_content:
        if len(fname) > 0:
            if fname[0] not in ['#']:
                print(f'\n* Reading: {fname}')
                with fits.open(fname) as hdul:
                    if extname_image not in hdul:
                        raise ValueError(f'Expected {extname_image} extension not found')
                    hdu2d_image = hdul[extname_image].copy()
                    if verbose:
                        print(f"{hdu2d_image.header['NAXIS1']=}")
                        print(f"{hdu2d_image.header['NAXIS2']=}")
                    if extname_mask is not None:
                        if extname_mask in hdul:
                            hdu2d_mask = hdul[extname_mask].copy()
                            if verbose:
                                print(f'image mask: {extname_mask}')
                            bitpix_mask = hdu2d_mask.header['BITPIX']
                            if bitpix_mask != 8:
                                raise ValueError(f'BITPIX (mask): {bitpix_mask} is not 8')
                            if hdu2d_image.data.shape != hdu2d_mask.data.shape:
                                raise ValueError(f'Shape of {extname_image} and {extname_mask} are different!')
                        else:
                            hdu2d_mask = None
                    else:
                        hdu2d_mask = None
                    # generate mask from np.nan when necessary
                    if hdu2d_mask is None:
                        hdu2d_mask = fits.ImageHDU(data=np.isnan(hdu2d_image.data).astype(np.uint8))
                        hdu2d_mask.header.extend(WCS(hdu2d_image.header).to_header(), update=True)
                        if verbose:
                            print(f'generating mask from np.nan values')
                    if verbose:
                        print(f'Number of masked pixels / total: {np.sum(hdu2d_mask.data)} / {hdu2d_mask.size}')
                    # store tuple with image and associated footprint
                    list_of_hdu2d_images.append(hdu2d_image)
                    list_of_hdu2d_masks.append(hdu2d_mask)

    nimages = len(list_of_hdu2d_images)
    if verbose:
        print(f'Total number of images to be combined: {nimages}')

    if nimages < 1:
        raise ValueError('Number of images = 0')

    # compute optimal WCS for combined image
    wcs_mosaic2d, shape_mosaic2d = find_optimal_celestial_wcs(list_of_hdu2d_images)
    if verbose:
        print(f'\n{wcs_mosaic2d=}')
        print(f'{shape_mosaic2d=}')
    naxis2_mosaic, naxis1_mosaic = shape_mosaic2d

    # transform individual images and store result in auxiliary 3D cube
    shape3d = nimages, naxis2_mosaic, naxis1_mosaic
    stack3d = ma.array(np.zeros(shape3d), mask=np.full(shape3d, fill_value=False))

    # generate mosaic
    if verbose:
        print(f'Reprojection method: {reproject_method}')
    for i in range(nimages):
        hdu2d_image = list_of_hdu2d_images[i]
        hdu2d_mask = list_of_hdu2d_masks[i]
        if reproject_method == 'interp':
            image2d_resampled, footprint_image2d_resampled = reproject_interp(
                hdu2d_image,
                wcs_mosaic2d,
                shape_out=shape_mosaic2d
            )
            if hdu2d_mask is not None:
                mask2d_resampled, _ = reproject_interp(
                    hdu2d_mask,
                    wcs_mosaic2d,
                    shape_out=shape_mosaic2d
                )
            else:
                mask2d_resampled = None
        elif reproject_method == 'adaptive':
            image2d_resampled, footprint_image2d_resampled = reproject_adaptive(
                hdu2d_image,
                wcs_mosaic2d,
                shape_out=shape_mosaic2d,
                conserve_flux=True,
                kernel='Gaussian'
            )
            if hdu2d_mask is not None:
                mask2d_resampled, _ = reproject_adaptive(
                    hdu2d_mask,
                    wcs_mosaic2d,
                    shape_out=shape_mosaic2d,
                    conserve_flux=True,
                    kernel='Gaussian'
                )
            else:
                mask2d_resampled = None
        elif reproject_method == 'exact':
            image2d_resampled, footprint_image2d_resampled = reproject_exact(
                hdu2d_image,
                wcs_mosaic2d,
                shape_out=shape_mosaic2d,
            )
            if hdu2d_mask is not None:
                mask2d_resampled, _ = reproject_exact(
                    hdu2d_mask,
                    wcs_mosaic2d,
                    shape_out=shape_mosaic2d,
                )
            else:
                mask2d_resampled = None
        else:
            raise ValueError(f'Unexpected {reproject_method=}')
        stack3d[i, :, :] = image2d_resampled
        if mask2d_resampled is None:
            mask2d_resampled = np.full(shape_mosaic2d, fill_value=False)
            stack3d[i, :, :].mask = mask2d_resampled
        else:
            # merge footprints
            stack3d[i, :, :].mask = np.logical_or(
                np.logical_or(mask2d_resampled, mask2d_resampled==np.nan),
                (footprint_image2d_resampled == 0)
            )

    # generate mosaic
    if combination_function == 'mean':
        mosaic = stack3d.mean(axis=0)
    elif combination_function == 'median':
        mosaic = stack3d.median(axis=0)
    elif combination_function == 'sum':
        mosaic = stack3d.sum(axis=0)
    elif combination_function == 'std':
        mosaic = stack3d.std(axis=0)
    elif combination_function[:9] == 'sigmaclip':
        array2d_sigclip_mean, array2d_sigclip_median, array2d_sigclip_stddev = \
            sigma_clipped_stats(stack3d, axis=0, maxiters=None)
        if combination_function == 'sigmaclip_mean':
            mosaic = array2d_sigclip_mean
        elif combination_function == 'sigmaclip_median':
            mosaic = array2d_sigclip_median
        elif combination_function == 'sigmaclip_stddev':
            mosaic = array2d_sigclip_stddev
        else:
            raise ValueError(f'Unexpected {combination_function=}')
    else:
        raise ValueError(f'Unexpected {combination_function=}')

    if verbose:
        print(f'Combination function: {combination_function}')

    # set to np.nan the masked values
    mosaic.data[mosaic.mask] = np.nan

    # save result
    hdu = fits.PrimaryHDU(mosaic.data.astype(np.float32))
    hdu.header.extend(wcs_mosaic2d.to_header(), update=True)
    hdu_mask = fits.ImageHDU(data=mosaic.mask.astype(np.uint8))
    hdu_mask.header['EXTNAME'] = 'MASK'
    hdu_mask.header.extend(wcs_mosaic2d.to_header(), update=True)
    hdul = fits.HDUList([hdu, hdu_mask])
    if verbose:
        print(f'\nSaving combined 2D image: {output_filename}')
    hdul.writeto(output_filename, overwrite='yes')

    # save 3D stack if requested
    if output_3d_stack is not None:
        hdu = fits.PrimaryHDU(stack3d.data.astype(np.float32))
        hdu.header.extend(wcs_mosaic2d.to_header(), update=True)
        hdu_mask = fits.ImageHDU(data=stack3d.mask.astype(np.uint8))
        hdu_mask.header['EXTNAME'] = 'MASK'
        hdu_mask.header.extend(wcs_mosaic2d.to_header(), update=True)
        hdul = fits.HDUList([hdu, hdu_mask])
        if verbose:
            print(f'Saving 3D stack.........: {output_3d_stack}')
        hdul.writeto(output_3d_stack, overwrite='yes')


if __name__ == "__main__":

    main()
