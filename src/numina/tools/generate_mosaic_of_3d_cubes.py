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
from astropy.wcs import WCS
import numpy as np
from reproject import reproject_interp, reproject_adaptive, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs
import sys

from numina.array.array_size_32bits import array_size_8bits, array_size_32bits
from .ctext import ctext

REPROJECT_METHODS = ['interp', 'adaptive', 'exact']
COMBINATION_FUNCTIONS = ['mean', 'median', 'sum', 'std','sigmaclip_mean', 'sigmaclip_median', 'sigmaclip_stddev']


# ToDo: revisar generación de wcs_mosaic3d final (mezcla de CELESTIAL y SPECTRAL)
#       Ver 20240722_atmospheric_differential_refraction/correct_adr.ipynb
# ToDo: hacer uso de combination_function
def generate_mosaic_of_3d_cubes(
        list_of_hdu3d_images,
        list_of_hdu3d_masks,
        islice1,
        islice2,
        final_celestial_wcs,
        reproject_method,
        combination_function,
        output_celestial_2d_wcs,
        verbose
):
    """Combine 3D cubes using their WCS information.

    Parameters
    ----------
    list_of_hdu3d_images : list of HDU images
        List of 3D HDU instances containing the images to be combined.
    list_of_hdu3d_masks : list or None
        List of 3D HDU instances containing the masks associated to
        the images to be combined. If this list is None, this function
        computes a particular mask for each 3D image prior to the
        combination, looking for np.nan values.
    islice1 : int
        First slice to be combined (from 1 to NAXIS3).
    islice2 : int or None
        Last slice to be combined (from 1 to NAXIS3).
    final_celestial_wcs : str, file-like, `pathlib.Path` or None
        FITS filename with final celestial WCS. If None,
        compute output celestial WCS for current list of 3D cubes.
    reproject_method : str
        Reprojection method. See 'REPROJECT_METHODS' above.
    combination_function : str
        Combination function. See 'COMBINATION_FUNCTIONS' above.
    output_celestial_2d_wcs : str, file-like, `pathlib.Path` or None
        Path to output 2D celestial WCS.
    verbose : bool
        If True, display additional information.

    Returns
    -------
    mosaic : numpy.ma.MaskedArray
        Masked array containing the combined image and the resulting
        mask.
    wcs_mosaic3d: astropy.wcs.WCS
        Instance representing the World Coordinate System of the
        combined image.
    """
    # protections
    if not isinstance(list_of_hdu3d_images, list):
        raise TypeError('list_of_hdu3d_images must be a list')
    if not isinstance(list_of_hdu3d_masks, list) and list_of_hdu3d_masks is not None:
        raise TypeError('list_of_hdu3d_masks must be a list or None')

    nimages = len(list_of_hdu3d_images)
    if verbose:
        print(f'Total number of images to be combined: {nimages}')

    if nimages < 1:
        raise ValueError('Number of images = 0')

    # check masks and generate them if not present
    if list_of_hdu3d_masks is None:
        list_of_hdu3d_masks = []
        for hdu3d_image in list_of_hdu3d_images:
            hdu3d_mask = fits.ImageHDU(data=np.isnan(hdu3d_image.data).astype(np.uint8))
            hdu3d_mask.header.extend(WCS(hdu3d_image.header).to_header(), update=True)
            list_of_hdu3d_masks.append(hdu3d_mask)
    else:
        if len(list_of_hdu3d_images) != len(list_of_hdu3d_masks):
            raise ValueError(f'Unexpected {len(list_of_hdu3d_images)=} != {len(list_of_hdu3d_masks)=}')

    # check compatibility between image and mask dimensions
    for hdu3d_image, hdu3d_mask in zip(list_of_hdu3d_images, list_of_hdu3d_masks):
        if hdu3d_image.shape != hdu3d_mask.shape:
            raise ValueError(f'Unexpected shape {hdu3d_image.shape=} != {hdu3d_mask.shape=}')

    # check the wavelength sampling is the same in all the images
    wcs1d_spectral_ini = None
    for i, hdu in enumerate(list_of_hdu3d_images):
        if i == 0:
            wcs1d_spectral_ini = WCS(hdu.header).spectral
        else:
            wcs1d_spectral = WCS(hdu.header).spectral
            if wcs1d_spectral.__str__() != wcs1d_spectral_ini.__str__():
                print(f'{list_of_hdu3d_images[i]}')
                raise ValueError('ERROR: spectral sampling is different!')

    # optimal 2D WCS (celestial part) for combined image
    if final_celestial_wcs is None:
        # compute final celestial WCS for the ensemble of 3D cubes
        list_of_inputs = []
        for hdu in list_of_hdu3d_images:
            header3d = hdu.header
            wcs2d = WCS(header3d).celestial
            list_of_inputs.append( ( (header3d['NAXIS2'], header3d['NAXIS1']), wcs2d) )
        wcs_mosaic2d, shape_mosaic2d = find_optimal_celestial_wcs(list_of_inputs)
    else:
        # make use of an external celestial WCS projection
        with fits.open(final_celestial_wcs) as hdul_mosaic2d:
            wcs_mosaic2d = WCS(hdul_mosaic2d[0].header)
            shape_mosaic2d = hdul_mosaic2d[0].header['NAXIS2'], hdul_mosaic2d[0].header['NAXIS1']
    if verbose:
        print(f'\n{wcs_mosaic2d=}')
        print(f'\n{shape_mosaic2d=}')
    if output_celestial_2d_wcs is not None:
        header_2d_wcs = wcs_mosaic2d.to_header()
        hdu = fits.PrimaryHDU(np.zeros(shape_mosaic2d, dtype=np.uint8), header=header_2d_wcs)
        hdu.writeto(output_celestial_2d_wcs, overwrite=True)

    # generate 3D mosaic
    if verbose:
        print(f'Reprojection method: {reproject_method}')
    wcs3d = WCS(list_of_hdu3d_images[0].header)
    naxis3_mosaic3d_ini = wcs3d.pixel_shape[-1]
    if islice2 is None:
        islice2 = naxis3_mosaic3d_ini
    naxis3_mosaic3d = islice2 - islice1 + 1
    naxis2_mosaic3d, naxis1_mosaic3d = shape_mosaic2d
    mosaic3d_cube_by_cube = np.zeros((naxis3_mosaic3d, naxis2_mosaic3d, naxis1_mosaic3d))
    footprint3d = np.zeros((naxis3_mosaic3d, naxis2_mosaic3d, naxis1_mosaic3d))
    if verbose:
        print(f'\nNAXIS1, NAXIS2, NAXIS3 of 3D mosaic: {naxis1_mosaic3d}, {naxis2_mosaic3d}, {naxis3_mosaic3d}')
        size1 = array_size_32bits(mosaic3d_cube_by_cube)
        size2 = array_size_8bits(footprint3d)
        print(f'Combined image will require {size1 + size2:.2f}')

    nimages = len(list_of_hdu3d_images)
    for i in range(nimages):
        # select [islice1:islice2] following FITS convention
        data_ini3d = list_of_hdu3d_images[i].data[(islice1-1):islice2]
        wcs_ini3d = WCS(list_of_hdu3d_images[i].header)
        wcs_ini2d = wcs_ini3d.celestial
        if reproject_method == 'interp':
            temp3d, footprint_temp3d = reproject_interp(
                (data_ini3d, wcs_ini2d),
                wcs_mosaic2d,
                shape_out=shape_mosaic2d
            )
        elif reproject_method == 'adaptive':
            temp3d, footprint_temp3d = reproject_adaptive(
                (data_ini3d, wcs_ini2d),
                wcs_mosaic2d,
                shape_out=shape_mosaic2d,
                conserve_flux=True,
                kernel='Gaussian'
            )
        elif reproject_method == 'exact':
            temp3d, footprint_temp3d = reproject_exact(
                (data_ini3d, wcs_ini2d),
                wcs_mosaic2d,
                shape_out=shape_mosaic2d
            )
        else:
            raise ValueError(f'Unexpected {reproject_method=}')
        valid_region = footprint_temp3d > 0
        mosaic3d_cube_by_cube[valid_region] += temp3d[valid_region]
        footprint3d += footprint_temp3d

    valid_region = footprint3d > 0
    mosaic3d_cube_by_cube[valid_region] /= footprint3d[valid_region]

    input("Solve ToDo here...!")
    # ToDo: ver cómo ajustar el WCS spectral para tener en cuenta que hemos usamos [islice1:islice2]
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

    return mosaic3d_cube_by_cube, footprint3d, wcs_mosaic3d


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list",
                        help="TXT file with list of 3D images to be combined", type=str)
    parser.add_argument('output_filename',
                        help='filename of output FITS image', type=str)
    parser.add_argument("--islice1", help="First slice (from 1 to NAXIS3), default 1",
                        type=int, default=1)
    parser.add_argument("--islice2", help="Last slice (from 1 to NAXIS3), default NAXIS3",
                        type=int, default=None)
    parser.add_argument("--final_celestial_wcs",
                        help="Final celestial WCS projection. Default None (compute for current 3D cube)",
                        type=str, default=None)
    parser.add_argument('--reproject_method',
                        help='Reprojection method (interp, adaptive, exact)',
                        type=str, choices=REPROJECT_METHODS, default='adaptive')
    parser.add_argument('--extname_image',
                        help='Extension name for image in input files. Default value: PRIMARY',
                        default='PRIMARY', type=str)
    parser.add_argument('--extname_mask',
                        help="Extension name for mask in input files. Default 'None': use np.nan in image",
                        default=None, type=str)
    parser.add_argument("--combination_function", help='Combination function. Default: mean',
                        type=str, default='mean', choices=COMBINATION_FUNCTIONS)
    parser.add_argument("--output_celestial_2d_wcs",
                        help="filename for output celestial 2D WCS",
                        type=str, default=None)
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

    input_list = args.input_list
    output_filename = args.output_filename
    islice1 = args.islice1
    islice2 = args.islice2
    if islice1 < 0:
        raise ValueError('islice1 must be >= 0')
    if islice2 is not None:
        if islice2 < islice1:
            raise ValueError('islice2 must be >= islice1')
    final_celestial_wcs = args.final_celestial_wcs
    reproject_method = args.reproject_method
    combination_function = args.combination_function
    output_celestial_2d_wcs = args.output_celestial_2d_wcs
    verbose = args.verbose

    # define extensions for image and mask
    extname_image = args.extname_image
    extname_mask = args.extname_mask
    if extname_mask is not None:
        if extname_mask.lower() == 'none':
            extname_mask = None

    # read input file with list of 3D images to be combined
    with open(input_list) as f:
        file_content = f.read().splitlines()

    # list of HDU
    list_of_hdu3d_images = []
    list_of_hdu3d_masks = []
    for fname in file_content:
        if len(fname) > 0:
            if fname[0] not in ['#']:
                print(f'\n* Reading: {fname}')
                with fits.open(fname) as hdul:
                    if extname_image not in hdul:
                        raise ValueError(f'Expected {extname_image} extension not found')
                    hdu3d_image = hdul[extname_image].copy()
                    if verbose:
                        print(f"{hdu3d_image.header['NAXIS1']=}")
                        print(f"{hdu3d_image.header['NAXIS2']=}")
                        print(f"{hdu3d_image.header['NAXIS3']=}")
                    if extname_mask is not None:
                        if extname_mask in hdul:
                            hdu3d_mask = hdul[extname_mask].copy()
                            if verbose:
                                print(f'image mask: {extname_mask}')
                            bitpix_mask = hdu3d_mask.header['BITPIX']
                            if bitpix_mask != 8:
                                raise ValueError(f'BITPIX (mask): {bitpix_mask} is not 8')
                            if hdu3d_image.data.shape != hdu3d_mask.data.shape:
                                raise ValueError(f'Shape of {extname_image} and {extname_mask} are different!')
                        else:
                            hdu3d_mask = None
                    else:
                        hdu3d_mask = None
                    # generate mask from np.nan when necessary
                    if hdu3d_mask is None:
                        hdu3d_mask = fits.ImageHDU(data=np.isnan(hdu3d_image.data).astype(np.uint8))
                        hdu3d_mask.header.extend(WCS(hdu3d_image.header).to_header(), update=True)
                        if verbose:
                            print(f'generating mask from np.nan values')
                    if verbose:
                        print(f'Number of masked pixels / total: {np.sum(hdu3d_mask.data)} / {hdu3d_mask.size}')
                    # store image and associated footprint
                    list_of_hdu3d_images.append(hdul[extname_image].copy())
                    list_of_hdu3d_masks.append(hdu3d_mask)

    # combine images
    mosaic3d_cube_by_cube, footprint3d, wcs_mosaic3d = generate_mosaic_of_3d_cubes(
        list_of_hdu3d_images=list_of_hdu3d_images,
        list_of_hdu3d_masks=list_of_hdu3d_masks,
        islice1=islice1,
        islice2=islice2,
        final_celestial_wcs=final_celestial_wcs,
        reproject_method=reproject_method,
        combination_function=combination_function,
        output_celestial_2d_wcs=output_celestial_2d_wcs,
        verbose=verbose
    )

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
