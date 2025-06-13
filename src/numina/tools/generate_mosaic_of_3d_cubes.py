#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Generate a 3D mosaic from individual 3D cubes.
"""

import argparse
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
from reproject import reproject_interp, reproject_adaptive, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs
import sys

from numina.array.array_size_32bits import array_size_8bits, array_size_32bits
from .ctext import ctext
from .resample_wave_3d_cube import resample_wave_3d_cube

REPROJECT_METHODS = ['interp', 'adaptive', 'exact']
COMBINATION_FUNCTIONS = ['mean', 'median', 'sum', 'std','sigmaclip_mean', 'sigmaclip_median', 'sigmaclip_stddev']


# TODO: hacer uso de combination_function
# TODO: hacer uso de list_of_hdu3d_masks (ver generate_mosaic_of_2d_images.py)
# TODO: metodo 'adaptive': ¿es necesario usar Gaussian kernel?
def generate_mosaic_of_3d_cubes(
        list_of_hdu3d_images,
        list_of_hdu3d_masks,
        crval3out,
        cdelt3out,
        naxis3out,
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
    crval3out : `astropy.units.Quantity` or None
        Minimum wavelength (in m) for the output image. If None,
        use the minimum wavelength of the input images.
    cdelt3out : `astropy.units.Quantity` or None
        Wavelength step (in m/pixel) for the output image. If None,
        the output image will use the minimum wavelength step of the input images.
    naxis3out : int or None
        Number of slices in the output image. If None, the output image
        will have the required number of slices to cover the full wavelength
        range of the input images.
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
    output_hdul : `astropy.io.fits.HDUList`
        Instance of HDUList with two HDU:
        - PRIMARY: mosaic image
        - FOOTPRINT: array with final footprint
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

    # compute crval3out, cdelt3out and naxis3out if not provided
    crval3out_ = None
    wavemax = None   # maximum wavelength (at the center of the last pixel)
    cdelt3out_ = None
    for i, hdu in enumerate(list_of_hdu3d_images):
        wcs1d_spectral = WCS(hdu.header).spectral
        wave = wcs1d_spectral.pixel_to_world(np.arange(hdu.data.shape[0]))
        if crval3out_ is None:
            crval3out_ = wave[0]
        else:
            if wave[0] < crval3out_:
                crval3out_ = wave[0]
        if wavemax is None:
            wavemax = wave[-1]
        else:
            if wave[-1] > wavemax:
                wavemax = wave[-1]
        if cdelt3out_ is None:
            cdelt3out_ = np.diff(wave).min()
        else:
            if np.diff(wave).min() < cdelt3out_:
                cdelt3out_ = np.diff(wave).min()
    if crval3out is None:
        crval3out = crval3out_
    if cdelt3out is None:
        cdelt3out = cdelt3out_ / u.pix
    if naxis3out is None:
        naxis3out = int(np.round((wavemax.value - crval3out.value) / cdelt3out.value)) + 1
    wavemax = crval3out + cdelt3out * (naxis3out - 1) * u.pix
    # define 1D WCS for the spectral axis of the combined mosaic
    header_spectral_mosaic = fits.Header()
    header_spectral_mosaic['NAXIS'] = 1
    header_spectral_mosaic['NAXIS1'] = naxis3out
    header_spectral_mosaic['CRPIX1'] = 1.0
    header_spectral_mosaic['CDELT1'] = cdelt3out.to(u.m/u.pix).value
    header_spectral_mosaic['CRVAL1'] = crval3out.to(u.m).value
    header_spectral_mosaic['CUNIT1'] = 'm'
    header_spectral_mosaic['CTYPE1'] = 'WAVE'
    wcs1d_spectral_mosaic = WCS(header_spectral_mosaic)
    if verbose:
        print(f'\n{crval3out=}\n{cdelt3out=}\n{naxis3out=}\n{wavemax=}\n')
        print(f'{wcs1d_spectral_mosaic=}')

    # optimal 2D WCS (celestial part) for combined mosaic
    if final_celestial_wcs is None:
        if verbose:
            print(f'\nCelestial scales:')
        # compute final celestial WCS for the ensemble of 3D cubes
        list_of_inputs = []
        for i, hdu in enumerate(list_of_hdu3d_images):
            header3d = hdu.header
            wcs2d = WCS(header3d).celestial
            scales = proj_plane_pixel_scales(wcs2d)
            if verbose:
                print(f'Image {i+1}: {scales[0]*3600:.3f} arcsec, {scales[1]*3600:.3f} arcsec')
            list_of_inputs.append( ( (header3d['NAXIS2'], header3d['NAXIS1']), wcs2d) )
        wcs_mosaic2d, shape_mosaic2d = find_optimal_celestial_wcs(list_of_inputs)
        scales = proj_plane_pixel_scales(wcs_mosaic2d)
        if verbose:
            print(f'Mosaic : {scales[0]*3600:.3f} arcsec, {scales[1]*3600:.3f} arcsec')
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

    # initialize arrays to store combination
    naxis3_mosaic3d = naxis3out
    naxis2_mosaic3d, naxis1_mosaic3d = shape_mosaic2d
    mosaic3d_cube_by_cube = np.zeros((naxis3_mosaic3d, naxis2_mosaic3d, naxis1_mosaic3d))
    footprint3d = np.zeros(shape=(naxis3_mosaic3d, naxis2_mosaic3d, naxis1_mosaic3d))
    if verbose:
        print(f'\nNAXIS1, NAXIS2, NAXIS3 of 3D mosaic: {naxis1_mosaic3d}, {naxis2_mosaic3d}, {naxis3_mosaic3d}')
        size1 = array_size_32bits(mosaic3d_cube_by_cube)
        size2 = array_size_8bits(footprint3d)
        print(f'Combined image will require {size1 + size2:.2f}')
        input('Press Enter to continue...')

    # generate 3D mosaic
    if verbose:
        print(f'Reprojection method: {reproject_method}')
    nimages = len(list_of_hdu3d_images)
    for i in range(nimages):
        single_hdu3d = resample_wave_3d_cube(list_of_hdu3d_images[i], crval3out, cdelt3out, naxis3out)
        data_ini3d = single_hdu3d.data
        wcs_ini3d = WCS(single_hdu3d.header)
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

    # generate resulting 3D WCS object
    header_spectral_single = WCS(list_of_hdu3d_images[0].header).spectral.to_header()  # to get keyword comments
    header3d_corrected = wcs_mosaic2d.to_header()
    header3d_corrected['WCSAXES'] = 3
    for item in ['CRPIX', 'CDELT', 'CUNIT', 'CTYPE', 'CRVAL']:
        # insert {item}3 after {item}2 to preserve the order in the header
        header3d_corrected.insert(
            f'{item}2',
            (f'{item}3', header_spectral_mosaic[f'{item}1'], header_spectral_single.comments[f'{item}1']),
            after=True)
    # fix slice in the spectral direction
    if header3d_corrected['CRPIX3'] != 1:
        raise ValueError(f"Expected CRPIX3=1 but got {header3d_corrected['CRPIX3']=}")

    # generate result
    hdu = fits.PrimaryHDU(mosaic3d_cube_by_cube.astype(np.float32))
    hdu.header.update(header3d_corrected)
    hdu_footprint = fits.ImageHDU(footprint3d.astype(np.uint8))
    hdu_footprint.header['EXTNAME'] = 'FOOTPRINT'
    hdu_footprint.header.update(header3d_corrected)
    output_hdul = fits.HDUList([hdu, hdu_footprint])

    return output_hdul


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(
        description="Generate a 3D mosaic from individual 3D cubes."
    )
    parser.add_argument("input_list",
                        help="TXT file with list of 3D images to be combined or single FITS file", type=str)
    parser.add_argument('output_filename',
                        help='filename of output FITS image', type=str)
    parser.add_argument("--crval3out", help="Minimum wavelength (in m) for the output image",
                        type=float, default=None)
    parser.add_argument("--cdelt3out", help="Wavelength step (in m/pixel) for the output image",
                        type=float, default=None)
    parser.add_argument("--naxis3out", help="Number of slices in the output image", type=int, default=None)
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
    crval3out = args.crval3out
    if crval3out is not None:
        crval3out = crval3out * u.m
    cdelt3out = args.cdelt3out
    if cdelt3out is not None:
        cdelt3out = cdelt3out * u.m / u.pix
    naxis3out = args.naxis3out
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

    # check if input file is a single FITS file or a list
    if input_list.endswith('.fits'):
        file_content = [input_list]
    else:
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
    #mosaic3d_cube_by_cube, footprint3d, wcs_mosaic3d = generate_mosaic_of_3d_cubes(
    output_hdul = generate_mosaic_of_3d_cubes(
        list_of_hdu3d_images=list_of_hdu3d_images,
        list_of_hdu3d_masks=list_of_hdu3d_masks,
        crval3out=crval3out,
        cdelt3out=cdelt3out,
        naxis3out=naxis3out,
        final_celestial_wcs=final_celestial_wcs,
        reproject_method=reproject_method,
        combination_function=combination_function,
        output_celestial_2d_wcs=output_celestial_2d_wcs,
        verbose=verbose
    )

    # save result
    if verbose:
        print(f'Saving: {output_filename}')
    output_hdul.writeto(output_filename, overwrite='yes')


if __name__ == "__main__":
    main()
