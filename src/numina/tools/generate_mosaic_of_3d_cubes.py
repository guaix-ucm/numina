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
from datetime import datetime
import numpy as np
from pathlib import Path
from reproject import reproject_interp, reproject_adaptive, reproject_exact
from reproject.mosaicking import find_optimal_celestial_wcs
import sys

from numina.array.array_size_32bits import array_size_8bits, array_size_32bits
from numina.instrument.simulation.ifu.define_3d_wcs import header3d_after_merging_wcs2d_celestial_and_wcs1d_spectral
from numina.instrument.simulation.ifu.define_3d_wcs import wcs_to_header_using_cd_keywords

from .add_script_info_to_fits_history import add_script_info_to_fits_history
from .ctext import ctext
from .resample_wave_3d_cube import resample_wave_3d_cube

REPROJECT_METHODS = ['interp', 'adaptive', 'exact']


def generate_mosaic_of_3d_cubes(
        list_of_fits_files,
        extname_image,
        crval3out,
        cdelt3out,
        naxis3out,
        desired_celestial_2d_wcs,
        reproject_method,
        parallel,
        output_celestial_2d_wcs,
        footprint=False,
        verbose=False
):
    """Combine 3D cubes using their WCS information.

    Parameters
    ----------
    list_of_fits_files : list of str
        List of paths to FITS files containing the 3D cubes to be combined.
    extname_image : str
        Extension name for the image in input files.
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
    desired_celestial_2d_wcs : str, file-like, `pathlib.Path` or None
        FITS filename with desired 2D celestial WCS. If None,
        compute output 2D celestial WCS for current list of 3D cubes.
    reproject_method : str
        Reprojection method. See 'REPROJECT_METHODS' above.
    parallel : bool
        If True, use parallel processing for reprojection.
    output_celestial_2d_wcs : str, file-like, `pathlib.Path` or None
        Path to output 2D celestial WCS.
    footprint : bool
        If True, generate a FOOTPRINT extension with the final footprint.
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
    if not isinstance(list_of_fits_files, list):
        raise TypeError('list_of_fits_files must be a list')
    if crval3out is not None and not isinstance(crval3out, u.Quantity):
        raise TypeError('crval3out must be an astropy Quantity')
    if cdelt3out is not None and not isinstance(cdelt3out, u.Quantity):
        raise TypeError('cdelt3out must be an astropy Quantity')
    if naxis3out is not None and not isinstance(naxis3out, int):
        raise TypeError('naxis3out must be an integer')

    nimages = len(list_of_fits_files)
    if verbose:
        print(f'Total number of images to be combined: {nimages}')

    if nimages < 1:
        raise ValueError('Number of images = 0')

    # compute crval3out, cdelt3out and naxis3out if not provided
    crval3out_ = None
    wavemax = None   # maximum wavelength (at the center of the last pixel)
    cdelt3out_ = None
    for fname in list_of_fits_files:
        with fits.open(fname) as hdul:
            if extname_image not in hdul:
                    raise ValueError(f'Expected {extname_image} extension not found')
            hdu = hdul[extname_image]
            if verbose:
                print(f'{hdu.header["NAXIS1"]=}, {hdu.header["NAXIS2"]=}, {hdu.header["NAXIS3"]=}')
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
        print(f'\n{crval3out=}\n{cdelt3out=}\n{naxis3out=}\n{wavemax  =}\n')
        print(f'{wcs1d_spectral_mosaic=}')

    # optimal 2D WCS (celestial part) for combined mosaic
    if desired_celestial_2d_wcs is None:
        if verbose:
            print(f'\nCelestial scales:')
        # compute final celestial WCS for the ensemble of 3D cubes
        list_of_inputs = []
        for i, fname in enumerate(list_of_fits_files):
            with fits.open(fname) as hdul:
                hdu = hdul[extname_image]
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
        if verbose:
            print(ctext(f'\nUsing external celestial WCS: {desired_celestial_2d_wcs}', fg='green'))
        with fits.open(desired_celestial_2d_wcs) as hdul_mosaic2d:
            wcs_mosaic2d = WCS(hdul_mosaic2d[0].header)
            shape_mosaic2d = hdul_mosaic2d[0].header['NAXIS2'], hdul_mosaic2d[0].header['NAXIS1']
    if verbose:
        print(f'\n{wcs_mosaic2d=}')
        print(f'\n{shape_mosaic2d=}')
    if output_celestial_2d_wcs is not None:
        header_2d_wcs = wcs_to_header_using_cd_keywords(wcs_mosaic2d)
        hdu = fits.PrimaryHDU(np.zeros(shape_mosaic2d, dtype=np.uint8), header=header_2d_wcs)
        if verbose:
            print(f'Saving resulting celestial 2D WCS to: {output_celestial_2d_wcs}')
        hdu.writeto(output_celestial_2d_wcs, overwrite=True)

    # initialize arrays to store combination
    naxis3_mosaic3d = naxis3out
    naxis2_mosaic3d, naxis1_mosaic3d = shape_mosaic2d
    mosaic3d_cube_by_cube = np.zeros((naxis3_mosaic3d, naxis2_mosaic3d, naxis1_mosaic3d))
    footprint3d = np.zeros(shape=(naxis3_mosaic3d, naxis2_mosaic3d, naxis1_mosaic3d))
    if verbose:
        print(f'\nNAXIS1, NAXIS2, NAXIS3 of 3D mosaic: {naxis1_mosaic3d}, {naxis2_mosaic3d}, {naxis3_mosaic3d}')
        size_output = array_size_32bits(mosaic3d_cube_by_cube)
        if footprint:
            size_output += array_size_8bits(footprint3d)
        print(ctext(f'Combined image will require {size_output:.2f}', fg='red'))

    # generate 3D mosaic
    for fname in list_of_fits_files:
        time_ini = datetime.now()
        if verbose:
            print(f'\n* Working with: {fname}')
        with fits.open(fname) as hdul:
            hdu = hdul[extname_image]
            single_hdu3d = resample_wave_3d_cube(
                hdu3d_image=hdu,
                crval3out=crval3out,
                cdelt3out=cdelt3out,
                naxis3out=naxis3out,
                verbose=verbose
            )
        data_ini3d = single_hdu3d.data
        wcs_ini3d = WCS(single_hdu3d.header)
        wcs_ini2d = wcs_ini3d.celestial
        if verbose:
            print(f'Celestial WCS reprojection method: {reproject_method}')
        if reproject_method == 'interp':
            temp3d, footprint_temp3d = reproject_interp(
                (data_ini3d, wcs_ini2d),
                wcs_mosaic2d,
                shape_out=shape_mosaic2d,
                parallel=parallel
            )
        elif reproject_method == 'adaptive':
            temp3d, footprint_temp3d = reproject_adaptive(
                (data_ini3d, wcs_ini2d),
                wcs_mosaic2d,
                shape_out=shape_mosaic2d,
                conserve_flux=True,
                kernel='Gaussian',
                parallel=parallel
            )
        elif reproject_method == 'exact':
            temp3d, footprint_temp3d = reproject_exact(
                (data_ini3d, wcs_ini2d),
                wcs_mosaic2d,
                shape_out=shape_mosaic2d,
                parallel=parallel
            )
        else:
            raise ValueError(f'Unexpected {reproject_method=}')
        valid_region = (footprint_temp3d > 0)
        mosaic3d_cube_by_cube[valid_region] += temp3d[valid_region]
        footprint3d += footprint_temp3d
        time_end = datetime.now()
        if verbose:
            print(f'Processing time for {fname}: {time_end - time_ini}')

    valid_region = (footprint3d > 0)
    mosaic3d_cube_by_cube[valid_region] /= footprint3d[valid_region]
    invalid_region = (footprint3d == 0)
    mosaic3d_cube_by_cube[invalid_region] = np.nan  # set invalid pixels to NaN
 
    # generate result
    hdu = fits.PrimaryHDU(mosaic3d_cube_by_cube.astype(np.float32))
    header3d_corrected = header3d_after_merging_wcs2d_celestial_and_wcs1d_spectral(
        wcs2d_celestial=wcs_mosaic2d,
        wcs1d_spectral=wcs1d_spectral_mosaic
    )
    hdu.header.update(header3d_corrected)
    if footprint:
        hdu_footprint = fits.ImageHDU(footprint3d.astype(np.uint8))
        hdu_footprint.header['EXTNAME'] = 'FOOTPRINT'
        hdu_footprint.header.update(header3d_corrected)
        output_hdul = fits.HDUList([hdu, hdu_footprint])
    else:
        output_hdul = fits.HDUList([hdu])

    return output_hdul


def main(args=None):

    time_ini = datetime.now()
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
    parser.add_argument("--desired_celestial_2d_wcs",
                        help="Desired 2D celestial WCS projection. Default None (compute for current 3D cube combination)",
                        type=str, default=None)
    parser.add_argument('--reproject_method',
                        help='Reprojection method (interp, adaptive, exact)',
                        type=str, choices=REPROJECT_METHODS, default='adaptive')
    parser.add_argument('--parallel',
                        help='Use parallel processing for reprojection',
                        action='store_true')
    parser.add_argument('--extname_image',
                        help='Extension name for image in input files. Default value: PRIMARY',
                        default='PRIMARY', type=str)
    parser.add_argument("--output_celestial_2d_wcs",
                        help="filename for output 2D celestial WCS",
                        type=str, default=None)
    parser.add_argument("--footprint",
                        help="Generate a FOOTPRINT extension with the final footprint",
                        action="store_true")
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
    desired_celestial_2d_wcs = args.desired_celestial_2d_wcs
    reproject_method = args.reproject_method
    parallel = args.parallel
    if reproject_method not in REPROJECT_METHODS:
        raise ValueError(f'Unexpected reproject_method: {reproject_method}. Expected one of {REPROJECT_METHODS}')
    output_celestial_2d_wcs = args.output_celestial_2d_wcs
    footprint = args.footprint
    verbose = args.verbose

    # define extensions for image and mask
    extname_image = args.extname_image

    # check if input file is a single FITS file or a list
    if input_list.endswith('.fits'):
        file_content = [input_list]
    else:
        with open(input_list) as f:
            file_content = f.read().splitlines()

    list_of_fits_files = []
    for fname in file_content:
        if len(fname) > 0:
            if fname[0] not in ['#']:
                if not Path(fname).is_file():
                    raise ValueError(f'File {fname} does not exist or is not a valid file.')
                list_of_fits_files.append(fname)

    if len(list_of_fits_files) < 1:
        raise ValueError(f'No valid FITS files found in {input_list}. Please check the file content.')
    
    # combine images
    output_hdul = generate_mosaic_of_3d_cubes(
        list_of_fits_files=list_of_fits_files,
        extname_image=extname_image,
        crval3out=crval3out,
        cdelt3out=cdelt3out,
        naxis3out=naxis3out,
        desired_celestial_2d_wcs=desired_celestial_2d_wcs,
        reproject_method=reproject_method,
        parallel=parallel,
        output_celestial_2d_wcs=output_celestial_2d_wcs,
        footprint=footprint,
        verbose=verbose
    )

    # save result
    add_script_info_to_fits_history(output_hdul[0].header, args)
    if verbose:
        print(f'Saving: {output_filename}')
    output_hdul.writeto(output_filename, overwrite='yes')

    time_end = datetime.now()
    if verbose:
        print(f'\nTotal time: {time_end - time_ini}')
        print('Done!')

if __name__ == "__main__":
    main()
