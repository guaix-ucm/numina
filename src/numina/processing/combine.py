#
# Copyright 2016-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Combination routines"""

import datetime
import logging
import uuid
import contextlib

from astropy.io import fits

from numina.array import combine
from numina.datamodel import get_imgid


def basic_processing_with_combination(
        rinput, reduction_flow,
        method=combine.mean, method_kwargs=None,
        errors=True, prolog=None):

    return basic_processing_with_combination_frames(
        rinput.obresult.frames, reduction_flow,
        method=method, method_kwargs=method_kwargs,
        errors=errors, prolog=prolog
    )


def basic_processing_with_combination_frames(
        frames, reduction_flow,
        method=combine.mean, method_kwargs=None,
        errors=True, prolog=None):

    result = combine_frames(
        frames, method=method, method_kwargs=method_kwargs,
        errors=errors, prolog=prolog
    )

    hdulist = reduction_flow(result)

    return hdulist


def combine_frames(frames, method=combine.mean, method_kwargs=None, errors=True, prolog=None):
    """

    Parameters
    ----------
    frames
    method
    method_kwargs
    errors
    prolog

    Returns
    -------

    """

    with contextlib.ExitStack() as stack:
        hduls = [stack.enter_context(dframe.open()) for dframe in frames]
        result = combine_imgs(
            hduls, method=method, method_kwargs=method_kwargs,
            errors=errors, prolog=prolog
        )

    return result


def combine_imgs(hduls, method=combine.mean, method_kwargs=None, errors=True, prolog=None):
    """

    Parameters
    ----------
    hduls
    method
    method_kwargs
    errors
    prolog

    Returns
    -------

    """

    _logger = logging.getLogger(__name__)

    cnum = len(hduls)
    if cnum == 0:
        raise ValueError('number of HDUList == 0')

    first_image = hduls[0]
    base_header = first_image[0].header.copy()
    last_header = hduls[-1][0].header.copy()

    method_kwargs = method_kwargs or {}
    if 'dtype' not in method_kwargs:
        method_kwargs['dtype'] = 'float32'

    _logger.info(f"stacking {cnum:d} images using '{method.__name__}'")
    combined_data = method([d[0].data for d in hduls], **method_kwargs)

    hdu = fits.PrimaryHDU(combined_data[0], header=base_header)
    _logger.debug('update result header')
    if prolog:
        _logger.debug('write prolog')
        hdu.header['history'] = prolog
    hdu.header['history'] = f"Combined {cnum:d} images using '{method.__name__}'"
    hdu.header['history'] = f'Combination time {datetime.datetime.utcnow().isoformat()}'

    for img in hduls:
        hdu.header['history'] = f"Image {get_imgid(img)}"

    prevnum = base_header.get('NUM-NCOM', 1)
    hdu.header['NUM-NCOM'] = prevnum * cnum
    hdu.header['UUID'] = str(uuid.uuid1())

    # Copy extensions and then append 'variance' and 'map'
    result = fits.HDUList([hdu])
    for hdu in first_image[1:]:
        result.append(hdu.copy())

    # Headers of last image, this is an EMIRISM
    if 'TSUTC2' in hdu.header:
        hdu.header['TSUTC2'] = last_header['TSUTC2']
    # Append error extensions
    if errors:
        varhdu = fits.ImageHDU(combined_data[1], name='VARIANCE')
        result.append(varhdu)
        num = fits.ImageHDU(combined_data[2].astype('int16'), name='MAP')
        result.append(num)

    return result


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(prog='combine')
    parser.add_argument('-o', '--output', default='combined.fits')
    parser.add_argument('-e', '--errors', default=False, action='store_true')
    parser.add_argument('--method', default='mean', choices=['mean', 'median'])
    parser.add_argument('image', nargs='+')
    args = parser.parse_args(args)

    if args.method == 'mean':
        method = combine.mean
    elif args.method == 'median':
        method = combine.median
    else:
        raise ValueError(f'wrong method {args.method}')

    errors = args.errors
    with contextlib.ExitStack() as stack:
        hduls = [stack.enter_context(fits.open(fname)) for fname in args.image]
        result = combine_imgs(hduls, method=method, errors=errors, prolog=None)

    result.writeto(args.output, overwrite=True)


if __name__ == '__main__':
    main()
