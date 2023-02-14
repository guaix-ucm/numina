#
# Copyright 2016-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Fix points in an image given by a bad pixel mask """

import argparse
import sys

from astropy.io import fits
import numpy

from numina.array._bpm import _process_bpm_intl
from numina.tools.arg_file_is_new import arg_file_is_new


def process_bpm(method, arr, mask, hwin=2, wwin=2, fill=0, reuse_values=False, subs=False):
    """

    Parameters
    ----------
    method
    arr
    mask
    hwin
    wwin
    fill
    reuse_values
    subs

    Returns
    -------

    """
    # FIXME: we are not considering variance extension
    # If arr is not in native byte order, the cython extension won't work
    if arr.dtype.byteorder != '=':
        narr = arr.byteswap().newbyteorder()
    else:
        narr = arr
    out = numpy.empty_like(narr, dtype='double')

    # Casting, Cython doesn't support well type bool
    cmask = numpy.where(mask > 0, 1, 0).astype('uint8')

    _, proc = _process_bpm_intl(method, narr, cmask, out, hwin=hwin, wwin=wwin, fill=fill, reuse_values=reuse_values)
    if subs:
        return out, proc
    else:
        return out


def process_bpm_median(arr, mask, hwin=2, wwin=2, fill=0, reuse_values=False, subs=False):
    """

    Parameters
    ----------
    arr
    mask
    hwin
    wwin
    fill
    reuse_values
    subs

    Returns
    -------

    """
    import numina.array._combine

    method = numina.array._combine.median_method()

    return process_bpm(method, arr, mask, hwin=hwin, wwin=wwin, fill=fill, reuse_values=reuse_values, subs=subs)


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(
        description='description: apply bad-pixel-mask to image'
    )

    # positional arguments
    parser.add_argument("fitsfile",
                        help="Input FITS file name",
                        type=argparse.FileType('rb'))
    parser.add_argument("--bpm", required=True,
                        help="Bad pixel mask",
                        type=argparse.FileType('rb'))
    parser.add_argument("--outfile", required=True,
                        help="Output FITS file name",
                        type=lambda x: arg_file_is_new(parser, x, mode='wb'))

    # optional arguments
    parser.add_argument("--extnum",
                        help="Extension number in input FITS image "
                             "(default=0)",
                        default=0, type=int)
    parser.add_argument("--extnum_bpm",
                        help="Extension number in bad pixel mask image "
                             "(default=0)",
                        default=0, type=int)
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args=args)

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    # read input FITS file
    with fits.open(args.fitsfile, mode='readonly') as hdulist_image:
        image2d = hdulist_image[args.extnum].data

        naxis2, naxis1 = image2d.shape

        # read bad pixel mask
        # (mask > 0: masked pixels; mask = 0: unmasked pixel)
        with fits.open(args.bpm) as hdulist_bpm:
            image2d_bpm = hdulist_bpm[args.extnum_bpm].data
            if image2d_bpm.shape != (naxis2, naxis1):
                raise ValueError("NAXIS1, NAXIS2 of FITS image and mask do "
                                 "not match")

            # apply bad pixel mask
            hdulist_image[args.extnum].data = process_bpm_median(
                arr=image2d,
                mask=image2d_bpm,
                reuse_values=True
            )

        # save output FITS file
        hdulist_image.writeto(args.outfile)


if __name__ == "__main__":

    main()
