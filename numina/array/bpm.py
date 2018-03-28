#
# Copyright 2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

"""Fix points in an image given by a bad pixel mask """

from __future__ import division

import argparse
from astropy.io import fits
import numpy
import sys

from numina.array._bpm import _process_bpm_intl
from numina.tools.arg_file_is_new import arg_file_is_new


def process_bpm(method, arr, mask, hwin=2, wwin=2, fill=0):

    # FIXME: we are not considering variance extension
    # If arr is not in native byte order, the cython extension won't work
    if arr.dtype.byteorder != '=':
        narr = arr.byteswap().newbyteorder()
    else:
        narr = arr
    out = numpy.empty_like(narr, dtype='double')

    # Casting, Cython doesn't support well type bool
    cmask = numpy.where(mask > 0, 1, 0).astype('uint8')

    _process_bpm_intl(method, narr, cmask, out, hwin=hwin, wwin=wwin, fill=fill)
    return out


def process_bpm_median(arr, mask, hwin=2, wwin=2, fill=0):
    import numina.array.combine

    method = numina.array.combine.median_method()

    return process_bpm(method, arr, mask, hwin=hwin, wwin=wwin, fill=fill)


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()

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
                mask=image2d_bpm
            )

        # save output FITS file
        hdulist_image.writeto(args.outfile)


if __name__ == "__main__":

    main()