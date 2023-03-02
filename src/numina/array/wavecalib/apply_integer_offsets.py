#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Apply integer (pixel) offsets to an image"""

import argparse
import astropy.io.fits as fits
import numpy as np
import sys


def apply_integer_offsets(image2d, offx, offy):
    """Apply global (integer) offsets to image.

    Parameters
    ----------
    image2d : numpy array
        Input image
    offx : int
        Offset in the X direction (must be integer).
    offy : int
        Offset in the Y direction (must be integer).

    Returns
    -------
    image2d_shifted : numpy array
        Shifted image

    """

    # protections
    if type(offx) != int or type(offy) != int:
        raise ValueError('Invalid non-integer offsets')

    # image dimensions
    naxis2, naxis1 = image2d.shape

    # initialize output image
    image2d_shifted = np.zeros((naxis2, naxis1))

    # handle negative and positive shifts accordingly
    non = lambda s: s if s < 0 else None
    mom = lambda s: max(0,s)

    # shift image
    image2d_shifted[mom(offy):non(offy), mom(offx):non(offx)] = \
        image2d[mom(-offy):non(-offy), mom(-offx):non(-offx)]

    # return shifted image
    return image2d_shifted


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser()
    # required parameters
    parser.add_argument("infile",
                        help="Input FITS image",
                        type=argparse.FileType('rb'))
    parser.add_argument("outfile",
                        help="Output FITS image",
                        type=argparse.FileType('wb'))
    # optional arguments
    parser.add_argument("--offx",
                        help="Offset in the X direction (integer)",
                        default=0, type=int)
    parser.add_argument("--offy",
                        help="Offset in the Y direction (integer)",
                        default=0, type=int)
    parser.add_argument("--extension",
                        help="Extension number in FITS image (0=first "
                             "extension)",
                        default=0, type=int)
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")
    args = parser.parse_args(args=args)

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    # ---

    # read input FITS file
    with fits.open(args.infile) as hdulist:
        image2d = hdulist[args.extension].data
        hdulist[args.extension].data = apply_integer_offsets(
            image2d=image2d,
            offx=args.offx,
            offy=args.offy
        )

    # save output FITS file
    hdulist.writeto(args.outfile, overwrite=True)


if __name__ == "__main__":

    main()