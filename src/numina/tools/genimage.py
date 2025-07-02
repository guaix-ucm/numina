#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Generate a 2D or 3D FITS image with a constant value.
"""
import argparse
from astropy.io import fits
import numpy as np

from .add_script_info_to_fits_history import add_script_info_to_fits_history


def genimage(naxis1, naxis2, naxis3, constant, dtype):
    """Generate a 2D or 3D numpy array with a constant value.
    
    Parameters
    ----------
    naxis1 : int
        Number of pixels in the first axis.
    naxis2 : int
        Number of pixels in the second axis.
    naxis3 : int
        Number of pixels in the third axis.
    constant : float
        Constant value for the array.
    dtype : str
        Data type of the output array.
        
    Returns
    -------
    array :  numpy.ndarray
        The generated image array.
    """
    if naxis1 <= 0 or naxis2 <= 0:
        raise ValueError("naxis1 and naxis2 must be greater than 0 for a 2D image.")
    if naxis3 < 0:
        raise ValueError("naxis3 cannot be negative.")

    if naxis3 == 0:
        # Generate a 2D array
        array = np.full((naxis2, naxis1), constant, dtype=dtype)
    else:
        # Generate a 3D array
        array = np.full((naxis3, naxis2, naxis1), constant, dtype=dtype)

    return array


def main(args=None):
    """Main function to parse arguments and call genimage."""
    parser = argparse.ArgumentParser(description="Generate a 2D or 3D FITS image with a constant value.")
    parser.add_argument('output', type=str, help='Output FITS file name.')
    parser.add_argument('--naxis1', help='Number of pixels for NAXIS1', type=int, default=0)
    parser.add_argument('--naxis2', help='Number of pixels for NAXIS2', type=int, default=0)
    parser.add_argument('--naxis3', help='Number of pixels for NAXIS3', type=int, default=0)
    parser.add_argument('--constant', help='Constant value for the array', type=float, default=0.0)
    parser.add_argument("--dtype", 
                        help="Data type of the output image (default: float32)",
                        type=str, 
                        choices=['uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64',
                                 'float32', 'float64'],
                        default='float32')
    parser.add_argument('--overwrite', help='Overwrite the output file if it exists', action='store_true')

    args = parser.parse_args()

    array = genimage(
        args.naxis1, 
        args.naxis2, 
        args.naxis3, 
        args.constant,
        args.dtype
    )

    # save output file
    hdu = fits.PrimaryHDU(array) 
    add_script_info_to_fits_history(hdu.header, args)
    hdu.writeto(args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
