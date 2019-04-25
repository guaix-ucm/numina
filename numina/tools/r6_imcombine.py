"""Combine several FITS images listed in a TXT file"""

from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import numpy as np

from numina.array.display.fileinfo import list_fileinfo_from_txt
from .subsets_of_fileinfo_from_txt import subsets_of_fileinfo_from_txt


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list",
                        help="TXT file with list of images to be coadded")
    # if output_fits_filename == "@" the script expects a line in
    # the file input_list of the form:
    # @ <number> <label>
    # where <number> is the number of images to be coadded and
    # <label> is the name of the output file
    parser.add_argument('output_fits_filename',
                        help='filename of output FITS image, or @ symbol')
    parser.add_argument('--method',
                        help='Combination method: sum (default), mean, median',
                        default='sum',
                        type=str,
                        choices=['sum', 'mean', 'median'])
    parser.add_argument('--extnum',
                        help='Extension number in input files (note that ' +
                             'first extension is 1 = default value)',
                        default=1, type=int)
    parser.add_argument('--add_header',
                        help='Add header of first image',
                        action='store_true')
    parser.add_argument('--noclobber',
                        help='Avoid overwriting existing file',
                        action='store_true')
    args = parser.parse_args(args)

    # first extension is number 1 for the user
    extnum = args.extnum - 1

    if args.output_fits_filename == "@":
        dict_of_fileinfo = subsets_of_fileinfo_from_txt(args.input_list)
    else:
        list_of_fileinfo = list_fileinfo_from_txt(args.input_list)
        dict_of_fileinfo = {}
        dict_of_fileinfo[0] = {
            'label': args.output_fits_filename,
            'list_of_fileinfo': list_of_fileinfo
        }

    number_of_output_files = len(dict_of_fileinfo)

    # main loop in number of output files
    for iout in range(number_of_output_files):
        # list of files to be combined
        tmpdict = dict_of_fileinfo[iout]
        output_fits_filename = tmpdict['label']
        list_of_files = [tmp.filename for tmp in tmpdict['list_of_fileinfo']]

        # number of files to be combined
        number_of_files = len(list_of_files)

        # declare auxiliary arrays to store image basic parameters
        naxis1 = np.zeros(number_of_files, dtype=np.int)
        naxis2 = np.zeros(number_of_files, dtype=np.int)

        # read basic parameters for all the images
        for i in range(number_of_files):
            infile = list_of_files[i]
            hdulist = fits.open(infile)
            image_header = hdulist[extnum].header
            hdulist.close()
            naxis1[i] = image_header['naxis1']
            naxis2[i] = image_header['naxis2']

        # check that NAXIS1 is the same for all the images
        naxis1_comp = np.repeat(naxis1[0], number_of_files)
        if not np.allclose(naxis1, naxis1_comp):
            raise ValueError("NAXIS1 values are different")

        # check that NAXIS2 is the same for all the images
        naxis2_comp = np.repeat(naxis2[0], number_of_files)
        if not np.allclose(naxis2, naxis2_comp):
            raise ValueError("NAXIS2 values are different")

        # declare output array
        image2d = np.zeros((naxis2[0], naxis1[0]))

        image_header_first_frame = None    # avoid PyCharm warning

        if args.method in ['sum', 'mean']:
            # add all the individual images
            for i in range(number_of_files):
                infile = list_of_files[i]
                print("<--" + infile + " (image " + str(i + 1) + " of " +
                      str(number_of_files) + ')')
                hdulist = fits.open(infile)
                data = hdulist[extnum].data
                if i == 0:
                    image_header_first_frame = hdulist[extnum].header
                hdulist.close()
                image2d += data
            # average result when required
            if args.method == 'mean':
                image2d /= number_of_files
        elif args.method == 'median':
            # declare temporary cube to store all the images
            image3d = np.zeros((number_of_files, naxis2[0], naxis1[0]))
            # read all the individual images
            for i in range(number_of_files):
                infile = list_of_files[i]
                print("<--" + infile + " (image " + str(i + 1) + " of " +
                      str(number_of_files) + ')')
                hdulist = fits.open(infile)
                data = hdulist[extnum].data
                if i == 0:
                    image_header_first_frame = hdulist[extnum].header
                hdulist.close()
                image3d[i, :, :] += data
            # compute median
            image2d = np.median(image3d, axis=0)
        else:
            raise ValueError('Unexpected combination method!')

        # save results
        print("==> Generating output file: " + output_fits_filename + "...")
        if args.add_header:
            hdu = fits.PrimaryHDU(image2d, image_header_first_frame)
        else:
            hdu = fits.PrimaryHDU(image2d)
        hdu.writeto(output_fits_filename, overwrite=(not args.noclobber))


if __name__ == "__main__":

    main()
