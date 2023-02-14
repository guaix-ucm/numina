"""Coadd several FITS images or subsets of FITS images listed in a TXT file"""

import argparse

from astropy.io import fits
import numpy as np

from numina.array.display.fileinfo import list_fileinfo_from_txt
from .subsets_of_fileinfo_from_txt import subsets_of_fileinfo_from_txt


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("input_list",
                        help="txt file with list of images to be coadded")
    # if output_fits_filename == "@" the script expects a line in
    # the file input_list of the form:
    # @ <number> <label>
    # where <number> is the number of images to be coadded and
    # <label> is the name of the output file
    parser.add_argument("output_fits_filename",
                        help="filename of output FITS image, or @ symbol")
    parser.add_argument("--factorcol",
                        help="Column number with multiplicative factors",
                        type=int)
    parser.add_argument("--offsetcol",
                        help="Column number with offsets:\n "
                             "final scan = old scan + offset (default=None)",
                        type=int)
    parser.add_argument("--average",
                        help="average result",
                        action="store_true")
    parser.add_argument("--add_header",
                        help="add header of first image: yes/no (default=yes)",
                        default="yes")
    parser.add_argument("--noclobber",
                        help="avoid overwriting existing file",
                        action='store_true')
    parser.add_argument("--out_nsum",
                        help="filename for output nsum FITS image "
                             "(default=None)",
                        type=argparse.FileType('w'),
                        default=None)
    parser.add_argument("--debugplot",
                        help="integer indicating plotting/debugging" +
                        " (default=10)",
                        type=int, default=10,
                        choices=[0, 1, 2, 10, 11, 12, 21, 22])
    args = parser.parse_args(args)

    debugplot = int(args.debugplot)

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
        list_of_infos = [tmp.fileinfo for tmp in tmpdict['list_of_fileinfo']]

        # number of files to be combined
        number_of_files = len(list_of_files)

        # if there are offsets between the images, check that all
        # the numbers are provided
        offsets = np.zeros(number_of_files, dtype=int)
        if args.offsetcol is not None:
            for i in range(number_of_files):
                offsets[i] = list_of_infos[i][args.offsetcol - 2]

        # if there are multiplicative factors, check that all
        # the numbers are provided
        multfactors = np.ones(number_of_files, dtype=float)
        if args.factorcol is not None:
            for i in range(number_of_files):
                multfactors[i] = list_of_infos[i][args.factorcol - 2]

        # declare auxiliary arrays to store image basic parameters
        naxis1 = np.zeros(number_of_files, dtype=int)
        naxis2 = np.zeros(number_of_files, dtype=int)

        # read basic parameters for all the images
        for i in range(number_of_files):
            infile = list_of_files[i]
            hdulist = fits.open(infile)
            image_header = hdulist[0].header
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

        # declare output arrays
        image2d = np.zeros((naxis2[0], naxis1[0]))
        image2d_nsum = np.zeros((naxis2[0], naxis1[0]), dtype=int)

        image_header_first_frame = None    # avoid PyCharm warning

        # add all the individual images
        for i in range(number_of_files):
            infile = list_of_files[i]
            if abs(debugplot) >= 10:
                print("<--" + infile +
                      " (image " + str(i + 1) + " of " +
                      str(number_of_files) + ")  offset: " +
                      str(offsets[i]) + "  factor: " +
                      str(multfactors[i])
                      )
            hdulist = fits.open(infile)
            data = hdulist[0].data * multfactors[i]
            if i == 0:
                image_header_first_frame = hdulist[0].header
            hdulist.close()
            # determine the image region to be coadded taking into
            # account the corresponding offset for the i-th image
            nsmin_in = 1 + offsets[i]
            nsmax_in = naxis2[0] + offsets[i]
            if nsmin_in < 1:
                nsmin_in = 1
            if nsmax_in > naxis2[0]:
                nsmax_in = naxis2[0]
            nsmin_out = 1 - offsets[i]
            nsmax_out = naxis2[0] - offsets[i]
            if nsmin_out < 1:
                nsmin_out = 1
            if nsmax_out > naxis2[0]:
                nsmax_out = naxis2[0]
            image2d[(nsmin_in-1):nsmax_in, ] += data[(nsmin_out-1):nsmax_out, ]
            image2d_nsum[(nsmin_in-1):nsmax_in, ] += 1

        # average result when requested
        if args.average:
            image2d /= image2d_nsum
            print("==> Computing average of " + str(number_of_files) +
                  " images")
        else:
            print("==> Coadding " + str(number_of_files) +
                  " images")

        # save results
        if abs(debugplot) >= 10:
            print("==> Generating output file: " +
                  output_fits_filename + "...")
        if args.add_header == "yes":
            hdu = fits.PrimaryHDU(image2d, image_header_first_frame)
        else:
            hdu = fits.PrimaryHDU(image2d)
        hdu.writeto(output_fits_filename, overwrite=(not args.noclobber))

        if args.out_nsum is not None:
            if abs(debugplot) >= 10:
                print("==> Generating output file: " +
                      args.out_nsum.name + "...")
            hdu = fits.PrimaryHDU(image2d_nsum.astype(np.int16))
            hdu.writeto(args.out_nsum, overwrite=(not args.noclobber))

        if abs(debugplot) >= 10:
            if args.average == "yes":
                print("average OK!")
            else:
                print("sum OK!")


if __name__ == "__main__":

    main()
