"""Combine several FITS images listed in a TXT file"""

import argparse
import datetime
import sys

from astropy.io import fits
import numpy as np

from numina.array.display.fileinfo import list_fileinfo_from_txt
from numina.processing.combine import combine
from .subsets_of_fileinfo_from_txt import subsets_of_fileinfo_from_txt


def compute_abba_result(list_of_fileinfo, outfile, extnum=0, noheader=False,
                        save_partial=False, debugplot=0):
    """Compute A-B-(B-A).

    Note that 'list_of_fileinfo' must contain 4 file names.

    Parameters
    ----------
    list_of_fileinfo : list of strings
        List of the for files corresponding to the ABBA observation.
    outfile : string
        Base name for the output FITS file name.
    extnum : int
        Extension number to be read (default value 0 = primary extension).
    noheader : bool
        If True, the header of the first image in the ABBA sequence is
        not copied in the resulting image(s).
    save_partial : bool
        If True, the partial (A-B) and -(B-A) images are also saved.
    debugplot : integer or None
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    """

    # check number of images
    if len(list_of_fileinfo) != 4:
        raise ValueError("Unexpected number of ABBA files: " +
                         str(len(list_of_fileinfo)))

    # avoid PyCharm warnings
    # (local variable might be referenced before assignment)
    naxis1 = 0
    naxis2 = 0
    image_header = None

    # check image dimensions
    for i in range(4):
        hdulist = fits.open(list_of_fileinfo[i].filename)
        if i == 0:
            image_header = hdulist[extnum].header
            naxis1 = image_header['naxis1']
            naxis2 = image_header['naxis2']
            hdulist.close()
        else:
            image_header_ = hdulist[extnum].header
            naxis1_ = image_header_['naxis1']
            naxis2_ = image_header_['naxis2']
            hdulist.close()
            if naxis1 != naxis1_ or naxis2 != naxis2_:
                print('>>> naxis1, naxis2..:', naxis1, naxis2)
                print('>>> naxis1_, naxis2_:', naxis1_, naxis2_)
                raise ValueError("Image dimensions do not agree!")

    # initialize outpuf array(s)
    if save_partial:
        result_ini = np.zeros((naxis2, naxis1), dtype=float)
        result_end = np.zeros((naxis2, naxis1), dtype=float)
    else:
        result_ini = None
        result_end = None
    result = np.zeros((naxis2, naxis1), dtype=float)

    # read the four images and compute arithmetic combination
    for i in range(4):
        if abs(debugplot) >= 10:
            print('Reading ' + list_of_fileinfo[i].filename + '...')
        hdulist = fits.open(list_of_fileinfo[i].filename)
        image2d = hdulist[extnum].data.astype(float)
        hdulist.close()
        if i == 0:
            if save_partial:
                result_ini += image2d
            result += image2d
        elif i == 1:
            if save_partial:
                result_ini -= image2d
            result -= image2d
        elif i == 2:
            if save_partial:
                result_end -= image2d
            result -= image2d
        elif i == 3:
            if save_partial:
                result_end += image2d
            result += image2d
        else:
            raise ValueError('Unexpected image number: ' + str(i))

    # save results
    if noheader:
        image_header = None
    if save_partial:
        if abs(debugplot) >= 10:
            print("==> Generating output file: " + outfile + "_sub1.fits...")
        hdu = fits.PrimaryHDU(result_ini.astype(float), image_header)
        hdu.writeto(outfile + '_sub1.fits', overwrite=True)
        if abs(debugplot) >= 10:
            print("==> Generating output file: " + outfile + "_sub2.fits...")
        hdu = fits.PrimaryHDU(result_end.astype(float), image_header)
        hdu.writeto(outfile + '_sub2.fits', overwrite=True)
    if abs(debugplot) >= 10:
        print("==> Generating output file: " + outfile + ".fits...")
    hdu = fits.PrimaryHDU(result.astype(float), image_header)
    hdu.writeto(outfile + '.fits', overwrite=True)


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
                        help='Combination method: sum (default), ' +
                             'mean, median, sigmaclip, abba, abba_partial',
                        default='sum',
                        type=str,
                        choices=['sum', 'mean', 'median', 'sigmaclip',
                                 'abba', 'abba_partial'])
    parser.add_argument('--method_kwargs',
                        help='Arguments for method sigmaclip; must be a'
                             'Python dictionary between double quotes, e.g.: '
                             '"{' + "'high': 2.5, 'low': 2.5}" +'")',
                        type=str)
    parser.add_argument('--extnum',
                        help='Extension number in input files (note that ' +
                             'first extension is 1 = default value)',
                        default=1, type=int)
    parser.add_argument('--noheader',
                        help='Do not include header of first image in ' +
                             'outpuf file(s)',
                        action='store_true')
    parser.add_argument('--noclobber',
                        help='Avoid overwriting existing file',
                        action='store_true')
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=12)",
                        default=12, type=int,
                        choices=[0, 1, 2, 10, 11, 12, 21, 22])
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args)

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

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
        naxis1 = np.zeros(number_of_files, dtype=int)
        naxis2 = np.zeros(number_of_files, dtype=int)

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

        image_header_first_frame = None  # avoid PyCharm warning
        image2d = None                   # avoid PyCharm warning

        if args.method in ['sum', 'mean']:
            image2d = np.zeros((naxis2[0], naxis1[0]), dtype=float)
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
        elif args.method == 'sigmaclip':
            list_data = []
            for i in range(number_of_files):
                infile = list_of_files[i]
                print("<--" + infile + " (image " + str(i + 1) + " of " +
                      str(number_of_files) + ')')
                hdulist = fits.open(infile)
                data = hdulist[extnum].data
                if i == 0:
                    image_header_first_frame = hdulist[extnum].header
                hdulist.close()
                list_data.append(data)
            if args.method_kwargs is None:
                image2d = combine.sigmaclip(list_data)[0]
            else:
                method_kwargs = eval(args.method_kwargs)
                image2d = combine.sigmaclip(list_data, **method_kwargs)[0]
        elif args.method == 'median':
            # declare temporary cube to store all the images
            image3d = np.zeros((number_of_files, naxis2[0], naxis1[0]),
                               dtype=float)
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
        elif args.method in ['abba', 'abba_partial']:
            save_partial = False
            if args.method == 'abba_partial':
                save_partial = True
            compute_abba_result(tmpdict['list_of_fileinfo'],
                                output_fits_filename,
                                extnum=extnum,
                                noheader=args.noheader,
                                save_partial=save_partial,
                                debugplot=args.debugplot)
        else:
            raise ValueError('Unexpected combination method!')

        # save results (except for ABBA combinations, which results
        # have already been saved)
        if args.method in ['sum', 'mean', 'median', 'sigmaclip']:
            if args.noheader:
                image_header = fits.Header()
            else:
                image_header = image_header_first_frame
            if abs(args.debugplot) >= 10:
                print("==> Generating output file: " + output_fits_filename +
                      "...")
            image_header.add_history("---")
            image_header.add_history("Image generated using:")
            image_header.add_history(" ".join(sys.argv))
            image_header.add_history("---")
            image_header.add_history('Combination time: {}'.format(
                datetime.datetime.utcnow().isoformat()))
            image_header.add_history(f"Contents of {args.input_list} file:")
            for i in range(number_of_files):
                image_header.add_history(list_of_files[i])
            image_header.add_history("---")
            hdu = fits.PrimaryHDU(image2d.astype(float), image_header)
            hdu.writeto(output_fits_filename, overwrite=(not args.noclobber))


if __name__ == "__main__":

    main()
