#
# Copyright 2015 Universidad Complutense de Madrid
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

from __future__ import division
from __future__ import print_function

import argparse
from astropy.io import fits
import os.path

from .pause_debugplot import pause_debugplot
from .ximshow import ximshow


if __name__ == "__main__":

    # parse command-line options
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        help="FITS file or txt file with list of FITS files")
    parser.add_argument("--z1z2",
                        help="tuple z1,z2")
    parser.add_argument("--bbox",
                        help="bounding box tuple nc1,nc2,ns1,ns2")
    parser.add_argument("--pdffile",
                        help="ouput PDF file name")
    parser.add_argument("--debugplot",
                        help="Integer indicating plotting/debugging" +
                             " (default=12)",
                        default=12)
    args = parser.parse_args()

    # read z1, z2
    if args.z1z2 is None:
        z1z2 = None
    else:
        tmp_str = args.z1z2.split(",")
        z1z2 = float(tmp_str[0]), float(tmp_str[1])

    # read pdffile
    pdffile = args.pdffile
    if pdffile is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(pdffile)
    else:
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        pdf = None

    # read debugplot value
    debugplot = int(args.debugplot)

    # check for input file
    filename = args.filename
    if not os.path.isfile(filename):
        raise ValueError("File " + filename + " not found!")

    # if input file is a txt file, assume it is a list of FITS files
    if filename[-4:] == ".txt":
        with open(filename) as f:
            file_content = f.read().splitlines()
        list_fits_files = []
        for line in file_content:
            if len(line) > 0:
                if line[0] != '#':
                    tmpfile = line.split()[0]
                    if not os.path.isfile(tmpfile):
                        raise ValueError("File " + tmpfile + " not found!")
                    list_fits_files.append(tmpfile)
    else:
        list_fits_files = [filename]

    for ifile, file in enumerate(list_fits_files):
        # read input FITS file
        hdulist = fits.open(file)
        image_header = hdulist[0].header
        image2d = hdulist[0].data
        hdulist.close()

        naxis1 = image_header['naxis1']
        naxis2 = image_header['naxis2']
        grism = image_header['grism']
        spfilter = image_header['filter']
        rotang = image_header['rotang']

        if image2d.shape != (naxis2, naxis1):
            raise ValueError("Unexpected error with NAXIS1, NAXIS2")
        else:
            print('>>> File..:', file)
            print('>>> NAXIS1:', naxis1)
            print('>>> NAXIS2:', naxis2)

        # read bounding box
        if args.bbox is None:
            nc1 = 1
            nc2 = naxis1
            ns1 = 1
            ns2 = naxis2
            bbox = (1, naxis1, 1, naxis2)
        else:
            tmp_bbox = args.bbox.split(",")
            nc1 = int(tmp_bbox[0])
            nc2 = int(tmp_bbox[1])
            ns1 = int(tmp_bbox[2])
            ns2 = int(tmp_bbox[3])
            if nc1 < 1:
                nc1 = 1
            if nc2 > naxis1:
                nc2 = naxis1
            if ns1 < 1:
                ns1 = 1
            if ns2 > naxis2:
                ns2 = naxis2

        # display full image
        ax = ximshow(image2d=image2d[ns1-1:ns2, nc1-1:nc2], show=False,
                     title=file + "\ngrism=" + grism +
                           ", filter=" + spfilter +
                           ", rotang=" + str(round(rotang, 2)),
                     z1z2=z1z2,
                     image_bbox=(nc1, nc2, ns1, ns2), debugplot=debugplot)
        if pdf is not None:
            pdf.savefig()
        else:
            plt.show(block=False)
            plt.pause(0.001)
            pause_debugplot(debugplot)

    if pdf is not None:
        pdf.close()

    if len(list_fits_files) > 1:
        pause_debugplot(12, optional_prompt="Press RETURN to STOP")
