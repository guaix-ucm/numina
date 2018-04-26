#
# Copyright 2015-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from __future__ import division
from __future__ import print_function

import argparse
import glob
import os.path


class FileInfo(object):
    """Auxiliary class to store filename and associated information.

    Parameters
    ----------
    filename : string
        File name.
    fileinfo : list of strings or None
        Associated file information.

    Attributes
    ----------
    The same as the parameters.

    """

    def __init__(self, filename, fileinfo=None):
        self.filename = filename
        self.fileinfo = fileinfo

    def __str__(self):
        """Printable representation of a FileInfo instance."""

        output = "<FileInfo instance>\n" + \
                 "- filename: " + self.filename + "\n" + \
                 "- fileinfo: " + str(self.fileinfo)

        return output


def list_fileinfo_from_txt(filename):
    """Returns a list of FileInfo instances if filename is a TXT file.
    
    Parameters
    ----------
    filename : string
        Name of a file (wildcards are acceptable) or a TXT file
        containing a list of files. Empty Lines, and lines starting by
        a hash or a at symbol in the TXT file are ignored.
    
    Returns
    -------
    output : list of FileInfo instances
        List of FileInfo instances containing the name of the files
        and additional file information.
        
    """
    
    # check for input file
    if not os.path.isfile(filename):
        # check for wildcards
        list_fits_files = glob.glob(filename)
        if len(list_fits_files) == 0:
            raise ValueError("File " + filename + " not found!")
        else:
            output = [FileInfo(tmpfile) for tmpfile in list_fits_files]
            return output

    # if input file is a txt file, assume it is a list of FITS files
    if filename[-4:] == ".txt":
        with open(filename) as f:
            file_content = f.read().splitlines()
        output = []
        for line in file_content:
            if len(line) > 0:
                if line[0] not in ['#', '@']:
                    tmplist = line.split()
                    tmpfile = tmplist[0]
                    if len(tmplist) > 1:
                        tmpinfo = tmplist[1:]
                    else:
                        tmpinfo = None
                    if not os.path.isfile(tmpfile):
                        raise ValueError("File " + tmpfile + " not found!")
                    output.append(FileInfo(tmpfile, tmpinfo))
    else:
        output = [FileInfo(filename)]

    return output


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(prog='fileinfo')
    parser.add_argument("txt_file",
                        help="txt file with list files")
    args = parser.parse_args(args)

    # execute function
    list_fileinfo = list_fileinfo_from_txt(args.txt_file)
    for item in list_fileinfo:
        print(item)

if __name__ == "__main__":

    main()
