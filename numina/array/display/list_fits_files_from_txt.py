#
# Copyright 2015-2016 Universidad Complutense de Madrid
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

import glob
import os.path


def list_fits_files_from_txt(filename):
    """Returns a list of FITS files if filename is a TXT file.
    
    Parameters
    ----------
    filename : string
        Name of a FITS file or a TXT file containing a list of
        FITS files. In the first case the function returns the
        a list with a single elemente: the same filename.
        In the second case, the function returns a list of the
        FITS file names. Lines starting by a hash symbol in the
        TXT file are ignored.
    
    Returns
    -------
    list_of_fits_files : list of strings
        List of file FITS file names.
        
    """
    
    # check for input file
    if not os.path.isfile(filename):
        # check for wildcards
        list_fits_files = glob.glob(filename)
        if len(list_fits_files) == 0:
            raise ValueError("File " + filename + " not found!")
        else:
            return list_fits_files

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

    return list_fits_files

