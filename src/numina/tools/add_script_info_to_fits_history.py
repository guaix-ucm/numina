#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Add script information to the FITS history of a file."""

from astropy.io import fits
from datetime import datetime
from io import BufferedReader
from pathlib import Path
import platform
import sys

def add_script_info_to_fits_history(header, args):
    """Add script information to the FITS header history.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        The FITS header to which the script information will be added.
    args : `argparse.Namespace`
        The arguments parsed from the command line.

    Returns
    -------
    None
    """
    if not isinstance(header, fits.Header):
        raise ValueError("The header must be an instance of astropy.io.fits.Header.")
    
    header.add_history('-' * 25)
    header.add_history(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header.add_history('-' * 25)
    header.add_history(f'Node: {platform.uname().node}')
    header.add_history(f'Python: {sys.executable}')
    header.add_history(f'$ {Path(sys.argv[0]).name}')
    for arg, value in vars(args).items():
        # filename read as argparse.FileType()
        if isinstance(value, BufferedReader):
            value = value.name if hasattr(value, 'name') else str(value)
        header.add_history(f'--{arg} {value}')
