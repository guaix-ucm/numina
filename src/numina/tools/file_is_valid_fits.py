#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Check if a file is a valid FITS file."""

import gzip


def file_is_valid_fits(filename):
    """Return True if filename looks like a FITS file (also .fits.gz).

    Parameters
    ----------
    filename : str
        Path to the file to check.
    Returns
    -------
    bool
        True if the file is a valid FITS file, False otherwise.
    """

    try:
        with open(filename, "rb") as f:
            start = f.read(2)
        opener = gzip.open if start == b"\x1f\x8b" else open
        with opener(filename, "rb") as f:
            header_start = f.read(30)
    except OSError:
        return False
    return header_start.startswith(b"SIMPLE  =") and header_start[29:30] == b"T"
