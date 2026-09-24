#
# Copyright 2025-2026 Universidad Complutense de Madrid
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


def _map_dest_to_option(parser):
    """Return a dict mapping each argparse dest to its option string.

    Long options (--xxx) are preferred. Positional arguments are
    mapped to None.
    """
    mapping = {}
    for action in parser._actions:
        long_options = [s for s in action.option_strings if s.startswith("--")]
        if long_options:
            mapping[action.dest] = long_options[0]
        elif action.option_strings:
            mapping[action.dest] = action.option_strings[0]
        else:
            mapping[action.dest] = None  # positional argument
    return mapping


def add_script_info_to_fits_history(header, args, parser=None, title=None):
    """Add script information to the FITS header history.

    Note that this function does not save the FITS file;
    it only modifies the header in memory.

    The arguments are added to the history in the order they were
    defined in the parser.

    The argument names containing underscores are converted to hyphens
    when they are used to match the command line options.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        The FITS header to which the script information will be added.
    args : `argparse.Namespace`
        The arguments parsed from the command line.
    parser : `argparse.ArgumentParser`, optional
        The parser used to obtaing `args`. If provided, the original
        option names (e.g. --output-dir) are used instead of the
        attribute names (e.g. output_dir).
    title : str, optional
        The title for the history entry.

    Returns
    -------
    None
    """
    if not isinstance(header, fits.Header):
        raise ValueError("The header must be an instance of astropy.io.fits.Header.")

    dest_to_option = _map_dest_to_option(parser) if parser is not None else {}

    header.add_history("*" * 71)
    if title is not None:
        header.add_history(f"{title}")
        header.add_history("-" * 71)
    header.add_history(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    header.add_history(f"Node: {platform.uname().node}")
    header.add_history(f"Python: {sys.executable}")
    header.add_history(f"$ {Path(sys.argv[0]).name}")
    for arg, value in vars(args).items():
        # Handle BufferedReader objects
        if isinstance(value, BufferedReader):
            value = value.name if hasattr(value, "name") else str(value)
        option = dest_to_option.get(arg, f"--{arg}")
        if option is not None:
            header.add_history(f"{option} {value}")
        else:
            header.add_history(f"{value}")  # Positional argument
