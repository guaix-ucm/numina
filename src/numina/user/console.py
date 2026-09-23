#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from rich.console import Console
from rich.text import Text


# Define NuminaConsole class for enriched output, inheriting from
# the Console class in the rich module
class NuminaConsole(Console):
    """Numina console for user interaction."""

    def __init__(self, **kwargs):
        """Initialize the Numina console with custom settings."""
        if isinstance(kwargs, dict):
            if "force_terminal" not in kwargs:
                kwargs["force_terminal"] = True
        super().__init__(**kwargs)


def print_table(console, table, header_style="red"):
    """Print an astropy Table/QTable through a rich Console.

    This function prints an astropy Table or QTable to a rich Console,
    highlighting the header lines as Table.pprint() does.

    Parameters
    ----------
    console : NuminaConsole
        The console object to which the table will be printed.
    table : astropy.table.Table or astropy.table.QTable
        The table to be printed.
    header_style : str, optional
        The style to be applied to the header lines. Default is "red".
    """
    lines = table.pformat(max_lines=-1, max_width=-1)

    # The header ends with the separator line made of dashes
    n_header = next(
        (i + 1 for i, line in enumerate(lines) if line.strip() and set(line) <= {"-", " "}),
        0,
    )

    text = Text()
    for i, line in enumerate(lines):
        text.append(line + "\n", style=header_style if i < n_header else None)

    console.print(text, highlight=False, soft_wrap=True, end="")
