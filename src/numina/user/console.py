#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from rich.console import Console


# Define NuminaConsole class for enriched output, inheriting from
# the Console class in the rich module
class NuminaConsole(Console):
    """Numina console for user interaction."""

    def __init__(self, **kwargs):
        """Initialize the Numina console with custom settings."""
        if isinstance(kwargs, dict):
            if 'force_terminal' not in kwargs:
                kwargs['force_terminal'] = True
        super().__init__(**kwargs)
