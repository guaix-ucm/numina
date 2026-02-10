#
# Copyright 2024-2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from rich.console import Console


def raise_ValueError(msg):
    """Raise exception showing a coloured message."""
    console = Console()
    console.print(msg, style="bold red")
    raise ValueError(msg)
