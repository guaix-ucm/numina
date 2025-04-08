#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from numina.tools.ctext import ctext


def raise_ValueError(msg):
    """Raise exception showing a coloured message."""
    raise ValueError(ctext(msg, fg='red'))
