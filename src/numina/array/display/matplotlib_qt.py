#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Deprecated, the module is numina.array.display.matplotlib_utils"""

import warnings

from . import matplotlib_utils as _matplotlib_utils
from .matplotlib_utils import new_figure, parse_geometry, set_window_geometry  # noqa: F401

warnings.warn(
    "numina.array.display.matplotlib_qt is deprecated, use numina.array.display.matplotlib_utils",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    # plt is loaded by matplotlib_utils when it is first used
    return getattr(_matplotlib_utils, name)
