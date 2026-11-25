#
# Copyright 2018-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Frame utils"""

import copy
import warnings


def copy_img(img):
    """Copy an HDUList

    .. deprecated:: 0.43
        Use ``copy.deepcopy(img)``, which makes the same copy. It will be
        removed in 0.44. ``img.copy()`` is not a replacement: it is a
        shallow copy that shares the HDUs.
    """
    warnings.warn(
        "copy_img is deprecated, use copy.deepcopy",
        DeprecationWarning,
        stacklevel=2,
    )
    return copy.deepcopy(img)
