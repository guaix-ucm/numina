#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Apply thresholding to CR mask"""

from scipy import ndimage


def apply_threshold_cr(bool_crmask2d, bool_threshold2d):
    """Apply thresholding to CR mask.

    Apply threshold only when none of the pixels within a
    grouped CR feature exceeds the threshold condition.
    """

    new_bool_crmask2d = bool_crmask2d.copy()
    structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    cr_labels, num_crs = ndimage.label(bool_crmask2d, structure=structure)
    for i in range(1, num_crs + 1):
        mask_cr = cr_labels == i
        if not bool_threshold2d[mask_cr].any():
            new_bool_crmask2d[mask_cr] = False

    return new_bool_crmask2d
