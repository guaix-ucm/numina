#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Tests for numina.array.wavecalib.arccalibration.match_wv_arrays"""

import numpy as np
import pytest

from numina.array.wavecalib.arccalibration import match_wv_arrays

# expected wavelengths of the line peaks
WV_PEAKS = np.array([100.0, 200.0, 300.0])


@pytest.mark.parametrize(
    "wv_master, expected",
    [
        # several master lines close to the first peak (j == 0 branch)
        ([99.0, 99.9], [99.9, 0.0, 0.0]),
        # several master lines close to the last peak (j == len branch)
        ([299.9, 301.0], [0.0, 0.0, 299.9]),
        # several master lines close to an intermediate peak, the
        # closest one found first
        ([199.9, 200.5], [0.0, 199.9, 0.0]),
        # same, the closest one found last
        ([199.0, 199.9], [0.0, 199.9, 0.0]),
        # master lines on both sides of an intermediate peak
        ([199.5, 200.2], [0.0, 200.2, 0.0]),
    ],
)
def test_match_wv_arrays_closest_line(wv_master, expected):
    """Each peak is assigned the closest master line within the range."""
    result = match_wv_arrays(np.array(wv_master), WV_PEAKS, delta_wv_max=2.0)
    np.testing.assert_array_equal(result, expected)


def test_match_wv_arrays_outside_range():
    """Master lines farther than delta_wv_max are not assigned."""
    result = match_wv_arrays(np.array([95.0, 203.0, 310.0]), WV_PEAKS, delta_wv_max=2.0)
    np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])
