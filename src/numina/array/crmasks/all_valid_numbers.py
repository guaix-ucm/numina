#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Check if all elements in a sequence are valid numbers (not NaN or Inf)."""
import math


def is_valid_number(x):
    """Check if x is a valid number (not NaN or Inf)."""
    return isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)


def all_valid_numbers(seq):
    """Check if all elements in seq are valid numbers."""
    if not isinstance(seq, (list, tuple)):
        raise TypeError("Input must be a list or tuple.")
    return all(is_valid_number(x) for x in seq)
