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


def all_valid_numbers(seq, fixed_length=None):
    """Check if all elements in seq are valid numbers.

    Parameters
    ----------
    seq : list or tuple
        Sequence of numbers to check.
    fixed_length : int or None, optional
        If provided, the sequence must have this exact length.

    Returns
    -------
    bool
        True if all elements are valid numbers (and length matches
        if fixed_length is set), False otherwise.
    """
    if not isinstance(seq, (list, tuple)):
        raise TypeError("Input must be a list or tuple.")
    if fixed_length is not None and len(seq) != fixed_length:
        print(f"Sequence {seq} has length {len(seq)}, expected {fixed_length}.")
        raise ValueError(f"Input sequence must have length {fixed_length}.")
    return all(is_valid_number(x) for x in seq)
