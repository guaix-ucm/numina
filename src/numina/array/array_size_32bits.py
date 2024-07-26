#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compute 32bits-array size in a human-readable format"""

from astropy.units import Unit


def bytes_to(size_in_bytes, to, bsize=1024):
    if to != 'k':
        to = to.upper()
    s = {'k': 1, 'M': 2, 'G': 3, 'T': 4, 'P': 5, 'E': 6}
    if to in s:
        return size_in_bytes / (bsize ** s[to])
    else:
        raise ValueError(f'Unexpected {to=}. Invalid size unit.')


def array_size_32bits(array):
    """Return 32bits-array size in a human-readable format"""
    size_in_bytes = array.__sizeof__()
    if array.dtype == 'float64':
        size_in_bytes /= 2
    elif array.dtype == 'float32':
        pass
    else:
        raise ValueError(f'Invalid {array.dtype=}')
    for s in 'EPTGMk':
        size = bytes_to(size_in_bytes, to=s)
        if size > 1:
            return size * Unit(f'{s}byte')
    raise ValueError(f'Size: {size_in_bytes} bytes is too large!')
