#
# Copyright 2008-2014 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import division

import logging
from itertools import product

import six
from six.moves import range

if six.PY2:
    from itertools import imap
else:
    imap = map

from numpy.lib.stride_tricks import as_strided as ast

_logger = logging.getLogger("numina.array")


def blockgen1d(block, size):
    '''Compute 1d block intervals to be used by combine.

    blockgen1d computes the slices by recursively halving the initial
    interval (0, size) by 2 until its size is lesser or equal than block

    :param block: an integer maximum block size
    :param size: original size of the interval,
    it corresponds to a 0:size slice
    :return: a list of slices

    Example:

        >>> blockgen1d(512, 1024)
        [slice(0, 512, None), slice(512, 1024, None)]

    '''
    def numblock(block, x):
        '''Compute recursively the numeric intervals
        '''
        a, b = x
        if b - a <= block:
            return [x]
        else:
            result = []
            d = int(b - a) // 2
            for i in imap(numblock, [block, block], [(a, a + d), (a + d, b)]):
                result.extend(i)
            return result

    return [slice(*l) for l in numblock(block, (0, size))]


def blockgen(blocks, shape):
    '''Generate a list of slice tuples to be used by combine.

    The tuples represent regions in an N-dimensional image.

    :param blocks: a tuple of block sizes
    :param shape: the shape of the n-dimensional array
    :return: an iterator to the list of tuples of slices

    Example:

        >>> blocks = (500, 512)
        >>> shape = (1040, 1024)
        >>> for i in blockgen(blocks, shape):
        ...     print i
        (slice(0, 260, None), slice(0, 512, None))
        (slice(0, 260, None), slice(512, 1024, None))
        (slice(260, 520, None), slice(0, 512, None))
        (slice(260, 520, None), slice(512, 1024, None))
        (slice(520, 780, None), slice(0, 512, None))
        (slice(520, 780, None), slice(512, 1024, None))
        (slice(780, 1040, None), slice(0, 512, None))
        (slice(780, 1040, None), slice(512, 1024, None))

    '''
    iterables = [blockgen1d(l, s) for (l, s) in zip(blocks, shape)]
    return product(*iterables)


def blk_coverage_1d(blk, size):
    '''Return the part of a 1d array covered by a block.

    :param blk: size of the 1d block
    :param shape: size of the 1d a image
    :return: a tuple of size covered and remaining size

    Example:

        >>> blk_coverage_1d(7, 100)
        (98, 2)

    '''
    rem = size % blk
    maxpix = size - rem
    return maxpix, rem


def max_blk_coverage(blk, shape):
    '''Return the maximum shape of an array covered by a block.

    :param blk: the N-dimensional shape of the block
    :param shape: the N-dimensional shape of the array
    :return: the shape of the covered region

    Example:

        >>> max_blk_coverage(blk=(7, 6), shape=(100, 43))
        (98, 42)


    '''
    return tuple(blk_coverage_1d(b, s)[0] for b, s in zip(blk, shape))


def blk_1d_short(blk, shape):
    '''Iterate through the slices that recover a line.

    This function is used by :func:`blk_nd_short` as a base 1d case.

    The function stops yielding slices when the size of
    the remaining slice is lesser than `blk`.

    :param blk: the size of the block
    :param shape: the size of the array
    :return: a generator that yields the slices
    '''
    maxpix, _ = blk_coverage_1d(blk, shape)
    for i in range(0, maxpix, blk):
        yield slice(i, i + blk)


def blk_nd_short(blk, shape):
    '''Iterate trough the blocks that strictly cover an array.

    Iterate trough the blocks that recover the part of the array
    given by max_blk_coverage.

    :param blk: the N-dimensional shape of the block
    :param shape: the N-dimensional shape of the array
    :return: a generator that yields the blocks

    Example:

        >>> result = list(blk_nd_short(blk=(5,3), shape=(11, 11)))
        >>> result[0]
        (slice(0, 5, None), slice(0, 3, None))
        >>> result[1]
        (slice(0, 5, None), slice(3, 6, None))
        >>> result[-1]
        (slice(5, 10, None), slice(6, 9, None))

        In this case, the output of max_blk_coverage
        is (10, 9), so only this part of the array is covered


    .. seealso::

        :py:func:`blk_nd`
          Yields blocks of blk size until the remaining part is
          smaller than `blk` and the yields smaller blocks.

    '''
    internals = (blk_1d_short(b, s) for b, s in zip(blk, shape))
    return product(*internals)


def blk_1d(blk, shape):
    '''Iterate through the slices that recover a line.

    This function is used by :func:`blk_nd` as a base 1d case.

    The last slice  is returned even if is lesser than blk.

    :param blk: the size of the block
    :param shape: the size of the array
    :return: a generator that yields the slices

    '''
    maxpix, rem = blk_coverage_1d(blk, shape)
    for i in range(0, maxpix, blk):
        yield slice(i, i + blk)

    if rem != 0:
        yield slice(maxpix, shape)


def blk_nd(blk, shape):
    '''Iterate through the blocks that cover an array.

    This function first iterates trough the blocks that recover
    the part of the array given by max_blk_coverage
    and then iterates with smaller blocks for the rest
    of the array.

    :param blk: the N-dimensional shape of the block
    :param shape: the N-dimensional shape of the array
    :return: a generator that yields the blocks

    Example:

        >>> result = list(blk_nd(blk=(5,3), shape=(11, 11)))
        >>> result[0]
        (slice(0, 5, None), slice(0, 3, None))
        >>> result[1]
        (slice(0, 5, None), slice(3, 6, None))
        >>> result[-1]
        (slice(10, 11, None), slice(9, 11, None))

    The generator yields blocks of size blk until
    it covers the part of the array given by
    :func:`max_blk_coverage` and then yields
    smaller blocks until it covers the full array.

    .. seealso::

        :py:func:`blk_nd_short`
          Yields blocks of fixed size

    '''
    internals = (blk_1d(b, s) for b, s in zip(blk, shape))
    return product(*internals)


def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array.

    No error checking made. Therefore meaningful (as implemented) only for
    blocks strictly compatible with the shape of A.

    """

    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0] // block[0], A.shape[1] // block[1]) + block
    strides = (block[0] * A.strides[0], block[1] * A.strides[1]) + A.strides
    return ast(A, shape=shape, strides=strides)

if __name__ == '__main__':
    from numpy import arange
    import six
    A = arange(144).reshape(12, 12)
    six.print_(block_view(A)[0, 0])
    # [[ 0  1  2]
    # [12 13 14]
    # [24 25 26]]
    six.print_(block_view(A, (2, 6))[0, 0])
    # [[ 0  1  2  3  4  5]
    # [12 13 14 15 16 17]]
    six.print_(block_view(A, (3, 12))[0, 0])
    # [[ 0  1  2  3  4  5  6  7  8  9 10 11]
    # [12 13 14 15 16 17 18 19 20 21 22 23]
    # [24 25 26 27 28 29 30 31 32 33 34 35]]
