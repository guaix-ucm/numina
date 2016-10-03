

import numpy

from .. import combine_shapes_alt


def test_combine_shapes_alt():

    value = 100
    shapes = [(100, 100), (100, 100), (20, 130)]

    # refs must go in (z,y,x) order...
    refs = [(33, 22), (12, 78), (5, 5)]
    arrn = [numpy.zeros(shape, dtype='int') for shape in shapes]

    for arr, ref in zip(arrn, refs):
        arr[ref] = value

    finalshape, slices = combine_shapes_alt(shapes, refs)
    final = numpy.zeros(finalshape, dtype='int')

    assert len(slices) == len(shapes)

    for arr, sl in zip(arrn, slices):
        final[sl] += arr

    assert final.max() == value * len(shapes)
    assert finalshape == (121, 203)

