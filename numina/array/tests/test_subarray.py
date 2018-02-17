
import numpy

from .. import combine_shapes


def test_combine_shapes():

    value = 100
    shapes = [(100, 100), (100, 100), (20, 130)]

    # refs must go in (z,y,x) order...
    refs = [(33, 22), (12, 78), (5, 5)]
    arrn = [numpy.zeros(shape, dtype='int') for shape in shapes]

    for arr, ref in zip(arrn, refs):
        arr[ref] = value

    finalshape, slices, finalref = combine_shapes(shapes, refs)
    final = numpy.zeros(finalshape, dtype='int')

    assert len(slices) == len(shapes)

    for arr, sl in zip(arrn, slices):
        final[sl] += arr

    assert final.max() == value * len(shapes)
    assert finalshape == (121, 203)


def test_combine_shapes2():

    value1 = 1
    value2 = 1
    shapes = [(40, 99), (103, 100), (100, 109)]

    # refs must go in (z,y,x) order...
    fshape = (128, 182)
    refs = [(33, 22), (12, 78), (5, 5)]
    #refs = [(0, 0), (5, 0), (0, 0)]

    off = [(ref[0] - refs[0][0], ref[1] - refs[0][1]) for ref in refs]

    arrn = [numpy.zeros(shape, dtype='int') for shape in shapes]

    for arr, ref in zip(arrn, refs):
        arr[:] = value1
        arr[ref] += value2

    finalshape, slices, finalref = combine_shapes(shapes, refs)

    assert finalshape == fshape

    final = numpy.zeros(finalshape, dtype='int')

    for arr, sl in zip(arrn, slices):
        final[sl] +=  arr

    assert final[finalref] == (value1 + value2) * len(shapes)
