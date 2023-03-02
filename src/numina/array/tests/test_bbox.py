
import pytest

import numina.array.bbox as bbox


def test_pixin1d1():
    bb = bbox.PixelInterval1D(1, 3)
    assert bb.shape == 2
    assert bb.slice == slice(1, 3, None)


def test_pixin1d2():
    bb = bbox.PixelInterval1D.from_coordinates(1.0, 2.0)
    assert bb.shape == 2
    assert bb.slice == slice(1, 3, None)


def test_pixin1d3():
    bb = bbox.PixelInterval1D.from_coordinates(1.0, 2.0)
    bc = bbox.PixelInterval1D(1, 3)
    assert bb == bc


def test_pixin1d4():
    bb = bbox.PixelInterval1D.from_coordinates(1.0, 2.0)
    bc = bbox.PixelInterval1D(1, 4)
    assert bb != bc


def test_pixin1d5():
    bb = bbox.PixelInterval1D(2, 2)
    assert bb.shape == 0
    assert bb.slice == slice(2, 2, None)


def test_pixin1d6():
    with pytest.raises(ValueError):
        bbox.PixelInterval1D(1, 0)


def test_pixin1d7():
    with pytest.raises(TypeError):
        bbox.PixelInterval1D(1, 1.5)

    with pytest.raises(TypeError):
        bbox.PixelInterval1D(1.5, 2)


def test_pixin1d8():
    assert len(bbox.PixelInterval1D(1, 12)) == 11
    assert len(bbox.PixelInterval1D(3, 3)) == 0


def test_pixin1d9():
    assert bool(bbox.PixelInterval1D(1, 12))

    assert not bool(bbox.PixelInterval1D(3, 3))


def test_pixin1d10():
    b1 = bbox.PixelInterval1D(1, 12)
    b2 = bbox.PixelInterval1D(8, 14)
    b3 = b1.union(b2)
    assert b3 == bbox.PixelInterval1D(1, 14)

    b1 = bbox.PixelInterval1D(1, 12)
    b2 = bbox.PixelInterval1D(20, 22)
    b3 = b1.union(b2)
    assert b3 == bbox.PixelInterval1D(1, 22)


def test_pixin1d11():
    b1 = bbox.PixelInterval1D(3, 3)
    b2 = bbox.PixelInterval1D(8, 14)
    b3 = b1.union(b2)
    assert b3 == bbox.PixelInterval1D(8, 14)

    b3 = b2.union(b1)
    assert b3 == bbox.PixelInterval1D(8, 14)

    b1 = bbox.PixelInterval1D(5, 5)
    b2 = bbox.PixelInterval1D(9, 9)
    b3 = b1.union(b2)
    assert not bool(b3)


def test_pixin1d12():
    b1 = bbox.PixelInterval1D(1, 4)
    b2 = bbox.PixelInterval1D(2, 6)
    b3 = b1.intersection(b2)
    assert b3 == bbox.PixelInterval1D(2, 4)

    b3 = b2.intersection(b1)
    assert b3 == bbox.PixelInterval1D(2, 4)

    b1 = bbox.PixelInterval1D(1, 8)
    b2 = bbox.PixelInterval1D(3, 5)
    b3 = b1.intersection(b2)
    assert b3 == bbox.PixelInterval1D(3, 5)

    b3 = b2.intersection(b1)
    assert b3 == bbox.PixelInterval1D(3, 5)

    b1 = bbox.PixelInterval1D(1, 4)
    b2 = bbox.PixelInterval1D(5, 9)
    b3 = b1.intersection(b2)
    assert not bool(b3)


def test_pixin1d13():
    b1 = bbox.PixelInterval1D(1, 4)
    b2 = bbox.PixelInterval1D(2, 14)
    b3 = b1.intersection(b2)
    assert b3 == bbox.PixelInterval1D(2, 4)

    b3 = b2.intersection(b1)
    assert b3 == bbox.PixelInterval1D(2, 4)

    b1 = bbox.PixelInterval1D(5, 5)
    b2 = bbox.PixelInterval1D(9, 9)
    b3 = b1.intersection(b2)
    assert not bool(b3)


def test_pixin1d14():
    # Empty intervals are considered equal
    b1 = bbox.PixelInterval1D(1, 1)
    b2 = bbox.PixelInterval1D(2, 2)

    assert b1 == b2


def test_pixin1():
    bb = bbox.PixelInterval((2, 80), (1, 3))
    assert bb.shape == (78, 2)
    assert bb.slice == (slice(2, 80, None), slice(1, 3, None))


def test_pixin2():
    bb = bbox.PixelInterval.from_coordinates((2.0, 79.0), (1.0, 2.0))
    assert bb.shape == (78, 2)
    assert bb.slice == (slice(2, 80, None), slice(1, 3, None))


def test_pixin3():
    bb = bbox.PixelInterval.from_coordinates((2.0, 79.0), (1.0, 2.0))
    bc = bbox.PixelInterval((2, 80), (1, 3))
    assert bb == bc


def test_pixin4():
    bb = bbox.PixelInterval.from_coordinates((2.0, 79.0), (1.0, 2.0))
    bc = bbox.PixelInterval((2, 80), (1, 4))
    assert bb != bc


def test_pixin5():
    bb = bbox.PixelInterval((2,2), (1, 1))
    assert bb.shape == (0, 0)
    assert bb.slice == (slice(2, 2, None), slice(1, 1, None))

    import numpy as np
    a =  np.zeros((10, 10))
    assert (a[bb.slice].shape) == bb.shape


def test_pixin6():
    with pytest.raises(ValueError) as einfo:
        bbox.PixelInterval((1, 0), (2, 3))
    assert str(einfo.value) == "'pix2' must be >= 'pix1' in axis 0"

    with pytest.raises(ValueError) as einfo:
        bbox.PixelInterval((1, 2), (2, 1))

    assert str(einfo.value) == "'pix2' must be >= 'pix1' in axis 1"


def test_pixin7():
    with pytest.raises(TypeError) as einfo:
        bbox.PixelInterval((1, 1.5), (2, 3))

    assert str(einfo.value) == "'pix2' must be integer in axis 0"

    with pytest.raises(TypeError) as einfo:
        bbox.PixelInterval((1, 2), (2.5, 3))

    assert str(einfo.value) == "'pix1' must be integer in axis 1"


def test_pixin8():
    bb = bbox.PixelInterval((2, 80), (1, 3))
    assert len(bb) == 156

    bb = bbox.PixelInterval((2, 80), (1, 1))
    assert len(bb) == 0

    bb = bbox.PixelInterval()
    assert len(bb) == 0


def test_pixin9():
    bb = bbox.PixelInterval((2, 80), (1, 3))
    assert bool(bb)

    bb = bbox.PixelInterval((2, 80), (1, 1))
    assert not bool(bb)

    bb = bbox.PixelInterval()
    assert not bool(bb)


def test_pixin10():
    bb = bbox.PixelInterval((2, 80), (1, 3))
    assert bb.ndim == 2

    bb = bbox.PixelInterval((2, 80), (1, 1), (4,4))
    assert bb.ndim == 3

    bb = bbox.PixelInterval()
    assert bb.ndim == 0


def test_pixin11():
    b1 = bbox.PixelInterval((1, 12), (3, 6))
    b2 = bbox.PixelInterval((8, 14), (1, 9))
    b3 = b1.union(b2)
    assert b3 == bbox.PixelInterval((1, 14), (1, 9))

    b1 = bbox.PixelInterval((1, 12), (4, 7))
    b2 = bbox.PixelInterval((20, 22), (15, 18))
    b3 = b1.union(b2)
    assert b3 == bbox.PixelInterval((1, 22), (4, 18))


def test_pixin12():
    b1 = bbox.PixelInterval((4, 9), (3, 3))
    b2 = bbox.PixelInterval((4, 9), (8, 14))
    b3 = b1.union(b2)
    assert b3 == bbox.PixelInterval((4, 9), (8, 14))

    b3 = b2.union(b1)
    assert b3 == bbox.PixelInterval((4, 9), (8, 14))

    b1 = bbox.PixelInterval((5, 5), (3, 12))
    b2 = bbox.PixelInterval((9, 9), (8, 12))
    b3 = b1.union(b2)
    assert not bool(b3)


def test_pixin13():
    b1 = bbox.PixelInterval((3, 8), (1, 4))
    b2 = bbox.PixelInterval((2, 6), (2, 6))
    b3 = b1.intersection(b2)
    assert b3 == bbox.PixelInterval((3, 6), (2, 4))

    b3 = b2.intersection(b1)
    assert b3 == bbox.PixelInterval((3, 6), (2, 4))

    b1 = bbox.PixelInterval((1, 8), (4, 9))
    b2 = bbox.PixelInterval((3, 5), (4, 9))
    b3 = b1.intersection(b2)
    assert b3 == bbox.PixelInterval((3, 5), (4, 9))

    b3 = b2.intersection(b1)
    assert b3 == bbox.PixelInterval((3, 5), (4, 9))

    b1 = bbox.PixelInterval((1, 4), (3, 6))
    b2 = bbox.PixelInterval((5, 9), (3, 7))
    b3 = b1.intersection(b2)
    assert not bool(b3)


def test_pixin14():
    b1 = bbox.PixelInterval((1, 4))
    b2 = bbox.PixelInterval((2, 14))
    b3 = b1.intersection(b2)
    assert b3 == bbox.PixelInterval((2, 4))

    b3 = b2.intersection(b1)
    assert b3 == bbox.PixelInterval((2, 4))

    b1 = bbox.PixelInterval((5, 5))
    b2 = bbox.PixelInterval((9, 9))
    b3 = b1.intersection(b2)
    assert not bool(b3)


def test_pixin15():
    """Empty intervals are considered equal"""
    b1 = bbox.PixelInterval((1, 1), (3, 4))
    b2 = bbox.PixelInterval((2, 2), (5, 6))

    assert b1 == b2


def test_pixin16():
    # Empty intervals are considered equal
    i1 = bbox.PixelInterval1D(1, 2)
    i2 = bbox.PixelInterval1D(4, 5)
    b1 = bbox.PixelInterval.from_intervals([i1, i2])
    assert b1 == bbox.PixelInterval((1, 2), (4, 5))


def test_bbox1():
    bb = bbox.BoundingBox(1, 3, 2, 80)
    assert bb.shape == (78, 2)
    assert bb.slice == (slice(2, 80, None), slice(1, 3, None))


def test_bbox2():
    bb = bbox.BoundingBox.from_coordinates(1.0, 2.0, 2.0, 79.0)
    assert bb.shape == (78, 2)
    assert bb.slice == (slice(2, 80, None), slice(1, 3, None))


def test_bbox3():
    bb = bbox.BoundingBox.from_coordinates(1.0, 2.0, 2.0, 79.0)
    bc = bbox.BoundingBox(1, 3, 2, 80)
    assert bb == bc


def test_bbox4():
    bb = bbox.BoundingBox.from_coordinates(1.0, 2.0, 2.0, 79.0)
    bc = bbox.BoundingBox(1, 3, 3, 80)
    assert bb != bc


def test_bbox5():
    bc = bbox.BoundingBox(1, 3, 3, 80)
    assert bc.extent == (0.5, 2.5, 2.5, 79.5)
