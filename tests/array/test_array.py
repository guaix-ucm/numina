#
# Copyright 2008-2014 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import unittest

import pytest

import numpy

import numina.array as array


class ArrayTestCase(unittest.TestCase):

    def test_subarray_match(self):
        """Test subarray_match."""

        # Shapes don't intersect
        minor, major = array.subarray_match((3, 4, 5), (10, 10, 10), (2, 2, 2))

        #  Returns None if shapes don't intersect
        self.assertTrue(
            minor is None,
        )
        self.assertTrue(major is None)

        small = 100

        # One is contained inside the other
        minor, major = array.subarray_match((2048, 2048), (0, 0), (small, small))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(0, small, None), slice(0, small, None)))
        self.assertEqual(major, (slice(0, small, None), slice(0, small, None)))

        # One is contained inside the other, with offsets
        minor, major = array.subarray_match((2048, 2048), (30, 40), (small, small))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(30, small + 30, None), slice(40, small + 40, None)))
        self.assertEqual(major, (slice(0, small, None), slice(0, small, None)))

        # Both are equal, with offsets
        minor, major = array.subarray_match((100, 100), (30, 40), (small, small))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(30, small, None), slice(40, small, None)))
        self.assertEqual(major, (slice(0, small - 30, None), slice(0, small - 40, None)))

        # Equal offsets in both sides
        minor, major = array.subarray_match((100, 100), (30, 40), (100, 100), (30, 40))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(0, small, None), slice(0, small, None)))
        self.assertEqual(major, (slice(0, small, None), slice(0, small, None)))

        # Different offsets in both sides
        minor, major = array.subarray_match((100, 100), (31, 42), (100, 100), (10, 20))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(21, small, None), slice(22, small, None)))
        self.assertEqual(major, (slice(0, 79, None), slice(0, 78, None)))

        # If we interchange the arrays and change the sign of the offset,
        # we get the same result
        minor, major = array.subarray_match((100, 100), (10, 20), (200, 100))
        cminor, cmajor = array.subarray_match((200, 100), (-10, -20), (100, 100))
        self.assertEqual(cminor, major)
        self.assertEqual(cmajor, minor)


class FixpixTestCase(unittest.TestCase):
    def test_exception(self):
        data = numpy.zeros((10, 10))
        mask = numpy.zeros((100, 100))

        self.assertRaises(ValueError, array.fixpix, data, mask)

    def test_array_is_the_same(self):
        data = numpy.zeros((10, 10))
        mask = numpy.zeros((10, 10))

        data2 = array.fixpix(data, mask)

        self.assertIs(data2, data)

    def test_simple_interpolation(self):
        data = numpy.array([[1.0, -1000, 3.0], [4.0, -1000, 6.0], [7.0, -10000, 9.0]])
        mask = numpy.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])

        data2 = array.fixpix(data, mask)
        for i, v in zip(data2[:, 1], [2.0, 5.0, 8.0]):
            self.assertAlmostEqual(i, v)


@pytest.mark.parametrize("shape", [(3, 4), (4, 3)])
def test_fixpix_not_square(shape):
    data = numpy.tile(numpy.arange(shape[1], dtype="float64"), (shape[0], 1))
    mask = numpy.zeros(shape, dtype="bool")
    mask[1, 1] = True
    data[1, 1] = -1000

    result = array.fixpix(data, mask)
    assert result[1, 1] == 1.0


def test_fixpix2_plane():
    yy, xx = numpy.mgrid[0:12, 0:10]
    plane = 3.0 + 0.5 * xx - 0.25 * yy
    data = plane.copy()
    mask = numpy.zeros_like(data, dtype="uint8")
    mask[3:5, 4:6] = 1
    mask[8, 2] = 1
    data[mask != 0] = -1000

    result = array.fixpix2(data, mask)

    assert result is not data
    assert numpy.allclose(result, plane)
    assert data[3, 4] == -1000


def test_fixpix2_out():
    data = numpy.ones((6, 6))
    mask = numpy.zeros((6, 6))
    mask[2, 3] = 1
    data[2, 3] = 100

    result = array.fixpix2(data, mask, out=data, iterations=1)

    assert result is data
    assert numpy.allclose(data, 1)


def test_combine_shape():
    finalshape, offsetsp = array.combine_shape([(10, 20), (10, 20), (10, 20)], [(0, 0), (-2, 3), (4, -1)])
    assert finalshape == (16, 24)
    assert numpy.array_equal(offsetsp, [(2, 1), (0, 4), (6, 0)])


def test_combine_shapes_rc():
    finalshape, subshapes, ref = array.combine_shapes([(10, 10), (8, 12)], [(0, 0), (2, 3)])
    assert finalshape == (12, 13)
    assert subshapes == [(slice(2, 12), slice(3, 13)), (slice(0, 8), slice(0, 12))]
    assert ref == (2, 3)


def test_combine_shapes_xy():
    finalshape, subshapes, ref = array.combine_shapes([(10, 10), (8, 12)], [(0, 0), (3, 2)], order="xy")
    assert finalshape == (12, 13)
    assert subshapes == [(slice(2, 12), slice(3, 13)), (slice(0, 8), slice(0, 12))]
    assert ref == (3, 2)


def test_rebin():
    data = numpy.arange(16).reshape((4, 4))
    assert numpy.array_equal(array.rebin(data, (2, 2)), [[0, 2], [8, 10]])


@pytest.mark.parametrize("shape, scale", [((3, 3), 3), ((7, 5), 3), ((13, 11), 7)])
def test_rebin_scale(shape, scale):
    data = numpy.arange(numpy.prod(shape)).reshape(shape)
    result = array.rebin_scale(data, scale)
    assert result.shape == (shape[0] * scale, shape[1] * scale)
    assert numpy.array_equal(result[::scale, ::scale], data)
    assert numpy.array_equal(result[scale - 1 :: scale, scale - 1 :: scale], data)


def test_resize_array():
    data = numpy.arange(6, dtype="float32").reshape((2, 3))
    region = (slice(1, 3), slice(2, 5))
    result = array.resize_array(data, (4, 6), region, fill=-1.0)
    assert result.dtype == numpy.float32
    assert numpy.array_equal(result[region], data)
    assert result.sum() == data.sum() - (24 - 6)


def test_resize_array_window_dtype():
    data = numpy.arange(12).reshape((3, 4))
    window = (slice(1, 3), slice(0, 2))
    region = (slice(0, 2), slice(0, 2))
    result = array.resize_array(data, (3, 3), region, window=window, dtype="float64")
    assert result.dtype == numpy.float64
    assert numpy.array_equal(result[region], data[window])


@pytest.mark.parametrize("conserve, factor", [(True, 0.25), (False, 1.0)])
def test_resize_array_scale(conserve, factor):
    data = numpy.arange(1, 5, dtype="float64").reshape((2, 2))
    region = (slice(0, 4), slice(1, 5))
    result = array.resize_array(data, (5, 5), region, scale=2, conserve=conserve)
    assert numpy.allclose(result[region], factor * array.rebin_scale(data, 2))
    assert result.sum() == pytest.approx(4 * factor * data.sum())


def test_resize_arrays():
    arrays = [numpy.full((2, 3), 1.0), numpy.full((2, 3), 2.0)]
    finalshape, offsetsp = array.combine_shape([(2, 3), (2, 3)], [(0, 0), (1, 2)])
    result, regions = array.resize_arrays(arrays, (2, 3), offsetsp, finalshape, fill=numpy.nan)

    assert regions == [(slice(0, 2), slice(0, 3)), (slice(1, 3), slice(2, 5))]
    for data, newdata, region in zip(arrays, result, regions):
        assert newdata.shape == finalshape
        assert numpy.array_equal(newdata[region], data)
        assert numpy.isnan(newdata).sum() == 15 - 6

    result_alt = array.resize_arrays_alt(arrays, regions, finalshape, fill=numpy.nan)
    for newdata, newdata_alt in zip(result, result_alt):
        assert numpy.array_equal(newdata, newdata_alt, equal_nan=True)


@pytest.mark.parametrize(
    "function, expected",
    [
        (array.correct_dark, 8.0),
        (array.correct_sky, 8.0),
        (array.correct_flatfield, 5.0),
    ],
)
def test_correct(function, expected):
    data = numpy.full((2, 2), 10.0)
    calib = numpy.full((2, 2), 2.0)
    result = function(data, calib)
    assert result.dtype == numpy.float32
    assert numpy.allclose(result, expected)
    assert function(data, calib, dtype="float64").dtype == numpy.float64


def test_correct_nonlinearity():
    data = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    result = array.correct_nonlinearity(data, [1.0, 0.0, 2.0])
    assert result.dtype == numpy.float32
    assert numpy.allclose(result, data**2 + 2)


def test_compute_sky_advanced():
    data = numpy.array([[1.0, 2.0, 100.0], [3.0, 4.0, 5.0]])
    mask = numpy.array([[0, 0, 1], [0, 0, 0]])
    result = array.compute_sky_advanced([data, data + 1], [mask, mask])
    assert result.shape == data.shape
    assert numpy.allclose(result, 3.0)


def test_compute_median_background():
    data = numpy.array([[1.0, 2.0, 100.0], [3.0, 4.0, 5.0], [50.0, 60.0, 70.0]])
    mask = numpy.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    region = (slice(0, 2), slice(0, 3))
    assert array.compute_median_background(data, mask, region) == 3.0


def test_numberarray():
    result = array.numberarray(2.5, (2, 3))
    assert result.shape == (2, 3)
    assert numpy.all(result == 2.5)
    values = [1, 2, 3]
    assert array.numberarray(values, (2, 3)) is values
