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

import unittest

import numpy

import numina.array as array


class ArrayTestCase(unittest.TestCase):

    def test_subarray_match(self):
        '''Test subarray_match.'''

        # Shapes don't intersect
        minor, major = array.subarray_match((3, 4, 5),
                                            (10, 10, 10),
                                            (2, 2, 2))

        #  Returns None if shapes don't intersect
        self.assertTrue(minor is None,)
        self.assertTrue(major is None)

        small = 100

        # One is contained inside the other
        minor, major = array.subarray_match((2048, 2048),
                                            (0, 0), (small, small))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(0, small, None),
                                 slice(0, small, None)))
        self.assertEqual(major, (slice(0, small, None),
                                 slice(0, small, None)))

        # One is contained inside the other, with offsets
        minor, major = array.subarray_match((2048, 2048), (30, 40),
                                            (small, small))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(30, small + 30, None),
                                 slice(40, small + 40, None)))
        self.assertEqual(major, (slice(0, small, None),
                                 slice(0, small, None)))

        # Both are equal, with offsets
        minor, major = array.subarray_match((100, 100),
                                            (30, 40), (small, small))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(30, small, None),
                                 slice(40, small, None)))
        self.assertEqual(major, (slice(0, small - 30, None),
                                 slice(0, small - 40, None)))

        # Equal offsets in both sides
        minor, major = array.subarray_match((100, 100), (30, 40),
                                            (100, 100), (30, 40))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(0, small, None),
                                 slice(0, small, None)))
        self.assertEqual(major, (slice(0, small, None),
                                 slice(0, small, None)))

        # Different offsets in both sides
        minor, major = array.subarray_match((100, 100), (31, 42),
                                            (100, 100), (10, 20))
        # If one array is inside the other
        # the outputs is equal to the small array
        self.assertEqual(minor, (slice(21, small, None),
                                 slice(22, small, None)))
        self.assertEqual(major, (slice(0, 79, None), slice(0, 78, None)))

        # If we interchange the arrays and change the sign of the offset,
        # we get the same result
        minor, major = array.subarray_match((100, 100), (10, 20),
                                            (200, 100))
        cminor, cmajor = array.subarray_match((200, 100), (-10, -20),
                                              (100, 100))
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
        data = numpy.array([[1.0, -1000, 3.0], [4.0, -1000, 6.0],
                            [7.0, -10000, 9.0]])
        mask = numpy.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])

        data2 = array.fixpix(data, mask)
        for i, v in zip(data2[:, 1], [2.0, 5.0, 8.0]):
            self.assertAlmostEqual(i, v)
