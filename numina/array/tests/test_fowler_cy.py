#
# Copyright 2008-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import unittest

import numpy

from numina.array.nirproc import fowler_array

MASK_GOOD = 0
MASK_SATURATION = 3


class FowlerTestCase(unittest.TestCase):
    def setUp(self):
        self.fdata = numpy.empty((10, 1, 1))
        rows = 3
        columns = 4
        self.emptybp = numpy.zeros((rows, columns), dtype='uint8')
        self.data = numpy.arange(10, dtype='int32')
        self.data = numpy.tile(self.data, (columns, rows, 1)).T
        self.blank = 1
        self.saturation = 65536
        self.sgain = 1.0
        self.sron = 1.0

    def test_exception(self):
        '''Test we raise exceptions for invalid inputs in Fowler mode.'''

        # Dimension must be 3
        self.assertRaises(ValueError, fowler_array,
                          numpy.empty((2,)))
        self.assertRaises(ValueError, fowler_array,
                          numpy.empty((2, 2)))
        self.assertRaises(ValueError, fowler_array,
                          numpy.empty((2, 3, 4, 5)))
        # saturation in good shape
        self.assertRaises(ValueError, fowler_array,
                          self.fdata, saturation=-100)
        self.assertRaises(ValueError, fowler_array,
                          self.fdata, saturation=0)
        # 0-axis must be even
        self.assertRaises(ValueError, fowler_array,
                          numpy.empty((5, 2, 0)))
        # gain must be positive
        self.assertRaises(ValueError, fowler_array,
                          self.fdata, gain=-1.0)
        self.assertRaises(ValueError, fowler_array,
                          self.fdata, gain=0)
        # RON must be positive
        self.assertRaises(ValueError, fowler_array,
                          self.fdata, ron=-1.0)
        self.assertRaises(ValueError, fowler_array,
                          self.fdata, ron=0)

    def test_saturation0(self):
        '''Test we count correctly saturated pixels in Fowler mode.'''

        # No points
        self.data[:] = 50000  # - 32768
        saturation = 40000  # - 32767

        res = fowler_array(self.data, saturation=saturation, blank=self.blank)

        # Number of points
        for nn in res[2].flat:
            self.assertEqual(nn, 0)

        # Mask value
        for n in res[3].flat:
            self.assertEqual(n, MASK_SATURATION)

        # Variance
        for v in res[1].flat:
            self.assertEqual(v, self.blank)

        # Values
        for v in res[0].flat:
            self.assertEqual(v, self.blank)

    def test_saturation1(self):
        '''Test we count correctly saturated pixels in Fowler mode.'''

        saturation = 50000
        self.data[7:, ...] = saturation

        res = fowler_array(self.data,
                           saturation=saturation,
                           blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 2)

        for n in res[3].flat:
            self.assertEqual(n, MASK_GOOD)

        for v in res[1].flat:
            self.assertAlmostEqual(v, 2.0/2)

        for v in res[0].flat:
            self.assertAlmostEqual(v, 5)

    def test_dtypes0(self):
        '''Test output is float64 by default'''
        inttypes = ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']
        floattypes = ['float32', 'float64', ]
        if hasattr(numpy, 'float128'):
            floattypes.append('float128')
        mdtype = numpy.dtype('uint8')
        ddtype = numpy.dtype('float64')
        rows = 3
        columns = 4
        for dtype in inttypes:
            data = numpy.zeros(10, dtype=dtype)
            data[5:] = 10
            data = numpy.tile(self.data, (rows, columns, 1))
            res = fowler_array(data)
            self.assertIs(res[0].dtype, ddtype)
            self.assertIs(res[1].dtype, ddtype)
            self.assertIs(res[2].dtype, mdtype)
            self.assertIs(res[3].dtype, mdtype)

        for dtype in floattypes:
            data = numpy.zeros(10, dtype=dtype)
            data[5:] = 10
            data = numpy.tile(self.data, (rows, columns, 1))
            res = fowler_array(data)
            self.assertIs(res[0].dtype, ddtype)
            self.assertIs(res[1].dtype, ddtype)
            self.assertIs(res[2].dtype, mdtype)
            self.assertIs(res[3].dtype, mdtype)

    def test_badpixel0(self):
        '''Test we ignore badpixels in Fowler mode.'''
        mask_val = 2
        self.emptybp[...] = mask_val

        res = fowler_array(self.data,
                           saturation=self.saturation,
                           badpixels=self.emptybp,
                           blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 0)

        for n in res[3].flat:
            self.assertEqual(n, mask_val)

        for v in res[1].flat:
            self.assertEqual(v, self.blank)

        for v in res[0].flat:
            self.assertAlmostEqual(v, self.blank)

    def test_badpixel1(self):
        '''Test we handle correctly None badpixel mask.'''
        self.emptybp[...] = 0
        values = [2343, 2454, 2578, 2661, 2709, 24311, 24445,
                  24405, 24612, 24707]
        self.data = numpy.empty((10, 3, 4), dtype='int32')
        for i in range(10):
            self.data[i, ...] = values[i]
        arr = self.data[5:, 0, 0] - self.data[:5, 0, 0]
        mean = arr.mean()

        res = fowler_array(self.data,
                           saturation=self.saturation,
                           badpixels=self.emptybp,
                           blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 5)

        for n in res[3].flat:
            self.assertEqual(n, 0)

        # We have ti = texp = ts =0
        # So the noise is 2*sigma / N_p
        for v in res[1].flat:
            self.assertAlmostEqual(v, 2.0 / 5)

        for v in res[0].flat:
            self.assertAlmostEqual(v, mean)

        self.emptybp = None
        res = fowler_array(self.data,
                           saturation=self.saturation,
                           badpixels=self.emptybp,
                           blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 5)

        for n in res[3].flat:
            self.assertEqual(n, 0)

        for v in res[1].flat:
            self.assertAlmostEqual(v, 2.0/5)

        for v in res[0].flat:
            self.assertAlmostEqual(v, mean)

        res = fowler_array(self.data,
                           saturation=self.saturation,
                           blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 5)

        for n in res[3].flat:
            self.assertEqual(n, 0)

        for v in res[1].flat:
            self.assertAlmostEqual(v, 2.0/5)

        for v in res[0].flat:
            self.assertAlmostEqual(v, mean)

    def test_badpixel2(self):
        '''Test we don't accept badpixel mask with incompatible shape.'''
        self.assertRaises(ValueError, fowler_array, self.fdata,
                          badpixels=numpy.empty((10, 10)))
        self.assertRaises(ValueError, fowler_array, self.fdata,
                          badpixels=numpy.empty((1, 1, 1)))

    def test_badpixel3(self):
        '''Test we don't accept badpixel mask with incompatible dtype.'''
        self.assertRaises(ValueError, fowler_array, self.fdata,
                          badpixels=numpy.empty((1, 1), dtype='int'))

    def test_results1(self):
        '''Test we obtain correct values in Fowler mode'''

        data = numpy.zeros((10, 4, 5), dtype='int32')
        vals = numpy.array([10, 13, 15, 17, 20, 411, 412, 414, 417, 422])
        ovals = vals[5:] - vals[:5]
        mean = ovals.mean()
        var = ovals.var()

        for i in range(10):
            data[i, ...] = vals[i]

        res = fowler_array(data,
                           saturation=self.saturation,
                           blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 5)

        for n in res[3].flat:
            self.assertEqual(n, 0)

        for v in res[1].flat:
            self.assertAlmostEqual(v, 2.0 / 5)

        for v in res[0].flat:
            self.assertAlmostEqual(v, mean)
