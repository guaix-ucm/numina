#
# Copyright 2008-2022 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import unittest

import numpy

from numina.array.nirproc import ramp_array


class FollowUpTheRampTestCase(unittest.TestCase):
    def setUp(self):
        self.ramp = numpy.empty((10, 1, 1))
        self.ti = 9.0
        self.sgain = 1.0
        self.sron = 1.0

    def test_exception(self):
        self.assertRaises(ValueError, ramp_array, self.ramp,
                          self.ti, -1.0, self.sron)
        self.assertRaises(ValueError, ramp_array, self.ramp,
                          self.ti, 0, self.sron)
        self.assertRaises(ValueError, ramp_array, self.ramp,
                          self.ti, self.sgain, -1.0)
        self.assertRaises(ValueError, ramp_array, self.ramp,
                          -1.0, self.sgain, self.sron)
        self.assertRaises(ValueError, ramp_array, self.ramp,
                          0.0, self.sgain, self.sron)
        self.assertRaises(ValueError, ramp_array, self.ramp,
                          self.ti, self.sgain, self.sron, saturation=-100)
        self.assertRaises(ValueError, ramp_array, self.ramp,
                          self.ti, self.sgain, self.sron, saturation=0)


class RampReadoutAxisTestCase(unittest.TestCase):

    def setUp(self):
        rows = 3
        columns = 4
        self.emptybp = numpy.zeros((rows, columns), dtype='uint8')
        self.data = numpy.arange(10, dtype='int32')
        self.data = numpy.tile(self.data, (columns, rows, 1)).T
        assert self.data.shape == (10, 3, 4)

        self.saturation = 65536
        self.ti = 9.0
        self.gain = 1.0
        self.ron = 1.0
        self.blank = 0

    def test_saturation0(self):
        '''Test we count correctly saturated pixels in RAMP mode.'''

        MASK_SATURATION = 3

        # No points
        self.data[:] = 50000  # - 32768
        saturation = 40000  # - 32767

        res = ramp_array(self.data, self.ti, self.gain, self.ron,
                         saturation=saturation,
                         blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 0)

        for n in res[3].flat:
            self.assertEqual(n, MASK_SATURATION)

        for v in res[1].flat:
            self.assertEqual(v, 0)

        for v in res[0].flat:
            self.assertEqual(v, self.blank)

    def test_saturation1(self):
        '''Test we count correctly saturated pixels in RAMP mode.'''

        MASK_GOOD = 0

        saturation = 50000
        self.data[7:, ...] = saturation

        res = ramp_array(self.data, self.ti, self.gain, self.ron,
                         saturation=saturation,
                         blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 7)

        for n in res[3].flat:
            self.assertEqual(n, MASK_GOOD)

        # for v in res[1].flat:
        #    self.assertEqual(v, 0.2142857142857143)

        for v in res[0].flat:
            self.assertAlmostEqual(v, 1)

    def test_dtypes0(self):
        '''Test output is float64 by default'''
        inttypes = ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']
        floattypes = ['float32', 'float64']
        if hasattr(numpy, 'float128'):
            floattypes.append('float128')
        mdtype = numpy.dtype('uint8')
        ddtype = numpy.dtype('float64')
        rows = 3
        columns = 4
        for dtype in inttypes:
            data = numpy.arange(10, dtype=dtype)
            data = numpy.tile(self.data, (columns, rows, 1)).T
            res = ramp_array(data, self.ti, self.gain, self.ron)
            self.assertIs(res[0].dtype, ddtype)
            self.assertIs(res[1].dtype, ddtype)
            self.assertIs(res[2].dtype, mdtype)
            self.assertIs(res[3].dtype, mdtype)

        for dtype in floattypes:
            data = numpy.arange(10, dtype=dtype)
            data = numpy.tile(self.data, (columns, rows, 1)).T
            res = ramp_array(data, self.ti, self.gain, self.ron)
            self.assertIs(res[0].dtype, ddtype)
            self.assertIs(res[1].dtype, ddtype)
            self.assertIs(res[2].dtype, mdtype)
            self.assertIs(res[3].dtype, mdtype)

    def test_badpixel0(self):
        '''Test we ignore badpixels in RAMP mode.'''
        self.emptybp[...] = 1

        res = ramp_array(self.data, self.ti, self.gain, self.ron,
                         saturation=self.saturation,
                         badpixels=self.emptybp,
                         blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 0)

        for n in res[3].flat:
            self.assertEqual(n, 1)

        for v in res[1].flat:
            self.assertEqual(v, 0)

        for v in res[0].flat:
            self.assertEqual(v, self.blank)

    def test_badpixel1(self):
        '''Test we handle correctly None badpixel mask.'''
        rows = 3
        columns = 4
        self.emptybp[...] = 0
        self.data = numpy.arange(10, dtype='int32')
        self.data = numpy.tile(self.data, (columns, rows, 1)).T
        value = 1.0
        variance = 0.13454545454545455

        # Badpixels is [0,..., 0]
        res = ramp_array(self.data, self.ti, self.gain, self.ron,
                         saturation=self.saturation,
                         badpixels=self.emptybp,
                         blank=self.blank)

        for n in res[3].flat:
            self.assertEqual(n, 0)

        for nn in res[2].flat:
            self.assertEqual(nn, 10)

        # for v in res[1].flat:
        #    self.assertAlmostEqual(v, variance)

        for v in res[0].flat:
            self.assertAlmostEqual(v, value)

        # Badpixels is None
        res = ramp_array(self.data, self.ti, self.gain, self.ron,
                         saturation=self.saturation,
                         badpixels=None,
                         blank=self.blank)

        # for n in res[4].flat:
        #    self.assertEqual(n, 0)

        for n in res[3].flat:
            self.assertEqual(n, 0)

        for nn in res[2].flat:
            self.assertEqual(nn, 10)

        for v in res[1].flat:
            self.assertAlmostEqual(v, variance)

        for v in res[0].flat:
            self.assertAlmostEqual(v, value)

        # Badpixels has default value
        res = ramp_array(self.data, self.ti, self.gain, self.ron,
                         saturation=self.saturation,
                         blank=self.blank)

        # for n in res[4].flat:
        #    self.assertEqual(n, 0)

        for n in res[3].flat:
            self.assertEqual(n, 0)

        for nn in res[2].flat:
            self.assertEqual(nn, 10)

        for v in res[1].flat:
            self.assertAlmostEqual(v, variance)

        for v in res[0].flat:
            self.assertAlmostEqual(v, value)

    def test_results1(self):
        '''Test we obtain correct values in RAMP mode'''

        res = ramp_array(self.data, self.ti, self.gain, self.ron,
                         saturation=self.saturation,
                         blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 10)

        for n in res[3].flat:
            self.assertEqual(n, 0)

#        for v in res[1].flat:
#            self.assertEqual(v, 0.13454545454545455)

        for v in res[0].flat:
            self.assertEqual(v, 1.0)

    def test_results2(self):
        '''Test we obtain correct values in RAMP mode'''

        self.data *= 12
        self.data[:, 1, 1] = 70000
        # self.data[5:,2,2] += 1300

        self.emptybp[0, 0] = 1

        res = ramp_array(self.data, self.ti, self.gain, self.ron,
                         dtype='float32',
                         saturation=self.saturation,
                         badpixels=self.emptybp,
                         blank=self.blank)

        res0 = 12 * numpy.ones((3, 4), dtype='float32')
        res0[0, 0] = 0
        res0[1, 1] = 0
        res1 = 1.4812121212 * numpy.ones((3, 4), dtype='float32')
        res1[0, 0] = 0
        res1[1, 1] = 0
        res1[2, 2] = 1.61
        res2 = 10 * numpy.ones((3, 4), dtype='uint8')
        res2[0, 0] = 0
        res2[1, 1] = 0
        res3 = numpy.zeros((3, 4), dtype='uint8')
        res3[0, 0] = 1
        res3[1, 1] = 3
        res4 = numpy.zeros((3, 4), dtype='uint8')
        res4[2, 2] = 5

        for xx, yy in zip(res[0].flat, res0.flat):
            self.assertAlmostEqual(xx, yy)
        # for xx, yy in zip(res[1].flat, res1.flat):
        #    self.assertAlmostEqual(xx, yy)
        for xx, yy in zip(res[2].flat, res2.flat):
            self.assertEqual(xx, yy)
        for xx, yy in zip(res[3].flat, res3.flat):
            self.assertEqual(xx, yy)
        # for xx, yy in zip(res[4].flat, res4.flat):
        #    self.assertEqual(xx, yy)
