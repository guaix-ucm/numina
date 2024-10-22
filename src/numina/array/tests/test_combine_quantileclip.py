#
# Copyright 2008-2024 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import unittest
import numpy

import numina.array.combine2 as c


class QuantileClipTestCase(unittest.TestCase):
    """Test case for the quantileclip rejection method."""

    def setUp(self):
        self.nimages = 10
        self.data = [numpy.ones((2, 2))] * self.nimages

    def testBasic0(self):
        """Test ValueError is raised if fraction of points rejected is < 0.0"""
        self.assertRaises(ValueError, c.quantileclip, self.data, fclip=-0.01)

    def testBasic1(self):
        """Test ValueError is raised if fraction of points rejected is > 0.4"""
        self.assertRaises(ValueError, c.quantileclip, self.data, fclip=0.41)

    def testBasic2(self):
        """Test integer rejections"""
        r = c.quantileclip(self.data, fclip=0.0)
        for v in r[0].flat:
            self.assertEqual(v, 1)
        for v in r[1].flat:
            self.assertEqual(v, 0)
        for v in r[2].flat:
            self.assertEqual(v, 10)

        r = c.quantileclip(self.data, fclip=0.1)
        for v in r[0].flat:
            self.assertEqual(v, 1)
        for v in r[1].flat:
            self.assertEqual(v, 0)
        for v in r[2].flat:
            self.assertEqual(v, 8)

        r = c.quantileclip(self.data, fclip=0.2)
        for v in r[0].flat:
            self.assertEqual(v, 1)
        for v in r[1].flat:
            self.assertEqual(v, 0)
        for v in r[2].flat:
            self.assertEqual(v, 6)

    def testBasic3(self):
        """Test simple fractional rejections"""
        r = c.quantileclip(self.data, fclip=0.23)
        for v in r[0].flat:
            self.assertEqual(v, 1)
        for v in r[1].flat:
            self.assertEqual(v, 0)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 5.0)

    def testBasic4(self):
        """Test complex fractional rejections"""
        data = [numpy.array([i]) for i in range(10)]

        r = c.quantileclip(data, fclip=0.0)
        for v in r[0].flat:
            self.assertAlmostEqual(v, 4.5)
        # for v in r[1].flat:
        #    self.assertAlmostEqual(v, 0.9166666666)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 10)

        r = c.quantileclip(data, fclip=0.1)
        for v in r[0].flat:
            self.assertAlmostEqual(v, 4.5)
        # for v in r[1].flat:
        #    self.assertAlmostEqual(v, 0.75)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 8)

        r = c.quantileclip(data, fclip=0.2)
        for v in r[0].flat:
            self.assertAlmostEqual(v, 4.5)
        # for v in r[1].flat:
        #    self.assertAlmostEqual(v, 0.58333333333333337)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 6)

        r = c.quantileclip(data, fclip=0.09)
        for v in r[0].flat:
            self.assertAlmostEqual(v, 4.5)
        # for v in r[1].flat:
        #    self.assertAlmostEqual(v, 0.76666666666666672)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 9.0)

    def testResults5(self):
        """Test deviant points are ignored"""
        data = [numpy.array([1.0]) for _ in range(22)]
        data[0][0] = 89
        data[12][0] = -89

        r = c.quantileclip(data, fclip=0.15)
        for v in r[0].flat:
            self.assertAlmostEqual(v, 1.0)
        for v in r[1].flat:
            self.assertAlmostEqual(v, 0.0)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 15.0)
