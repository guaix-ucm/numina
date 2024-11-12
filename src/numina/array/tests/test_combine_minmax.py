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

import numina.array.combine as c


class MinMaxTestCase(unittest.TestCase):
    """Test case for the minmax rejection method."""

    def setUp(self):
        self.nimages = 10
        self.data = [numpy.ones((2, 2))] * self.nimages

    def testBasic1(self):
        """Test value if points rejected are less than the images."""
        for nmin in range(0, self.nimages):
            for nmax in range(0, self.nimages - nmin):
                out = c.minmax(self.data, nmin=nmin, nmax=nmax)
            for v in out[0].flat:
                self.assertEqual(v, 1)
            for v in out[1].flat:
                self.assertEqual(v, 0)
            for v in out[2].flat:
                self.assertEqual(v, self.nimages - nmin - nmax)

    def testBasic2(self):
        """Test value if points rejected are equal to the images."""
        for nmin in range(0, self.nimages):
            nmax = self.nimages - nmin
            out = c.minmax(self.data, nmin=nmin, nmax=nmax)
            for v in out[0].flat:
                self.assertEqual(v, 0)
            for v in out[1].flat:
                self.assertEqual(v, 0)
            for v in out[2].flat:
                self.assertEqual(v, 0)

    @unittest.skip("requires fixing generic_combine routine")
    def testBasic3(self):
        """Test ValueError is raised if points rejected are more than images"""
        for nmin in range(0, self.nimages):
            nmax = self.nimages - nmin + 1

            self.assertRaises(ValueError, c.minmax, self.data,
                              nmin=nmin, nmax=nmax)
