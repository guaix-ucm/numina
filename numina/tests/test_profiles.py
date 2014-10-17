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

'''Unit test for GaussProfile.'''


import unittest
import math

import numpy as np
from scipy.special import erf

from numina.instrument.profiles import GaussProfile

M_SQRT1_2 = math.sqrt(1. / 2.)


class GaussProfileTestCase(unittest.TestCase):
    '''Test of the GaussProfile class.'''

    @staticmethod
    def unidim(x, m, s):
        l = M_SQRT1_2 * (x - m) / s
        u = l + M_SQRT1_2 / s
        return 0.5 * (erf(u) - erf(l))

    def testOneDimensional(self):
        means = [0.3]
        covar = np.eye(1) * [25]
        gp3 = GaussProfile(means, covar)
        array = gp3.kernel

        m = gp3.selfcenter
        s = (5,)

        for i, v in np.ndenumerate(array):
            e = self.unidim(i[0], m[0], s[0])
            self.assertAlmostEqual(e, v)

    def testBiDimensional1(self):
        means = [34.1, 29.1]
        covar = np.eye(2) * [25, 25]

        gp1 = GaussProfile(means, covar)
        array = gp1.kernel
        center = gp1.selfcenter
        s = (5, 5)
        for i, v in np.ndenumerate(array):
            e1 = self.unidim(i[0], center[0], s[0])
            e2 = self.unidim(i[1], center[1], s[1])
            self.assertAlmostEqual(v, e1 * e2, 3)

    def testBiDimensional2(self):
        means = [0, 0]
        covar = np.eye(2) * [2, 1]

        gp1a = GaussProfile(means[0:1], covar[0:1, 0:1])
        gp1b = GaussProfile(means[1:2], covar[1:2, 1:2])
        gp2 = GaussProfile(means, covar)
        array1 = np.outer(gp1a.kernel, gp1b.kernel)
        array2 = gp2.kernel

        for i, v in np.ndenumerate(array1):
            self.assertAlmostEqual(v, array2[i])

    def testOneDimensionalNormalized(self):
        means = [0.3]
        covar = np.eye(1) * [25]
        gp3 = GaussProfile(means, covar, scale=10)
        array = gp3.kernel

        self.assertAlmostEqual(array.sum(), 1)

    def testBiDimensionalNormalized(self):
        means = [34.1, 29.1]
        covar = np.eye(2) * [25, 25]

        gp1 = GaussProfile(means, covar, scale=10)
        array = gp1.kernel

        self.assertAlmostEqual(array.sum(), 1)


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(GaussProfileTestCase))
    return suite

if __name__ == '__main__':
    unittest.main()
