#
# Copyright 2008 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyEmir is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# PyEmir is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with PyEmir.  If not, see <http://www.gnu.org/licenses/>.
# 

# $Id$ 

import unittest

import scipy

import numina.image.methods as mt

__version__ = '$Revision$'

__metaclass__ = type

class CombineMethodTestMixin:
    def function(self, values):
        return (0.0, 0.0, 0) 
    multiValues = []
    
    def setUp(self):
        self.emptyResult = (0,0,0)
        self.emptyInput = []
        self.oneInput = [2.0]
        self.oneResult = (self.oneInput[0], 0.0, 1)

    def tearDown(self):
        pass
    
    def test01Combine(self):
        self.assertEqual(self.function(self.emptyInput), self.emptyResult)
    
    def test02Combine(self):
        result = self.function(self.oneInput)
        self.assertAlmostEqual(result[0], self.oneResult[0])
        self.assertAlmostEqual(result[1], self.oneResult[1])
        self.assertEqual(result[2], self.oneResult[2])
    
    def test03Combine(self):
        for i in self.multiValues:
            result = self.function(i[0])
            self.assertAlmostEqual(result[0], i[1][0])
            self.assertAlmostEqual(result[1], i[1][1])
            self.assertEqual(result[2], i[1][2])

class MeanTestCase(CombineMethodTestMixin, unittest.TestCase):
    scipy.random.seed(93492094239423)
    randvalues = scipy.rand(100)
    multiValues = [
                   ([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1], 
                    (4.5999999999999996, 8.25, 10)),
                    (randvalues, (0.48745997782792883, 0.080160769637505264, 100))
                   ]
    def function(self, values):
        return mt.mean(values)

class SigmaClippingTestCase(CombineMethodTestMixin, unittest.TestCase):
    scipy.random.seed(93492094239423)
    randvalues = scipy.rand(100)
    randvalues[-1] = 100.
    multiValues = [
                   (randvalues, 
                    (0.48713429684450515, 0.080959867571022026, 99),)
                   ]
    def function(self, values):
        return mt.sigmaclip(values)

#class QuantileClippingTestCase(CombineMethodTestMixin, unittest.TestCase):
# TODO: Write this tests
class QuantileClippingTestCase(CombineMethodTestMixin):
    multiValues = [
                   ([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1], 
                    (4.1, 0, 10)),
                   ([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1], 
                    (4.1, 0, 10))
                   ]
    def function(self, values):
        return mt.quantileclip(values)



def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(MeanTestCase))
    suite.addTest(unittest.makeSuite(SigmaClippingTestCase))
    #suite.addTest(unittest.makeSuite(QuantileClippingTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')