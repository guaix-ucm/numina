#
# Copyright 2008-2010 Sergio Pascual
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

import unittest

import numpy

from numina.array.combine import mean, combine
from numina.array.combine import CombineError
        
class CombineTestCase(unittest.TestCase):
    def setUp(self):
        data = numpy.array([[1, 2], [1, 2]])
        self.validImages = [data]
        mask = numpy.array([[True, False], [True, False]])
        self.validMasks = [mask]
        databig = numpy.zeros((256, 256))
        self.validBig = [databig] * 100
        
    def tearDown(self):
        pass
         
    def testBaseCombineError(self):
        '''test CombineError is raised for different operations.'''
        # method is not an allowed string
        self.assertRaises(CombineError, combine, self.validImages, method='dummy')
        # reject is not an allowed string
        self.assertRaises(CombineError, combine, self.validImages, reject='dummy')
        # inputs is empty
        self.assertRaises(CombineError, combine, [])
        # images don't have the same shape
        self.assertRaises(CombineError, combine, [self.validImages[0], self.validBig[0]])
        # incorrect number of offsets
        self.assertRaises(CombineError, combine, self.validImages, offsets=[])
        # incorrect number of masks
        self.assertRaises(CombineError, combine, self.validImages, self.validMasks * 2)
        # mask and image have different shape
        self.assertRaises(CombineError, combine, self.validImages, [numpy.array([True, False])])
        # output has wrong shape
        self.assertRaises(CombineError, combine, self.validImages, out=self.validImages[0])
        # scales must be != 0
        self.assertRaises(CombineError, combine, self.validImages, scales=[0])        
        # weights must be >= 0'
        self.assertRaises(CombineError, combine, self.validImages, weights=[-1])
        # must be one dimensional
        self.assertRaises(CombineError, combine, self.validImages, zeros=[[1, 2]])
        self.assertRaises(CombineError, combine, self.validImages, scales=[[1, 2]])
        self.assertRaises(CombineError, combine, self.validImages, weights=[[1, 2]])
        # incorrect number of elements
        self.assertRaises(CombineError, combine, self.validImages, zeros=[1, 2])
        self.assertRaises(CombineError, combine, self.validImages, scales=[1, 2])
        self.assertRaises(CombineError, combine, self.validImages, weights=[1, 2])
        
    def test03Exception(self):
        '''mean combine: CombineError is raised if inputs have different lengths.'''
        self.assertRaises(CombineError, mean, self.validImages, self.validMasks * 2)
        
    def test04(self):
        '''mean combine: TypeError is raised if inputs aren't convertible to numpy.array.'''
        self.assertRaises(TypeError, mean, ["a"])

    def testCombineMaskMean(self):
        '''mean combine: combination of integer arrays with masks.'''
        input1 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input2 = numpy.array([[3, 2, 8, 4], [6, 2, 0, 4], [1, 3, 3, 4]])
        input3 = numpy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
        inputs = [input1, input2, input3]
        
        mask1 = numpy.array([[False, False, False, True],
                             [False, True, False, False],
                             [False, True, False, False]])
        mask2 = numpy.array([[False, False, False, False],
                             [False, True, False, False],
                             [False, True, False, False]])
        mask3 = numpy.array([[False, False, False, False],
                             [False, True, False, False],
                             [False, False, True, False]])
        masks = [mask1, mask2, mask3]
        rres = numpy.array([[3.66666667, 2., 4., 4.],
                            [2.6666666666666665, 0., 1. , 4.],
                            [18., 2., 1.5 , 2.66666667]])
        rvar = numpy.array([[9.3333333333333339, 0.0, 13.0, 0.0],
                            [8.3333333333333339, 0.0, 3.0, 0.0],
                            [523.0, 0.0, 4.5, 5.3333333333333339]])
        rnum = numpy.array([[3, 3, 3, 2],
                            [3, 0, 3, 3],
                            [3, 1, 2, 3]])
        
        out = combine(inputs, masks, method='average')
        for cal, precal in zip(out[0].flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[1].flat, rvar.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[2].flat, rnum.flat):
            self.assertEqual(cal, precal)

    def testCombineMean(self):
        '''mean combine: combination of integer arrays.'''
        
        # Inputs
        input1 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input2 = numpy.array([[3, 2, 8, 4], [6, 2, 0, 4], [1, 3, 3, 4]])
        input3 = numpy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
        inputs = [input1, input2, input3]
        
        # Results
        rres = numpy.array([[3.66666667, 2., 4. , 4.0], [2.6666666666666665, 2., 1. , 4.],
                          [18., 2.33333333, 1.666666667 , 2.66666667]])
        rvar = numpy.array([[9.3333333333333339, 0., 13.0, 0.],
                            [8.3333333333333339, 0., 3.00000000, 0.],
                            [523.0, 0.33333333333333337, 2.333333333333333, 5.3333333333333339]])
        rnum = numpy.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])
        
        out = combine(inputs, method='average')
                
        # Checking
        for cal, precal in zip(out[0].flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[1].flat, rvar.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[2].flat, rnum.flat):
            self.assertEqual(cal, precal)
    
class MinMaxTestCase(unittest.TestCase):
    '''Test case for the minmax rejection method.'''
    def setUp(self):
        self.nimages = 10
        self.data = [numpy.ones((2, 2))] * self.nimages
        
    def testBasic1(self):
        '''Test value if points rejected are less than the images.'''
        for nlow in xrange(0, self.nimages):
            for nhigh in xrange(0, self.nimages - nlow):
                r = combine(self.data, reject='minmax', nlow=nlow, nhigh=nhigh)
            for v in r[0].flat:
                self.assertEqual(v, 1)
            for v in r[1].flat:
                self.assertEqual(v, 0)
            for v in r[2].flat:
                self.assertEqual(v, self.nimages - nlow - nhigh)
                
    def testBasic2(self):
        '''Test value if points rejected are equal to the images.'''
        for nlow in xrange(0, self.nimages):
            nhigh = self.nimages - nlow
            r = combine(self.data, reject='minmax', nlow=nlow, nhigh=nhigh)
            for v in r[0].flat:
                self.assertEqual(v, 0)
            for v in r[1].flat:
                self.assertEqual(v, 0)
            for v in r[2].flat:
                self.assertEqual(v, 0)
                
    def testBasic3(self):
        '''Test CombineError is raised if points rejected are more than images.'''
        for nlow in xrange(0, self.nimages):
            nhigh = self.nimages - nlow + 1
            self.assertRaises(CombineError, combine, self.data, 
                              reject='minmax', nlow=nlow, nhigh=nhigh)
      
      
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CombineTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
    
