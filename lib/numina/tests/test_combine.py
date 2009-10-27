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

from numina.image.combine import combine as combine
from numina.exceptions import Error

__version__ = '$Revision$'

class CombineFilter1TestCase:
    def __init__(self):
        pass
    
    def setUp(self):
        data = scipy.array([[1, 2], [1, 2]])
        mask = scipy.array([[True, False], [True, False]])
        self.validImages = [data]
        self.validMasks = [mask]
            
class CombineTestCase(unittest.TestCase):
    def setUp(self):
        self.method = "mean"
        data = scipy.array([[1, 2], [1, 2]])
        self.validImages = [data]
        mask = scipy.array([[True, False], [True, False]])
        self.validMasks = [mask]
        databig = scipy.zeros((256, 256))
        self.validBig = [databig] * 100
        databig = scipy.zeros((256, 256))
        self.validBig = [databig] * 100
        
    def tearDown(self):
        pass
    
    def test01Exception(self):
        '''combine: TypeError is raised if method is not string'''
        nofun = 1
        self.assertRaises(TypeError, combine, nofun, self.validImages)
      
    def test02Exception(self):
        '''combine: numina.Error is raised if images list is empty'''
        self.assertRaises(Error, combine, self.method, [])
    
    def test03Exception(self):
        '''combine: numina.Error is raised if inputs have different lengths'''
        self.assertRaises(Error, combine, self.method, self.validImages, self.validMasks * 2)
        
    def test04(self):
        '''combine: TypeError is raised if inputs aren't convertible to scipy.array'''
        self.assertRaises(TypeError, combine, self.method, ["a"])
    
    def test05Exception(self):
        '''combine: TypeError is raised if inputs aren't 2D'''
        self.assertRaises(TypeError, combine, self.method, [1], self.validMasks)
          
    def test07Exception(self):
        '''combine: TypeError is raised if masks aren't convertible to scipy.array bool'''
        self.assertRaises(TypeError, combine, self.method, self.validImages, [["a"]])

    def test08Exception(self):
        '''combine: TypeError is raised if masks aren't 2D'''
        self.assertRaises(TypeError, combine, self.method, self.validImages, [[True, False]])

    def testCombineMaskMean(self):
        '''Combination of float arrays and masks, mean method'''
        input1 = scipy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input2 = scipy.array([[3, 2, 8, 4], [6.5, 2, 0, 4], [1, 3, 3, 4]])
        input3 = scipy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
        inputs = [input1, input2, input3]
        
        mask1 = scipy.array([[False, False, False, True], 
                             [False, True, False, False], 
                             [False, True, False, False]])
        mask2 = scipy.array([[False, False, False, False], 
                             [False, True, False, False], 
                             [False, True, False, False]])
        mask3 = scipy.array([[False, False, False, False], 
                             [False, True, False, False], 
                             [False, False, True, False]])
        masks = [mask1, mask2, mask3]
        rres = scipy.array([[3.66666667, 2., 4., 4.], 
                            [2.83333333, 0., 1. , 4.],
                            [18., 2., 1.5 , 2.66666667]])
        rvar = scipy.array([[6.22222222222, 0.0, 8.66666666667, 0.0],
                            [6.72222222222, 0.0, 2.0, 0.0],
                            [348.666666667, 0.0, 2.25, 3.55555555556]])
        rnum = scipy.array([[3, 3, 3, 2], 
                            [3, 0, 3, 3], 
                            [3, 1, 2, 3]])
        
        (res, var, num) = combine(self.method, inputs, masks)
        for cal, precal in zip(res.flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(var.flat, rvar.flat):
            pass #self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(num.flat, rnum.flat):
            self.assertEqual(cal, precal)

    def testCombineMean(self):
        '''Combination of float arrays, mean method'''
        
        # Inputs
        input1 = scipy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input2 = scipy.array([[3, 2, 8, 4], [6.5, 2, 0, 4], [1, 3, 3, 4]])
        input3 = scipy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
        inputs = [input1, input2, input3]
        
        # Results
        rres = scipy.array([[3.66666667, 2., 4. , 4.0], [2.83333333, 2., 1. , 4.],
                          [18., 2.33333333, 1.666666667 , 2.66666667]])
        rvar = scipy.array([[6.22222222, 0., 8.66666667, 0.],
                            [6.72222222, 0., 2.00000000, 0.],
                            [3.4866666666667e2, 0.222222222, 1.55555556, 3.55555556]])
        rnum = scipy.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])
        
        (res, var, num) = combine(self.method, inputs)
                
        # Checking
        for cal, precal in zip(res.flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(var.flat, rvar.flat):
            self.assertAlmostEqual(cal, precal) # TODO: problem with variance definition (N or N -1)
        for cal, precal in zip(num.flat, rnum.flat):
            self.assertEqual(cal, precal)

    def testCombineOffsetMean(self):
        '''Combination of float arrays with offsets, mean method'''
        # Inputs
        input1 = scipy.ones((4, 4))
        
        inputs = [input1, input1, input1, input1]
        offsets = [(1, 1), (1, 0), (0, 0), (0, 1)]
        
        # Results
        rres = scipy.ones((5, 5))
        rvar = scipy.zeros((5, 5))
        rnum = scipy.array([[1, 2, 2, 2, 1],
                            [2, 4, 4, 4, 2],
                            [2, 4, 4, 4, 2],
                            [2, 4, 4, 4, 2],
                            [1, 2, 2, 2, 1]])
        
        (res, var, num) = combine(self.method, inputs, offsets=offsets)

        # Checking
        for cal, precal in zip(res.flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(var.flat, rvar.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(num.flat, rnum.flat):
            self.assertEqual(cal, precal)
      
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CombineTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
    
