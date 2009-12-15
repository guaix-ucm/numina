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

__version__ = '$Revision$'

import unittest

import scipy

from numina.image.combine import mean
from numina.image.combine import CombineError

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
         
    def test02Exception(self):
        '''mean combine: CombineError is raised if images list is empty'''
        self.assertRaises(CombineError, mean, [])
    
    def test03Exception(self):
        '''mean combine: CombineError is raised if inputs have different lengths'''
        self.assertRaises(CombineError, mean, self.validImages, self.validMasks * 2)
        
    def test04(self):
        '''mean combine: TypeError is raised if inputs aren't convertible to scipy.array'''
        self.assertRaises(TypeError, mean, ["a"])

    def testCombineMaskMean(self):
        '''mean combine: combination of integer arrays with masks'''
        input1 = scipy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input2 = scipy.array([[3, 2, 8, 4], [6, 2, 0, 4], [1, 3, 3, 4]])
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
                            [2.6666666666666665, 0., 1. , 4.],
                            [18., 2., 1.5 , 2.66666667]])
        rvar = scipy.array([[6.22222222222, 0.0, 8.66666666667, 0.0],
                            [5.5555555555555554, 0.0, 2.0, 0.0],
                            [348.666666667, 0.0, 2.25, 3.55555555556]])
        rnum = scipy.array([[3, 3, 3, 2], 
                            [3, 0, 3, 3], 
                            [3, 1, 2, 3]])
        
        out = mean(inputs, masks, dof=0)
        for cal, precal in zip(out[0].flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[1].flat, rvar.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[2].flat, rnum.flat):
            self.assertEqual(cal, precal)

    def testCombineMean(self):
        '''mean combine: combination of integer arrays'''
        
        # Inputs
        input1 = scipy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input2 = scipy.array([[3, 2, 8, 4], [6, 2, 0, 4], [1, 3, 3, 4]])
        input3 = scipy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
        inputs = [input1, input2, input3]
        
        # Results
        rres = scipy.array([[3.66666667, 2., 4. , 4.0], [2.6666666666666665, 2., 1. , 4.],
                          [18., 2.33333333, 1.666666667 , 2.66666667]])
        rvar = scipy.array([[6.22222222, 0., 8.66666667, 0.],
                            [5.5555555555555554, 0., 2.00000000, 0.],
                            [3.4866666666667e2, 0.222222222, 1.55555556, 3.55555556]])
        rnum = scipy.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])
        
        out = mean(inputs, dof=0)
                
        # Checking
        for cal, precal in zip(out[0].flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[1].flat, rvar.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[2].flat, rnum.flat):
            self.assertEqual(cal, precal)
      
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CombineTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
    
