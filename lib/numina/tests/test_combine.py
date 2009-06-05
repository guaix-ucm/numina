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

import numpy
import scipy

from numina.image.combine import combine as combine
from numina.image.methods import mean as fun
from numina.exceptions import Error

__version__ = '$Revision$'

#class CombineFilter1TestCase(unittest.TestCase):
class CombineFilter1TestCase:
    def setUp(self):
        data = numpy.array([[1, 2], [1, 2]])
        mask = numpy.array([[True, False], [True, False]])
        self.validImages = [data]
        self.validMasks = [mask]
    
    def tearDown(self):
        pass
    
    def test04Exception(self):
        '''An exception is raised if inputs aren't convertible to numpy.array float'''
        self.assertRaises(TypeError, combine, ["a"], self.validMasks, fun)
    
    def test05Exception(self):
        '''An exception is raised if inputs aren't 2D'''
        self.assertRaises(TypeError, combine, [1], self.validMasks, fun)
    
    def test06Exception(self):
        '''An exception is raised if inputs aren't the same size'''
        self.assertRaises(TypeError, combine, [numpy.array([[1, 1, 1], [1, 1, 1]]), numpy.array([[1, 1], [1, 1]])],
                          self.validMasks * 2, fun)
    
    def test07Exception(self):
        '''An exception is raised if masks aren't convertible to numpy.array bool'''
        self.assertRaises(TypeError, combine, self.validImages, [["a"]], fun)
    
    def test08Exception(self):
        '''An exception is raised if masks aren't 2D'''
        self.assertRaises(TypeError, combine, self.validImages, [[True, False]], fun)
    
    def testCombineMean(self):
        '''Combination of float arrays and masks, mean method'''
        input1 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input2 = numpy.array([[3, 2, 8, 4], [6.5, 2, 0, 4], [1, 3, 3, 4]])
        input3 = numpy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
        inputs = [input1, input2, input3]
        
        mask1 = numpy.array([[False, False, False, True], [False, True, False, False], [False, True, False, False]])
        mask2 = numpy.array([[False, False, False, False], [False, True, False, False], [False, True, False, False]])
        mask3 = numpy.array([[False, False, False, False], [False, True, False, False], [False, False, True, False]])
        masks = [mask1, mask2, mask3]
        rres = numpy.array([[3.66666667, 2., 4. , 3.5], [2.83333333, 0., 1. , 4.],
                          [18., 9., 1.5 , 2.66666667]])
        rvar = numpy.array([[9.33333333, 0., 1.30e+01, 5.0e-01],
                            [1.00833333e+01 , 0., 3., 0.],
                            [5.23e+02 , 0., 4.5, 5.33333333]])
        rnum = numpy.array([[3, 3, 3, 2], [3, 0, 3, 3], [3, 1, 2, 3]])
        (res, var, num) = combine(inputs, masks, "mean", ())
        for cal, precal in zip(res.flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(var.flat, rvar.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(num.flat, rnum.flat):
            self.assertEqual(cal, precal)

class CombineTestCase(unittest.TestCase):
    def setUp(self):
        data = scipy.array([[1, 2], [1, 2]])
        self.validImages = [data]
        mask = numpy.array([[True, False], [True, False]])
        self.validMasks = [mask]
        
    def tearDown(self):
        pass
    
    def test01Exception(self):
        '''combine: TypeError is raised if method is not callable'''
        nofun = 1
        self.assertRaises(TypeError, combine, nofun, self.validImages)
      
    def test02Exception(self):
        '''combine: numina.Error is raised if images list is empty'''
        self.assertRaises(Error, combine, fun, [])
    
    def test03Exception(self):
        '''combine: numina.Error is raised if iputs have different lengths'''
        self.assertRaises(Error, combine, fun, self.validImages, self.validMasks * 2)
      
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CombineTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
    
