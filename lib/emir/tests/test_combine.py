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
from emir.image.combine import new_combine as n
from emir.image.combine import method_mean as fun
import numpy

class CombineFilter1TestCase(unittest.TestCase):
    def setUp(self):
        data = numpy.array([[1,2],[1,2]])
        mask = numpy.array([[True,False],[True,False]])
        self.validInput = [data]
        self.validMask = [mask]
    def tearDown(self):
        pass
    def test01Exception(self):
        '''Test that an exception is raised if method is not callable'''
        nonfun = 1
        self.assertRaises(TypeError, n, self.validInput, self.validMask, nonfun)
    def test02Exception(self):
        '''Test that an exception is raised if inputs list is empty'''
        self.assertRaises(TypeError, n, [], self.validMask, fun)
    def test03Exception(self):
        '''Test that an exception is raised if masks list is empty'''
        self.assertRaises(TypeError, n, self.validInput, [], fun)
    def test031Exception(self):
        '''Test that an exception is raised if inputs and masks lists have different length'''
        self.assertRaises(TypeError, n, self.validInput, self.validMask * 2, fun)
    def test04Exception(self):
        '''Test that an exception is raised if inputs aren't convertible to numpy.array float'''
        self.assertRaises(TypeError, n, ["a",["a"]], self.validMask, fun)
    def test05Exception(self):
        '''Test that an exception is raised if inputs aren't 2D'''
        self.assertRaises(TypeError, n, [1, 1], self.validMask, fun)
    def test06Exception(self):
        '''Test that an exception is raised if inputs aren't the same size'''
        self.assertRaises(TypeError,n, [numpy.array([[1,1,1],[1,1,1]]),numpy.array([[1,1],[1,1]])], 
                          self.validMask, fun)
    def test07Exception(self):
        '''Test that an exception is raised if masks aren't convertible to numpy.array bool'''
        self.assertRaises(TypeError, n, self.validInput, ["a",["a"]], fun)
    def test08Exception(self):
        '''Test that an exception is raised if masks aren't 2D'''
        self.assertRaises(TypeError, n, self.validInput, [[True,False]], fun)
    def testCombineMean(self):
        '''Combination of float arrays and masks, mean method'''
        input1 = numpy.array([[1,2,3,4],[1,2,3,4],[9,2,0,4]])
        input2 = numpy.array([[3,2,8,4],[6.5,2,0,4],[1,3,3,4]])
        input3 = numpy.array([[7,2,1,4],[1,2,0,4],[44,2,2,0]])
        inputs = [input1, input2, input3]
        
        mask1 = numpy.array([[False,False,False,True],[False,True,False,False],[False,True,False,False]])
        mask2 = numpy.array([[False,False,False,False],[False,True,False,False],[False,True,False,False]])
        mask3 = numpy.array([[False,False,False,False],[False,True,False,False],[False,False,True,False]])
        masks = [mask1, mask2, mask3]
        rres = numpy.array([[3.66666667, 2., 4. ,3.5], [2.83333333, 0., 1. ,4.],
                          [18., 9., 1.5 , 2.66666667]])
        rvar = numpy.array([[9.33333333, 0.,  1.30e+01,  5.e-01],
                            [1.00833333e+01 , 0., 3., 0.],
                            [5.23e+02 ,  0., 4.5, 5.33333333]])
        rnum = numpy.array([[3, 3, 3, 2],[3, 0, 3, 3],[3, 1, 2, 3]])
        (res,var,num) = n(inputs, masks, fun, ())
        for i in zip(res.flat, rres.flat):
            self.assertAlmostEqual(i[0], i[1])
        for i in zip(var.flat, rvar.flat):
            self.assertAlmostEqual(i[0], i[1])
        for i in zip(num.flat, rnum.flat):
            self.assertEqual(i[0], i[1])
        
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CombineFilter1TestCase))
    return suite

if __name__ == '__main__':
    # unittest.main(defaultTest='test_suite')
    unittest.TextTestRunner(verbosity=2).run(test_suite())
