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
import emir.image._combine as c
from emir.image.combine import new_combine as n
import numpy

class CombineFilter1TestCase(unittest.TestCase):
    def setUp(self):
        data = numpy.array([[1,2],[1,2]])
        mask = numpy.array([[True,False],[True,False]])
        self.validInput = [data]
        self.validMask = [mask]
    def tearDown(self):
        pass
    def fun(self, list):
        '''A generic callable'''
        return (1.,2.,1)
    def test01Exception(self):
        '''Test that an exception is raised if method is not callable'''
        self.assertRaises(TypeError, c.test1, 1, self.validInput, self.validMask)
    def test02Exception(self):
        '''Test that an exception is raised if inputs list is empty'''
        self.assertRaises(TypeError, n, [], self.validMask, self.fun)
    def test03Exception(self):
        '''Test that an exception is raised if masks list is empty'''
        self.assertRaises(TypeError, n, self.validInput, [], self.fun)
    def test031Exception(self):
        '''Test that an exception is raised if inputs and masks lists have different length'''
        self.assertRaises(TypeError, n, self.validInput, self.validMask * 2, self.fun)
    def test04Exception(self):
        '''Test that an exception is raised if inputs aren't convertible to numpy.array float'''
        self.assertRaises(TypeError, c.test1, self.fun, ["a",["a"]], self.validMask)
    def test05Exception(self):
        '''Test that an exception is raised if inputs aren't 2D'''
        self.assertRaises(TypeError, c.test1, self.fun, [1, 1], self.validMask)
    def test06Exception(self):
        '''Test that an exception is raised if inputs aren't the same size'''
        self.assertRaises(TypeError, c.test1, self.fun, 
                          [numpy.array([[1,1,1],[1,1,1]]),numpy.array([[1,1],[1,1]])], self.validMask)
#    def test06Exception(self):
#        '''Test that an exception is raised if masks aren't convertible to numpy.array bool'''
#        self.assertRaises(TypeError, c.test1, self.fun, self.validInput, ["a",["a"]])
#    def test07Exception(self):
#        '''Test that an exception is raised if masks aren't 2D'''
#        self.assertRaises(TypeError, c.test1, self.fun, self.validInput, [[True,False]])

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CombineFilter1TestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')