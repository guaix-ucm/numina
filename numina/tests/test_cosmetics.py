#
# Copyright 2008-2013 Universidad Complutense de Madrid
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

import unittest

import numpy

from numina.array.cosmetics import cosmetics, ccdmask

class CosmeticsTestCase(unittest.TestCase):
    
    def setUp(self):
        # Two different sized grids
        
        self.flat1 = numpy.zeros((2,2))
        self.flat2 = self.flat1
        self.mask = numpy.zeros((2,2), dtype='int')
                              
    def test_exceptions(self):
        '''Test exceptions on invalid inputs of cosmetics'''
        
        self.assertRaises(ValueError, cosmetics, self.flat1, self.flat2, self.mask, lowercut=0.0)
        self.assertRaises(ValueError, cosmetics, self.flat1, self.flat2, self.mask, uppercut=0.0)
        self.assertRaises(ValueError, cosmetics, self.flat1, self.flat2, self.mask, siglev=0.0)

        self.assertRaises(ValueError, cosmetics, self.flat1, self.flat2, self.mask, lowercut=-1.0)
        self.assertRaises(ValueError, cosmetics, self.flat1, self.flat2, self.mask, uppercut=-10.0)
        self.assertRaises(ValueError, cosmetics, self.flat1, self.flat2, self.mask, siglev=-20.0)
        
class CcdmaskTestCase(unittest.TestCase):
    
    def setUp(self):
        # Two different sized grids
        
        self.flat1 = numpy.zeros((2,2))
        self.flat2 = self.flat1
        self.mask = numpy.zeros((2,2), dtype='int')
                              
    def test_exceptions(self):
        '''Test exceptions on invalid inputs of ccdmask'''
        
        self.assertRaises(ValueError, ccdmask, self.flat1, self.flat2, self.mask, lowercut=0.0)
        self.assertRaises(ValueError, ccdmask, self.flat1, self.flat2, self.mask, uppercut=0.0)
        self.assertRaises(ValueError, ccdmask, self.flat1, self.flat2, self.mask, siglev=0.0)

        self.assertRaises(ValueError, ccdmask, self.flat1, self.flat2, self.mask, lowercut=-1.0)
        self.assertRaises(ValueError, ccdmask, self.flat1, self.flat2, self.mask, uppercut=-10.0)
        self.assertRaises(ValueError, ccdmask, self.flat1, self.flat2, self.mask, siglev=-20.0)