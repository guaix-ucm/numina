#
# Copyright 2008-2011 Sergio Pascual
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

import math
import random
import unittest

from numina.image.aperture.elliptical import aperture

class EllipticalApertureTestCase(unittest.TestCase):
    
    def test_axis_is_positive(self):
        '''Test an exception is raised if semiaxis is not positive'''
        for a, b in [(-1, 3), (0, 3), (3,-1), (3, 0)]:
            generator = aperture(a, b, 0.0, 0.0)
            self.assertRaises(ValueError, generator.next)
            
            
    def test_full_area(self):
        '''Test the area of the full ellipse.'''
        
        for _ in range(100):
            a = 10**random.uniform(-1, 2)
            b = 10**random.uniform(-1, 2)
            x0 = random.uniform(-2, 2)
            y0 = random.uniform(-2, 2)
            sum = 0.0
            for _,_,w in aperture(a, b, x0, y0):
                sum += w
            
            self.assertAlmostEqual(sum, math.pi * a * b)
            
            
        for a in range(1, 50):
            for b in range(1, 50):
                sum = 0.0
                for _,_,w in aperture(a, b, 0.0, 0.0):
                    sum += w
            
                self.assertAlmostEqual(sum, math.pi * a * b)
            
            
            
            