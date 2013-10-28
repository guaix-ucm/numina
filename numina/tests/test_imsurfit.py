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

from numina.array.imsurfit import imsurfit

class ImsurfitTestCase(unittest.TestCase):
    
    def setUp(self):
        # Two different sized grids
        xx, yy = numpy.mgrid[-1:1:100j,-1:1:100j]
        self.grid0 = [1, xx, yy, xx ** 2, xx * yy, yy ** 2, 
                      xx ** 3, xx ** 2 * yy, xx * yy ** 2, yy ** 3,
                      xx ** 4, xx ** 3 * yy, xx ** 2 *  yy ** 2, xx * yy ** 3, yy ** 4,]
        xx, yy = numpy.mgrid[-1:1:1203j,-1:1:458j]
        self.grid1 = [1, xx, yy, xx ** 2, xx * yy, yy ** 2, 
                      xx ** 3, xx ** 2 * yy, xx * yy ** 2, yy ** 3,
                      xx ** 4, xx ** 3 * yy, xx ** 2 *  yy ** 2, xx * yy ** 3, yy ** 4,]        
        

    def test_linear(self):
        '''Test linear fitting.'''
        
        order = 1
        results0 = numpy.array([456.0, 0.3, -0.9])
        
        z = numpy.sum(cf * v for cf, v in zip(results0.flat, self.grid0))
        results, = imsurfit(z, order=order)
        
        for i0, i in zip(results0.flat, results.flat):
            self.assertAlmostEqual(i0, i)
            
        z = numpy.sum(cf * v for cf, v in zip(results0.flat, self.grid1))
        results, = imsurfit(z, order=order)
        
        for i0, i in zip(results0.flat, results.flat):
            self.assertAlmostEqual(i0, i)

    def test_cuadratic(self):
        '''Test cuadratic fitting.'''
        
        order=2
        
        results0 = numpy.array([1.0, 0.1, 12, 3.0, -11.8, 9.2])
       
        z = numpy.sum(cf * v for cf, v in zip(results0.flat, self.grid0))
        results, = imsurfit(z, order=order)
        
        for i0, i in zip(results0.flat, results.flat):
            self.assertAlmostEqual(i0, i)
            
        z = numpy.sum(cf * v for cf, v in zip(results0.flat, self.grid1))
        results, = imsurfit(z, order=order)
        
        for i0, i in zip(results0.flat, results.flat):
            self.assertAlmostEqual(i0, i)
            
            
    def test_cubic(self):
        '''Test cubic fitting.'''
        
        order=3
        
        results0 = numpy.array([456.0, 0.3, -0.9, 0.03, -0.01, 0.07, 0.0, -10, 0.0, 0.04])
        z = numpy.sum(cf * v for cf, v in zip(results0.flat, self.grid0))
        results, = imsurfit(z, order=order)
        
        for i0, i in zip(results0.flat, results.flat):
            self.assertAlmostEqual(i0, i)
            
        z = numpy.sum(cf * v for cf, v in zip(results0.flat, self.grid1))
        results, = imsurfit(z, order=order)
        
        for i0, i in zip(results0.flat, results.flat):
            self.assertAlmostEqual(i0, i)
            
    def test_cuartic(self):
        '''Test cuartic fitting.'''
        
        order=4
        
        results0 = numpy.array([-11.0, -1.5, -0.1, 0.14, -15.03, 0.07, 
                                0.448, -0.28, 1.4, 1.24,
                                -3.2, -1.2, 2.24, -8.1, -0.03])
        z = numpy.sum(cf * v for cf, v in zip(results0.flat, self.grid0))
        results, = imsurfit(z, order=order)
        
        for i0, i in zip(results0.flat, results.flat):
            self.assertAlmostEqual(i0, i)
            
        z = numpy.sum(cf * v for cf, v in zip(results0.flat, self.grid1))
        results, = imsurfit(z, order=order)
        
        for i0, i in zip(results0.flat, results.flat):
            self.assertAlmostEqual(i0, i)
                        
        
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ImsurfitTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
