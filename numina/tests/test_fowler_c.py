#
# Copyright 2008-2012 Universidad Complutense de Madrid
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

from numina.array import fowler_array

class FowlerTestCase(unittest.TestCase):
    def setUp(self):
        self.fdata = numpy.empty((1, 1, 10))
        rows = 3
        columns = 4
        self.emptybp = numpy.zeros((rows, columns), dtype='uint8')
        self.data = numpy.arange(10, dtype='int32')
        self.data = numpy.tile(self.data, (rows, columns, 1))
        self.blank = 1
        self.saturation = 65536
            
    def test_exception(self):
        
        # Dimension must be 3
        self.assertRaises(ValueError, fowler_array, numpy.empty((2,)))
        self.assertRaises(ValueError, fowler_array, numpy.empty((2,2)))
        self.assertRaises(ValueError, fowler_array, numpy.empty((2,3,4,5)))
        # saturation in good shape
        self.assertRaises(ValueError, fowler_array, self.fdata, saturation=-100)
        self.assertRaises(ValueError, fowler_array, self.fdata, saturation=0)
        # 2-axis must be even
        self.assertRaises(ValueError, fowler_array, numpy.empty((2,2,5)))
        
    def test_saturation0(self):        
        '''Test we count correctly saturated pixels in Fowler mode.'''
        
        MASK_SATURATION = 3
    
        # No points 
        self.data[:] = 50000 #- 32768
        saturation = 40000 #- 32767
        
        res = fowler_array(self.data, saturation=saturation, blank=self.blank)

        # Number of points
        for nn in res[2].flat:
            self.assertEqual(nn, 0)

        # Mask value            
        for n in res[3].flat:
            self.assertEqual(n, MASK_SATURATION)
            
        # Variance
        for v in res[1].flat:
            self.assertEqual(v, self.blank)
        
        # Values
        for v in res[0].flat:
            self.assertEqual(v, self.blank)
            

    def test_saturation1(self):        
        '''Test we count correctly saturated pixels in Fowler mode.'''
        
        MASK_GOOD = 0
            
        saturation = 50000
        self.data[..., 7:] = saturation 
        
        res = fowler_array(self.data, 
                    saturation=saturation,
                    blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 2)

        for n in res[3].flat:
            self.assertEqual(n, MASK_GOOD)
            
        for v in res[1].flat:
            self.assertEqual(v, 0)
            
        for v in res[0].flat:
            self.assertEqual(v, 5)
            
            
        
    def test_badpixel(self):
        '''Test we ignore badpixels in Fowler mode.'''
        self.emptybp[...] = 1

        res = fowler_array(self.data, 
                    saturation=self.saturation,
                    badpixels=self.emptybp,
                    blank=self.blank)

        for nn in res[2].flat:
            self.assertEqual(nn, 0)

        for n in res[3].flat:
            self.assertEqual(n, 1)
            
        for v in res[1].flat:
            self.assertEqual(v, self.blank)
            
        for v in res[0].flat:
            self.assertEqual(v, self.blank)

    def test_results1(self):
        '''Test we obtain correct values in Fowler mode'''
        
        data = numpy.zeros((4, 5, 10), dtype='int32')
        vals = numpy.array([10, 13, 15, 17, 20, 411, 412, 414, 417, 422])
        ovals = vals[5:] - vals[:5]
        mean = ovals.mean()
        var = ovals.var()
        
        for i in range(10):
            data[..., i] = vals[i]
        
        res = fowler_array(data, 
                    saturation=self.saturation, 
                    blank=self.blank)
        
        for nn in res[2].flat:
            self.assertEqual(nn, 5)

        for n in res[3].flat:
            self.assertEqual(n, 0)
            
        for v in res[1].flat:
            self.assertAlmostEqual(v, var)
            
        for v in res[0].flat:
            self.assertAlmostEqual(v, mean)         

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FowlerTestCase))    
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
