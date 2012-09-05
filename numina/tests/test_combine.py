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
import itertools

import numpy # pylint: disable-msgs=E1101

from numina.array.combine import generic_combine
from numina.array.combine import mean, median, minmax, quantileclip
from numina.array.combine import mean_method
from numina.array.combine import CombineError
        
class CombineTestCase(unittest.TestCase):
    def setUp(self):
        data = numpy.array([[1, 2], [1, 2]])
        self.validImages = [data]
        mask = numpy.array([[True, False], [True, False]])
        self.validMasks = [mask]
        self.out = numpy.empty((3, 2, 2))
        databig = numpy.zeros((256, 256))
        self.validBig = [databig] * 100
        self.validOut = numpy.empty((3, 256, 256))
        
    def test_CombineError(self):
        '''Test CombineError is raised for different operations.'''
        
        # method is not valid
        self.assertRaises(TypeError, generic_combine, "dum", self.validImages)
        
        # inputs is empty
        self.assertRaises(CombineError, generic_combine, 
                          mean_method, [], out=self.out)
        # images don't have the same shape
        self.assertRaises(CombineError, generic_combine, 
                          mean_method, [self.validImages[0], self.validBig[0]],
                          out=self.validOut
                          )

        # incorrect number of masks
        self.assertRaises(CombineError, generic_combine, 
                          mean_method, self.validImages, 
                          out=self.validOut,
                          masks=self.validMasks)
        
        # mask and image have different shape
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages,                           
                          out=self.validOut,
                          masks=[numpy.array([True, False])]
                          )
        # output has wrong shape
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages, 
                          out=numpy.empty((3, 80, 12)))
        # scales must be != 0
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages, 
                          out=self.validOut, scales=numpy.zeros(1))        
        # weights must be >= 0'
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages, 
                          out=self.validOut, weights=numpy.array([-1]))
        # must be one dimensional
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages, 
                          out=self.validOut, zeros=[[1, 2]])
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages, 
                          out=self.validOut, scales=[[1, 2]])
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages, 
                          out=self.validOut, weights=[[1, 2]])
        # incorrect number of elements
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages, 
                          out=self.validOut, zeros=[1, 2])
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages, 
                          out=self.validOut, scales=[1, 2])
        self.assertRaises(CombineError, generic_combine, mean_method, self.validImages, 
                          out=self.validOut, weights=[1, 2])
        
    def test_CombineException(self):
        '''Combine: CombineError is raised if inputs have different lengths.'''
        self.assertRaises(CombineError, mean, self.validImages, self.validMasks * 2)
        
    def test_TypeError(self):
        '''Combine: TypeError is raised if inputs aren't convertible to numpy.array.'''
        self.assertRaises(TypeError, mean, ["a"])

    def testCombineMaskAverage(self):
        '''Average combine: combination of integer arrays with masks.'''
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
        rvar = numpy.array([[3 * 3.11111111, 0., 3 * 4.33333333, 0.],
                            [3 * 2.77777778, 0., 3 * 1.,  0.],
                            [3 * 174.33333333, 0., 2 * 2.25, 3 * 1.77777778]
                            ])                            
        
        rnum = numpy.array([[3, 3, 3, 2],
                            [3, 0, 3, 3],
                            [3, 1, 2, 3]])
        
        out = mean(inputs, masks)
        for cal, precal in zip(out[0].flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[1].flat, rvar.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[2].flat, rnum.flat):
            self.assertEqual(cal, precal)

    def testCombineAverage(self):
        '''Average combine: combination of integer arrays.'''
        
        # Inputs
        input1 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input2 = numpy.array([[3, 2, 8, 4], [6, 2, 0, 4], [1, 3, 3, 4]])
        input3 = numpy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
        inputs = [input1, input2, input3]
        
        # Results
        rres = numpy.array([[3.66666667, 2., 4. , 4.0], [2.6666666666666665, 2., 1. , 4.],
                          [18., 2.33333333, 1.666666667 , 2.66666667]])
        rvar = 3 * numpy.array([[9.3333333333333339, 0., 13.0, 0.],
                            [8.3333333333333339, 0., 3.00000000, 0.],
                            [523.0, 0.33333333333333337, 2.333333333333333, 
                             5.3333333333333339]]) / len(inputs)
        rnum = numpy.array([[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]])
        
        out = mean(inputs)       
        # Checking
        for cal, precal in zip(out[0].flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[1].flat, rvar.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[2].flat, rnum.flat):
            self.assertEqual(cal, precal)
            
    def test_median(self):
        '''Median combine: combination of integer arrays.'''
        # Inputs
        input1 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input2 = numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input3 = numpy.array([[7, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input4 = numpy.array([[7, 2, 3, 4], [1, 2, 3, 4], [9, 2, 0, 4]])
        input5 = numpy.array([[7, 2, 1, 4], [1, 2, 0, 4], [44, 2, 2, 0]])
        inputs = [input1, input2, input3, input4, input5]
        
        out = median(inputs)
    
        rres = input3
        rvar = [16.954474097331239, 0.0, 1.2558869701726849, 
                0.0, 0.0, 0.0, 2.8257456828885403, 0.0, 
                384.61538461538458, 0.0, 1.2558869701726847, 
                5.0235478806907397]
        
        # Checking
        for cal, precal in zip(out[0].flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        print list(out[1].flat)
        for cal, precal in zip(out[1].flat, rvar):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[2].flat, itertools.repeat(5)):
            self.assertEqual(cal, precal)
            
            
    def test_median2(self):
        '''Median combine: combination an even number of integer arrays.'''
        # Inputs
        input1 = numpy.array([[1, 2, 3, -4]])
        input2 = numpy.array([[1, 2, 6, 4]])
        input3 = numpy.array([[7, 3, 8, -4]])
        input4 = numpy.array([[7, 2, 3, 4]])
        inputs = [input1, input2, input3, input4]
        
        out = median(inputs)
    
        rres = numpy.array([[4, 2, 4.5, 0.0]], dtype='float')
        rvar = [18.838304552590266, 0.39246467817896391, 9.419152276295133, 
                33.490319204604916]
        
        # Checking
        for cal, precal in zip(out[0].flat, rres.flat):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[1].flat, rvar):
            self.assertAlmostEqual(cal, precal)
        for cal, precal in zip(out[2].flat, itertools.repeat(4)):
            self.assertEqual(cal, precal)
    
    
class MinMaxTestCase(unittest.TestCase):
    '''Test case for the minmax rejection method.'''
    def setUp(self):
        self.nimages = 10
        self.data = [numpy.ones((2, 2))] * self.nimages
        self.out = numpy.empty((3, 2, 2))
        
    def testBasic1(self):
        '''Test value if points rejected are less than the images.'''
        for nmin in xrange(0, self.nimages):
            for nmax in xrange(0, self.nimages - nmin):
                minmax(self.data, out=self.out, nmin=nmin, nmax=nmax)
            for v in self.out[0].flat:
                self.assertEqual(v, 1)
            for v in self.out[1].flat:
                self.assertEqual(v, 0)
            for v in self.out[2].flat:
                self.assertEqual(v, self.nimages - nmin - nmax)
                
    def testBasic2(self):
        '''Test value if points rejected are equal to the images.'''
        for nmin in xrange(0, self.nimages):
            nmax = self.nimages - nmin
            minmax(self.data, out=self.out, nmin=nmin, nmax=nmax)
            for v in self.out[0].flat:
                self.assertEqual(v, 0)
            for v in self.out[1].flat:
                self.assertEqual(v, 0)
            for v in self.out[2].flat:
                self.assertEqual(v, 0)
    
    @unittest.skip("requires fixing generic_combine routine")     
    def testBasic3(self):
        '''Test ValueError is raised if points rejected are more than images.'''
        for nmin in xrange(0, self.nimages):
            nmax = self.nimages - nmin + 1
            
            self.assertRaises(ValueError, minmax, self.data, 
                              nmin=nmin, nmax=nmax)

class QuantileClipTestCase(unittest.TestCase):
    '''Test case for the quantileclip rejection method.'''
    def setUp(self):
        self.nimages = 10
        self.data = [numpy.ones((2, 2))] * self.nimages
        
    def testBasic0(self):
        '''Test ValueError is raised if fraction of points rejected is < 0.0.'''
        self.assertRaises(ValueError, quantileclip, self.data, fclip=-0.01)        
        
    def testBasic1(self):
        '''Test ValueError is raised if fraction of points rejected is > 0.4.'''
        self.assertRaises(ValueError, quantileclip, self.data, fclip=0.41)
        
    def testBasic2(self):
        '''Test integer rejections'''
        r = quantileclip(self.data, fclip=0.0)
        for v in r[0].flat:
            self.assertEqual(v, 1)
        for v in r[1].flat:
            self.assertEqual(v, 0)
        for v in r[2].flat:
            self.assertEqual(v, 10)
  
        r = quantileclip(self.data, fclip=0.1)
        for v in r[0].flat:
            self.assertEqual(v, 1)
        for v in r[1].flat:
            self.assertEqual(v, 0)
        for v in r[2].flat:
            self.assertEqual(v, 8)
        
        r = quantileclip(self.data, fclip=0.2)
        for v in r[0].flat:
            self.assertEqual(v, 1)
        for v in r[1].flat:
            self.assertEqual(v, 0)
        for v in r[2].flat:
            self.assertEqual(v, 6)

    def testBasic3(self):
        '''Test simple fractional rejections'''
        r = quantileclip(self.data, fclip=0.23)
        for v in r[0].flat:
            self.assertEqual(v, 1)
        for v in r[1].flat:
            self.assertEqual(v, 0)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 5.4)
            
    def testBasic4(self):
        '''Test complex fractional rejections'''
        data = [numpy.array([i]) for i in range(10)]

        r = quantileclip(data, fclip=0.0)
        for v in r[0].flat:
            self.assertAlmostEqual(v, 4.5)
        #for v in r[1].flat:
        #    self.assertAlmostEqual(v, 0.9166666666)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 10)
            
        r = quantileclip(data, fclip=0.1)
        for v in r[0].flat:            
            self.assertAlmostEqual(v, 4.5)
        #for v in r[1].flat:
        #    self.assertAlmostEqual(v, 0.75)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 8)            
            
        r = quantileclip(data, fclip=0.2)
        for v in r[0].flat:            
            self.assertAlmostEqual(v, 4.5)
        #for v in r[1].flat:
        #    self.assertAlmostEqual(v, 0.58333333333333337)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 6)

        r = quantileclip(data, fclip=0.09)
        for v in r[0].flat:
            self.assertAlmostEqual(v, 4.5)
        #for v in r[1].flat:
        #    self.assertAlmostEqual(v, 0.76666666666666672)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 8.2)
            
    def testResults5(self):
        '''Test deviant points are ignored'''
        data = [numpy.array([1.0]) for _ in range(22)]
        data[0][0] = 89
        data[12][0] = -89
        
        r = quantileclip(data, fclip=0.15)
        for v in r[0].flat:
            self.assertAlmostEqual(v, 1.0)
        for v in r[1].flat:
            self.assertAlmostEqual(v, 0.0)
        for v in r[2].flat:
            self.assertAlmostEqual(v, 15.4)        
        
        
      
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(CombineTestCase))
    suite.addTest(unittest.makeSuite(MinMaxTestCase))
    suite.addTest(unittest.makeSuite(QuantileClipTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
    
