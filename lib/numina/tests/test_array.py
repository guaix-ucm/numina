#
# Copyright 2008-2010 Sergio Pascual
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

import numina.array as array

class ArrayTestCase(unittest.TestCase):
    
    def test_subarray_match(self):
        '''Test subarray_match'''
        
        # Shapes don't intersect
        minor, major = array.subarray_match((3,4,5), (10,10,10), (2,2,2))
        
        #  Returns None if shapes don't intersect
        self.assertTrue(minor is None,)
        self.assertTrue(major is None)
        
        small = 100
        
        # One is contained inside the other
        minor, major = array.subarray_match((2048, 2048), (0,0), (small, small))
        # If one array is inside the other, the outputs is equal to the small array
        self.assertEqual(minor, (slice(0, small, None), slice(0, small, None)))
        self.assertEqual(major, (slice(0, small, None), slice(0, small, None)))
        
        # One is contained inside the other, with offsets
        minor, major = array.subarray_match((2048, 2048), (30, 40), (small, small))
        # If one array is inside the other, the outputs is equal to the small array
        self.assertEqual(minor, (slice(30, small + 30, None), slice(40, small + 40, None)))
        self.assertEqual(major, (slice(0, small, None), slice(0, small, None)))
        
        # Both are equal, with offsets
        minor, major = array.subarray_match((100, 100), (30, 40), (small, small))
        # If one array is inside the other, the outputs is equal to the small array
        self.assertEqual(minor, (slice(30, small, None), slice(40, small, None)))
        self.assertEqual(major, (slice(0, small - 30, None), slice(0, small - 40, None)))
        
        # Equal offsets in both sides
        minor, major = array.subarray_match((100, 100), (30, 40), (100, 100), (30, 40))
        # If one array is inside the other, the outputs is equal to the small array
        self.assertEqual(minor, (slice(0, small, None), slice(0, small, None)))
        self.assertEqual(major, (slice(0, small, None), slice(0, small, None)))
        
        # Different offsets in both sides
        minor, major = array.subarray_match((100, 100), (31, 42), (100, 100), (10, 20))
        # If one array is inside the other, the outputs is equal to the small array
        self.assertEqual(minor, (slice(21, small, None), slice(22, small, None)))
        self.assertEqual(major, (slice(0, 79, None), slice(0, 78, None)))
        
        # If we interchange the arrays and change the sign of the offset,
        # we get the same result
        minor, major = array.subarray_match((100, 100), (10, 20), (200, 100))
        cminor, cmajor = array.subarray_match((200, 100), (-10, -20), (100, 100))
        self.assertEqual(cminor, major)
        self.assertEqual(cmajor, minor)
        
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(ArrayTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')