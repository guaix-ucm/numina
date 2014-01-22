#
# Copyright 2008-2014 Universidad Complutense de Madrid
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

from numina.generic import generic

class GenericTestCase(unittest.TestCase):
    
    def test_generic_is_callable(self):
        '''Test if decorated function is callable.'''
        
        @generic
        def testfunc(obj):
            return obj
        
        self.assertTrue(callable(testfunc))
        
        
    def test_register(self):
        '''Test if register adds new types.'''
        
        @generic
        def testfunc(obj):
            return 0
        
        class B(object): 
            value = 1
        
        @testfunc.register(B)
        def test_b(obj):
            return obj.value
        
        b = B()
        self.assertEqual(testfunc(b), 1)
        
    def test_unregister(self):
        '''Test if unregister removes existing types.'''
        
        @generic
        def testfunc(obj):
            return 0
        
        class B(object): 
            value = 1
        
        @testfunc.register(B)
        def test_b(obj):
            return obj.value
        
        
        testfunc.unregister(B)
        b = B()
        c = testfunc(b)
        self.assertEqual(c, 0)
        
    def test_inheritance(self):
        '''Test if generic follows the inheritance graph.'''
        
        @generic
        def testfunc(obj):
            return obj
        
        class B(object): 
            value = 1
        
        @testfunc.register(B)
        def test_b(obj):
            return obj.value

        class C(B):
            value = 2
    
        c = C()
        d = testfunc(c)
        self.assertEqual(d, 2)

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(GenericTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
