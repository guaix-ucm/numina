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
from numina import Null

__version__ = '$Revision: 411 $'

class NullTestCase(unittest.TestCase):
    '''A test of the Null class.'''
    def setUp(self):
        '''Set up TestCase.'''
        self.null = Null()
    
    def testIgnoreInitArgs(self):
        '''Null ignores the arguments of __init__.'''
        a = Null()
        a = Null(key1="2", key2=[])
        a = Null("", [], (), nkey1="2", nkey2=[])
    
    
    def testIgnoreCallArgs(self):
        '''Null ignores the arguments of __call__.'''
        self.assertEqual(self.null, self.null())
        self.assertEqual(self.null, self.null(key1="2", key2=[]))
        self.assertEqual(self.null, self.null("", [], (),
                                              nkey1="2", nkey2=[]))
        
        
    def testIgnoreGetAttr(self):
        '''Null ignores attribute getting.'''
        self.assertEqual(self.null, self.null.prueba)
        self.assertEqual(self.null, self.null.otro)
        self.assertEqual(self.null, self.null.metodo)
    
    def testIgnoreSetAttr(self):
        '''Null ignores attribute getting.'''        
        for i in [[], "44f342", 1234, (4.5, [])]:
            self.null.atributo = i
            self.assertEqual(self.null, self.null.atributo)

    def testIgnoreDelAttr(self):
        '''Null ignores attribute deleting.'''        
        for i in [[], "44f342", 1234, (4.5, [])]:
            self.null.atributo = i
            del self.null.atributo
            self.assertEqual(self.null, self.null.atributo)

    def testRepr(self):
        '''Null repr is <Null>'''
        self.assertEqual(repr(self.null), "<Null>")
            
    def testStr(self):
        '''Null srt is Null'''
        self.assertEqual(str(self.null), "Null")
        
            
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(NullTestCase))
    return suite

if __name__ == '__main__':
    # unittest.main(defaultTest='test_suite')
    unittest.TextTestRunner(verbosity=2).run(test_suite())
