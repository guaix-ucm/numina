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
        self.null = Null()

    def tearDown(self):
        pass
    
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
            
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(NullTestCase))
    return suite

if __name__ == '__main__':
    # unittest.main(defaultTest='test_suite')
    unittest.TextTestRunner(verbosity=2).run(test_suite())
