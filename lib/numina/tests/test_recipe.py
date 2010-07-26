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
 
'''Unit test for RecipeBase.'''

import unittest
from numina.recipes import RecipeBase

class Test(RecipeBase):
    '''Minimal class that implements RecipeBase.'''
    def __init__(self, param, runinfo):
        super(Test, self).__init__(param, runinfo)
        
    def run(self):
        return {}


class RecipeTestCase(unittest.TestCase):
    '''Test of the Recipebase class.'''
    def setUp(self):
        '''Set up TestCase.'''
        self.rc = Test({}, {})
        
    def test1(self):
        self.assertTrue(callable(self.rc))
            
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(RecipeTestCase))
    return suite

if __name__ == '__main__':
    # unittest.main(defaultTest='test_suite')
    unittest.TextTestRunner(verbosity=2).run(test_suite())
