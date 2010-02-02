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

'''A module used to run all the numina unit tests'''

__version__ = '$Revision: 411 $'

import sys, os
import unittest


def print_info():
    '''Print information about the system and emir version.'''
    print ''
    print 'python version', sys.version
    print 'my pid', os.getpid()
    sys.stdout.flush()
    
class PrintInfoFakeTest(unittest.TestCase):
    '''Test case that calls print_info if the module is called as an unittest.'''
    def testPrintVersions(self):
        print_info()

def suite():
    test_modules = ['test_recipe', 'test_methods', 'test_image']
    alltests = unittest.TestSuite()
    for name in test_modules:
        m = __import__("numina.tests." + name, globals(), locals(), [""], -1)
        alltests.addTest(m.test_suite())
    return alltests

def test_suite():
    msuite = unittest.TestSuite()
    msuite.addTest(unittest.makeSuite(PrintInfoFakeTest))
    return msuite

if __name__ == '__main__':
    print_info()
    unittest.main(defaultTest='suite')