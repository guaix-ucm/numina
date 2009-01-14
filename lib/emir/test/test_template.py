#
# Copyright 2008 Sergio Pascual
# 
# This file is part of PyEmir
# 
# PyMilia is free software: you can redistribute it and/or modify
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

class SimpleTestCase(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test01Simple(self):
        self.assertAlmostEqual(3.0,3.0)
    def test02Simple(self):
        self.assertEqual([1,2],[1,2])


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(SimpleTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')