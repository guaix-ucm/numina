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
import emir.image._combine as c

class CombineMethodTestCase(unittest.TestCase):
    function = None
    multiValues = None
    def setUp(self):
        self.emptyResult = (0,0,0)
        self.emptyInput = []
        self.oneInput = [2.0]
        self.oneResult = (self.oneInput[0], 0.0, 1)

    def tearDown(self):
        pass
    def test01Mean(self):
        self.assertEqual(self.function(self.emptyInput), self.emptyResult)
    def test02Mean(self):
        result=self.function(self.oneInput)
        self.assertAlmostEqual(result[0], self.oneResult[0])
        self.assertAlmostEqual(result[1], self.oneResult[1])
        self.assertEqual(result[2], self.oneResult[2])
    def test03Mean(self):
        for i in self.multiValues:
            result = self.function(i[0])
            self.assertAlmostEqual(result[0], i[1][0])
            self.assertAlmostEqual(result[1], i[1][1])
            self.assertEqual(result[2], i[1][2])

class MeanTestCase(CombineMethodTestCase):
    function = c.method_mean
    multiValues = [
                   ([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1], 
                    (4.5999999999999996, 9.1666666666666661, 10)),
                   ([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1], 
                    (4.5999999999999996, 9.1666666666666661, 10))
                   ]

class MedianTestCase(CombineMethodTestCase):
    function = c.method_median
    multiValues = [
                   ([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1], 
                    (4.1, 0, 10)),
                   ([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1], 
                    (4.1, 0, 10))
                   ]


def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(MeanTestCase))
    suite.addTest(unittest.makeSuite(MedianTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')