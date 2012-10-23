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

from numina.array.nirproc import ramp_array

class FollowUpTheRampTestCase(unittest.TestCase):
    def setUp(self):
        self.ramp = numpy.empty((1, 1, 10))
        self.sdt = 1.0
        self.sgain = 1.0
        self.sron = 1.0
    
    def test_exception(self):
        self.assertRaises(ValueError, ramp_array, self.ramp, self.sdt, -1.0, self.sron)
        self.assertRaises(ValueError, ramp_array, self.ramp, self.sdt, 0, self.sron)
        self.assertRaises(ValueError, ramp_array, self.ramp, self.sdt, self.sgain, -1.0)
        self.assertRaises(ValueError, ramp_array, self.ramp, self.sdt, self.sgain, self.sron, nsig=-1.0)
        self.assertRaises(ValueError, ramp_array, self.ramp, self.sdt, self.sgain, self.sron, nsig=0)
        self.assertRaises(ValueError, ramp_array, self.ramp, -1.0, self.sgain, self.sron, nsig=-1.0)
        self.assertRaises(ValueError, ramp_array, self.ramp, 0.0, self.sgain, self.sron, nsig=-1.0)
        self.assertRaises(ValueError, ramp_array, self.ramp, self.sdt, self.sgain, self.sron, saturation=-100)
        self.assertRaises(ValueError, ramp_array, self.ramp, self.sdt, self.sgain, self.sron, saturation=0)
                
def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(FollowUpTheRampTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')
