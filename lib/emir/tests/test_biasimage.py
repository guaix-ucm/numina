#
# Copyright 2008-2011 Sergio Pascual
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

import datetime
import unittest

import numpy

from numina.simulation import run_counter
from emir.instrument.detector import EmirDetector, CdsReadoutMode
from emir.dataproducts import create_raw

# Classes are new style
__metaclass__ = type

class BiasImageTestCase(unittest.TestCase):
    '''Test case of the EMIR bias image recipe.'''
    def setUp(self):
        # Create some 'bias' images
        

        self.detector = EmirDetector()
                
        self._repeat = 1
        romode = CdsReadoutMode()
        self.detector.configure(romode)
        
        self.input_ = numpy.zeros(self.detector.shape())
        self.detector.exposure(0.0)
        
        self.runcounter = run_counter("r%05d")
        
        
        self.nimages = 10
        
        def image_generator(upto):
            i = 0
            while i < upto:
                output = self.detector.lpath(self.input_)
                run = self.runcounter.next()
                now = datetime.datetime.now()
                nowstr = now.strftime('%FT%T')
                headers = {'RUN': run, 'DATE': nowstr, 'DATE-OBS':nowstr}
                headers.update(self.detector.metadata())
                hdulist = create_raw(output, headers)
                yield hdulist
                i += 1
        
        self.images = list(image_generator(self.nimages))
        

    def tearDown(self):
        pass
    
    def test_combine(self):
        '''The result bias is compatible with its inputs.'''
        

def test_suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(BiasImageTestCase))
    return suite

if __name__ == '__main__':
    unittest.main(defaultTest='test_suite')