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

import scipy

from numina.simulation import run_counter
from emir.instrument.detector import Hawaii2
from emir.dataproducts import create_raw

# Classes are new style
__metaclass__ = type

class BiasImageTestCase(unittest.TestCase):
    '''Test case of the EMIR bias image recipe.'''
    def setUp(self):
        # Create some 'bias' images
        detector_conf = {'ron': 1, 'dark': 1, 'gain':1, 'flat':1, 'well': 65000}
        detector_conf['shape'] = (100, 100)

        self.detector = Hawaii2(**detector_conf)
    
        readout_opt = {'exposure':0, 'reads':1, 'repeat':1,
                       'mode':'fowler', 'scheme':'perline'}
                
        self._repeat = 1

        self.detector.configure(readout_opt)
        
        self.input_ = scipy.zeros(detector_conf['shape'])
        self.detector.exposure(readout_opt['exposure'])
        
        self.runcounter = run_counter("r%05d")
        
        
        self.nimages = 100
        
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