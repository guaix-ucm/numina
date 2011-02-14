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

import scipy
import numpy.random

from numina.array import numberarray
from numina.exceptions import DetectorElapseError

# Classes are new style
__metaclass__ = type

class Detector:
    '''A generic nIR bidimensional detector.'''
    def __init__(self, shape, gain=1.0, ron=0.0, dark=1.0, 
                 well=65535, pedestal=200., flat=1.0, 
                 resetval=0, resetnoise=0.0):
        self._shape = shape
        self._detector = scipy.zeros(self._shape)
        self._gain = numberarray(gain, self._shape)
        self._ron = numberarray(ron, self._shape)
        self._dark = numberarray(dark, self._shape)
        self._pedestal = numberarray(pedestal, self._shape)
        self._well = numberarray(well, self._shape)
        self._flat = numberarray(flat, self._shape)
        self._reset_noise = resetnoise
        self._reset_value = resetval
        self._time = 0
        
        self.readout_time = 0
        self.reset_time = 0
        self.type = 'int16'
        self.outtype = 'int16'
        
    def elapse(self, time, source=None):
        '''
        :raises: DetectorElapseError
        '''
        etime = time - self._time
        if etime < 0:
            msg = "Elapsed time is %ss, it's larger than %ss" % (self._time, time)
            raise DetectorElapseError(msg)
        self._detector += numpy.random.poisson(self._dark * etime).astype('float')
        if source is not None:
            self._detector += numpy.random.poisson(self._flat * source * etime).astype('float')
        self._time = time
        
    def reset(self):
        '''Reset the detector.'''
        self._time = 0
        self._detector[:] = self._reset_value
        # Considering normal reset noise
        self._detector += numpy.random.standard_normal(self._shape) * self._reset_noise
        self._time += self.reset_time
        
    def read(self, time=None, source=None):
        '''Read the detector.'''
        if time is not None:
            self.elapse(time, source)
        self._time += self.readout_time
        # Read is non destructive
        result = self._detector.copy()
        result[result < 0] = 0
        # Gain per channel
        result /= self._gain
        # Readout noise
        result += numpy.random.standard_normal(self._shape) * self._ron
        result += self._pedestal
        # result[result > self._well] = self._well
        return result.astype(self.type)
    
    def data(self):
        '''Return the current content of the detector.'''
        return self._detector
    
    def shape(self):
        '''Return the shape of the detector.'''
        return self._shape
    
    def time_since_last_reset(self):
        return self._time
