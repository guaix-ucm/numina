#
# Copyright 2008-2011 Universidad Complutense de Madrid
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

from collections import namedtuple
from datetime import datetime

import numpy
from numpy.random import normal, poisson

from numina.treedict import TreeDict

Amplifier = namedtuple('Amplifier', ['shape', 'gain', 'ron', 'wdepth'])


class CCDDetector(object):
    def __init__(self, shape, amplifiers, bias=100, dark=0.0):        
        self.shape = shape
        self.amplifiers = amplifiers
        self.bias = bias
        self.dark = dark
                
        self.buffer = numpy.zeros(self.shape)
        
        self.light_source = None

        self.meta = TreeDict()
        self.meta['readmode'] = 'fast'
        self.meta['readscheme'] = 'perline'
        self.meta['exposed'] = 0
        self.meta['gain'] = 3.6
        self.meta['readnoise'] = 2.16

    def reset(self):
        self.buffer.fill(0)

    def expose(self, exposure):
        now = datetime.now()
        # Recording time of start of exposure
        self.meta['exposed'] = exposure
        self.meta['dateobs'] = now.isoformat()
        self.meta['mjdobs'] = 1 # datetime_to_mjd(now)

#        if self.light_source is not None:
#            self.buffer += self.light_source * exposure

        self.buffer = poisson(self.buffer)
        self.buffer += self.dark * exposure

    def readout(self):
        data = self.buffer.copy()
        for amp in self.amplifiers:
            if amp.ron > 0:
                data[amp.shape] = normal(self.buffer[amp.shape], amp.ron)
            data[amp.shape] /= amp.gain
        data += self.bias
        data = data.astype('int32')
        # readout destroys data
        self.buffer.fill(0)
        return data

    def mode(self, name):
        self.meta['readmode'] = name
    
