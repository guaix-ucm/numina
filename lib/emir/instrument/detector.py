#
# Copyright 2008-2009 Sergio Pascual
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

import numpy
import numpy.random

from emir.exceptions import DetectorElapseError, DetectorReadoutError

__version__ = "$Id$"

# Classes are new style
__metaclass__ = type

def numberarray(x, shape = (5,5)):
    try:
        iter(x)
    except TypeError:
        return numpy.ones(shape) * x
    else:
        return x


class Detector:
    def __init__(self, shape=(5,5), gain=1.0, ron=0.0, dark=1.0, 
                 well=65535, pedestal=200.,flat=1.0, resetval=0, resetnoise=0.0):
        self._shape = shape
        self._detector = numpy.zeros(self._shape)
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
        etime = time - self._time
        if etime < 0:
            msg = "Elapsed time is %ss, it's larger than %ss" % (self._time, time)
            raise DetectorElapseError(msg)
        self._detector += numpy.random.poisson(self._dark * etime).astype('float')
        if source is not None:
            self._detector += numpy.random.poisson(self._flat * source * etime).astype('float')
        self._time = time
        
    def reset(self):
        self._time = 0
        self._detector[:] = self._reset_value
        # Considering normal reset noise
        self._detector += numpy.random.standard_normal(self._shape) * self._reset_noise
        self._time += self.reset_time
        
    def read(self, time=None, source=None):
        if time is not None:
            self.elapse(time, source)
        self._time += self.readout_time
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
        return self._detector
    
    def size(self):
        return self._shape
    
    def time_since_last_reset(self):
        return self._time
    
    
class EmirDetector(Detector):
    def __init__(self, shape=(5, 5), gain=1.0, ron=0.0, dark=1.0, well=65535, 
                 pedestal=200.,flat=1.0, resetval=0, resetnoise=0.0):
        super(EmirDetector, self).__init__(shape, gain, ron, dark, well, 
                                           pedestal, flat, resetval, resetnoise)
        
    def configure(self, options):
        self.options = options
    
    def exposure(self, exposure):
        self.exposure = exposure
        self.events = self.generate_events(exposure)
    
    def path(self, input=None):
        self.reset()
        images = [self.read(t, input) for t in self.events]
        # Process the images according to the mode
        final = self.process(images, self.events)
        return final.astype(self.outtype)

    def generate_events(self, exposure):
        # generate read events according to the mode        
        events_function = getattr(self, "events_%s" % self.options['mode'], self.events_wrong)
        return events_function(exposure)    
    
    def events_wrong(self, exposure):
        msg = 'Readout mode %s doesn\'t exist' % self.options['mode']
        raise DetectorReadoutError(msg)
    
    def events_single(self, exposure):
        return [exposure]
    
    def events_cds(self, exposure):
        return [0, exposure]
    
    def events_fowler(self, exposure):
        dt = self.readout_time
        nsamples = int(self.options['reads'])
        reads = [i * dt for i in range(nsamples)]
        reads += [i * dt + exposure for i in reads]
        return reads
    
    def events_ramp(self, exposure):
        nsamples = int(self.options['reads'])
        dt = exposure / (nsamples - 1.)
        return [dt * i for i in range(nsamples)]
    
    def process(self, images, events):
        process_function = getattr(self, "process_%s" % self.options['mode'], 
                                   self.process_wrong)
        return process_function(images, events)
    
    def process_wrong(self, images, events):
        msg = "Readout mode %s doesn't exist" % self.options['mode']
        raise DetectorReadoutError(msg)
        
    def process_ramp(self, images, events):
        def slope(y, xcenter, varx, time):
            return ((y - y.mean())*xcenter).sum() / varx * time
        
        events = numpy.array(events)
        images = numpy.array(images)
        meanx = events.mean()
        sxx = events.var() * events.shape[0]
        xcenter = events - meanx
        images = numpy.dstack(images)
        return numpy.apply_along_axis(slope, 2, images, xcenter, sxx, events[-1])
    
    def process_single(self, images, events):
        return images[0]
    
    def process_cds(self, images, events):
        return images[1] - images[0]
    
    def process_fowler(self, images, events):
         # Subtracting correlated reads
         nsamples = len(images) / 2
         # nsamples has to be odd
         reduced = numpy.array([images[nsamples + i] - images[i] for i in range(nsamples)])
         # Final mean
         return reduced.mean(axis=0)     
    
    def metadata(self):
        mtdt = {'EXPOSED':self.exposure, 'EXPTIME':self.exposure, 
                'ELAPSED':self.time_since_last_reset(), 
                'DARKTIME':self.time_since_last_reset(),
                'READMODE':self.options['mode'].upper(), 
                'READSCHM':self.options['scheme'].upper(),
                'READNUM':self.options['reads'], 
                'READREPT':self.options['repeat']}
        return mtdt
    
    
    
    
