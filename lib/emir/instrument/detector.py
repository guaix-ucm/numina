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


import numpy

import numina.instrument.detector
from numina.exceptions import DetectorReadoutError

# Classes are new style
__metaclass__ = type


class EmirDetector(numina.instrument.detector.Detector):
    '''The EMIR detector.'''
    def __init__(self, shape=(5, 5), gain=1.0, ron=0.0, dark=1.0, well=65535,
                 pedestal=200., flat=1.0, resetval=0, resetnoise=0.0):
        super(EmirDetector, self).__init__(shape, gain, ron, dark, well,
                                           pedestal, flat, resetval, resetnoise)
        self.events = None
        self.options = None
        self._exposure = 0
        
    def configure(self, options):
        self.options = options
    
    def exposure(self, exposure):
        self._exposure = exposure
        self.events = self.generate_events(exposure)
    
    def lpath(self, input_=None):
        self.reset()
        images = [self.read(t, input_) for t in self.events]
        # Process the images according to the mode
        final = self.process(images, self.events)
        return final.astype(self.outtype)

    def generate_events(self, exposure):
        # generate read events according to the mode        
        events_function = getattr(self, "events_%s" % self.options['mode'],
                                  self.events_wrong)
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
        return numpy.apply_along_axis(slope, 2, images, xcenter, sxx, events[ - 1])
    
    def process_single(self, images, events):
        return images[0]
    
    def process_cds(self, images, events):
        return images[1] - images[0]
    
    def process_fowler(self, images, events):
        # Subtracting correlated reads
        nsamples = len(images) / 2
        # nsamples has to be odd
        reduced = numpy.array([images[nsamples + i] - images[i] 
                               for i in range(nsamples)])
        # Final mean
        return reduced.mean(axis=0)     
    
    def metadata(self):
        '''Return metadata exported by the EMIRDetector.'''
        mtdt = {'EXPOSED':self._exposure, 'EXPTIME':self._exposure,
                'ELAPSED':self.time_since_last_reset(),
                'DARKTIME':self.time_since_last_reset(),
                'READMODE':self.options['mode'].upper(),
                'READSCHM':self.options['scheme'].upper(),
                'READNUM':self.options['reads'],
                'READREPT':self.options['repeat']}
        return mtdt
    
    
    
    
