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
import emir.exceptions as exception

def numberarray(x, size = (5,5)):
    try:
        iter(x)
    except TypeError:
        return numpy.ones(size) * x
    else:
        return x


class DetectorElapseError(exception.Error):
    def __init__(self, txt):
        exception.Error.__init__(self, txt)


class DetectorReadoutError(exception.Error):
    def __init__(self, txt):
        exception.Error.__init__(self, txt)

class Detector:
    def __init__(self, size=(5,5), gain=1.0, ron=0.0, dark=1.0, well=65535, pedestal=200.,flat=1.0, resetval=0,resetnoise=0.0):
        self.__size = size
        self.__detector = numpy.zeros(self.__size)
        self.__gain = numberarray(gain, self.__size)
        self.__ron = numberarray(ron, self.__size)
        self.__dark = numberarray(dark, self.__size)
        self.__pedestal = numberarray(pedestal, self.__size)
        self.__well = numberarray(well, self.__size)
        self.__flat = numberarray(flat, self.__size)
        self.__reset_noise = resetnoise
        self.__reset_value = resetval
        self.__time = 0
        self.readout_time = 0
        self.reset_time = 0
        self.type = 'int16'
        self.outtype = 'int16'
        
    def elapse(self, time, source = None):
        etime = time - self.__time
        if etime < 0:
            raise DetectorElapseError(("Elapsed time is %ss, it's larger than %ss" % (self.__time, time)))
        self.__detector += numpy.random.poisson(self.__dark * etime).astype('float')
        if source is not None:
            self.__detector += numpy.random.poisson(self.__flat * source * etime).astype('float')
        self.__time = time
        
    def reset(self):
        self.__time = 0
        self.__detector[:] = self.__reset_value
        # Considering normal reset noise
        self.__detector += numpy.random.standard_normal(self.__size) * self.__reset_noise
        self.__time += self.reset_time
        
    def read(self, time = None, source = None):
        if time is not None:
            self.elapse(time, source)
        self.__time += self.readout_time
        result = self.__detector.copy()
        result[result < 0] = 0
        # Gain per channel
        result /= self.__gain
        # Readout noise
        result += numpy.random.standard_normal(self.__size) * self.__ron
        result += self.__pedestal
       # result[result > self.__well] = self.__well
        return result.astype(self.type)
    
    def data(self):
        return self.__detector
    
    def size(self):
        return self.__size
    
    def time_since_last_reset(self):
        return self.__time
    
    
class EmirDetector(Detector):
    def __init__(self, size=(5,5), gain=1.0, ron=0.0, dark=1.0, well=65535, pedestal=200.,flat=1.0, resetval=0,resetnoise=0.0):
        Detector.__init__(self, size, gain, ron, dark, well, pedestal, flat, resetval,resetnoise)
        
    def configure(self, options):
        self.options = options
    
    def exposure(self, exposure):
        self.exposure = exposure
        self.events = self.generate_events(exposure)
    
    def path(self, input = None):
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
        raise DetectorReadoutError('Readout mode %s doesn\'t exist' % self.options['mode'])
    
    def events_single(self, exposure):
        return [exposure]
    
    def events_cds(self, exposure):
        return [0,exposure]
    
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
        process_function = getattr(self, "process_%s" % self.options['mode'], self.process_wrong)
        return process_function(images, events)
    
    def process_wrong(self,images, events):
        raise DetectorReadoutError('Readout mode %s doesn\'t exist' % self.options['mode'])
        
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
        mtdt = {'EXPOSED':self.exposure, 'EXPTIME':self.exposure, 'ELAPSED':self.time_since_last_reset(), 'DARKTIME':self.time_since_last_reset(),
        'READMODE':self.options['mode'].upper(), 'READSCHM':self.options['scheme'].upper(),
        'READNUM':self.options['reads'], 'READREPT':self.options['repeat']}
        return mtdt
    
    
    
    
