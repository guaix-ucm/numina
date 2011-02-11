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

import itertools as ito

import numpy # pylint: disable-msgs=E1101

import numina.instrument.detector
from numina.exceptions import DetectorReadoutError

# Classes are new style
__metaclass__ = type

def _channel_gen1(beg, end, step):
    return ito.imap(lambda x: (x, x + step), xrange(beg, end, step))

def _channel_gen2(beg, end, step):
    return ito.imap(lambda x: (x - step, x), xrange(beg, end, -step))

def _ch1():
    return ito.izip(ito.repeat(slice(1024, 2048)), ito.starmap(slice, _channel_gen2(1024, 0, 128)))

def _ch2():  
    return ito.izip(ito.starmap(slice, _channel_gen2(1024, 0, 128)), ito.repeat(slice(0, 1024)))

def _ch3():
    return ito.izip(ito.repeat(slice(0, 1024)), ito.starmap(slice, _channel_gen1(1024, 2048, 128)))

def _ch4():
    return ito.izip(ito.starmap(slice, _channel_gen1(1024, 2048, 128)), ito.repeat(slice(1024, 2048)))

# FIXME: move this to numina
def braid(*iterables):
    '''Return the elements of each iterator in turn until some is exhausted.
    
    This function is similar to the roundrobin example 
    in itertools documentation.
    
    >>> a = iter([1,2,3,4])
    >>> b = iter(['a', 'b'])
    >>> c = iter([1,1,1,1,'a', 'c'])
    >>> d = iter([1,1,1,1,1,1])
    >>> list(braid(a, b, c, d))
    [1, 'a', 1, 1, 2, 'b', 1, 1]
    '''
    buffer = [next(i) for i in iterables]
    while 1:
        for i, idx in ito.izip(iterables, ito.count(0)):
            yield buffer[idx]
            buffer[idx] = next(i, None)
        if any(map(lambda x: x is None, buffer)):
            raise StopIteration

# Channels are listed per quadrant and then in fast readout order
CHANNELS = list(ito.chain(_ch1(), _ch2(), _ch3(), _ch4()))

# Channels in read out order
CHANNELS_READOUT = list(braid(_ch1(), _ch2(), _ch3(), _ch4()))

# Quadrants are listed starting at left-top and counter-clockwise then
QUADRANTS = [(slice(1024, 2048), slice(0, 1024)),
             (slice(0, 1024), slice(0, 1024)),
             (slice(0, 1024), slice(1024, 2048)),
             (slice(1024, 2048), slice(1024, 2048))
             ]

class ReadoutMode:
    def __init__(self, mode, scheme, reads, repeat):
        self.mode = mode
        self.scheme = scheme
        self.reads = reads
        self.repeat = repeat
        
class SingleReadoutMode(ReadoutMode):
    def __init__(self, repeat=1):
        ReadoutMode.__init__(self, 'single', 'perline', 1, repeat)
        
    def events(self, exposure):
        return [exposure]
    
    def process(self, images, events):
        return images[0]
    
class CdsReadoutMode(ReadoutMode):
    def __init__(self, repeat=1):
        ReadoutMode.__init__(self, 'CDS', 'perline', 1, repeat)
        
    def events(self, exposure):
        return [0, exposure]
    
    def process(self, images, events):
        return images[1] - images[0]
    
class FowlerReadoutMode(ReadoutMode):
    def __init__(self, reads, repeat=1):
        ReadoutMode.__init__(self, 'Fowler', 'perline', reads, repeat)
        self.readout_time = 0.0
        
    def events(self, exposure):
        dt = self.readout_time
        nsamples = int(self.reads)
        vreads = [i * dt for i in range(nsamples)]
        vreads += [i * dt + exposure for i in vreads]
        return vreads
    
    def process(self, images, events):
        # Subtracting correlated reads
        nsamples = len(images) / 2
        # nsamples has to be odd
        reduced = numpy.array([images[nsamples + i] - images[i] 
                               for i in range(nsamples)])
        # Final mean
        return reduced.mean(axis=0)

class RampReadoutMode(ReadoutMode):
    def __init__(self, reads, repeat=1):
        ReadoutMode.__init__(self, 'Ramp', 'perline', reads, repeat)
        
    def events(self, exposure):
        nsamples = int(self.reads)
        dt = exposure / (nsamples - 1.)
        return [dt * i for i in range(nsamples)]
    
    def process(self, images, events):
        
        def slope(y, xcenter, varx, time):
            return ((y - y.mean()) * xcenter).sum() / varx * time
        
        events = numpy.asarray(events)
        images = numpy.asarray(images)
        meanx = events.mean()
        sxx = events.var() * events.shape[0]
        xcenter = events - meanx
        images = numpy.dstack(images)
        return numpy.apply_along_axis(slope, 2, images, xcenter, sxx, events[ - 1])
    

class Hawaii2(numina.instrument.detector.Detector):
    '''The EMIR detector.'''
    
    AMP1 = QUADRANTS # 1 amplifier per quadrant
    AMP8 = CHANNELS # 8 amplifiers per quadrant
    
    
    def __init__(self, gain=1.0, ron=0.0, dark=1.0, well=65535,
                 pedestal=200., flat=1.0, resetval=0, resetnoise=0.0,
                 mode='8'):
        super(Hawaii2, self).__init__((2048, 2048), gain, ron, dark, well,
                                           pedestal, flat, resetval, resetnoise)
        
        if mode not in ['1', '8']:
            raise ValueError('mode must be "1" or "8"')
        
        self.mode = mode
        # Amplifier region
        self.amp = self.AMP1 if mode == '1' else self.AMP8
        
        # Gain and RON per amplifier
        self._ron = numpy.asarray(ron)
        self._gain = numpy.asarray(gain)
        
        self.ronmode = SingleReadoutMode()
        
        self.events = None
        
        self._exposure = 0
        
    def read(self, time=None, source=None):
        '''Read the detector.'''
        if time is not None:
            self.elapse(time, source)
        self._time += self.readout_time
        result = self._detector.copy()
        result[result < 0] = 0
        
        # Gain and RON per amplifier        
        ampgain = ito.cycle(self._gain.flat)
        ampron = ito.cycle(self._ron.flat)
        
        for amp, gain, ron in zip(self.amp, ampgain, ampron):
            data = result[amp]
            data /= gain            
            # Readout noise
            data += numpy.random.standard_normal(data.shape) * ron
        
        result += self._pedestal
        # result[result > self._well] = self._well
        return result.astype(self.type)        
        
    def configure(self, ronmode):
        self.ronmode = ronmode
    
    def exposure(self, exposure):
        self._exposure = exposure
        self.events = self.ronmode.events(exposure)
    
    def lpath(self, input_=None):
        self.reset()
        images = [self.read(t, input_) for t in self.events]
        # Process the images according to the mode
        final = self.ronmode.process(images, self.events)
        return final.astype(self.outtype)
     
    def metadata(self):
        '''Return metadata exported by the EMIRDetector.'''
        mtdt = {'EXPOSED':self._exposure, 
                'EXPTIME':self._exposure,
                'ELAPSED':self.time_since_last_reset(),
                'DARKTIME':self.time_since_last_reset(),
                'READMODE':self.ronmode.mode.upper(),
                'READSCHM':self.ronmode.scheme.upper(),
                'READNUM':self.ronmode.reads,
                'READREPT':self.ronmode.repeat}
        return mtdt
