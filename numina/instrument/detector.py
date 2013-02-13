#
# Copyright 2008-2013 Universidad Complutense de Madrid
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

import numpy.random

from numina.treedict import TreeDict
from numina.array import numberarray
from numina.exceptions import DetectorElapseError
from numina.astrotime import datetime_to_mjd

from .base import BaseConectable
from .mapping import Mapper

Channel = namedtuple('Channel', ['shape', 'gain', 'ron', 'wdepth'])

PIXEL_VALID = 0
PIXEL_DEAD = 1
PIXEL_HOT = 2

class DAS(object):
    '''Data Acquisition System.'''
    def __init__(self, detector):
        self.detector = detector
        self._meta = TreeDict()
        
        self.readmode(SingleReadoutMode())

    def readmode(self, mode):
        self.mode = mode
        self._meta['readmode'] = self.mode.mode
        self._meta['readscheme'] = self.mode.scheme
        self._meta['readnum'] = self.mode.reads
        self._meta['readrept'] = self.mode.repeat

    def acquire(self, exposure):
        self._meta['elapsed'] = exposure
        self._meta['darktime'] = exposure
        now = datetime.now()
        self._meta['dateobs'] = now.isoformat()
        self._meta['mjdobs'] = datetime_to_mjd(now)
        
        alldata = []
        events = self.mode.events(exposure)
        
        self.detector.reset()
        
        for time in events:
            self.detector.expose_until(time)
            data = self.detector.readout()
            alldata.append(data)

        return alldata, events

    @property
    def meta(self):
        self._meta['detector'] = self.detector.meta
        return self._meta

class ReadoutMode(object):
    '''Base class for readout modes.'''
    def __init__(self, mode, scheme, reads, repeat):
        self.mode = mode
        self.scheme = scheme
        self.reads = reads
        self.repeat = repeat

class SingleReadoutMode(ReadoutMode):
    def __init__(self, repeat=1):
        super(SingleReadoutMode, self).__init__('single', 'perline', 1, repeat)
        
    def events(self, exposure):
        return [exposure]
    
    def process(self, images, events):
        return images[0]
    
class CdsReadoutMode(ReadoutMode):
    '''Correlated double sampling readout mode.'''
    def __init__(self, repeat=1):
        super(CdsReadoutMode, self).__init__('CDS', 'perline', 1, repeat)
        
    def events(self, exposure):
        '''
        >>> cds_rm = CdsReadoutMode()
        >>> cds_rm.events(10.0)
        [0.0, 10.0]
        '''
        return [0.0, exposure]
    
    def process(self, images, events):
        return images[1] - images[0]
    
class FowlerReadoutMode(ReadoutMode):
    '''Fowler sampling readout mode.'''
    def __init__(self, reads, repeat=1, readout_time=0.0):
        super(FowlerReadoutMode, self).__init__('Fowler', 'perline', reads, repeat)
        self.readout_time = readout_time
        
    def events(self, exposure):
        '''
        
        >>> frm = FowlerReadoutMode(reads=3, readout_time=0.8)
        >>> frm.events(10.0)
        [0.0, 0.8, 1.6, 10.0, 10.8, 11.6]
        '''
        
        dt = self.readout_time
        vreads = [i * dt for i in range(self.reads)]
        vreads += [t + exposure for t in vreads]
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
    '''"Up the ramp" sampling readout mode.'''
    def __init__(self, reads, repeat=1):
        super(RampReadoutMode, self).__init__('Ramp', 'perline', reads, repeat)
        
    def events(self, exposure):
        dt = exposure / (self.reads - 1.)
        return [dt * i for i in range(self.reads)]
    
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

class ArrayDetector(BaseConectable):
    '''A bidimensional detector.'''
    def __init__(self, shape, channels, 
                 bias=100.0, 
                 dark=0.0,
                 flat=1.0,
                 reset_value=0.0, reset_noise=0.0,
                 bad_pixel_mask=None,
                 readout_time=0.0,
                 out_dtype='uint16'):
        
        super(ArrayDetector, self).__init__()
        self.comp_dtype = 'int32'
        self.out_dtype = out_dtype
        self.out_dtype_info = numpy.iinfo(self.out_dtype)
        self.shape = shape
        self.channels = channels
        self.bias = bias
        self.dark = dark
        self.flat = flat
        self.reset_value = reset_value
        self.reset_noise = reset_noise
        self.reset_time = 0.0
        
        self.readout_time = readout_time
        self._last_read = 0.0
        self.bad_pixel_mask = bad_pixel_mask
        
        self.buffer = numpy.zeros(self.shape, dtype=self.comp_dtype)

        self.mapper = Mapper(shape)
        
        self.dead_pixel_value = self.out_dtype_info.min
        self.hot_pixel_value = self.out_dtype_info.max

        self.meta = TreeDict()
        
    def readout(self):
        '''Read the detector.'''
        data = self.buffer.copy()
        data[data < 0] = 0
        source = self.mapper.sample(self.source)
        source *= self.flat * self.readout_time
        data += numpy.random.poisson((self.dark + source) * self.reset_time)
        for amp in self.channels:
            if amp.ron > 0:
                data[amp.shape] = numpy.random.normal(self.buffer[amp.shape], amp.ron)
            data[amp.shape] /= amp.gain
        data += self.bias

        numpy.clip(data, self.out_dtype_info.min, self.out_dtype_info.max, out=data)

        data = data.astype(self.out_dtype)

        if self.bad_pixel_mask is not None:
            # processing badpixels:
            data = numpy.where(self.bad_pixel_mask == PIXEL_DEAD, self.dead_pixel_value, data)
            data = numpy.where(self.bad_pixel_mask == PIXEL_HOT, self.hot_pixel_value, data)

        self._last_read += self.readout_time

        return data
    
    def time_since_last_reset(self):
        return self._last_read

    def reset(self):
        '''Reset the detector.'''
        self.buffer.fill(self.reset_value)
        self._last_read = self.reset_time
        if self.reset_noise > 0.0:
            self.buffer += numpy.random.normal(self.shape) * self.reset_noise
            
    def expose_until(self, time):
        '''
        :raises: DetectorElapseError
        '''
        dt = time - self._last_read
        if dt < 0:
            msg = "Elapsed time is %ss, it's larger than %ss" % (self._last_read, time)
            raise DetectorElapseError(msg)
        self.expose(dt)

    def expose(self, dt):
        self._last_read += dt

            
class CCDDetector(ArrayDetector):
    def __init__(self, shape, channels, bias=100, dark=0.0, bad_pixel_mask=None):
        super(CCDDetector, self).__init__(shape, channels,
                    bias=bias, dark=dark, bad_pixel_mask=bad_pixel_mask)

        self.meta['readmode'] = 'fast'
        self.meta['readscheme'] = 'perline'

    def expose(self, dt, source=None):
        now = datetime.now()
        # Recording time of start of exposure
        self.meta['exposed'] = dt
        self.meta['dateobs'] = now.isoformat()
        self.meta['mjdobs'] = datetime_to_mjd(now)

        self._last_read += dt
        
        source = self.mapper.sample(self.source)
        source *= self.flat
        self.buffer += numpy.random.poisson((self.dark + source) * dt, 
                               size=self.shape).astype('float')
          
    def readout(self):
        '''Read the CCD detector.'''
        self._last_read += self.readout_time
        result = super(CCDDetector, self).readout()
        self.reset()
        return result

class nIRDetector(ArrayDetector):
    '''A generic nIR bidimensional detector.'''
    
    def __init__(self, shape, channels, dark=0.0, 
                 pedestal=0.0, flat=1.0, bad_pixel_mask=None,
                 resetval=1000, resetnoise=0.0):
        super(nIRDetector, self).__init__(shape, 
                    channels, 
                    pedestal, 
                    dark=dark,
                    flat=flat,
                    bad_pixel_mask=bad_pixel_mask)
        
        self.readout_time = 0
        self.reset_time = 0        

    def expose(self, dt, source=None):

        self._last_read += dt
        source = self.mapper.sample(self.source)
        source *= self.flat 
        sint = self.dark * numpy.ones(self.shape, dtype=self.comp_dtype) + source
        self.buffer += numpy.random.poisson(sint * dt).astype('float')
        

        
if __name__ == '__main__':
    
    
    shape = (10, 10)
    ampshape = (slice(0, 10), slice(0, 10))
    a = Amplifier(ampshape, 2.1, 4.5, 65000)
    det = CCDDetector(shape, [a], bias=100, dark=20.0)
    
    das1 = DAS(det)
    
    
    #print d
        
    ndet = nIRDetector(shape, [a], dark=20.0, pedestal=100)
    ndet.readout_time = 0.12
    das2 = DAS(ndet)
    mode = FowlerReadoutMode(5, 1, readout_time=ndet.readout_time)
    das2.readmode(mode)
    pp = das2.acquire(10)
    for key in das2.meta:
        if key == 'detector':
            for k in das2.meta['detector']:
                print 'detector.',k
                
        else:
            print key, das2.meta[key]
    print pp
    
