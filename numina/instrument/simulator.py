#
# Copyright 2011 Universidad Complutense de Madrid
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

'''
    Components for instrument simulation
'''

import logging

_logger = logging.getLogger('numina.simulator')

__all__ = ['find_recipe']

class Shutter(object):
    def __init__(self, cid=0):
        self.cid = cid
        self.opened = True

    def open(self):
        self.opened = True
        
    def close(self):
        self.opened = False

class CCDDetector(object):
    def __init__(self, description, cid=0):

        self.shape = description.shape
        self.model = description.model
        self.biaslevel = description.bias
        self.dark = description.dark
        self.gain = description.gain
        self.ron = description.ron
        self.buffer = numpy.zeros(self.shape)
        self.ls = 0.0
        self.amplifiers = description.amps
        self.meta = {}
        x = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
        y = [0.0, 0.79, 0.88, 0.92, 0.94, 0.85, 0.82, 0.79, 0.0]
        self.eff = interp1d(x, y)


#    @method(dbus_interface='es.ucm.Pontifex.Instrument.Detector',
#            in_signature='', out_signature='')
    def reset(self):
        return self.i_reset()

    def i_reset(self):
        self.buffer.fill(0)

#    @method(dbus_interface='es.ucm.Pontifex.Instrument.Detector',
#            in_signature='d', out_signature='')
    def expose(self, exposure):
        return self.i_expose(exposure)

    def i_expose(self, exposure):
        now = datetime.datetime.now()
        # Recording time of start of exposure
        self.meta['DATE-OBS'] = now.isoformat()
        self.meta['MDJ-OBS'] = datetime_to_mjd(now)
        #
        #
        #fun = lambda x: self.eff(x) * self.ls.sed(x) * x
        #val,_ = quad(fun, 200.0, 1000.0)
        val = 100.0
        #for i in range(2, 4096, 6):
        #    self.buffer[i:i+4,:] += val * exposure
        self.buffer[2046:2050,:] += val * exposure
        #
        # Poisson noise
        self.buffer = poisson(self.buffer)
        #self.buffer = normal(self.buffer, scale=numpy.sqrt(self.buffer))

        # Dark current    
        self.buffer += self.dark * exposure


    def readout(self):        
        data = self.buffer.copy()
        for amp in self.amplifiers:            
            # Readout noise
            if amp.ron > 0:
                try:
                    data[amp.shape] = normal(self.buffer[amp.shape], amp.ron)
                except Exception as e:
                    _logger.error(str(e))
            # gain
            data[amp.shape] /= amp.gain
        # bias level
        data += self.biaslevel
        data = data.astype('int32')
        # readout destroys data
        self.buffer.fill(0)
        return data

        i_readout = readout

    def illum(self, ls):
        self.ls = ls
        return 0.0

class Grism(object):
    def __init__(self, description, cid=0):
        self.name = description.name

class InstrumentWheel(object):
    def __init__(self, description, cid=0):

        self.cid = cid
        self.fwpos = 0
        self.fwmax = len(description.grisms)

        self.elements = []

        for cid, grism in enumerate(description.grisms):
            el = Grism(grism, cid=cid)
            self.elements.append(el)

#    @method(dbus_interface='es.ucm.Pontifex.Wheel',
#            in_signature='i', out_signature='i')
    def turn(self, position):
        self.fwpos += (position % self.fwmax)
        return self.fwpos


#    @method(dbus_interface='es.ucm.Pontifex.Wheel',
#            in_signature='i', out_signature='i')
    def set_position(self, position):
        self.fwpos = (position % self.fwmax)
        return self.fwpos

    def illum(self, ls):
        return ls

    def current(self):
        return self.elements[self.fwpos]

#    @method(dbus_interface='es.ucm.Pontifex.Wheel',
#            in_signature='', out_signature='o')
    def current_element(self):
        return self.current()._object_path

class CMOSDetector(object):
    '''A generic nIR bidimensional detector.'''
    def __init__(self, shape, gain=1.0, ron=0.0, dark=1.0, 
                 well=65535, pedestal=200., flat=1.0, 
                 resetval=0, resetnoise=0.0):
        self.shape = shape
         
        self._detector = numpy.zeros(self.shape)
        self._gain = numberarray(gain, self.shape)
        self._ron = numberarray(ron, self.shape)
        self._dark = numberarray(dark, self.shape)
        self._dark[self._dark < 0] = 0.0 
        self._pedestal = numberarray(pedestal, self.shape)
        self._well = numberarray(well, self.shape)
        self._flat = numberarray(flat, self.shape)
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
        self._detector += numpy.random.standard_normal(self.shape) * self._reset_noise
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
        result += numpy.random.standard_normal(self.shape) * self._ron
        result += self._pedestal
        # result[result > self._well] = self._well
        return result.astype(self.type)
    
    def data(self):
        '''Return the current content of the detector.'''
        return self._detector
        
    def time_since_last_reset(self):
        return self._time
    
