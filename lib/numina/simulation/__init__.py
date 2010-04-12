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

import os
from os.path import join as pjoin
from cPickle import dump, load
import logging
import math

import numpy
import scipy.stats.mvn as mvn

from numina.array import subarray_match

# Classes are new style
__metaclass__ = type

_logger = logging.getLogger("numina.storage")

class RunCounter:
    '''Run number counter'''
    def __init__(self, template, suffix='.fits',
                 dir=None, last=1):
        self.template = template
        self.suffix = suffix
        self.last = last
        if dir is not None:
            self.complete = pjoin(dir, self.template + self.suffix)
        else:
            self.complete = self.template + self.suffix
           
    def runstring(self):
        '''Return the run number and the file name.'''
        run = self.template % self.last
        cfile = self.complete % self.last
        self.last += 1
        return (run, cfile)

class PersistentRunCounter:
    '''Persistent run number counter'''
    def __init__(self, template, suffix='.fits',
                 dir=None, pstore='index.pkl', last=1):
        self.template = template
        self.suffix = suffix
        self.pstore = pstore
        self.last = last
        if dir is not None:
            self.complete = pjoin(dir, self.template + self.suffix)
            if not os.access(dir, os.F_OK):
                os.mkdir(dir)
        else:
            self.complete = self.template + self.suffix
        try:
            pkl_file = open(self.pstore, 'rb')
            try:                
                self.last = load(pkl_file)
            finally:
                pkl_file.close() 
        except IOError, strrerror:            
            _logger.error(strrerror)
                
    def store(self):
        try:
            pkl_file = open(self.pstore, 'wb')
            try:                
                dump(self.last, pkl_file)
            finally:
                pkl_file.close()
        except IOError, strrerror:            
            _logger.error(strrerror)
            
    def runstring(self):
        '''Return the run number and the file name.'''
        run = self.template % self.last
        cfile = self.complete % self.last
        self.last += 1
        return (run, cfile)
        
class Profile:
    '''Base class for profiles'''
    def __init__(self, center):
        self.center = numpy.asarray(center)
        self.offset = self.center.astype('int')
        self.peak = self.center - self.offset

    def array(self):
        y, x = numpy.indices(self.shape)
        return self.area(x, y)

    def area(self, x ,y):
        return numpy.ones(self.shape)

class GaussProfile(Profile):
    '''A N-dimensional Gaussian profile.'''
    def __init__(self, center, covar, scale=5):
        super(GaussProfile, self).__init__(center)
        # TODO check that center and covar are compatible
        
        self.covar = numpy.asarray(covar)
        halfsize = numpy.round(scale * numpy.sqrt(covar.diagonal())).astype('int')
        self.selfcenter = self.peak +  halfsize
        self._shape = tuple(2 * halfsize + 1)
        self.density = self.mvnu(self.selfcenter, self.covar)
        vfun = numpy.vectorize(self.density)
        self._kernel = numpy.fromfunction(vfun, self._shape)
        
    @classmethod
    def mvnu(self, means, covar):
        def myfun(*args):
            low = numpy.asarray(args)
            up = low + 1
            v, _i = mvn.mvnun(lower=low, upper=up, means=means, covar=covar)
            return v
    
        return myfun
    
    @property
    def kernel(self):
        '''An array representing the Gaussian kernel.'''
        return self._kernel
    
    @property
    def shape(self):
        '''Shape of the Gaussian kernel.'''
        return self._shape

#class SlitProfile(Profile):
#    ''' A rectangular slit'''
#    def __init__(self, blc, urc):
#        self.urc = numpy.array(urc)
#        self.blc = numpy.array(blc)
#        self.eurc = self.urc.astype('int')
#        self.eblc = self.blc.astype('int')
#        self.lurc = self.urc - self.eblc
#        self.lblc = self.blc - self.eblc
#        Profile.__init__(self, shape=tuple(self.eurc - self.eblc + 1), offset=self.eblc)
#    def area(self, x, y):                
#        v = minimum(x + 1, self.lurc[1]) - maximum(x, self.lblc[1])
#        w = minimum(y + 1, self.lurc[0]) - maximum(y, self.lblc[0])
#        return v * w

def add_profile(im, profile, intensity=1000.):
    '''Add the profile into a given array.'''
    ss = profile.kernel
    i,j = subarray_match(im.shape, profile.offset, ss.shape)
    im[i] += intensity * ss[j]
    return im

def add_gaussian(im, center, covar, scale=5, intensity=1000):
    '''Add a Gaussian profile using add_profile'''
    profile = GaussProfile(center=center, covar=covar, scale=scale)
    return add_profile(im, profile, intensity=intensity)
