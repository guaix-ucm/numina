
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

'''Simulation package'''

import math

import numpy
from scipy.special import erf
from scipy import maximum, minimum

from emir.image import subarray_match

__version__ = "$Revision$"

# Classes are new style
__metaclass__ = type

class Profile:
    '''Base class for 2D profiles'''
    def __init__(self, shape, offset):
        self.shape = shape
        # Offset corresponds to the point [0,0] of array
        self.offset = offset
    def array(self):
        y, x = numpy.indices(self.shape)
        return self.area(x, y)
    def area(self, x ,y):
        return numpy.ones(self.shape)

class GaussProfile(Profile):
    '''Simetrical gaussian profile'''
    M_SQRT1_2 = math.sqrt(1. / 2.) 
    def __init__(self, sigma, center, scale=5):       
        self.center = numpy.array(center)
        self.sigma = sigma
        l = int(round(scale * sigma))
        # Offset corresponds to the point [0,0] of array
        offset = self.center.astype('int') - numpy.array([l,l]) 
        Profile.__init__(self, shape=(2 * l + 1, 2 * l + 1), offset=offset)
    @staticmethod
    def _compute1(v, center, sigma):
        return GaussProfile.M_SQRT1_2 * (v - center) / sigma
    @staticmethod
    def _compute2(v, center, sigma):
        tipV1 = GaussProfile._compute1(v, center, sigma)
        tipV2 = GaussProfile._compute1(v + 1, center, sigma) 
        return 0.5 * (erf(tipV2) - erf(tipV1))
    def area(self, x, y):      
        v = GaussProfile._compute2(x, self.center[1] - self.offset[1], self.sigma)
        w = GaussProfile._compute2(y, self.center[0] - self.offset[0], self.sigma)
        return  v * w

class SlitProfile(Profile):
    ''' A rectangular slit'''
    def __init__(self, blc, urc):
        self.urc = numpy.array(urc)
        self.blc = numpy.array(blc)
        self.eurc = self.urc.astype('int')
        self.eblc = self.blc.astype('int')
        self.lurc = self.urc - self.eblc
        self.lblc = self.blc - self.eblc
        Profile.__init__(self, shape=tuple(self.eurc - self.eblc + 1), offset=self.eblc)
    def area(self, x, y):                
        v = minimum(x + 1, self.lurc[1]) - maximum(x, self.lblc[1])
        w = minimum(y + 1, self.lurc[0]) - maximum(y, self.lblc[0])
        return v * w

def add_profile(im, profile, intensity=1000.):
    '''Adds the array given by profile.array() in the position profile.offset 
    multiplied by value of intensity'''
    ss = profile.array()
    i,j = subarray_match(im.shape, profile.offset, ss.shape)
    im[i] += intensity * ss[j]
    return im

def add_gaussian(im, sigma, center, scale = 5, intensity=1000.):
    '''Adds a Gaussian profile using add_profile'''
    profile = GaussProfile(sigma=sigma, center = center, scale=scale)
    return add_profile(im, profile, intensity=intensity)

if __name__ == "__main__":
    from numdisplay import display
    
    def fun1():
        im = numpy.random.normal(loc=1000., scale=1.0, size=(1000,1000))
        numpy.random.seed(250)
        npos = 1000
        y = numpy.random.uniform(high=1000,size=npos)
        x = numpy.random.uniform(high=1000,size=npos)
        lx = 50.
        ly = 12.
        for i in zip(y,x):
            profile = GaussProfile(sigma=1.4, center = i)
            #blc = (i[0] - ly / 2, i[1] - lx / 2)
            #urc = (i[0] + ly / 2, i[1] + lx / 2)
            #profile = SlitProfile(blc, urc)
            subref = add_profile(im, profile)
        display(subref)
    
    def fun2():
        im = numpy.random.normal(loc=1000., scale=1.0, size=(1000,1000))
        numpy.random.seed(250)
        npos = 1000
        y = numpy.random.uniform(high=1000,size=npos)
        x = numpy.random.uniform(high=1000,size=npos)
        lx = 50.
        ly = 12.
        for i in zip(y,x):            
            blc = (i[0] - ly / 2, i[1] - lx / 2)
            urc = (i[0] + ly / 2, i[1] + lx / 2)
            profile = SlitProfile(blc, urc)
            subref = add_profile(im, profile)
        display(subref)
    
    fun1()
    fun2()