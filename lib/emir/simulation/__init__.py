
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

# $Id: detector.py 386 2009-01-20 18:10:57Z spr $

import numpy
import math
from scipy.special import erf
from scipy import maximum, minimum
from emir.image import subarray_match

M_SQRT1_2 = math.sqrt(1. / 2.)

class GaussProfile:
  def __init__(self, sigma, fcenter= (0.,0.), scale = 5):
    self.sigma = sigma
    self.fcenter = fcenter
    self.scale = scale
  def side(self):
    l = int(round(self.scale * self.sigma))
    return (l, l)
  def shape(self):
    l = 2 * int(round(self.scale * self.sigma)) + 1
    return (l, l)
  def area(self, x, y):
    def compute1(v, center, sigma):
      return M_SQRT1_2 * (v - center) / sigma
    def compute2(v, center, sigma):
      tipV1 = compute1(v, center, sigma)
      tipV2 = compute1(v + 1, center, sigma)
      return 0.5 * (erf(tipV2) - erf(tipV1))
      
    side = self.side()
    return compute2(x, self.fcenter[1] + side[1] + 1, self.sigma) * compute2(y, self.fcenter[0] + side[0] + 1,
 self.sigma)
  def array(self):
      y, x = numpy.indices(self.shape())
      return self.area(x, y)

class SlitProfile:
    def __init__(self, sides, center = (0.5, 0.5)):
        self.blc = numpy.array([0,0])        
        self.urc = numpy.array(sides, dtype='int') + 1
    def shape(self):
        return (int(self.urc[0]) - int(self.blc[0]) + 1, int(self.urc[1]) - int(self.blc[1]) + 1)
    def area(self, x, y):
        return (minimum(x + 1, self.urc[1]) - maximum(x, self.blc[1])) * (minimum(y + 1, self.urc[0]) - maximum(y, self.blc[0]))
    def set_center(self, center):
        self.blc = numpy.array([center[0] - sides[0] / 2. , center[1] - sides[1] / 2.], dtype='int')        
        self.urc = numpy.array([center[0] + sides[0] / 2. + 1 , center[1] + sides[1] / 2. + 1], dtype='int')        
    def get_center(self):
        return ((self.blc[0] + self.urc[0]) / 2.,(self.blc[1] + self.urc[1]) / 2.)
    center = property(get_center, set_center)
    def array(self):
      print self.shape()
      print self.urc
      print self.blc
      y, x = numpy.indices(self.shape())
      return self.area(x, y)


def add_profile(im, center0, center1, profile, intensity=1000.):
  center = numpy.array([center0, center1])
  rcenter = map(int, center)
  pcenter = center - rcenter
  profile.center = pcenter
  ss = profile.array()
  i,j = subarray_match(im.shape, rcenter, ss.shape)
  im[i] += intensity * ss[j]
  return im

if __name__ == "__main__":
    from numdisplay import display
    def fun1():
        im = numpy.random.normal(loc=1000., scale=1.0, size=(1000,1000))
        numpy.random.seed(250)
        npos = 1000
        y = numpy.random.uniform(high=1000,size=npos)
        x = numpy.random.uniform(high=1000,size=npos)
        profile = [GaussProfile(1.4)] * npos
        for i in zip(y,x,profile):
          subref = add_profile(im, *i)
        display(subref)
    def fun2():
        
        im = numpy.random.normal(loc=1000., scale=1.0, size=(1000,1000))
        numpy.random.seed(250)
        npos = 50
        y = numpy.random.uniform(high=1000,size=npos)
        x = numpy.arange(0,1000,20)
        profile = [SlitProfile((15.5, 23.8))] * npos
        for i in zip(y,x,profile):
          subref = add_profile(im, *i)
        display(subref)
        #p = SlitProfile((15, 23))
        #display(p.array())
    def fun3(): 
        p = GaussProfile(sigma=5, fcenter=(0.5, 0.5))
        im = numpy.random.normal(loc=1000., scale=1.0, size=(1000,1000))
        a = p.array()
        ain, bin = subarray_match(im.shape, (990, 25), a.shape)
        im[ain] -= 2000 * a[bin] 
        display(im)
        add_profile(im, 45.89, 95.99, p, intensity=1000.)
        
    fun2()