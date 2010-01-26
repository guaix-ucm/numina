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

# $Id$

__version__ = "$Revision$"
 
import math
from math import sin, cos, tan
import sys
import os

from scipy import array
import scipy.interpolate as sil
from scipy import loadtxt

from emir.exceptions import Error

# Classes are new style
__metaclass__ = type

def _find(pathname, matchfunc=os.path.isfile):
    for dirname in sys.path:
        candidate = os.path.join(dirname, pathname)
        if matchfunc(candidate):
            return candidate
    raise Error("Can't find file %s" % pathname)

def findFile(pathname):
    return _find(pathname)

def findDir(pathname):
    return _find(pathname, matchfunc=os.path.isdir)

class Counts:
    INTEGRAL_COUNTS = 0,
    DIFFERENTIAL_COUNTS = 1
    
class PhotometricFilter:
    FILTER_NONE = 0
    FILTER_B = 1
    FILTER_V = 2
    FILTER_Z = 3
    FILTER_J = 4
    FILTER_H = 5
    FILTER_K = 6
        
class RBModel:
    '''Ratnatunga & Bahcall model
    
    Number of stars per square arc minute
    '''
    _counts = array([
                     [12, 14, 16, 18, 20, 22, 24, 26, 28],
                     [0.0, 1.3e-2, 1.8e-2, 4.8e-2, 1.1e-1, 9.0e-2, 2.7e-2, 3.4e-3, 1.6e-4],
                     [0.0, 8.2e-3, 2.6e-2, 2.6e-2, 4.5e-2, 1.2e-1, 1.1e-1, 3.9e-2, 5.6e-3],
                     [0.0, 2.0e-3, 1.0e-2, 4.4e-2, 1.1e-1, 2.2e-1, 4.6e-1, 8.1e-1, 1.1e-0]
                     ])
    # Order 3 spline
    _spl = [sil.splrep(_counts[0], _counts[i]) for i in (1, 2, 3)]
    _min = _counts[0, 0]
    del _counts
    
    _null = [0.0, 0.0, 0.0]
    _colorsVK = [4.7, 2.5, 1.4]
    
    
    name = 'Ratnatunga & Bahcall'
    filters = [PhotometricFilter.FILTER_K]
        
    def differential_counts(self, mag, filter_=PhotometricFilter.FILTER_K):
        if filter_ == PhotometricFilter.FILTER_V:
            return self._differential_counts_color(mag, self._null)
        elif filter_ == PhotometricFilter.FILTER_K:
            return self._differential_counts_color(mag, self._colorsVK)    
            
    def integral_counts(self, mag, filter_=PhotometricFilter.FILTER_K):
        if filter_ == PhotometricFilter.FILTER_K:
            return self._integral_counts_color(mag, self._colorsVK)
        return 0.
    
    @classmethod
    def _integral_counts_color(cls, mag, colors):        
        def subint(i):
            return max(sil.splint(cls._min + colors[i], mag + colors[i], cls._spl[i]), 0.0)
        return sum(subint(i) for i in (0, 1, 2))
    
    @classmethod
    def _differential_counts_color(cls, mag, colors):
        def subspl(i):
            return max(sil.splev(mag, cls._spl[i]), 0.0)
        return sum(subspl(i) for i in (0, 1, 2))



class SpagnaModel:
    '''Spagna1999
    
    Number of stars per square arc minute
    '''
    
    _J_counts_filename = findFile("emir/simulation/spagna-J.dat")
    _K_counts_filename = findFile("emir/simulation/spagna-K.dat")
    f = open(_J_counts_filename)
    try:
        _J_counts_data = loadtxt(f)
        # Data in file is for square degree
        _J_counts_data[:, 1:3] /= 3600.0
        _spl_J_1 = sil.splrep(_J_counts_data[:, 0], _J_counts_data[:, 1])
        _spl_J_2 = sil.splrep(_J_counts_data[:, 0], _J_counts_data[:, 2])
        del _J_counts_data
    finally:
        f.close()
    f = open(_K_counts_filename)
    try: 
        _K_counts_data = loadtxt(f)
        # Data in file is for square degree
        _K_counts_data[:, 1:3] /= 3600.0
        _spl_K_1 = sil.splrep(_K_counts_data[:, 0], _K_counts_data[:, 1])
        _spl_K_2 = sil.splrep(_K_counts_data[:, 0], _K_counts_data[:, 2])
        del _K_counts_data
    finally:
        f.close()
        
    name = 'Spagna 1999'    
    filters = [PhotometricFilter.FILTER_K, PhotometricFilter.FILTER_J]
    
    @classmethod
    def integral_counts(cls, mag, filter=PhotometricFilter.FILTER_K):
        if filter == PhotometricFilter.FILTER_J:
            return sil.splev(mag, cls._spl_J_2)
        elif filter == PhotometricFilter.FILTER_K:
            return sil.splev(mag, cls._spl_K_2)
        else:
            return 0.
        return 0.
    
    @classmethod
    def differential_counts(cls, mag, filter=PhotometricFilter.FILTER_K):
        if filter == PhotometricFilter.FILTER_J:
            return sil.splev(mag, cls._spl_J_1) / 3600.0
        elif filter == PhotometricFilter.FILTER_K:
            return sil.splev(mag, cls._spl_K_1) / 3600.0
        else:
            return 0.
        return 0.

class BSModel:
    '''Bahcall & Soneira
    
    Number of stars per square arc minute
    '''
    name = "Bahcall & Soneira"
    params = array([[200, - 0.2, 0.01, 2, 15, 400, - 0.26, 0.065, 1.5, 17.5],
                    [925, - 0.132, 0.035, 3, 15.75, 1050, - 0.18, 0.087, 2.5, 17.5],
                    [235, - 0.227, 0.0, 1.5, 17, 370, - 0.175, 0.06, 2.0, 18],
                    [950, - 0.124, 0.027, 3.1, 16.60, 910, - 0.167, 0.083, 2.5, 18]])
    
    @classmethod
    def integral_counts(cls, mag, filter=PhotometricFilter.FILTER_V):
        if filter == PhotometricFilter.FILTER_V:
            return cls.dFunction(mag, 0., 0.5 * math.pi, *cls.params[1])
        elif filter == PhotometricFilter.FILTER_B:
            return cls.dFunction(mag, 0., 0.5 * math.pi, *cls.params[4])
        else:
            return 0.
        return 0.
    
    @classmethod
    def differential_counts(cls, mag, filter=PhotometricFilter.FILTER_V):
        if filter == PhotometricFilter.FILTER_V:
            return cls.dFunction(mag, 0., 0.5 * math.pi, *cls.params[0])
        elif filter == PhotometricFilter.FILTER_B:
            return cls.dFunction(mag, 0., 0.5 * math.pi, *cls.params[3])
        else:
            return 0.
        return 0.
    
    @staticmethod
    def mu(magnitude):
        '''Parameter \mu of Appendix B'''
        if magnitude <= 12.:
            return 0.03
        if magnitude <= 20:
            return 0.0075 * (magnitude - 12.) + 0.03
        #if magnitude > 20:
        return 0.09
    
    @staticmethod
    def gamma(magnitude):
        if magnitude <= 12.:
            return 0.36
        if magnitude <= 20:
            return 0.04 * (12. - magnitude) + 0.36
        #if magnitude > 20:
        return 0.04
    
    @staticmethod
    def sigma(galacticLongitude, galacticLatitude):
        return 1.45 - 0.20 * cos(galacticLongitude) * cos(galacticLatitude)

    @classmethod
    def dFunction(cls, mag, glongitude, glatitude, C1, alpha, beta, delta, mStar, C2, kappa, eta, lambdax, mDagger):
        firstTerm = C1 * pow(10., beta * (mag - mStar)) / pow(1 
 + pow(10., alpha * (mag - mStar)), delta) / pow(sin(glatitude) * (1 
 - cls.mu(mag) / tan(glatitude) * cos(glongitude)), 3.0 - 5 * cls.gamma(mag))
        secondTerm = C2 * pow(10., eta * (mag 
 - mDagger)) / pow(1 + pow(10., kappa * (mag - mDagger)), lambdax) / pow(1 
 - cos(glatitude) * cos(glongitude), cls.sigma(glongitude, glatitude))
        return (firstTerm + secondTerm) / 3600.0

if __name__ == '__main__':
    rbmodel = RBModel()
    #print rbmodel.integral_counts(18)
    #print rbmodel.differential_counts(18)    
    sgmodel = SpagnaModel()
    #print sgmodel.integral_counts(18)
    #print sgmodel.differential_counts(18)
    bsmodel = BSModel()
    #print sgmodel.integral_counts(18)
    #print bsmodel.differential_counts(18)
  
    import numpy.random
  
    class GeneralRandom:
        ''' Recipe from http://code.activestate.com/recipes/576556/'''
        def __init__(self, x, p, Nrl=1000):
            self.x = x
            self.pdf = p / p.sum()
            self.cdf = self.pdf.cumsum()
            self.inversecdfbins = Nrl
            self.Nrl = Nrl
            y = numpy.arange(Nrl) / float(Nrl)
            delta = 1.0 / Nrl
            self.inversecdf = numpy.zeros(Nrl)
            self.inversecdf[0] = self.x[0]
            cdf_idx = 0
            for n in xrange(1, self.inversecdfbins):
                while self.cdf[cdf_idx] < y [n] and cdf_idx < Nrl:
                    cdf_idx += 1
                # Seems a linear interpolation    
                self.inversecdf[n] = self.x[cdf_idx - 1] + (self.x[cdf_idx] - self.x[cdf_idx - 1]) * (y[n] - self.cdf[cdf_idx - 1]) / (self.cdf[cdf_idx] - self.cdf[cdf_idx - 1])
                if cdf_idx >= Nrl:
                    break
            self.delta_inversecdf = numpy.concatenate((numpy.diff(self.inversecdf), [0]))

        def random(self, N=1000):
            idx_f = numpy.random.uniform(size=N, high=self.Nrl - 1)
            idx = numpy.array(idx_f, 'i')
            y = self.inversecdf[idx] + (idx_f - idx) * self.delta_inversecdf[idx]
            return y


        scmodel = sgmodel
        magmin = 12.0
        magmax = 25.0
  
        plate_scale = 0.2
        detector_shape = (2048, 2048)
        pixel_area = detector_shape
        detector_area = (plate_scale ** 2 * detector_shape[0] * detector_shape[1]) / 3600.
  
        nstars = int(round(detector_area * (scmodel.integral_counts(magmax) - scmodel.integral_counts(magmin))))  
        print nstars
  
        seeing = 1.0
        from math import sqrt, log
        scale = 2 * sqrt(2 * log(2))
  
        delta = 0.1
        steps = int(round((magmax - magmin) / delta))
        x = magmin + delta * numpy.arange(steps)
        p = numpy.array([scmodel.differential_counts(i) for i in x])
        
        g = GeneralRandom(x, p)
  
        mags = g.random(nstars)
        y = numpy.random.uniform(high=pixel_area[0], size=nstars)
        x = numpy.random.uniform(high=pixel_area[1], size=nstars)
  
        numpy.random.seed(100)
        x = numpy.random.uniform(high=2048, size=nstars)
        y = numpy.random.uniform(high=2048, size=nstars)
        sigmas = [seeing / scale / plate_scale] * nstars
        ints = [100] * nstars 
        mag = g.random(N=nstars)
        