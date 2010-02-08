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

'''
This module implements different combination methods 
that can be passed to the combination function.

The methods are functions and classes with the
__call__ method defined. A sequence is passed to
the object and a 3-tuple is returned. 

The first item is the combined value, the second is the variance of the
first value and the third the number of points used in the
calculus.
'''

__version__ = "$Revision$"

# Classes are new style
__metaclass__ = type

import scipy

def mean(values):
    '''Compute the mean, variance and number of points of a sequence.
    
    :param values: a sequence convertible to scipy.array
    :returns: the mean, the variance and the number of points used
    :rtype: a 3-tuple
    
    For example:
    
      >>> mean([1, 2 ,3, 4])
      (2.5, 1.25, 4)
    
      >>> mean([1,1,1,1,1,1,1,1,1,1,1])
      (1.0, 0.0, 11)
    
    '''
    if len(values) == 0:
        return (0., 0., 0)
    
    data = scipy.asarray(values).ravel()
    l = len(values)
    return (data.mean(), scipy.var(data), l)


def sigmaclip(values, low=4., high=4.):
    '''Compute the iterative sigma-clipping mean, variance and number of points of a sequence.
    
    :param values: a sequence convertible to scipy.array
    :param low: lower bound of sigma clipping 
    :param high: upper bound of sigma clipping
    :returns: the mean, the variance and the number of points used
    :rtype: a 3-tuple
    
    For example:
    
      >>> a = [5.1, 5.2, 5.1, 5., 5., 5.1, 5.5, 5.3, 5.6, 4.7, 15.]
      >>> sigmaclip(a, 2, 2)
      (5.1600000000000001, 0.060399999999999968, 10)
    
    
    '''
    if len(values) == 1:
        return (values[0], 0.0, 1)
    
    c = scipy.asarray(values).ravel()
    delta = 1
    while delta:
        c_std = c.std()
        c_mean = c.mean()
        size = c.size
        c = c[(c > c_mean - c_std * low ) & (c < c_mean + c_std * high)]
        delta = size - c.size
    return mean(c)


def quantileclip(values, low=1.5, high=1.5):
    '''Compute the mean, variance and number of points of a sequence, clipping outliers.
    
    :param values: a sequence convertible to scipy.array
    :param low: lower bound of lower quartile 
    :param high: upper bound of upper quartile
    :returns: the mean, the variance and the number of points used
    :rtype: a 3-tuple
    
    The lower and upper quartile (Q1 and Q3) are computed, and from them 
    the interquartile range (IQR).
    Outliers are either bellow Q1 - low * IQR or above Q3 + high * IQR.
    
    Example of usage:
    
      >>> a = [5.1, 5.2, 5.1, 5., 5., 5.1, 5.5, 5.3, 5.6, 4.7, 15.]
      >>> quantileclip(a)
      (5.1600000000000001, 0.060399999999999954, 10)
    
    '''
    c = scipy.asarray(values).ravel()
    # pylint: disable-msg=W0104
    c.sort();
    
    l = len(c)
    lq = c[0.25 * l - 1]
    uq = c[0.75 * l - 1]
    dinq = uq - lq
    # median = c[0.5 * l - 1]
    c = c[(c > lq - low * dinq) & (c < uq + high * dinq)]
    return mean(c) 
