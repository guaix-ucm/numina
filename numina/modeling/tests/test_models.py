#
# Copyright 2014 Universidad Complutense de Madrid
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

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ... import modeling


def test_EnclosedGaussian():
    
    A = 1234.02
    sigma = 3.1
    
    model = modeling.EnclosedGaussian(A, sigma)
    
    rad = np.arange(0, 15, 1.0)
    Fr_ref = np.array([    0.        ,    62.56332417,   231.85707671,   
                       461.41325153,
         697.25186078,   897.9551982 ,  1044.40731257,  1137.61050118,
        1189.84461703,  1215.77898974,  1227.232255  ,  1231.74380107,
        1233.33213537,  1233.83267152,  1233.97402596])
    
    Fr = model(rad)
    
    assert_allclose(Fr, Fr_ref, rtol=0, atol=1e-6)
    


def test_GaussBox():

    a = 34.5
    s = 1.5        
    xp = np.arange(-5, 6, 1.0)
    yp_ref = np.array([ 0.00122703,  0.00846543,  0.03797502,  0.1108649 ,  
                       0.21078609,
        0.26111732,  0.21078609,  0.1108649 ,  0.03797502,  0.00846543,
        0.00122703]) * a
    

    model = modeling.GaussBox(a, 0.0, s, 0.5)
    yp = model(xp)
    
    assert_allclose(yp, yp_ref, rtol=0, atol=1e-6)


def test_sum_of_gaussians():
    N = 7
    amps = np.array([1e2, 1.01e2, 7.015e2, 1.417e2, 2.021e2, 1.022e2, 1.025e1])
    centers = [7.63, 14.24, 21.35, 28.1, 34.7, 41.9, 49.2]
    sigs = [1.48, 1.51, 1.46, 1.48, 1.51, 1.46, 1.50]
    back = 120.0
    
    xl = np.arange(0, 9*N, dtype='float')
    yl = np.zeros_like(xl)
    for a, l, s in zip(amps[:N], centers[:N], sigs[:N]):
        z = (xl - l) / s
        yl +=  a * np.exp(-0.5 * z**2)

    yl += back
    
    init = {}
    for i in range(N):
        init['amplitude_%d' % i] = amps[i]
        init['center_%d' % i] = centers[i]
        init['stddev_%d' % i] = sigs[i]
    init['background'] = back

    KS = modeling.sum_of_gaussian_factory(N)
    model_gs = KS(**init)
    yp = model_gs(xl)
    
    assert_allclose(yl, yp, rtol=0, atol=1e-6)
