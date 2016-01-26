#
# Copyright 2015 Universidad Complutense de Madrid
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

import cython
cimport cython

import numpy
cimport numpy

from libc.math cimport floor, ceil
from libc.math cimport fabs
from libc.math cimport log as logc
# from libc.stdio cimport printf
from libcpp.vector cimport vector

ctypedef fused FType:
    double
    float
    int
    long

cdef extern from "Trace.h" namespace "Numina":
    cdef cppclass InternalTrace:
        InternalTrace() except +
        void push_back(double x, double y, double p) nogil
        double predict(double x) nogil
        vector[double] xtrace
        vector[double] ytrace
        vector[double] ptrace
        void reverse() nogil


cdef vector[int] local_max(double* mm, size_t n, double background) nogil:
    '''Find local maximum points.'''

    # TODO: this can be a pure C++ function
    
    cdef vector[int] result
    cdef size_t i
    
    if mm[0] >= background:
        if mm[0] > mm[1]:
            result.push_back(0)

    for i in range(1, n-1):
        if mm[i] >= background:
            if (mm[i] > mm[i+1]) and (mm[i] > mm[i-1]):
                result.push_back(i)
                
    if mm[n-1] >= background:
        if mm[n-1] > mm[n-2]:
            result.push_back(n-1)
            
    return result

@cython.boundscheck(False)
cdef vector[double] fit_para_equal_spaced(FType[:] dd) nogil:
    '''Fit (-1, 0, 1) (dd0, dd1, dd2) to a 2nd degree polynomial.'''

    cdef vector[double] result
    cdef double A, B, C
    C = dd[1]
    B = 0.5 * (dd[2] - dd[0])
    A = 0.5 * (dd[0] + dd[2] - 2 * dd[1])
    
    result.push_back(A)
    result.push_back(B)
    result.push_back(C)
    return result


@cython.boundscheck(False)
cdef vector[double] fit_para_equal_spaced_alt(double y0, double y1, double y2) nogil:
    '''Fit (-1, 0, 1) (dd0, dd1, dd2) to a 2nd degree polynomial.'''

    cdef vector[double] result
    cdef double A, B, C
    C = y1
    B = 0.5 * (y2 - y0)
    A = 0.5 * (y0 + y2 - 2 * y1)

    result.push_back(A)
    result.push_back(B)
    result.push_back(C)
    return result


@cython.boundscheck(False)
cdef vector[double] interp_max_3(FType[:] dd) nogil:
    '''Parabola that passes through 3 points
        
    With X=[-1,0,1]
    '''

    cdef vector[double] result
    cdef vector[double] params
    cdef double A,B,C
    params = fit_para_equal_spaced(dd)
    A = params[0]
    B = params[1]
    C = params[2]
    if A == 0:
        result.push_back(0.0)
        result.push_back(C)
    else:
        result.push_back(-B / (2*A))
        result.push_back(C - B * B / (4*A))
    return result


@cython.boundscheck(False)
cdef vector[double] interp_max_3_alt(double y0, double y1, double y2) nogil:
    '''Parabola that passes through 3 points

    With X=[-1,0,1]
    '''

    cdef vector[double] result
    cdef vector[double] params
    cdef double A,B,C
    params = fit_para_equal_spaced_alt(y0, y1, y2)
    A = params[0]
    B = params[1]
    C = params[2]
    if A == 0:
        result.push_back(0.0)
        result.push_back(C)
    else:
        result.push_back(-B / (2*A))
        result.push_back(C - B * B / (4*A))
    return result


@cython.boundscheck(False)
cdef vector[double] interp_max_3_alt_log(double y0, double y1, double y2) nogil:
    '''Parabola that passes through 3 points

    With X=[-1,0,1]
    '''

    cdef vector[double] result
    cdef vector[double] params
    cdef double A,B,C
    params = fit_para_equal_spaced_alt(logc(y0), logc(y1), logc(y2))
    A = params[0]
    B = params[1]
    C = params[2]
    if A == 0:
        result.push_back(0.0)
        result.push_back(C)
    else:
        result.push_back(-B / (2*A))
        result.push_back(C - B * B / (4*A))
    return result


cdef int wc_to_pix(double x) nogil:
    return <int>floor(x + 0.5)


@cython.cdivision(True)
@cython.boundscheck(False)
cdef int colapse_mean(FType[:, :] arr, vector[double]& out) nogil:
    cdef size_t I = arr.shape[0]
    cdef size_t J = arr.shape[1]
    cdef double accum
    cdef size_t i, j
    for i in range(I):
        accum = 0.0
        for j in range(J):
            accum += arr[i, j]
        out[i] = accum / J
    
    return 0


@cython.cdivision(True)
@cython.boundscheck(False)
cdef InternalTrace _internal_tracing(FType[:, :] arr, 
                             InternalTrace& trace, double x, double y,
                             size_t step=1, size_t hs=1, size_t tol=2,
                             double maxdis=2.0, double background=150.0,
                             int direction=-1) nogil:

    cdef int col = wc_to_pix(x)
    cdef int row = wc_to_pix(y)
    
    cdef size_t pred_pix
    cdef double prediction
    
    cdef int axis = 1
    cdef size_t i
    cdef size_t tolcounter = tol
    cdef size_t axis_size = arr.shape[1]
    
    # Buffer
    cdef size_t regw = 1 + <int>ceil(maxdis)
    cdef size_t buffsize = 2 * regw + 1
    cdef size_t pred_off 
    cdef vector[double] pbuff
    # Peaks
    cdef vector[int] peaks
    cdef double dis, ndis
    cdef size_t ipeak
    cdef size_t nearp    
    cdef vector[double] result
    
    # Init pbuff
    for _ in range(buffsize):
        pbuff.push_back(0.0)
    
    
    while (col - step > hs) and (col + step + hs < axis_size):
        # printf("--------- %i\n", tolcounter)
        # printf("col %i dir %i step %i\n", col, direction, step)

        col += direction * step
        prediction = trace.predict(col)

        pred_pix = wc_to_pix(prediction)
        pred_off = pred_pix - regw
        # printf("col %i prediction %f pixel %i\n", col, prediction, pred_pix)

        # extract a region around the expected peak
        # and collapse it
        colapse_mean(arr[pred_pix-regw:pred_pix+regw + 1,col-hs:col+hs+1], pbuff)

        # printf("cut region %i : %i,  %i : %i\n", pred_pix-regw,pred_pix+regw,col-hs,col+hs)

        # Find the peaks
        peaks = local_max(&pbuff[0], buffsize, background=background)

        # find nearest peak to prediction
        dis = 40000.0 # a large number
        ipeak = -1
        for i in range(peaks.size()):
            ndis = fabs(peaks[i] + pred_off - prediction)
            if ndis < dis:
                dis = ndis
                ipeak = i

        # check the peak is not further than npixels'
        if ipeak < 0 or dis > maxdis:
            # printf("peak is not found %i\n", ipeak)
            # peak is not found
            if tolcounter > 0:
                # Try again
                # printf("%i tries remaining\n", tolcounter)
                tolcounter -= 1
                continue
            else:
                # No more tries
                # printf("%i tries remaining\n", 0)
                # Exit now
                return trace

        # Reset counter
        tolcounter = tol 

        nearp = peaks[ipeak] + pred_off

        # fit the peak with three points
        result = interp_max_3_alt_log(pbuff[peaks[ipeak]-1], pbuff[peaks[ipeak]], pbuff[peaks[ipeak]+1])

        if (result[0] > 0.5) or (result[0] < -0.5):
            # ignore the correction if it's larger than 0.5 pixel
            trace.push_back(col, nearp, pbuff[peaks[ipeak]])
        else:
            trace.push_back(col, result[0] + nearp, result[1])

    return trace


@cython.cdivision(True)
@cython.boundscheck(False)
def tracing(FType[:, :] arr, double x, double y, double p, size_t step=1, 
                     size_t hs=1, size_t tol=2, double background=150.0,
                     double maxdis=2.0):
    '''Trace peak in array starting in (x,y).

    Trace a peak feature in an array starting in position (x,y).

    Parameters
    ----------
    arr : array
         A 2D array
    x : float
        x coordinate of the initial position
    y : float
        y coordinate of the initial position
    p : float
        Intensity of the initial position
    step : int, optional
           Number of pixels to move (left and rigth)
           in each iteration

    Returns
    -------
    ndarray
        A nx3 array, with x,y,p of each point in the trace
    '''
    
    cdef InternalTrace trace 
    # Initial values    
    trace.push_back(x, y, p)

    _internal_tracing(arr, trace, x, y, step=step, hs=hs, tol=tol,
                      maxdis=maxdis, background=background,
                      direction=-1)
    trace.reverse()
    _internal_tracing(arr, trace, x, y, step=step, hs=hs, tol=tol,
                      maxdis=maxdis, background=background,
                      direction=+1)

    result = numpy.empty((trace.xtrace.size(), 3), dtype='float')
    for i in range(trace.xtrace.size()):
        result[i,0] = trace.xtrace[i]
        result[i,1] = trace.ytrace[i]
        result[i,2] = trace.ptrace[i]
    
    return result

