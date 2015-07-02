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


from ._traces import tracing

def trace(arr, x, y, p, axis=0, background=0.0,
          step=1, hs=1, tol=2, maxdis=2.0):
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
    axis : {0, 1}
           Spatial axis of the array (0 is Y, 1 is X).
    background: float
           Background level
    step : int, optional
           Number of pixels to move (left and rigth)
           in each iteration

    Returns
    -------
    ndarray
        A nx3 array, with x,y,p of each point in the trace
    '''

    # If arr is not in native byte order, the C-extension won't work

    if arr.dtype.byteorder != '=':
        arr2 = arr.byteswap().newbyteorder()
    else:
        arr2 = arr

    if axis == 0:
        arr3 = arr2
    elif axis == 1:
        arr3 = arr2.t
    else:
        raise ValueError("'axis' must be 0 or 1")

    result = tracing(arr3, x, y, p, background=background,
                     step=step, hs=hs, tol=tol, maxdis=maxdis)

    if axis == 1:
        # Flip X,Y columns
        return result[:,::-1]

    return result

def fit_trace_polynomial(trace, deg, axis=0):
    '''
    Fit a trace information table to a polynomial.

    Parameters
    ----------
    trace 
           A 2D array, 2 columns and n rows
    deg : int
           Degree of polynomial
    axis : {0, 1}
           Spatial axis of the array (0 is Y, 1 is X).
    '''

    dispaxis = axis_to_dispaxis(axis)

    # FIT to a polynomial
    pfit = numpy.polyfit(trace[:,0], trace[:,1], deg)
    start = trace[0,0]
    stop = trace[-1,0],
    return PolyTrace(start, stop, axis, pfit)

def axis_to_dispaxis(axis):
    if axis == 0:
        dispaxis = 'X'
    elif axis == 1:
        dispaxis = 'Y'
    else:
        raise ValueError("'axis' must be 0 or 1")
    return dispaxis


class GeometricTrace(object):
    def __init__(self, id, start, stop, axis, ttype, coeff):
        self.id = id
        self.start = start
        self.stop = stop
        self.axis = axis
        self.type = ttype
        self.dispaxis = axis_to_dispaxis(axis)
            

class PolyTrace(GeometricTrace):
    def __init__(self, id, start, stop, axis, coeff):
        super(PolyTrace, self).__init__(id, start, stop, axis,
                                       'poly', coeff)


class Tracemap(object):
    def __init__(self, instrument, traces):
        super(Tracemap, self).__init__(instrument, traces)
        self.instrument = instrument
        self.traces = traces


class Aperture(object):
    def __init__(self, id, bbox, axis, borders):
        self.id = id
        self.bbox = bbox
        self.axis = axis
        self.dispaxis = axis_to_dispaxis(axis)
        self.borders = borders

