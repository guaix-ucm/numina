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

import math

from .pixelize import pixelize, rpixelize, frac

class EllipticalAperture(object):
    '''Geometrical properties of the aperture'''
    def __init__(self, a, b, x0=0.0, y0=0.0):
        if a < 0:
            raise ValueError('a (semimajor axis) must be positive')
        if b < 0:
            raise ValueError('b (semiminor axis) must be positive')
        self._a = a
        self._b = b
        self.x0 = x0
        self.y0 = y0
        self._update_axis()
    
    def _update_axis(self):
        self._a2 = self._a**2
        self._b2 = self._b**2
        self._hip = math.sqrt(self._a2 + self._b2)

    def _set_a(self, a):
        self._a = a
        self._update_axis()

    a = property(lambda self: self._a, _set_a)

    def _set_b(self, b):
        self._b = b
        self._update_axis()

    b = property(lambda self: self._b, _set_b)

    def curve_y(self, x):
        return self.y0 + self.b * math.sqrt(1 - (x - self.x0)**2 / self._a2)

    def curve_x(self, y):
        return self.x0 + self.a * math.sqrt(1 - (y - self.y0)**2 / self._b2)

    def integral(self, e1, e2, e0, scale):
        t1 = math.asin((e1 - e0) / scale)
        t2 = math.asin((e2 - e0) / scale)

        def parent(x):
            return 0.5 * (x + 0.5 * math.sin(2 * x))

        return self.a * self.b * (parent(t2) - parent(t1))

    def integral1(self, x1, x2):
        return self.integral(x1, x2, self.x0, self.a)

    def integral2(self, y1, y2):
        return self.integral(y1, y2, self.y0, self.b)

    def area(self):
        return math.pi * self.a * self.b

    def indicator(self, x, y):
        return (x - self.x0)**2 / self._a2 + (y - self.y0)**2 / self._b2 - 1

    def slope_change(self):
        x = self.x0 + self._a2 / self._hip
        y = self.y0 + self._b2 / self._hip
        return x, y

def generate_arc_pixels(e, x1, x2, y1, y2):

    def pixel_weight(e, xmin, lcut, ucut, ymin, ymax):
        w1 = (ucut - xmin) * (ymax - ymin)
        w2 = e.integral1(ucut, lcut)
        w3 = (ymin - e.y0) * (ucut - lcut)
        return w1 + w2 + w3

    ypixgen = rpixelize(y1, y2)
    ymin, ymax = ypixgen.next()
    for xmin, xmax in pixelize(x1, x2):
        xpix = int(math.floor(xmin))
        sio = e.indicator(xmax, ymin)

        if sio >= 0: # curve cuts ymin inside the pixel
            lcut = xmin
            try:
                while True:                
                    lcut, ucut = e.curve_x(ymin), lcut
                    w = pixel_weight(e, xmin, lcut, ucut, ymin, ymax)
                    yield xpix, int(ymin), w
                    # next pixel, below
                    ymin, ymax = ypixgen.next()                   
                    sio = e.indicator(xmax, ymin)
                    if sio < 0: # curve cuts ymin outside the pixel
                        lcut, ucut = xmax, lcut
                        w = pixel_weight(e, xmin, lcut, ucut, ymin, ymax)
                        yield xpix, int(ymin), w
                        break
            except StopIteration:
                pass

        elif sio < 0:
            ucut, lcut = xmin, xmax
            w = pixel_weight(e, xmin, lcut, ucut, ymin, ymax)
            yield xpix, int(ymin), w


        # Fill the column
        for pix in generate_column(xmin, xmax, y1, ymin):
            yield pix

def generate_column(xmin, xmax, ymin, ymax):
    for ymin, ymax in rpixelize(ymin, ymax):
        yield int(xmin), int(ymin), (ymax - ymin) * (xmax - xmin)

def intersec_ellip_rect(ellip, x1, x2, y1, y2):

    ps = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    idcs = [ellip.indicator(x, y) for x, y in ps]

    if idcs[0] > 0: # rectangle is outside
        return ()

    if idcs[3] < 0: # rectangle is inside
        ucut = lcut = x2, y2
    else: # rectangle intersects
        if idcs[1] > 0:
            lcut = ellip.curve_x(y1), y1
        else:
            lcut = x2, ellip.curve_y(x2)

        if idcs[2] > 0:
            ucut = x1, ellip.curve_y(x1)
        else:
            ucut = ellip.curve_x(y2), y2

    return lcut, ucut

def ellipse_quadrant(a, b, x0, y0, xmin, xmax, ymin, ymax):
    '''Generate a quadrant of the ellipse'''
    def flip(pix):
        return pix[1], pix[0], pix[2]

    e = EllipticalAperture(a, b, x0, y0)
    changex, changey = e.slope_change()

    xmax = min(xmax, x0 + a)
    xmin = min(max(xmin, x0), x0 + a)
    ymax = min(ymax, y0 + b)
    ymin = min(max(ymin, y0), y0 + b)

    #print xmin, xmax, ymin, ymax
    if xmin >= xmax or ymin >= ymax:
        return
    
    cuts = intersec_ellip_rect(e, xmin, xmax, ymin, ymax)

    if not cuts:
        return
    #print 'cuts',cuts
    lcut, ucut = cuts    

    for x1, x2 in pixelize(xmin, ucut[0]):
        for y1, y2 in rpixelize(ymin, ucut[1]):
            yield int(x1), int(y1), (x2-x1)*(y2-y1)

    for y1, y2 in pixelize(ymin, lcut[1]):
        for x1, x2 in rpixelize(ucut[0], xmax):
            yield int(x1), int(y1), (x2-x1)*(y2-y1)

    for pix in generate_arc_pixels(e, x1=ucut[0], 
                                        x2=min(changex, lcut[0]), 
                                        y1=lcut[1], 
                                        y2=ucut[1]):
        yield pix

    eT = EllipticalAperture(b, a, y0, x0)
    for pix in generate_arc_pixels(eT, 
                                x1=lcut[1], 
                                x2=min(changey, ucut[1]),
                                y1=max(changex, ucut[0]), 
                                y2=lcut[0]):
        yield flip(pix)




def aperture(a, b, x0, y0, xmin=None, xmax=None, ymin=None, ymax=None):
    '''Generate the complete elliptical aperture'''

    def reflexion(center, offset=0.0):
        def reflexion_func(coord):
            return 2 * center - coord - offset

        return reflexion_func
    
    if a <= 0:
        raise ValueError('a (semimajor axis) must be positive')
    if b <= 0:
        raise ValueError('b (semiminor axis) must be positive')

    xmax = x0 + a if xmax is None else min(xmax, x0 + a)
    ymax = y0 + b if ymax is None else min(ymax, y0 + b)
    xmin = x0 - a if xmin is None else max(xmin, x0 - a)
    ymin = y0 - b if ymin is None else max(ymin, y0 - b)

    base_x_pix = int(math.floor(x0))
    base_y_pix = int(math.floor(y0))

    fracx = x0 - base_x_pix
    fracy = y0 - base_y_pix

    cox = frac(1 - fracx)
    coy = frac(1 - fracy)

    reflexion_func_x = reflexion(base_x_pix, math.trunc(1 - fracx))
    reflexion_func_y = reflexion(base_y_pix, math.trunc(1 - fracy))

    ref_func_cox0 = reflexion(base_x_pix + 0.5)
    ref_func_coy0 = reflexion(base_y_pix + 0.5)
    
    for pix in ellipse_quadrant(a, b, x0, y0, xmin, xmax, ymin, ymax):
        yield pix

    # second cuadrant. -x, +y    
    for pix in ellipse_quadrant(a, b, base_x_pix + cox, y0, 
                                ref_func_cox0(xmax), ref_func_cox0(xmin), 
                                ymin, ymax):
        x, y, w = pix
        affpix = reflexion_func_x(x)
        yield affpix, y, w

    # third cuadrant. -x, -y
    for pix in ellipse_quadrant(a, b, base_x_pix + cox, base_y_pix + coy,
                                ref_func_cox0(xmax), ref_func_cox0(xmin), 
                                ref_func_coy0(ymax), ref_func_coy0(ymin)):
        x, y, w = pix
        affpix = reflexion_func_x(x)
        affpiy = reflexion_func_y(y)
        yield affpix, affpiy, w
    
    # fourth cuadrant x, -y
    for pix in ellipse_quadrant(a, b, x0, base_y_pix + coy,
                                xmin, xmax, 
                                ref_func_coy0(ymax), ref_func_coy0(ymin)):
        x, y, w = pix
        affpiy = reflexion_func_y(y)
        yield x, affpiy, w
        
    
