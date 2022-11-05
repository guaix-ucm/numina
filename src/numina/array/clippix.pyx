#
# Copyright 2016 Universidad Complutense de Madrid
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

#cython: language_level=3

cimport cython
import numpy as np

# maximum number of intersections of two generic pixels
DEF NMAX_CORNERS = 8


cdef struct Polygon:
    int ncorners
    double x[NMAX_CORNERS]
    double y[NMAX_CORNERS]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef Polygon polygon4vert(double x1, double y1,
                          double x2, double y2,
                          double x3, double y3,
                          double x4, double y4):
    cdef Polygon p
    p.ncorners = 4
    p.x[0] = x1
    p.y[0] = y1
    p.x[1] = x2
    p.y[1] = y2
    p.x[2] = x3
    p.y[2] = y3
    p.x[3] = x4
    p.y[3] = y4
    return p


cdef Polygon empty_polygon():
    cdef Polygon p
    p.ncorners = 0
    return p


@cython.nonecheck(False)
cdef inline check_ncorners(Polygon p):
    if p.ncorners == NMAX_CORNERS:
        raise ValueError('Number of corners exceeds maximum value' +
                         str(NMAX_CORNERS))


# Sutherland–Hodgman algorithm implementation in Cython
# https://stackoverflow.com/questions/44765229/
# The original code has been modified to make use of the Polygon structure
# and the auxiliary functions (note that although the original lists are not
# employed, the variable names have been preserved)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef cclippix_area(Polygon subjectPolygon, Polygon clipPolygon):
    cdef:
        int i, ii
        double cp10, cp11, cp20, cp21
        double s0, s1, e0, e1
        double s_in, e_in
        double dc0, dc1
        double dp0, dp1
        double n1, n2, n3
        double xdum, ydum
        double x1, y1, x2, y2
        double area = 0.0

    outputList = subjectPolygon
    cp10 = clipPolygon.x[clipPolygon.ncorners-1]
    cp11 = clipPolygon.y[clipPolygon.ncorners-1]

    for i in range(clipPolygon.ncorners):
        cp20 = clipPolygon.x[i]
        cp21 = clipPolygon.y[i]

        inputList = outputList
        outputList = empty_polygon()
        s0 = inputList.x[inputList.ncorners-1]
        s1 = inputList.y[inputList.ncorners-1]
        s_in = (cp20 - cp10) * (s1 - cp11) - (cp21 - cp11) * (s0 - cp10)
        for ii in range(inputList.ncorners):
            e0 = inputList.x[ii]
            e1 = inputList.y[ii]
            e_in = (cp20 - cp10) * (e1 - cp11) - (cp21 - cp11) * (e0 - cp10)
            if e_in > 0:
                if s_in <= 0:
                    # compute intersection
                    dc0, dc1 = cp10 - cp20, cp11 - cp21
                    dp0, dp1 =  s0 - e0, s1 - e1
                    n1 = cp10 * cp21 - cp11 * cp20
                    n2 = s0 * e1 - s1 * e0
                    n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
                    xdum = (n1 * dp0 - n2 * dc0) * n3
                    ydum = (n1 * dp1 - n2 * dc1) * n3
                    check_ncorners(outputList)
                    outputList.x[outputList.ncorners] = xdum
                    outputList.y[outputList.ncorners] = ydum
                    outputList.ncorners += 1
                check_ncorners(outputList)
                outputList.x[outputList.ncorners] = e0
                outputList.y[outputList.ncorners] = e1
                outputList.ncorners += 1
            elif s_in > 0:
                # compute intersection
                dc0, dc1 = cp10 - cp20, cp11 - cp21
                dp0, dp1 =  s0 - e0, s1 - e1
                n1 = cp10 * cp21 - cp11 * cp20
                n2 = s0 * e1 - s1 * e0
                n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
                xdum = (n1 * dp0 - n2 * dc0) * n3
                ydum = (n1 * dp1 - n2 * dc1) * n3
                check_ncorners(outputList)
                outputList.x[outputList.ncorners] = xdum
                outputList.y[outputList.ncorners] = ydum
                outputList.ncorners += 1
            s0, s1, s_in = e0, e1, e_in
        if outputList.ncorners < 1:
            # no intersection: null area
            return 0.0
        cp10, cp11 = cp20, cp21

    # compute area
    x1 = outputList.x[outputList.ncorners-1]
    y1 = outputList.y[outputList.ncorners-1]
    for i in range(outputList.ncorners):
        x2 = outputList.x[i]
        y2 = outputList.y[i]
        area += x1 * y2 - x2 * y1
        x1, y1 = x2, y2
    area /= 2.0

    return area


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def _resample(double [:,:] image2d,
              double [:] xxx, double [:] yyy,
              long [:] ixx, long [:] iyy,
              int naxis1out=0, int naxis2out=0,
              int ioffx=0, int ioffy=0):
    cdef int naxis1, naxis2
    naxis1 = image2d.shape[1]
    naxis2 = image2d.shape[0]
    if naxis1out == 0:
        naxis1out = naxis1
    if naxis2out == 0:
        naxis2out = naxis2

    image2d_rect_np = np.zeros((naxis2out, naxis1out), dtype=np.double)
    cdef double [:,:] image2d_rect = image2d_rect_np  # memoryview

    cdef:
        long k, k4
        int kk
        int jmin, jmax, imin, imax
        int jdum, idum
        double [:] xvertices
        double [:] yvertices
        double xmin, xmax, ymin, ymax
        double xdum, ydum
        double pixel_area

    for k in range(naxis1out * naxis2out):
        k4 = k*4
        xvertices = xxx[(k4):(k4+4)]
        yvertices = yyy[(k4):(k4+4)]
        polygon1 = polygon4vert(xvertices[0], yvertices[0],
                                xvertices[1], yvertices[1],
                                xvertices[2], yvertices[2],
                                xvertices[3], yvertices[3])
        # limits to determine the pixels of the original (distorted) image
        # that can have intersection with polygon1
        xmin = xmax = xvertices[0]
        for kk in range(1,4):
            xdum = xvertices[kk]
            if xmax < xdum:
                xmax = xdum
            elif xmin > xdum:
                xmin = xdum
        ymin = ymax = yvertices[0]
        for kk in range(1,4):
            ydum = yvertices[kk]
            if ymax < ydum:
                ymax = ydum
            elif ymin > ydum:
                ymin = ydum
        jmin = int(xmin + 0.5)
        jmax = int(xmax + 0.5)
        imin = int(ymin + 0.5)
        imax = int(ymax + 0.5)
        # determine intersection of polygon1 with pixels in the original
        # (distorted) image
        pixel_area = 0
        for jdum in range(jmin, jmax + 1):
            if 0 <= jdum < naxis1:
                for idum in range(imin, imax + 1):
                    if 0 <= idum < naxis2:
                        # polygon corresponding to a particular pixel in
                        # the original (distorted) image
                        polygon2 = polygon4vert(jdum - 0.5, idum - 0.5,
                                                jdum + 0.5, idum - 0.5,
                                                jdum + 0.5, idum + 0.5,
                                                jdum - 0.5, idum + 0.5)
                        # add signal according to the area of the intersection
                        pixel_area += cclippix_area(polygon1, polygon2) * \
                            image2d[idum, jdum]
        image2d_rect[iyy[k], ixx[k]] = pixel_area

    return image2d_rect_np
