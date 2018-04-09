import numpy as np


# Sutherland–Hodgman algorithm implementation in Cython
# https://stackoverflow.com/questions/44765229/working-with-variable-sized-lists-with-cython
cdef compute_intersection(double cp10, double cp11, double cp20, double cp21,
                          double s0, double s1, double e0, double e1):
    dc0, dc1 = cp10 - cp20, cp11 - cp21
    dp0, dp1 =  s0 - e0, s1 - e1
    n1 = cp10 * cp21 - cp11 * cp20
    n2 = s0 * e1 - s1 * e0
    n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
    return (n1 * dp0 - n2 * dc0) * n3, (n1 * dp1 - n2 * dc1) * n3


cdef cclippix_area(subjectPolygon, clipPolygon):
    cdef double cp10, cp11, cp20, cp21
    cdef double s0, s1, e0, e1
    cdef double s_in, e_in

    outputList = subjectPolygon
    cp10, cp11 = clipPolygon[-1]

    for cp20, cp21 in clipPolygon:

        inputList = outputList
        outputList = []
        s0, s1 = inputList[-1]
        s_in = (cp20 - cp10) * (s1 - cp11) - (cp21 - cp11) * (s0 - cp10)
        for e0, e1  in inputList:
            e_in = (cp20 - cp10) * (e1 - cp11) - (cp21 - cp11) * (e0 - cp10)
            if e_in > 0:
                if s_in <= 0:
                    outputList.append(
                        compute_intersection(cp10, cp11, cp20, cp21,
                                             s0, s1, e0, e1)
                    )
                outputList.append((e0, e1))
            elif s_in > 0:
                outputList.append(
                    compute_intersection(cp10, cp11, cp20, cp21,
                                         s0, s1, e0, e1)
                )
            s0, s1, s_in = e0, e1, e_in
        if len(outputList) < 1:
            return 0.0
        cp10, cp11 = cp20, cp21

    cdef double x1, y1, x2, y2
    cdef double area = 0
    x1, y1 = outputList[-1]
    for x2, y2 in outputList:
        area += x1 * y2 - x2 * y1
        x1, y1 = x2, y2
    area /= 2

    return area


def _clippix_area(subjectPolygon, clipPolygon):
    return cclippix_area(subjectPolygon, clipPolygon)


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

    cdef long k
    cdef int kk
    cdef int jmin, jmax, imin, imax
    cdef int jdum, idum
    cdef double [:] xvertices
    cdef double [:] yvertices
    cdef double xmin, xmax, ymin, ymax
    cdef double xdum, ydum

    for k in range(naxis1out * naxis2out):
        xvertices = xxx[(k*4):(k*4+4)]
        yvertices = yyy[(k*4):(k*4+4)]
        polygon1 = ((xvertices[0], yvertices[0]),
                    (xvertices[1], yvertices[1]),
                    (xvertices[2], yvertices[2]),
                    (xvertices[3], yvertices[3]))
        # limits to determine the pixels of the original (distorted) image
        # that can have intersection with polygon1
        xmin = xmax = xvertices[0]
        for kk in range(3):
            xdum = xvertices[kk+1]
            if xmax < xdum:
                xmax = xdum
            elif xmin > xdum:
                xmin = xdum
        ymin = ymax = yvertices[0]
        for kk in range(3):
            ydum = yvertices[kk+1]
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
        image2d_rect[iyy[k], ixx[k]] = 0
        for jdum in range(jmin, jmax + 1):
            if 0 <= jdum < naxis1:
                for idum in range(imin, imax + 1):
                    if 0 <= idum < naxis2:
                        # polygon corresponding to a particular pixel in
                        # the original (distorted) image
                        polygon2 = ((jdum - 0.5, idum - 0.5),
                                    (jdum + 0.5, idum - 0.5),
                                    (jdum + 0.5, idum + 0.5),
                                    (jdum - 0.5, idum + 0.5))
                        # add signal according to the area of the intersection
                        image2d_rect[iyy[k], ixx[k]] += \
                            cclippix_area(polygon1, polygon2) * \
                            image2d[idum, jdum]

    return image2d_rect_np
