#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Utility functions to handle image distortions."""

import numpy as np
from skimage import transform

from numina.array._clippix import _resample
from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplotxy import ximplotxy

NMAX_ORDER = 4


def compute_distortion(x_orig, y_orig, x_rect, y_rect, order, debugplot):
    """Compute image distortion transformation.

    This function computes the following 2D transformation:
    x_orig = sum[i=0:order]( sum[j=0:i]( a_ij * x_rect**(i - j) * y_rect**j ))
    y_orig = sum[i=0:order]( sum[j=0:i]( b_ij * x_rect**(i - j) * y_rect**j ))

    Parameters
    ----------
    x_orig : numpy array
        X coordinate of the reference points in the distorted image
    y_orig : numpy array
        Y coordinate of the reference points in the distorted image
    x_rect : numpy array
        X coordinate of the reference points in the rectified image
    y_rect : numpy array
        Y coordinate of the reference points in the rectified image
    order : int
        Order of the polynomial transformation
    debugplot : int
        Determine whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------

    aij : numpy array
        Coefficients a_ij of the 2D transformation.
    bij : numpy array
        Coefficients b_ij of the 2D transformation.

    """

    # protections
    npoints = len(x_orig)
    for xdum in [y_orig, x_rect, y_rect]:
        if len(xdum) != npoints:
            raise ValueError('Unexpected different number of points')
    if order < 1 or order > NMAX_ORDER:
        raise ValueError("Invalid order=" + str(order))

    # normalize ranges dividing by the maximum, so that the transformation
    # fit will be computed with data points with coordinates in the range [0,1]
    x_scale = 1.0 / np.concatenate((x_orig, x_rect)).max()
    y_scale = 1.0 / np.concatenate((y_orig, y_rect)).max()
    x_orig_scaled = x_orig * x_scale
    y_orig_scaled = y_orig * y_scale
    x_inter_scaled = x_rect * x_scale
    y_inter_scaled = y_rect * y_scale

    # solve 2 systems of equations with half number of unknowns each
    if order == 1:
        a_matrix = np.vstack([np.ones(npoints),
                              x_inter_scaled,
                              y_inter_scaled]).T
    elif order == 2:
        a_matrix = np.vstack([np.ones(npoints),
                              x_inter_scaled,
                              y_inter_scaled,
                              x_inter_scaled ** 2,
                              x_inter_scaled * y_orig_scaled,
                              y_inter_scaled ** 2]).T
    elif order == 3:
        a_matrix = np.vstack([np.ones(npoints),
                              x_inter_scaled,
                              y_inter_scaled,
                              x_inter_scaled ** 2,
                              x_inter_scaled * y_orig_scaled,
                              y_inter_scaled ** 2,
                              x_inter_scaled ** 3,
                              x_inter_scaled ** 2 * y_inter_scaled,
                              x_inter_scaled * y_inter_scaled ** 2,
                              y_inter_scaled ** 3]).T
    elif order == 4:
        a_matrix = np.vstack([np.ones(npoints),
                              x_inter_scaled,
                              y_inter_scaled,
                              x_inter_scaled ** 2,
                              x_inter_scaled * y_orig_scaled,
                              y_inter_scaled ** 2,
                              x_inter_scaled ** 3,
                              x_inter_scaled ** 2 * y_inter_scaled,
                              x_inter_scaled * y_inter_scaled ** 2,
                              y_inter_scaled ** 3,
                              x_inter_scaled ** 4,
                              x_inter_scaled ** 3 * y_inter_scaled ** 1,
                              x_inter_scaled ** 2 * y_inter_scaled ** 2,
                              x_inter_scaled ** 1 * y_inter_scaled ** 3,
                              y_inter_scaled ** 4]).T
    else:
        raise ValueError("Invalid order=" + str(order))
    poltrans = transform.PolynomialTransform(
        np.vstack(
            [np.linalg.lstsq(a_matrix, x_orig_scaled, rcond=None)[0],
             np.linalg.lstsq(a_matrix, y_orig_scaled, rcond=None)[0]]
        )
    )

    # reverse normalization to recover coefficients of the
    # transformation in the correct system
    factor = np.zeros_like(poltrans.params[0])
    k = 0
    for i in range(order + 1):
        for j in range(i + 1):
            factor[k] = (x_scale ** (i - j)) * (y_scale ** j)
            k += 1
    aij = poltrans.params[0] * factor / x_scale
    bij = poltrans.params[1] * factor / y_scale

    # show results
    if abs(debugplot) >= 10:
        print(">>> u=u(x,y) --> aij:\n", aij)
        print(">>> v=v(x,y) --> bij:\n", bij)

    if abs(debugplot) % 10 != 0:
        ax = ximplotxy(x_orig_scaled, y_orig_scaled,
                       show=False,
                       **{'marker': 'o',
                          'label': '(u,v) coordinates', 'linestyle': ''})
        dum = list(zip(x_orig_scaled, y_orig_scaled))
        for idum in range(len(dum)):
            ax.text(dum[idum][0], dum[idum][1], str(idum + 1), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='bottom', color='black')
        ax.plot(x_inter_scaled, y_inter_scaled, 'o',
                label="(x,y) coordinates")
        dum = list(zip(x_inter_scaled, y_inter_scaled))
        for idum in range(len(dum)):
            ax.text(dum[idum][0], dum[idum][1], str(idum + 1), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='bottom', color='grey')
        xmin = np.concatenate((x_orig_scaled, x_inter_scaled)).min()
        xmax = np.concatenate((x_orig_scaled, x_inter_scaled)).max()
        ymin = np.concatenate((y_orig_scaled, y_inter_scaled)).min()
        ymax = np.concatenate((y_orig_scaled, y_inter_scaled)).max()
        dx = xmax - xmin
        xmin -= dx / 20
        xmax += dx / 20
        dy = ymax - ymin
        ymin -= dy / 20
        ymax += dy / 20
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel("pixel (normalized coordinate)")
        ax.set_ylabel("pixel (normalized coordinate)")
        ax.set_title("compute distortion")
        ax.legend()
        pause_debugplot(debugplot, pltshow=True)

    return aij, bij


def fmap(order, aij, bij, x, y):
    """Evaluate the 2D polynomial transformation.

    u = sum[i=0:order]( sum[j=0:i]( a_ij * x**(i - j) * y**j ))
    v = sum[i=0:order]( sum[j=0:i]( b_ij * x**(i - j) * y**j ))

    Parameters
    ----------
    order : int
        Order of the polynomial transformation.
    aij : numpy array
        Polynomial coefficents corresponding to a_ij.
    bij : numpy array
        Polynomial coefficents corresponding to b_ij.
    x : numpy array or float
        X coordinate values where the transformation is computed. Note
        that these values correspond to array indices.
    y : numpy array or float
        Y coordinate values where the transformation is computed. Note
        that these values correspond to array indices.

    Returns
    -------
    u : numpy array or float
        U coordinate values.
    v : numpy array or float
        V coordinate values.

    """

    u = np.zeros_like(x)
    v = np.zeros_like(y)

    k = 0
    for i in range(order + 1):
        for j in range(i + 1):
            u += aij[k] * (x ** (i - j)) * (y ** j)
            v += bij[k] * (x ** (i - j)) * (y ** j)
            k += 1

    return u, v


def ncoef_fmap(order):
    """Expected number of coefficients in a 2D transformation of a given order.

    Parameters
    ----------
    order : int
        Order of the 2D polynomial transformation.

    Returns
    -------
    ncoef : int
        Expected number of coefficients.

    """

    ncoef = 0
    for i in range(order + 1):
        for j in range(i + 1):
            ncoef += 1
    return ncoef


def order_fmap(ncoef):
    """Compute order corresponding to a given number of coefficients.

    Parameters
    ----------
    ncoef : int
        Number of coefficients.

    Returns
    -------
    order : int
        Order corresponding to the provided number of coefficients.

    """

    loop = True
    order = 1
    while loop:
        loop = not (ncoef == ncoef_fmap(order))
        if loop:
            order += 1
            if order > NMAX_ORDER:
                print('No. of coefficients: ', ncoef)
                raise ValueError("order > " + str(NMAX_ORDER) + " not implemented")
    return order


def rectify2d(image2d, aij, bij, resampling,
              naxis1out=None, naxis2out=None,
              ioffx=None, ioffy=None,
              debugplot=0):
    """Rectify image applying the provided 2D transformation.

    The rectified image correspond to the transformation given by:
        u = sum[i=0:order]( sum[j=0:i]( a_ij * x**(i - j) * y**j ))
        v = sum[i=0:order]( sum[j=0:i]( b_ij * x**(i - j) * y**j ))

    Parameters
    ----------
    image2d : 2d numpy array
        Initial image.
    aij : 1d numpy array
        Coefficients a_ij of the transformation.
    bij : 1d numpy array
        Coefficients b_ij of the transformation.
    resampling : int
        1: nearest neighbour, 2: flux preserving interpolation.
    naxis1out : int or None
        X-axis dimension of output image.
    naxis2out : int or None
        Y-axis dimension of output image.
    ioffx : int
        Integer offset in the X direction.
    ioffy : int
        Integer offset in the Y direction.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot

    Returns
    -------
    image2d_rect : 2d numpy array
        Rectified image.

    """

    # protections
    ncoef = len(aij)
    if len(bij) != ncoef:
        raise ValueError("aij and bij lengths are different!")

    # order of the polynomial transformation
    order = order_fmap(ncoef)
    if abs(debugplot) >= 10:
        print('--> rectification order:', order)

    # initial image dimension
    naxis2, naxis1 = image2d.shape

    # output image dimension
    if naxis1out is None:
        naxis1out = naxis1
    if naxis2out is None:
        naxis2out = naxis2
    if ioffx is None:
        ioffx = 0
    if ioffy is None:
        ioffy = 0

    if resampling == 1:
        # pixel coordinates (rectified image); since the fmap function
        # below requires floats, these arrays must use dtype=float
        j = np.arange(0, naxis1out, dtype=float) - ioffx
        i = np.arange(0, naxis2out, dtype=float) - ioffy
        # the cartesian product of the previous 1D arrays could be stored
        # as np.transpose([xx,yy]), where xx and yy are computed as follows
        xx = np.tile(j, (len(i),))
        yy = np.repeat(i, len(j))
        # compute pixel coordinates in original (distorted) image
        xxx, yyy = fmap(order, aij, bij, xx, yy)
        # round to the nearest integer and cast to integer; note that the
        # rounding still provides a float, so the casting is required
        ixxx = np.rint(xxx).astype(int)
        iyyy = np.rint(yyy).astype(int)
        # determine pixel coordinates within available image
        lxxx = np.logical_and(ixxx >= 0, ixxx < naxis1)
        lyyy = np.logical_and(iyyy >= 0, iyyy < naxis2)
        lok = np.logical_and(lxxx, lyyy)
        # assign pixel values to rectified image
        ixx = xx.astype(int)[lok]
        iyy = yy.astype(int)[lok]
        ixxx = ixxx[lok]
        iyyy = iyyy[lok]
        # initialize result
        image2d_rect = np.zeros((naxis2out, naxis1out), dtype=float)
        # rectified image
        image2d_rect[iyy + ioffy, ixx + ioffx] = image2d[iyyy, ixxx]
    elif resampling == 2:
        # coordinates (rectified image) of the four corners, sorted in
        # anticlockwise order, of every pixel
        j = np.array(
            [[k - 0.5 - ioffx, k + 0.5 - ioffx,
              k + 0.5 - ioffx, k - 0.5 - ioffx] for k in range(naxis1out)]
        )
        i = np.array(
            [[k - 0.5 - ioffy, k - 0.5 - ioffy,
              k + 0.5 - ioffy, k + 0.5 - ioffy] for k in range(naxis2out)]
        )
        xx = np.reshape(np.tile(j, naxis2out), naxis1out * naxis2out * 4)
        yy = np.concatenate([np.reshape(i, naxis2out * 4)] * naxis1out)
        # compute pixel coordinates in original (distorted) image
        xxx, yyy = fmap(order, aij, bij, xx, yy)
        # indices of pixels in the rectified image
        ixx = np.repeat(np.arange(naxis1out), naxis2out)
        iyy = np.tile(np.arange(naxis2out), (naxis1out,))
        # rectified image (using cython function)
        image2d_rect = _resample(image2d, xxx, yyy, ixx, iyy, naxis1out, naxis2out)
    else:
        raise ValueError("Sorry, resampling method must be 1 or 2")

    # return result
    return image2d_rect


def shift_image2d(image2d, xoffset=0.0, yoffset=0.0, resampling=2):
    """Shift image applying arbitray X and Y offsets.

    Parameters
    ----------
    image2d : 2d numpy array
        Initial image.
    xoffset : float
        Offset in the X direction.
    yoffset : float
        Offset in the Y direction.
    resampling : int
        1: nearest neighbour, 2: flux preserving interpolation.

    Returns
    -------
    image2d_shifted : 2d numpy array
        Rectified image.

    """

    aij = np.array([-xoffset, 1.0, 0.0], dtype=float)
    bij = np.array([-yoffset, 0.0, 1.0], dtype=float)
    image2d_shifted = rectify2d(image2d.astype('double'),
                                aij, bij, resampling=resampling)
    return image2d_shifted


def rotate_image2d(image2d, theta_deg, xcenter, ycenter, fscale=1.0, resampling=2):
    """Shift image applying arbitray X and Y offsets.

    Parameters
    ----------
    image2d : 2d numpy array
        Initial image.
    theta_deg : float
        Rotation angle (positive values correspond to counter-clockwise
        angles).
    xcenter : float
        X coordinate of center of rotation, in pixel coordinates (i.e.,
        the image X coordinates run from 0.5 to NAXIS1+0.5).
    ycenter : float
        Y coordinate of center of rotation, in pixel coordinates (i.e.,
        the image Y coordinates run from 0.5 to NAXIS2+0.5).
    fscale : float
        Scale factor (1.0: no change in scale).
    resampling : int
        1: nearest neighbour, 2: flux preserving interpolation.

    Returns
    -------
    image2d_shifted : 2d numpy array
        Rectified image.

    """

    f = 1/fscale
    theta_rad = theta_deg * np.pi/180
    costheta = np.cos(theta_rad)
    sintheta = np.sin(theta_rad)
    xc = xcenter - 1.0
    yc = ycenter - 1.0
    aij = [-f*xc*costheta-f*yc*sintheta+xc, f*costheta, f*sintheta]
    bij = [f*xc*sintheta-f*yc*costheta+yc, -f*sintheta, f*costheta]
    image2d_rotated = rectify2d(image2d.astype('double'), aij, bij, resampling=resampling)
    return image2d_rotated
