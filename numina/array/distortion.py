#
# Copyright 2015-2017 Universidad Complutense de Madrid
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

"""Utility functions to handle image distortions."""

from __future__ import division
from __future__ import print_function

import numpy as np
from skimage import transform

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplotxy import ximplotxy


def compute_distortion(x_orig, y_orig, x_rect, y_rect, order, debugplot):
    """Compute image distortion transformation.

    This function computes the following 2D transformation:
    x_orig = sum[i=0:order]( sum[j=0:j]( a_ij * x_rect**(i - j) * y_rect**j ))
    y_orig = sum[i=0:order]( sum[j=0:j]( b_ij * x_rect**(i - j) * y_rect**j ))

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

    # normalize ranges dividing by the maximum, so that the transformation
    # fit will be computed with data points with coordinates in the range [0,1]
    x_scale = 1.0 / np.concatenate((x_orig,
                                    x_rect)).max()
    y_scale = 1.0 / np.concatenate((y_orig,
                                    y_rect)).max()
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
            [np.linalg.lstsq(a_matrix, x_orig_scaled)[0],
             np.linalg.lstsq(a_matrix, y_orig_scaled)[0]]
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
                       **{'marker': 'o', # 'color': 'cyan',
                          'label': '(u,v) coordinates', 'linestyle': ''})
        dum = zip(x_orig_scaled, y_orig_scaled)
        for idum in range(len(dum)):
            ax.text(dum[idum][0], dum[idum][1], str(idum + 1), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='bottom', color='black')
        ax.plot(x_inter_scaled, y_inter_scaled, 'o',
                label="(x,y) coordinates")
        dum = zip(x_inter_scaled, y_inter_scaled)
        for idum in range(len(dum)):
            ax.text(dum[idum][0], dum[idum][1], str(idum + 1), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='bottom', color='grey')
        xmin = np.concatenate((x_orig_scaled,
                               x_inter_scaled)).min()
        xmax = np.concatenate((x_orig_scaled,
                               x_inter_scaled)).max()
        ymin = np.concatenate((y_orig_scaled,
                               y_inter_scaled)).min()
        ymax = np.concatenate((y_orig_scaled,
                               y_inter_scaled)).max()
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

    u = sum[i=0:order]( sum[j=0:j]( a_ij * x**(i - j) * y**j ))
    v = sum[i=0:order]( sum[j=0:j]( b_ij * x**(i - j) * y**j ))

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
