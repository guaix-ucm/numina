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

"""Utility functions to determine image distortions."""

import numpy as np
from skimage import transform

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplotxy import ximplotxy


def compute_distortion(x_orig, y_orig, x_rect, y_rect, order, debugplot):

    # protections
    npoints = len(x_orig)
    for xdum in [y_orig, x_rect, y_rect]:
        if len(xdum) != npoints:
            raise ValueError('Unexpected different number of points')

    # normalize ranges dividing by the maximum, so the
    # transformation fit will be computed with data points with
    # coordinates in the range [0,1]
    x_scale = 1.0 / np.concatenate((x_orig,
                                    x_rect)).max()
    y_scale = 1.0 / np.concatenate((y_orig,
                                    y_rect)).max()
    if abs(debugplot) >= 10:
        print("x_scale:", x_scale)
        print("y_scale:", y_scale)
    x_orig_scaled = x_orig * x_scale
    y_orig_scaled = y_orig * y_scale
    x_inter_scaled = x_rect * x_scale
    y_inter_scaled = y_rect * y_scale

    if abs(debugplot) % 10 != 0:
        ax = ximplotxy(x_orig_scaled, y_orig_scaled,
                       show=False,
                       **{'marker': 'o', 'color': 'cyan',
                          'label': 'original', 'linestyle': ''})
        dum = zip(x_orig_scaled, y_orig_scaled)
        for idum in range(len(dum)):
            ax.text(dum[idum][0], dum[idum][1], str(idum + 1), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='bottom', color='green')
        ax.plot(x_inter_scaled, y_inter_scaled, 'bo',
                label="rectified")
        dum = zip(x_inter_scaled, y_inter_scaled)
        for idum in range(len(dum)):
            ax.text(dum[idum][0], dum[idum][1], str(idum + 1), fontsize=10,
                    horizontalalignment='center',
                    verticalalignment='bottom', color='blue')
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
        ax.set_title("(estimate_tt_to_rectify #1)\n\n")
        # shrink current axis and put a legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.92])
        ax.legend(loc=3, bbox_to_anchor=(0., 1.02, 1., 0.07),
                  mode="expand", borderaxespad=0., ncol=4,
                  numpoints=1)
        pause_debugplot(debugplot, pltshow=True)

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
    ttd = transform.PolynomialTransform(
        np.vstack(
            [np.linalg.lstsq(a_matrix, x_orig_scaled)[0],
             np.linalg.lstsq(a_matrix, y_orig_scaled)[0]]
        )
    )

    # reverse normalization to recover coefficients of the
    # transformation in the correct system
    factor = np.zeros_like(ttd.params[0])
    k = 0
    for i in range(order + 1):
        for j in range(i + 1):
            factor[k] = (x_scale ** (i - j)) * (y_scale ** j)
            k += 1
    ttd_aij = ttd.params[0] * factor / x_scale
    ttd_bij = ttd.params[1] * factor / y_scale
    if abs(debugplot) >= 10:
        print("ttd_aij X:\n", ttd_aij)
        print("ttd_bij Y:\n", ttd_bij)

    return ttd_aij, ttd_bij


def fmap(order, aij, bij, x, y):
    """Evaluate the 2D polynomial transformation.

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
