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


"""Fit offset and rotation"""

import numpy
import numpy.linalg as linalg


def fit_offset_and_rotation(coords0, coords1):
    """Fit a rotation and a traslation between two sets points.

    Fit a rotation matrix and a traslation bewtween two matched sets
    consisting of M N-dimensional points

    Parameters
    ----------
    coords0 : (M, N) array_like
    coords1 : (M, N) array_lke

    Returns
    -------
    offset : (N, ) array_like
    rotation : (N, N) array_like

    Notes
    ------
    Fit offset and rotation using Kabsch's algorithm[1]_ [2]_

    .. [1] Kabsch algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm

    .. [2] Also here: http://nghiaho.com/?page_id=671

    """

    cP = coords0.mean(axis=0)
    cQ = coords1.mean(axis=0)

    P0 = coords0 - cP
    Q0 = coords1 - cQ

    crossvar = numpy.dot(P0.T, Q0)

    u, s, vt = linalg.svd(crossvar)

    d = linalg.det(u) * linalg.det(vt)

    if d < 0:
        s[-1] = -s[-1]
        vt[:, -1] = -vt[:, -1]

    rot = numpy.dot(vt, u)
    off = -numpy.dot(rot, cP) + cQ

    return off, rot
