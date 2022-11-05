#
# Copyright 2016-2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


"""Fit offset and rotation"""

import numpy
import numpy.linalg as linalg


def fit_offset_and_rotation(coords0, coords1):
    """Fit a rotation and a translation between two sets points.

    Fit a rotation matrix and a translation between two matched sets
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
    Fit offset and rotation using Kabsch's algorithm [1]_ [2]_

    .. [1] Kabsch algorithm: https://en.wikipedia.org/wiki/Kabsch_algorithm

    .. [2] Also here: http://nghiaho.com/?page_id=671

    """

    coords0 = numpy.asarray(coords0)
    coords1 = numpy.asarray(coords1)

    cp = coords0.mean(axis=0)
    cq = coords1.mean(axis=0)

    p0 = coords0 - cp
    q0 = coords1 - cq

    crossvar = numpy.dot(numpy.transpose(p0), q0)

    u, _, vt = linalg.svd(crossvar)

    d = linalg.det(u) * linalg.det(vt)

    if d < 0:
        u[:, -1] = -u[:, -1]

    rot = numpy.transpose(numpy.dot(u, vt))
    # Operation is
    # B - B0 = R(A - A0)
    # So off is B0 -R * A0
    # The inverse operation is
    # A - A0 = R* (B- B0)
    # So inverse off* is A - R* B0
    # where R* = transpose(R)
    #  R * off* = -off
    off = -numpy.dot(rot, cp) + cq
    return off, rot
