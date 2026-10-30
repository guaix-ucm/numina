#
# Copyright 2015-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#


import numpy

from ._extract import extract_simple_intl
from .traces import axis_to_dispaxis


class Aperture:
    """Spectroscopic aperture.

    Parameters
    ----------
    bbox
        Bounding box of the aperture.
    borders
        Borders of the aperture.
    axis : {0, 1}, optional
        Spatial axis of the array.
    id : optional
        Identifier of the aperture.
    """

    def __init__(self, bbox, borders, axis=0, id=None):
        self.bbox = bbox
        self.borders = borders
        self.id = id
        self.axis = axis
        self.dispaxis = axis_to_dispaxis(axis)


def extract_simple_rss(arr, borders, axis=0, out=None):
    """Extract the spectra between pairs of borders by summing the pixels along the spatial axis.

    Partial pixels at the borders are weighted by the fraction covered.

    Parameters
    ----------
    arr : numpy.ndarray
        2D image.
    borders : sequence of tuple
        Tuples ``(idx, b1, b2)``, where ``idx`` is the 1-based index of the
        spectrum and ``b1``, ``b2`` are callables that return the lower and
        upper border for each pixel along the dispersion axis.
    axis : {0, 1}
        Spatial axis of the array.
    out : numpy.ndarray, optional
        Output array. If None, a new array is created with as many rows as the
        ``idx`` of the last element of ``borders``.

    Returns
    -------
    numpy.ndarray
        Array with one extracted spectrum per row.

    Raises
    ------
    ValueError
        If ``axis`` is not 0 or 1, or ``borders`` is empty.
    """

    # If arr is not in native byte order, the C-extension won't work
    if arr.dtype.byteorder != "=":
        arr2 = arr.byteswap().view(arr.dtype.newbyteorder())
    else:
        arr2 = arr

    if axis == 0:
        arr3 = arr2
    elif axis == 1:
        arr3 = arr2.T
    else:
        raise ValueError("'axis' must be 0 or 1")

    if len(borders) == 0:
        raise ValueError("'borders' must be greater than 0")

    if out is None:
        out = numpy.zeros((borders[-1][0], arr3.shape[1]), dtype="float")

    xx = numpy.arange(arr3.shape[1])

    # Borders contains a list of function objects
    # idx starts in 1
    for idx, b1, b2 in borders:
        bb1 = b1(xx)
        bb1[bb1 < -0.5] = -0.5
        bb2 = b2(xx)
        bb2[bb2 > arr3.shape[0] - 0.5] = arr3.shape[0] - 0.5
        extract_simple_intl(arr3, xx, bb1, bb2, out[idx - 1])
    return out


def extract_simple_rss_apers(arr, apers, axis=0, out=None):
    """Extract the spectra of a sequence of ``Aperture`` by summing the pixels along the spatial axis.

    Each spectrum is extracted only in the range of its bounding box along the
    dispersion axis; the rest of its row in ``out`` is not changed.

    Parameters
    ----------
    arr : numpy.ndarray
        2D image.
    apers : sequence of Aperture
        Apertures, all with the same ``axis`` as the array.
    axis : {0, 1}
        Spatial axis of the array.
    out : numpy.ndarray, optional
        Output array. If None, a new array of zeros with one row per aperture is created.

    Returns
    -------
    numpy.ndarray
        Array with one extracted spectrum per row.

    Raises
    ------
    ValueError
        If ``axis`` is not 0 or 1, or it is different from the axis of an aperture.
    """

    # If arr is not in native byte order, the C-extension won't work
    if arr.dtype.byteorder != "=":
        arr2 = arr.byteswap().view(arr.dtype.newbyteorder())
    else:
        arr2 = arr

    if axis == 0:
        arr3 = arr2
        i1, i2 = 0, 2
    elif axis == 1:
        arr3 = arr2.T
        i1, i2 = 2, 4
    else:
        raise ValueError("'axis' must be 0 or 1")

    if out is None:
        out = numpy.zeros((len(apers), arr3.shape[1]), dtype="float")

    for idx, aper in enumerate(apers):
        if aper.axis != axis:
            raise ValueError("array 'axis' and aperture 'axis' are different")

        u1, u2 = aper.bbox[i1:i2]
        uu = numpy.arange(u1, u2 + 1)

        # Borders contains a list of function objects
        b1, b2 = aper.borders
        bb1 = b1(uu)
        bb1[bb1 < -0.5] = -0.5
        bb2 = b2(uu)
        bb2[bb2 > arr3.shape[0] - 0.5] = arr3.shape[0] - 0.5
        extract_simple_intl(arr3, uu, bb1, bb2, out[idx])

    return out
