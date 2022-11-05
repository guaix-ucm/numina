#
# Copyright 2018 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#


import math
import numbers


class PixelInterval1D(object):
    """One dimensional bounding box with integer indices.

    Upper value (pix2) is not included in the
    box, as in slices.

    Parameters
    ----------
    pix1, pix2 : int
    axis : int

    """
    def __init__(self, pix1, pix2, axis=0):

        for name in ['pix1', 'pix2']:
            value = locals()[name]
            if not isinstance(value, numbers.Integral):
                raise TypeError(f"'{name}' must be integer in axis {axis}")

        if pix2 < pix1:
            raise ValueError(f"'pix2' must be >= 'pix1' in axis {axis}")

        self.pix1 = pix1
        self.pix2 = pix2
        self.ndim = 1
        self.axis = axis
        self._empty = (self.pix2 == self.pix1)

    @property
    def shape(self):
        return self.pix2 - self.pix1

    @property
    def slice(self):
        return slice(self.pix1, self.pix2)

    @classmethod
    def from_coordinates(cls, x1, x2):
        pix1, pix2 = cls.pixel_range(x1, x2)
        return cls(pix1, pix2)

    @classmethod
    def pixel_range(cls, x1, x2):
        iws = [int(math.floor(w + 0.5)) for w in [x1, x2]]
        pix1, pix2 = iws
        return pix1, pix2 + 1

    def union(self, other):
        if self:
            if other:
                npix2 = max(self.pix2, other.pix2)
                npix1 = min(self.pix1, other.pix1)
                return PixelInterval1D(npix1, npix2, axis=self.axis)
            else:
                return self
        else:
            return other

    def intersection(self, other):
        if self:
            if other:
                if (other.pix1 < self.pix2) and (self.pix1 < other.pix2):
                    npix1 = max(self.pix1, other.pix1)
                    npix2 = min(self.pix2, other.pix2)
                    return PixelInterval1D(npix1, npix2, axis=self.axis)
                else:
                    return PixelInterval1D(self.pix2, self.pix2, axis=self.axis)
            else:
                return self
        else:
            return other

    def __eq__(self, other):
        if isinstance(other, type(self)):
            if not (self or other):
                # if both are empty, are equal
                return True
            if self.pix1 != other.pix1:
                return False
            if self.pix2 != other.pix2:
                return False
            return True
        else:
            return NotImplemented

    def __ne__(self, other):
        return not (self == other)

    def __len__(self):
        return self.shape


class PixelInterval(object):
    """N-dimensional bounding box with integer indices.

    Upper value (pix2) is not included in the
    box, as in slices.

    Parameters
    ----------
    args : N tuples of integer pairs (pix1, pix2)

    """
    def __init__(self, *args):
        self.ndim = len(args)
        self.intervals = [PixelInterval1D(a, b, axis) for axis, (a, b) in enumerate(args)]

    @property
    def shape(self):
        return tuple(intl.shape for intl in self.intervals)

    @property
    def slice(self):
        return tuple(intl.slice for intl in self.intervals)

    @classmethod
    def pixel_range(cls, *args):
        return tuple(PixelInterval1D.pixel_range(*arg)
                     for arg in args)

    @classmethod
    def from_coordinates(cls, *args):
        newargs = cls.pixel_range(*args)
        return cls(*newargs)

    @classmethod
    def from_intervals(cls, intervals):
        result = super(PixelInterval, cls).__new__(cls)
        result.intervals = list(intervals)
        # update axis
        for idx, intl in enumerate(result.intervals):
            intl.axis = idx
        result.ndim = len(result.intervals)
        return result

    def union(self, other):
        if self.ndim == other.ndim:
            intl = [s.union(o) for s,o in zip(self.intervals, other.intervals)]
            return self.from_intervals(intl)
        else:
            raise NotImplementedError

    def intersection(self, other):
        if self.ndim == other.ndim:
            intl = [s.intersection(o) for s,o in zip(self.intervals, other.intervals)]
            return self.from_intervals(intl)
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, type(self)):

            if self.ndim != other.ndim:
                return False
            if not (self or other):
                # if both are empty, are equal
                return True

            return all(m == o for m, o in zip(self.intervals, other.intervals))
        else:
            return NotImplemented

    def __ne__(self, other):
        return not (self == other)

    def __len__(self):
        if self.intervals:
            result = 1
            for intl in self.intervals:
                result *= len(intl)
        else:
            result = 0
        return result


class BoundingBox(PixelInterval):
    """Rectangular bounding box with integer indices.

    Upper values (ix2 and iy2) are not included in the
    box, as in slices.

    Parameters
    ----------
    ix1, ix2, iy1, iy2 : int

    """
    def __init__(self, ix1, ix2, iy1, iy2):
        newargs = (iy1, iy2), (ix1, ix2)
        super(BoundingBox, self).__init__(*newargs)

    @classmethod
    def from_coordinates(cls, x1, x2, y1, y2):
        newargs = (y1, y2), (x1, x2)
        dum = cls.pixel_range(*newargs)
        ix1, ix2 = dum[1]
        iy1, iy2 = dum[0]
        return cls(ix1, ix2, iy1, iy2)

    @property
    def extent(self):
        """Helper for matplotlib imshow"""
        return (
            self.intervals[1].pix1 - 0.5,
            self.intervals[1].pix2 - 0.5,
            self.intervals[0].pix1 - 0.5,
            self.intervals[0].pix2 - 0.5,
        )
