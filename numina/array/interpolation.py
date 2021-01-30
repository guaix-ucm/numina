#
# Copyright 2016-2020 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""A monotonic piecewise cubic interpolator."""

import numpy as np


class SteffenInterpolator(object):
    """
    A monotonic piecewise cubic 1-d interpolator.

    A monotonic piecewise cubic interpolator based on
    Steffen, M., Astronomy & Astrophysics, 239, 443-450 (1990)

    `x` and `y` are arrays of values used to approximate some function f: `y = f(x)`.
    This class returns an object whose call method uses
    monotonic cubic splines to find the value of new points.

    Parameters
    ----------
    x : (N,), array_like
        A 1-D array of real values, sorted monotonically increasing.
    y : (N,), array_like
        A 1-D array of real values.
    yp_0 : float, optional
        The value of the derivative in the first sample.
    yp_N : float, optional
        The value of the derivative in the last sample.
    extrapolate : str, optional
        Specifies the kind of extrapolation as a string
        ('extrapolate', 'zeros', 'raise', 'const', 'border')
        If 'raise' is set, when interpolated values are requested outside of the
        domain of the input data (x,y), a ValueError is raised.
        If 'const' is set, 'fill_value' is returned.
        If 'zeros' is set, '0' is returned.
        If 'border' is set, 'y[0]' is returned for values below 'x[0]'
        and 'y[N-1]' is returned for values above 'x[N-1]'
        If 'extrapolate' is set, the extreme polynomial are extrapolated
        outside of their ranges.
        Default is 'raise'.
    fill_value : float, optional
        If provided, then this value will be used to fill in for requested
        points outside of the data range when `extrapolation`is set to `const`.
        If not provided, then the default is NaN.

    """
    def __init__(self, x, y, yp_0=0.0, yp_N=0.0, extrapolate='raise', fill_value=np.nan):

        # 'zeros' is a shortcut
        if extrapolate == 'zeros':
            extrapolate = 'const'
            fill_value = 0.0

        self.fill_value = fill_value
        self._extrapolation(extrapolate)

        # Compute all coefficients

        N = len(x)

        # Steps
        h = self._create_h(x)
        # Secants
        s = self._create_s(y, h)

        abs_s = np.abs(s)

        # Check monotonicity given borders
        if yp_0 * s[0] < 0:
            raise ValueError('monotonicity not guaranteed')

        if abs(yp_0) > 3*abs_s[0]:
            raise ValueError('monotonicity not guaranteed')

        if yp_N * s[-2] < 0:
            raise ValueError('monotonicity not guaranteed')

        if abs(yp_N) > 3*abs_s[-2]:
            raise ValueError('monotonicity not guaranteed')

        p = self._create_p(s, h)

        # Derivatives
        yp = np.zeros((N+1,))
        yp[0] = yp_0
        yp[N] = yp_N
        sign_s = np.sign(s)
        min_abs_s = np.minimum(abs_s[1:], abs_s[:-1])
        yp[1:-1] = (sign_s[1:] + sign_s[:-1]) * np.minimum(min_abs_s, 0.5 * np.abs(p[1:]))

        # Polynomial coefficients
        self._d = y
        self._c = yp[:-1]
        self._b = (3 * s - 2 * yp[:-1] - yp[1:]) / h
        self._a = (yp[:-1] + yp[1:] - 2 * s) / h**2
        # Other values
        self._x = x

    def _extrapolation(self, extrapolate):
        """Check permitted values of extrapolation."""
        modes = ['extrapolate',
                 'raise',
                 'const',
                 'border']
        if extrapolate not in modes:
            msg = f'invalid extrapolation mode {extrapolate}'
            raise ValueError(msg)

        if extrapolate == 'raise':
            self.bounds_error = True
            self.extrapolate = False
        else:
            self.extrapolate = True
            self.bounds_error = False
        self.extrapolate_mode = extrapolate

    @staticmethod
    def _create_s(y, h):
        """Estimate secants"""
        s = np.zeros_like(y)
        s[:-1] = (y[1:] - y[:-1]) / h[:-1]
        s[-1] = 0.0
        return s

    @staticmethod
    def _create_h(x):
        """increase between samples"""
        h = np.zeros_like(x)
        h[:-1] = x[1:] - x[:-1]
        # border
        h[-1] = h[-2]
        return h

    @staticmethod
    def _create_p(s, h):
        """Parabolic derivative"""
        p = np.zeros_like(s)
        p[1:] = (s[:-1]*h[1:] + s[1:] * h[:-1]) / (h[1:] + h[:-1])
        return p

    def _eval(self, v, in_bounds, der):
        """Eval polynomial inside bounds."""
        result = np.zeros_like(v, dtype='float')
        x_indices = np.searchsorted(self._x, v, side='rigth')
        ids = x_indices[in_bounds] - 1
        u = v[in_bounds] - self._x[ids]
        result[in_bounds] = self._poly_eval(u, ids, der)
        return result

    def _extrapolate(self, result, v, below_bounds, above_bounds, der):
        """Extrapolate result based on extrapolation mode."""
        if self.extrapolate_mode == 'const':
            fill_b = fill_a = self.fill_value
        elif self.extrapolate_mode == 'border':
            fill_b = self._poly_eval(0, 0, der)
            fill_a = self._poly_eval(0, -1, der)
        elif self.extrapolate_mode == 'extrapolate':
            u = v[above_bounds] - self._x[-2]
            fill_a = self._poly_eval(u, -2, der)
            u = v[below_bounds] - self._x[0]
            fill_b = self._poly_eval(u, 0, der)
        else:
            raise ValueError("extrapolation method doesn't exist")

        result[below_bounds] = fill_b
        result[above_bounds] = fill_a

    def __call__(self, x_new, der=0):
        """
        Evaluate the Steffen interpolant

        Parameters
        ----------
        x_new : array-like
              Points to evaluate the interpolant at.

        der : int, optional
              Degree of the derivative. 0 is the plain interpolant.
              For der >= 4, the return value is 0
        Returns
        -------
        y : array-like
            Interpolated values.

        """

        v = np.asarray(x_new)

        below_bounds, above_bounds = self._check_bounds(v)
        in_bounds = np.logical_and(~below_bounds, ~above_bounds)

        result = self._eval(v, in_bounds, der)

        if self.extrapolate:
            self._extrapolate(result, v, below_bounds, above_bounds, der)
        return result

    def _poly_eval(self, u, ids, der=0):
        """Evaluate internal polynomial."""
        if der == 0:
            return self._poly_eval_0(u, ids)
        elif der == 1:
            return self._poly_eval_1(u, ids)
        elif der == 2:
            return self._poly_eval_2(u, ids)
        elif der == 3:
            return self._poly_eval_3(u, ids)
        elif der >= 4:
            return self._poly_eval_4(u, ids)
        else:
            raise ValueError(f"der={der} is impossible")

    def _poly_eval_0(self, u, ids):
        """Evaluate internal polynomial."""
        return u * (u * (self._a[ids] * u + self._b[ids]) + self._c[ids]) + self._d[ids]

    def _poly_eval_1(self, u, ids):
        """Evaluate internal polynomial."""
        return (u * (3*self._a[ids] * u + 2*self._b[ids]) + self._c[ids])

    def _poly_eval_2(self, u, ids):
        """Evaluate internal polynomial."""
        return 6 * self._a[ids] * u + 2*self._b[ids]

    def _poly_eval_3(self, u, ids):
        """Evaluate internal polynomial."""
        return 6 * self._a[ids]

    def _poly_eval_4(self, u, ids):
        """Evaluate internal polynomial."""
        return u *0.0

    def _check_bounds(self, v):
        """Check which values are out of bounds.

        Raises
        ------
        ValueError:

        """
        below_bounds = v < self._x[0]
        above_bounds = v > self._x[-1]

        if self.bounds_error and below_bounds.any():
            raise ValueError("A value in x_new is below the interpolation "
                "range.")
        if self.bounds_error and above_bounds.any():
            raise ValueError("A value in x_new is above the interpolation "
                "range.")

        return below_bounds, above_bounds

# A possible optimization if the points are in a regular grid
# Is a bit faster (~2 times)

# class SteffenInterpolatorGrid(SteffenInterpolator):
#     @staticmethod
#     def _create_h(x):
#         return x[1] - x[0]
#
#     @staticmethod
#     def _create_p(s, h):
#         p = np.zeros_like(s)
#         p[1:] = (s[:-1] + s[1:]) / 2.0
#         return p
#
#     @staticmethod
#     def _create_s(y, h):
#         s = np.zeros_like(y)
#         s[:-1] = (y[1:] - y[:-1]) / h
#         s[-1] = 0.0
#         return s
#
#     def __init__(self, x, y, yp_0=0.0, yp_N=0.0, extrapolate='raise',
#                  fill_value=np.nan):
#         super(SteffenInterpolatorGrid, self).__init__(x, y, yp_0, yp_N,
#                                                       extrapolate=extrapolate,
#                                                       fill_value=fill_value)
#
#         self.h = self._create_h(x)
#
#     def _eval(self, v, in_bounds):
#         result = np.zeros_like(v, dtype='float')
#         off = (v - self._x[0]) / self.h
#         x_indices1 = np.floor(off).astype('int')
#         ids = x_indices1[in_bounds]
#         u = v[in_bounds] - (self._x[0] + ids * self.h)
#         result[in_bounds] = self._poly_eval(u, ids)
#         return result
#