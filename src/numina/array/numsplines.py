#
# Copyright 2019 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Numerical spline fit using different strategies."""

from lmfit import Minimizer, Parameters
import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def fun_residuals(params, xnor, ynor, w, bbox, k, ext):
    """Compute fit residuals"""

    spl = LSQUnivariateSpline(
        x=xnor,
        y=ynor,
        t=[item.value for item in params.values()],
        w=w,
        bbox=bbox,
        k=k,
        ext=ext,
        check_finite=False
    )
    return spl.get_residual()


class AdaptiveLSQUnivariateSpline(LSQUnivariateSpline):
    """Extend scipy.interpolate.LSQUnivariateSpline.

    One-dimensional spline with explicit internal knots.

    This is actually a wrapper of
    `scipy.interpolate.LSQUnivariateSpline`
    with the addition of using adaptive knot location determined
    numerically (after normalising the x and y arrays before
    the minimisation process).

    Parameters
    ----------
    x : (N,) array_like
        Input dimension of data points -- must be increasing
    y : (N,) array_like
        Input dimension of data points
    t : (M,) array_like or int
        When integer it indicates the number of equidistant
        interior knots. When array_like it provides the location
        of the interior knots of the spline; must be in ascending
        order and::

         bbox[0] < t[0] < ... < t[-1] < bbox[-1]

    w : (N,) array_like, optional
        weights for spline fitting.  Must be positive.
        If None (default), weights are all equal.
    bbox : (2,) array_like, optional
        2-sequence specifying the boundary of the approximation
        interval. If None (default), ``bbox = [x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
        Default is k=3, a cubic spline.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite
        numbers, that the x array is increasing and that the
        x and y arrays are 1-D and with the same length.
        Disabling may give a performance gain, but may
        result in problems (crashes, non-termination or non-sensical
        results) if the inputs do contain infinities or NaNs.
        Default is True.
    adaptive : bool, optional
        Whether to optimise knot location following the procedure
        described in Cardiel (2009); see:
        http://cdsads.u-strasbg.fr/abs/2009MNRAS.396..680C
        Default is True.
    tolerance : float, optional
        Tolerance for Nelder-Mead minimisation process.

    Attributes
    ----------
    _params : instance of Parameters()
        Initial parameters before minimisation.
    _result : Minimizer output
        Result of the minimisation process.

    See also
    --------
    LSQUnivariateSpline : Superclass

    """

    def __init__(self, x, y, t, w=None, bbox=(None, None),
                 k=3, ext=0, check_finite=False, adaptive=True,
                 tolerance=1E-7):
        """One-dimensional spline with explicit internal knots."""

        if check_finite:
            # check here the arrays instead of in the base class
            # (note that in the call to super(...).__init(...) the
            # parameter check_finite is set to False)
            w_finite = np.isfinite(x).all() if w is not None else True
            if not np.isfinite(x).all() or not np.isfinite(y).all() or \
                    not w_finite:
                raise ValueError('Input(s) must not contain '
                                 'NaNs or infs.')

            if np.asarray(x).ndim != 1:
                raise ValueError('x array must have dimension 1')

            if np.asarray(y).ndim != 1:
                raise ValueError('y array must have dimension 1')

            if np.asarray(x).shape != np.asarray(y).shape:
                raise ValueError('x and y arrays must have the same length')

            if not all(np.diff(x) > 0.0):
                raise ValueError('x array must be strictly increasing')

        # initial inner knot location (equidistant or fixed)
        try:
            nknots = int(t)
            if nknots > 0:
                xmin = x[0]
                xmax = x[-1]
                deltax = (xmax - xmin) / float(nknots + 1)
                xknot = np.zeros(nknots)
                for i in range(nknots):
                    xknot[i] = (xmin + float(i + 1) * deltax)
            else:
                xknot = np.array([])
        except (ValueError, TypeError):
            xknot = np.asarray(t)
            if check_finite:
                if not np.isfinite(xknot).all():
                    raise ValueError('Interior knots must not contain '
                                     'NaNs or infs.')
                if xknot.ndim != 1:
                    raise ValueError('t array must have dimension 1')
            nknots = len(xknot)

        # adaptive knots
        if nknots > 0 and adaptive:
            xknot_backup = xknot.copy()

            # normalise the x and y arrays to the [-1, +1] interval
            xmin = x[0]
            xmax = x[-1]
            ymin = np.min(y)
            ymax = np.max(y)
            bx = 2.0 / (xmax - xmin)
            cx = (xmin + xmax) / (xmax - xmin)
            by = 2.0 / (ymax - ymin)
            cy = (ymin + ymax) / (ymax - ymin)
            xnor = bx * np.asarray(x) - cx
            ynor = by * np.asarray(y) - cy
            xknotnor = bx * xknot - cx
            params = Parameters()
            for i in range(nknots):
                if i == 0:
                    xminknot = bx * x[0] - cx
                    xmaxknot = (xknotnor[i] + xknotnor[i+1]) / 2.0
                elif i == nknots - 1:
                    xminknot = (xknotnor[i-1] + xknotnor[i]) / 2.0
                    xmaxknot = bx * x[-1] - cx
                else:
                    xminknot = (xknotnor[i-1] + xknotnor[i]) / 2.0
                    xmaxknot = (xknotnor[i] + xknotnor[i+1]) / 2.0
                params.add(
                    name=f'xknot{i:03d}',
                    value=xknotnor[i],
                    min=xminknot,
                    max=xmaxknot,
                    vary=True
                )
            self._params = params.copy()
            fitter = Minimizer(
                userfcn=fun_residuals,
                params=params,
                fcn_args=(xnor, ynor, w, bbox, k, ext)
            )
            try:
                self._result = fitter.scalar_minimize(
                    method='Nelder-Mead',
                    tol=tolerance
                )
                xknot = [item.value for item in self._result.params.values()]
                xknot = (np.asarray(xknot) + cx) / bx
            except ValueError:
                print('Error when fitting adaptive splines. '
                      'Reverting to initial knot location.')
                xknot = xknot_backup.copy()
                self._result = None
        else:
            self._params = None
            self._result = None

        # final fit
        super(AdaptiveLSQUnivariateSpline, self).__init__(
            x=x,
            y=y,
            t=xknot,
            w=w,
            bbox=bbox,
            k=k,
            ext=ext,
            check_finite=False
        )

    def get_params(self):
        """Return initial parameters for minimisation process."""

        return self._params

    def get_result(self):
        """Return result of minimisation process."""

        return self._result
