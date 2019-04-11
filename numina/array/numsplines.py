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
from scipy.interpolate import splrep, splev


def get_xyknots_from_params(params):
    """Auxiliary function to get xknot and yknot from params object.

    Parameters
    ----------
    params : :class:`~lmfit.parameter.Parameters`
        Parameters object with X and Y knot location.

    Returns
    -------
    xknot : numpy array
        X location of the knots.
    yknot : numpy array
        Y location of the knots.

    """

    nparams = len(params)

    if nparams % 2 != 0:
        raise ValueError('Unexpected number of values in Parameters object')

    nknots = nparams // 2

    xknot = np.zeros(nknots)
    yknot = np.zeros(nknots)

    for i in range(nknots):
        xknot[i] = params['xknot{:03d}'.format(i + 1)].value
        yknot[i] = params['yknot{:03d}'.format(i + 1)].value

    return xknot, yknot

class AdaptiveLSQUnivariateSpline(LSQUnivariateSpline):
    """Extend scipy.interpolate.LSQUnivariateSpline.

    """

    def __init__(self, x, y, t, w=None, bbox=(None, None),
                 k=3, ext=0, check_finite=False):
        """One-dimensional spline with explicit internal knots.

        This is actually a wrapper of
        `scipy.interpolate.LSQUnivariateSpline`
        with the addition of normalising the x and y arrays before
        the fit.

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
        normalised : bool, optional
            Whether to normalise the input arrays before fitting,
            following the procedure described in Appendix B1 of
            Cardiel (2009). See:
            http://cdsads.u-strasbg.fr/abs/2009MNRAS.396..680C
            Default is True.

        """

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

        try:
            nknots = int(t)
            if nknots > 0:
                xmin = x[0]
                xmax = x[-1]
                deltax = (xmax - xmin) / float(nknots + 1)
                xknots = np.zeros(nknots)
                for i in range(nknots):
                    xknots[i] = (xmin + float(i + 1) * deltax)
            else:
                xknots = np.array([])

        except:
            xknots = np.asarray(t)
            if check_finite:
                if not np.isfinite(xknots).all():
                    raise ValueError('Interior knots must not contain '
                                     'NaNs or infs.')

        super(AdaptiveLSQUnivariateSpline, self).__init__(
            x=x,
            y=y,
            t=xknots,
            w=w,
            bbox=bbox,
            k=k,
            ext=ext,
            check_finite=False
        )

