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
from scipy.interpolate import interp1d
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


class NumSplines(object):
    """Numerical spline fit class definition.

    Attributes
    ----------
    _xnor : (N,) numpy array
        1-D array of normalised independent input data.
    _ynor : (N,) numpy array
        1-D array of normalised dependent input data.
    _bx : float
        Normalisation factor: xnor = x * bx - cx.
    _cx : float
        Normalisation factor: xnor = x * bx - cx.
    _by : float
        Normalisation factor: ynor = y * by - cy.
    _cy : float
        Normalisation factor: ynor = y * by - cy.
    _nknots : int
        Number of knots fitted.
    _xknot : numpy array
        X location of fitted knots.
    _yknot : numpy array
        Y location of fitted knots.
    _tck : tuple
        A tuple (t,c,k) containing the vector of knots, the B-spline
        coefficients, and the degree of the spline.

    """

    def __init__(self, xfit, yfit, check_finite=True, normalised=True):
        """Class constructor.

        Parameters
        ----------
        xfit : (N,) array_like
            1-D array of independent input data. Must be increasing.
        yfit : (N,) array_like
            1-D array of dependent input data, of the same length
            as `x`.
        check_finite : bool, optional
            Whether to check that the input arrays contain only finite
            numbers. Disabling may give a performance gain, but may
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
            if not np.isfinite(xfit).all() or not np.isfinite(yfit).all():
                raise ValueError('xfit and yfit arrays must not contain '
                                 'NaNs or infs.')

        if not all(np.diff(xfit) > 0.0):
            raise ValueError('xfit must be strictly increasing')

        xmin = xfit[0]
        xmax = xfit[-1]
        ymin = np.min(yfit)
        ymax = np.max(yfit)

        if normalised:
            self._bx = 2.0 / (xmax - xmin)
            self._cx = (xmin + xmax) / (xmax - xmin)
            self._by = 2.0 / (ymax - ymin)
            self._cy = (ymin + ymax) / (ymax - ymin)
        else:
            self._bx = 1.0
            self._cx = 0.0
            self._by = 1.0
            self._cy = 0.0

        self._xnor = np.asarray(xfit) * self._bx - self._cx
        self._ynor = np.asarray(yfit) * self._by - self._cy

        self._nknots = 0
        self._xknot = None
        self._yknot = None
        self._tck = None

    @classmethod
    def fixed_knots(cls, xfit, yfit, xknots, tolerance=1E-5, normalised=True):
        """Spline fit with fixed knot location.

        Parameters
        ----------
        xfit : (N,) array_like
            1-D array of independent input data. Must be increasing.
        yfit : (N,) array_like
            1-D array of dependent input data, of the same length
            as `x`.
        xknots : array_like
            1-D array providing knot location. Must be increasing.
        tolerance : float
            Tolerance value for numerical minimisation.
        normalised : bool, optional
            Whether to normalise the input arrays before fitting,
            following the procedure described in Appendix B1 of
            Cardiel (2009). See:
            http://cdsads.u-strasbg.fr/abs/2009MNRAS.396..680C
            Default is True.

        """

        if len(xknots) > 1:
            if not all(np.diff(xknots) > 0.0):
                raise ValueError('xfit must be strictly increasing')

        self = NumSplines(
            xfit=xfit,
            yfit=yfit,
            normalised=normalised
        )

        xknots = np.asarray(xknots)
        if normalised:
            xknots = xknots * self._bx - self._cx

        # number of knots
        self._nknots = len(xknots)

        # interpolation function to evaluate dependent variable at
        # the location of the knots
        flinear_interp = interp1d(
            x=self._xnor,
            y=self._ynor,
            kind='linear',
            fill_value='extrapolate'
        )

        params = Parameters()
        for i in range(self._nknots):
            iknot = i + 1
            xvalue = xknots[i]
            yvalue = float(flinear_interp(xvalue))
            params.add('xknot{:03d}'.format(iknot), xvalue, vary=False)
            params.add('yknot{:03d}'.format(iknot), yvalue, vary=True)

        fitter = Minimizer(self._fun_residuals, params=params)
        result = fitter.scalar_minimize(method='Nelder-Mead', tol=tolerance)

        self._xknot, self._yknot = get_xyknots_from_params(result.params)

        # reverse normalisation
        self._xknot = (self._xknot + self._cx) / self._bx
        self._yknot = (self._yknot + self._cy) / self._by

        # spline fit to refined knot location
        self._tck = splrep(
            x=self._xknot,
            y=self._yknot,
            s=0
        )

        return self

    @classmethod
    def equidistant_knots(cls, xfit, yfit, nknots, tolerance=1E-5):
        """Spline fit with fixed number of equidistant knots.

        Parameters
        ----------
        xfit : (N,) array_like
            1-D array of independent input data. Must be increasing.
        yfit : (N,) array_like
            1-D array of dependent input data, of the same length
            as `x`.
        nknots : int
            Number of knots. Must be greather or equal to 2.
        tolerance : float
            Tolerance value for numerical minimisation.

        """

        xknots = [100, 200]

        return cls.fixed_knots(xfit, yfit, xknots, tolerance)

    def _fun_residuals(self, params):
        """Function to be minimised.

        """

        xknot, yknot = get_xyknots_from_params(params)

        tck = splrep(x=xknot, y=yknot, s=0)

        yfitted = splev(self._xnor, tck, der=0)

        residuals = np.sum((yfitted - self._ynor)**2)

        return residuals

    def __call__(self, x, der=0, ext=0):
        """Evaluate spline fit at `x`

        This is a simple interface to the scipy.interpolate.splev function.

        Parameters
        ----------
        x : array_like
            1-D array of points at which the fit needs to be
            evaluated.
        der : int, optional
            The order of derivative of the spline to compute
            (must be less than or equal to k --the spline order--).
        ext : int, optional
            Controls the value returned for elements of ``x`` not in the
            interval defined by the knot sequence.

            * if ext=0, return the extrapolated value.
            * if ext=1, return 0
            * if ext=2, raise a ValueError
            * if ext=3, return the boundary value.

            The default value is 0.

        Returns
        -------
        yfitted : numpy array

        """

        x = np.asarray(x)

        yfitted = splev(x=x, tck=self._tck, der=der, ext=ext)

        return np.asarray(yfitted)
