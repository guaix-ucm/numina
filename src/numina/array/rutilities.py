#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Utility classes and functions to employ R from python using rpy2."""

import numpy as np
from scipy.misc import comb
import rpy2.robjects as ro
from rpy2.robjects.packages import importr


class LinearModelYvsX:
    """Linear Model class to fit data using the R lm() function.

    Parameters
    ----------
    x : 1d numpy array, float
        X values.
    y : 1d numpy array, float
        Y values.
    degree: int
        Polynomial degree.
    w : 1d numpy array, float
        Optional vector of weights to be used in the fitting process.
    raw : bool
        If True, use raw and not orthogonal polynomials.
    renorm : bool
        If True, renormalize the data ranges to [-1,1] before the fit
        as explained in Appendix B1 of Cardiel (2009, MNRAS, 396, 680)
        in order to prevent numerical errors.
    rm_all : bool
        If True, remove all R objects to initialize session.
    debug : bool
        If True, intermediate information is displayed.

    Attributes
    ----------
    fit : rpy2.robjects.vectors.ListVector
        Object returned by call to lm(...) function.
    summary : rpy2.robjects.vectors.ListVector
        Object returned by call to summary(lm(...))) function.
    coeff_estimate : 1d numpy array, float
        Fitted coefficients.
    coeff_std_error: 1d numpy array, float
        Associated standard deviation for the coefficients.
    coeff_t_value : 1d numpy array, float
        t value returned by summary(lm(...))
    coeff_wald_test : 1d numpy array, float
        Pr(>|t|) returned by summary(lm(...))
    p_value : float
        Global p-value obtained in the F-test.

    Examples
    --------
    Simple fit to a 5th order polynomial

    >>> import numpy as np
    >>> x = np.arange(10)
    >>> y = 2 * x
    >>> w = np.ones(x.size)
    >>> myfit = LinearModelYvsX(x=x, y=y, degree=5, w=w)
    >>> print(myfit.coeff_estimate)
    [  1.12346671e-15   2.00000000e+00   8.55144851e-16  -1.17227862e-16
      -2.12064864e-19   5.50598230e-19]
    >>> print(myfit.coeff_std_error)
    [  6.28322594e-16   1.74104426e-15   1.37908919e-15   4.10725784e-16
       5.12878797e-17   2.26887586e-18]

    Note that relevant information of the resulting fit is returned in
    the class member 'fit'. Individual values are accessible using the
    method rx2.

    >>> print(myfit.fit.names)
     [1] "coefficients"  "residuals"     "fitted.values" "effects"
     [5] "weights"       "rank"          "assign"        "qr"
     [9] "df.residual"   "xlevels"       "call"          "terms"
    [13] "model"
    >>> print(myfit.fit.rx2("coefficients"))

    In addition, the result of the call to the R function
    summary(lm(...)) is also stored as the class member 'summary'.
    Individual values are also accessible using the method rx2.

    >>> print(myfit.summary.names)
     [1] "call"          "terms"         "weights"       "residuals"
     [5] "coefficients"  "aliased"       "sigma"         "df"
     [9] "r.squared"     "adj.r.squared" "fstatistic"    "cov.unscaled"
    >>> print(myfit.summary.rx2("sigma"))
    [1] 6.336625e-16

    """

    def __init__(self, x, y, degree, w=None,
                 raw=True, renorm=False,
                 rm_all=False, debug=False):
        """Fit polynomial using the R lm() function."""

        # remove all objects in R session
        if rm_all:
            ro.r('rm(list=ls())')

        # renormalize data ranges when requested
        if renorm:
            xmin = x.min()
            xmax = x.max()
            ymin = y.min()
            ymax = y.max()
            if xmin == xmax:
                bx = 1.0
                cx = 0.0
            else:
                bx = 2./(xmax - xmin)
                cx = (xmax + xmin)/(xmax - xmin)
            if ymin == ymax:
                by = 1.0
                cy = 0.0
            else:
                by = 2./(ymax - ymin)
                cy = (ymax + ymin)/(ymax - ymin)
            xx = bx * x - cx
            yy = by * y - cy
            if w is None:
                ww = None
            else:
                ww = by * w - cy
        else:
            xx = np.copy(x)
            yy = np.copy(y)
            if w is None:
                ww = None
            else:
                ww = np.copy(w)

        # declare x and y in R session
        ro.globalenv['x'] = ro.FloatVector(xx)
        ro.globalenv['y'] = ro.FloatVector(yy)

        # define R command line to be executed
        r_command = 'lm(y ~ poly(x, degree=' + str(degree)
        r_command += ', raw=' + str(raw).upper() + ')'
        if w is not None:  # weigthed fit
            # declare weights in R session
            ro.globalenv['w'] = ro.FloatVector(ww)
            r_command += ', weights=w'
        r_command += ')'

        if debug:
            print('r_command:\n', r_command)

        # execute command line and store resulting fit
        self.fit = ro.r(r_command)

        # store summary
        base = importr('base')
        self.summary = base.summary(self.fit)

        # store coefficients and associated statistics
        coeff_matrix = np.array(self.summary.rx2("coefficients"))
        self.coeff_estimate = coeff_matrix[:, 0]
        self.coeff_std_error = coeff_matrix[:, 1]
        self.coeff_t_value = coeff_matrix[:, 2]
        self.coeff_wald_test = coeff_matrix[:, 3]

        # when necessary, reverse the impact of renormalization of
        # the data ranges in the fitted coefficients
        if renorm:
            if cx == 0.0:
                coeff_estimate = np.copy(self.coeff_estimate)
                for i in range(degree + 1):
                    coeff_estimate[i] *= bx**i
            else:
                aa = np.copy(self.coeff_estimate)
                coeff_estimate = np.zeros_like(self.coeff_estimate)
                coeff_std_error = np.zeros_like(self.coeff_std_error)
                for i in range(degree + 1):
                    for j in range(i, degree + 1):
                        coeff_estimate[i] += aa[j] * comb(j, j-i, True) * \
                                             (bx**i)*((-cx)**(j-i))
            coeff_estimate[0] += cy
            coeff_estimate /= by
            self.coeff_estimate = coeff_estimate

        # compute global p-value, which is not stored in summary
        fstatistic = self.summary.rx2("fstatistic")
        stats = importr('stats')
        global_p_value = stats.pf(q=fstatistic[0],
                                  df1=fstatistic[1],
                                  df2=fstatistic[2],
                                  lower_tail=False)
        self.p_value = np.array(global_p_value)[0]
