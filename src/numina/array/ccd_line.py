#
# Copyright 2008-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from copy import deepcopy
import numbers
import numpy as np

from numpy.polynomial import Polynomial

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.display.polfit_residuals import \
    polfit_residuals_with_sigma_rejection


class CCDLine:
    """Generic CCD line definition."""

    def __init__(self):
        # set availability to False
        self.available = False
        # bounding box
        self.bb_nc1_orig = None
        self.bb_nc2_orig = None
        self.bb_ns1_orig = None
        self.bb_ns2_orig = None
        # extreme points
        self.xlower_line = None
        self.ylower_line = None
        self.xupper_line = None
        self.yupper_line = None
        # polynomial fit
        self.poly_funct = None

    def __str__(self):
        """Define printable representation of a CCDLine instance."""

        output = '<' + str(self.__class__) + " instance>"
        for key in sorted(self.__dict__.keys()):
            output += '\n=> ' + key + ': ' + str(self.__dict__[key])

        return output

    def fit(self, x, y, deg, w=None, y_vs_x=True,
            times_sigma_reject=None, title=None, debugplot=0):
        """Update the arc line from least squares fit to data.

        Parameters
        ----------
        x : 1d numpy array, float
            X coordinates of the data being fitted.
        y : 1d numpy array, float
            Y coordinates of the data being fitted.
        deg : int
            Degree of the fitting polynomial.
        w : 1d numpy array, float
            Weights to be employed in the polynomial fit.
        y_vs_x : bool
            If True, the fit is Y vs X. Otherwise, X vs Y is computed.
        times_sigma_reject : float
            If not None, deviant point are rejected.
        title : string
            Plot title.
        debugplot : int
            Determines whether intermediate computations and/or plots
            are displayed:
            00 : no debug, no plots
            01 : no debug, plots without pauses
            02 : no debug, plots with pauses
            10 : debug, no plots
            11 : debug, plots without pauses
            12 : debug, plots with pauses

        """

        # protections
        if type(x) is not np.ndarray:
            raise ValueError("x=" + str(x) + " must be a numpy.ndarray")
        if type(y) is not np.ndarray:
            raise ValueError("y=" + str(y) + " must be a numpy.ndarray")
        if x.size != y.size:
            raise ValueError("x.size != y.size")
        if w is not None:
            if type(w) is not np.ndarray:
                raise ValueError(
                    "w=" + str(w) + " must be None or a numpy.ndarray")
            if w.size != x.size:
                raise ValueError("w.size != x.size")
        if not isinstance(deg, numbers.Integral):
            raise ValueError("deg=" + str(deg) +
                             " is not a valid integer")

        # update bounding box of the CCD line
        self.bb_nc1_orig = min(x)
        self.bb_nc2_orig = max(x)
        self.bb_ns1_orig = min(y)
        self.bb_ns2_orig = max(y)

        # compute polynomial from fit to data
        if y_vs_x:
            if times_sigma_reject is None:
                # fit using the minimal domain that covers the x data
                poly_funct = \
                    Polynomial.fit(x=x, y=y, deg=deg, w=w,
                                   domain=None, window=None, full=False)
                # restore the class domain
                self.poly_funct = Polynomial.cast(poly_funct)
                # display resulting fit when requested
                if debugplot % 10 != 0:
                    polfit_residuals(x=x, y=y, deg=deg,
                                     title=title,
                                     debugplot=debugplot)
            else:
                self.poly_funct, yres_dum, reject_dum = \
                    polfit_residuals_with_sigma_rejection(
                        x=x, y=y, deg=deg,
                        title=title,
                        times_sigma_reject=times_sigma_reject,
                        debugplot=debugplot
                    )
            self.xlower_line = self.bb_nc1_orig
            self.ylower_line = self.poly_funct(self.xlower_line)
            self.xupper_line = self.bb_nc2_orig
            self.yupper_line = self.poly_funct(self.xupper_line)
        else:
            if times_sigma_reject is None:
                # fit using the minimal domain that covers the y data
                poly_funct = \
                    Polynomial.fit(x=y, y=x, deg=deg, w=w,
                                   domain=None, window=None, full=False)
                # restore the class domain
                self.poly_funct = Polynomial.cast(poly_funct)
                # display resulting fit when requested
                if debugplot % 10 != 0:
                    polfit_residuals(x=y, y=x, deg=deg,
                                     title=title, debugplot=debugplot)
            else:
                self.poly_funct, yres_dum, reject_dum = \
                    polfit_residuals_with_sigma_rejection(
                        x=y, y=x, deg=deg,
                        title=title,
                        times_sigma_reject=times_sigma_reject,
                        debugplot=debugplot
                    )
            self.ylower_line = self.bb_ns1_orig
            self.xlower_line = self.poly_funct(self.ylower_line)
            self.yupper_line = self.bb_ns2_orig
            self.xupper_line = self.poly_funct(self.yupper_line)

        # CCD line has been defined
        self.available = True

    def linspace_pix(self, start=None, stop=None, pixel_step=1, y_vs_x=None):
        """Return x,y values evaluated with a given pixel step.

        The returned values are computed within the corresponding
        bounding box of the line.

        Parameters
        ----------
        start : float
            Minimum pixel coordinate to evaluate the independent
            variable.
        stop : float
            Maximum pixel coordinate to evaluate the independent
            variable.
        pixel_step : float
            Pixel step employed to evaluate the independent variable.
        y_vs_x : bool
            If True, the polynomial fit is assumed to be Y vs X.
            Otherwise, X vs Y is employed.

        Returns
        -------
        x : 1d numpy array
            X coordinates.
        y : 1d numpy array
            Y coordinates.
        """

        if y_vs_x:
            if start is None:
                xmin = self.bb_nc1_orig
            else:
                xmin = start
            if stop is None:
                xmax = self.bb_nc2_orig
            else:
                xmax = stop
            num = int(float(xmax-xmin+1)/float(pixel_step)+0.5)
            x = np.linspace(start=xmin, stop=xmax, num=num)
            y = self.poly_funct(x)
        else:
            if start is None:
                ymin = self.bb_ns1_orig
            else:
                ymin = start
            if stop is None:
                ymax = self.bb_ns2_orig
            else:
                ymax = stop
            num = int(float(ymax-ymin+1)/float(pixel_step)+0.5)
            y = np.linspace(start=ymin, stop=ymax, num=num)
            x = self.poly_funct(y)

        return x, y

    def offset(self, offset_value):
        """Return a copy of self, shifted a constant offset.

        Parameters
        ----------
        offset_value : float
            Number of pixels to shift the CCDLine.

        """

        new_instance = deepcopy(self)
        new_instance.poly_funct.coef[0] += offset_value
        return new_instance


class ArcLine(CCDLine):
    """Arc line definition."""

    def __init__(self):
        # members of superclass CCDLine
        CCDLine.__init__(self)
        # rectified abscissa of arc line
        self.x_rectified = None

    def linspace_pix(self, start=None, stop=None, pixel_step=1, y_vs_x=False):
        """Return x,y values evaluated with a given pixel step."""
        return CCDLine.linspace_pix(self, start=start, stop=stop,
                                    pixel_step=pixel_step, y_vs_x=y_vs_x)


class SpectrumTrail(CCDLine):
    """Spectrum trail definition"""

    def __init__(self):
        # members of superclass CCDLine
        CCDLine.__init__(self)
        # rectified ordinate of spectrum trail
        self.y_rectified = None

    def linspace_pix(self, start=None, stop=None, pixel_step=1, y_vs_x=True):
        """Return x,y values evaluated with a given pixel step."""
        return CCDLine.linspace_pix(self, start=start, stop=stop,
                                    pixel_step=pixel_step, y_vs_x=y_vs_x)


def intersection_spectrail_arcline(spectrail, arcline):
    """Compute intersection of spectrum trail with arc line.

    Parameters
    ----------
    spectrail : SpectrumTrail object
        Instance of SpectrumTrail class.
    arcline : ArcLine object
        Instance of ArcLine class

    Returns
    -------
    xroot, yroot : tuple of floats
        (X,Y) coordinates of the intersection.

    """

    # approximate location of the solution
    expected_x = (arcline.xlower_line + arcline.xupper_line) / 2.0

    # composition of polynomials to find intersection as
    # one of the roots of a new polynomial
    rootfunct = arcline.poly_funct(spectrail.poly_funct)
    rootfunct.coef[1] -= 1
    # compute roots to find solution
    tmp_xroots = rootfunct.roots()

    # take the nearest root to the expected location
    xroot = tmp_xroots[np.abs(tmp_xroots - expected_x).argmin()]
    if np.isreal(xroot):
        xroot = xroot.real
    else:
        raise ValueError("xroot=" + str(xroot) +
                         " is a complex number")
    yroot = spectrail.poly_funct(xroot)

    return xroot, yroot

