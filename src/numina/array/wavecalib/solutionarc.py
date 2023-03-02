#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

"""Store the solution of a wavelength calibration"""

import math
import warnings

from numpy.polynomial import Polynomial


class CrLinear(object):
    """Store information concerning the linear wavelength calibration.

    Parameters
    ----------
    crpix : float
        CRPIX1 value employed in the linear wavelength calibration.
    crval : float
        CRVAL1 value corresponding tot he linear wavelength
        calibration.
    crmin : float
        CRVAL value at pixel number 1 corresponding to the linear
        wavelength calibration.
    crmax : float
        CRVAL value at pixel number NAXIS1 corresponding to the linear
        wavelength calibration.
    cdelt : float
        CDELT1 value corresponding to the linear wavelength
        calibration.

    Attributes
    ----------
    Identical to parameters.

    """

    def __init__(self, crpix, crval, crmin, crmax, cdelt):
        self.crpix = crpix
        self.crval = crval
        self.cdelt = cdelt
        self.crmin = crmin
        self.crmax = crmax

    def __getstate__(self):
        state = self.__dict__.copy()
        for k in state:
            state[k] = float(state[k])
        return state

    def __str__(self):
        """Printable representation of a CrLinear instance."""

        output = "<CrLinear instance>\n" + \
                 "crpix linear: " + str(self.crpix) + "\n" + \
                 "crval linear: " + str(self.crval) + "\n" + \
                 "cdelt linear: " + str(self.cdelt) + "\n" + \
                 "crmin linear: " + str(self.crmin) + "\n" + \
                 "crmax linear: " + str(self.crmax)

        return output


class WavecalFeature(object):
    """Store information concerning a particular line identification.

    Parameters
    ----------
    line_ok : bool
        True if the line has been properly identified.
    category : char
        Line identification type (A, B, C, D, E, R, T, P, K, I, X).
        See documentation embedded within the arccalibration_direct
        function for details.
    lineid : int
        Number of identified line within the master list.
    xpos : float
        Pixel x-coordinate of the peak of the line.
    ypos : float
        Pixel y-coordinate of the peak of the line.
    peak : float
        Flux of the peak of the line.
    fwhm : float
        FWHM of the line.
    reference : float
        Wavelength of the identified line in the master list.
    wavelength : float
        Wavelength of the identified line estimated from the wavelength
        calibration polynomial.
    funcost : float
        Cost function corresponding to each identified arc line.

    Attributes
    ----------
    Identical to parameters.

    """

    def __init__(self, line_ok, category, lineid, funcost, xpos, ypos=0.0,
                 peak=0.0, fwhm=0.0, reference=0.0, wavelength=0.0):
        self.line_ok = line_ok
        self.category = category
        self.lineid = lineid
        self.funcost = funcost
        self.xpos = xpos
        self.ypos = ypos
        self.peak = peak
        self.fwhm = fwhm
        self.reference = reference
        self.wavelength = wavelength

    def __getstate__(self):
        state = self.__dict__.copy()
        float_keys = ['funcost', 'xpos', 'ypos', 'peak', 'fwhm',
                      'reference', 'wavelength']
        for k in state:
            if k in float_keys:
                value = float(state[k])
                # translate infinities
                if math.isinf(value):
                    value = 1e50
                    warnings.warn(
                        f'Converting {k}=inf to {value}',
                        RuntimeWarning
                    )

                state[k] = value
            elif k in ['lineid']:
                state[k] = int(state[k])
            else:
                pass
        return state

    def __str__(self):
        if self.line_ok:
            sline_ok = 'True '
        else:
            sline_ok = 'False'
        output = "<WavecalFeature instance>\n" + \
                 " line_ok: " + sline_ok + "  " + \
                 "category: " + str(self.category) + "  " + \
                 "id: " + str(self.lineid) + "  " + \
                 "xpos: " + str(self.xpos) + "  " + \
                 "ypos: " + str(self.ypos) + "\n" + \
                 " peak: " + str(self.peak) + "  " + \
                 "fwhm: " + str(self.fwhm) + "  " + \
                 "reference: " + str(self.reference) + "  " + \
                 "wavelength: " + str(self.wavelength) + "  " + \
                 "funcost: " + str(self.funcost)

        return output


class SolutionArcCalibration(object):
    """Auxiliary class to store the arc calibration solution.

    Note that this class only stores the information concerning the
    arc lines that have been properly identified. The information
    about all the lines (including those initially found but at the
    end discarded) is stored in the list of WavecalFeature instances.

    Parameters
    ----------
    features : list (of WavecalFeature instances)
        A list of size equal to the number of identified lines, which
        elements are instances of the class WavecalFeature, containing
        all the relevant information concerning the line
        identification.
    coeff : 1d numpy array (float)
        Coefficients of the wavelength calibration polynomial.
    residual_std : float
        Residual standard deviation of the fit.
    cr_linear : instance of CrLinear
        Object containing the linear approximation parameters crpix,
        crval, cdelt, crmin and crmax.

    Attributes
    ----------
    Identical to parameters.

    """

    def __init__(self, features, coeff, residual_std, cr_linear):

        self.features = features
        self.coeff = [float(tmpcoeff) for tmpcoeff in coeff]
        # force residual_std to be a float and not a scalar numpy array
        self.residual_std = float(residual_std)
        self.cr_linear = cr_linear

        # use fitted polynomial to predict wavelengths at the line
        # locations given by xpos
        self.update_features(poly=Polynomial(self.coeff))

    def update_features(self, poly):
        """Evaluate wavelength at xpos using the provided polynomial."""

        for feature in self.features:
            feature.wavelength = poly(feature.xpos)

    @property
    def nlines_arc(self):
        return len([wvfeature for wvfeature in self.features
                    if wvfeature.line_ok])

    def __eq__(self, other):
        if isinstance(other, SolutionArcCalibration):
            return self.__getstate__() == other.__getstate__()
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self == other
    
    def __str__(self):
        """Printable representation of a SolutionArcCalibration instance."""

        output = "<SolutionArcCalibration instance>\n" + \
                 "- Number arc lines: " + str(self.nlines_arc) + "\n" + \
                 "- Coeff...........: " + str(self.coeff) + "\n" + \
                 "- Residual std....: " + str(self.residual_std) + "\n" + \
                 "- " + str(self.cr_linear) + "\n"

        for feature in self.features:
            output += "- " + str(feature) + "\n"

        # return string with all the information
        return output

    def __getstate__(self):
        result = self.__dict__.copy()
        result['features'] = [
            feature.__getstate__() for feature in self.features
            ]
        result['cr_linear'] = result['cr_linear'].__getstate__()
        return result

    def __setstate__(self, state):
        self.__dict__ = state.copy()
        features = []
        for feature in state['features']:
            newf = WavecalFeature.__new__(WavecalFeature)
            newf.__dict__ = feature
            features.append(newf)

        self.features = features
        self.cr_linear = CrLinear(**state['cr_linear'])
