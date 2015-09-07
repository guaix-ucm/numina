#
# Copyright 2015 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# Numina is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Numina is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Numina.  If not, see <http://www.gnu.org/licenses/>.
#


"""Peak finding routines from wavecal"""

from __future__ import division

import numpy as np

#------------------------------------------------------------------------------

def findPeaks_spectrum(sx, nwinwidth, data_threshold=0):
    """Find peaks in the 1d-numpy array sx.

    Note that nwinwidth must be an odd number. The function imposes that the
    fluxes in the nwinwidth/2 points to the left (and right) of the peak
    decrease monotonously as one moves away from the peak.

    Parameters
    ----------
    sx : 1d numpy array (float)
        Input 1D spectrum.
    nwinwidth : int
        Width of the window where the peak must be found. This number must be
        odd.
    data_threshold : float
        Minimum signal in the peak (optional).

    Returns
    -------
    ipeaks : 1d numpy array (int)
        Indices of the input array sx in which the peaks have been found. Note
        that these numbers are not channels, but array indices starting from
        zero.
        
    """

    sx_shape = sx.shape
    nmed = nwinwidth//2

    ipeaks = []

    if sx_shape[0] < nwinwidth:
        raise ValueError('findPeaks_spectrum> ERROR: invalid nwinwidth')
        return np.array(ipeaks)

    i = nmed
    while i < sx_shape[0]-nmed:
        if sx[i] > data_threshold:
            lpeakgood = True

            j = 0
            loop = True
            while loop:
                if sx[i-nmed+j] > sx[i-nmed+j+1]:
                    lpeakgood = False
                j += 1
                loop = (j < nmed) and lpeakgood

            if lpeakgood:
                j = nmed+1
                loop = True
                while loop:
                    if sx[i-nmed+j-1] < sx[i-nmed+j]: lpeakgood = False
                    j += 1
                    loop = (j < nwinwidth) and lpeakgood
                
            if lpeakgood: 
                ipeaks.append(i)
                i += nwinwidth-1
            else:
                i += 1
        else:
            i += 1

    npeaks = len(ipeaks)

    return np.array(ipeaks, dtype=np.int)

#------------------------------------------------------------------------------

def refinePeaks_spectrum(sx, ipeaks, nwinwidth, method=2):
    """Refine the peak location previously found by findPeaks_spectrum()

    Parameters
    ----------
    sx : 1d numpy array, float
        Input 1D spectrum.
    ipeaks : 1d numpy array (int)
        Indices of the input array sx in which the peaks were initially found.
    nwinwidth : int
        Width of the window where the peak must be found.
    method : int
        Indicates which function will be employed to refine the peak location.
        method  = 1 -> fit to 2nd order polynomial
        method != 1 -> fit to gaussian

    Returns
    -------
    xfpeaks : 1d numpy array (float)
        X-coordinates in which the refined peaks have been found.
 
    """
    nmed = nwinwidth//2

    xfpeaks = np.zeros(len(ipeaks))

    for iline in range(len(ipeaks)):
        jmax = ipeaks[iline]
        sx_peak_flux = sx[jmax]
        x_fit = np.zeros(0)
        y_fit = np.zeros(0)
        for j in range(jmax-nmed,jmax+nmed+1):
            x_fit = np.concatenate((x_fit, np.array([float(j-jmax)])))
            y_fit = np.concatenate((y_fit, np.array([sx[j]/sx_peak_flux])))

        if method == 1:
            poly = np.polyfit(x_fit, y_fit, 2)
            refined_peak = -poly[1]/(2.0*poly[0])+jmax
        else:
            poly = np.polyfit(x_fit, np.log(y_fit), 2)
            A = np.exp(poly[2]-poly[1]*poly[1]/(4*poly[0]))
            x0 = -poly[1]/(2*poly[0])
            sigma = np.sqrt(-1/(2*poly[0]))
            refined_peak = x0+jmax

        xfpeaks[iline] = refined_peak

    return xfpeaks

