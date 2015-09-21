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

from __future__ import division
from __future__ import print_function

import logging
import itertools

import six
import numpy as np
from numpy.polynomial import polynomial
import scipy.misc

#------------------------------------------------------------------------------

def my_round(x, d=0):
    """Round as in Python2"""


    # Python 2 and 3 round behaviour is different
    # In Python 2
    # round(4.5) == 5
    # round(3.5) == 4

    # In Python 3
    # round(4.5) == 4
    # round(3.5) == 4

    # See http://stackoverflow.com/questions/10825926/python-3-x-rounding-behavior

    import math
    p = 10 ** d
    return float(math.floor((x * p) + math.copysign(0.5, x)))/p


def fitTheilSen(x, y):
    """Compute a robust linear fit using the Theil-Sen method.

    See http://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator for details.
    This function "pairs up sample points by the rank of their x-coordinates
    (the point with the smallest coordinate being paired with the first point
    above the median coordinate, etc.) and computes the median of the slopes of
    the lines determined by these pairs of points".

    Parameters
    ----------
    x : 1d numpy array, float
        X coordinate.
    y : 1d numpy array, float
        Y coordinate.

    Returns
    -------
    intercept : float
        Intercept of the linear fit.
    slope : float
        Slope of the linear fit.

    """

    if x.ndim == y.ndim == 1:
        n = x.size
        if n == y.size:
            if n < 5:
                raise ValueError('FATAL ERROR #3: in fitTheilSen')
            result = []  # python list
            if (n % 2) == 0:
                iextra = 0
            else:
                iextra = 1
            for i in range(n//2):
                ii = i + n//2 + iextra
                deltax = x[ii]-x[i]
                deltay = y[ii]-y[i]
                result.append(deltay/deltax)
            slope = np.median(result)
            result = y - slope*x  # numpy array
            intercept = np.median(result)
            return intercept, slope
        else:
            raise ValueError('FATAL ERROR #2: in fitTheilSen')
    else:
        raise ValueError('FATAL ERROR #1: in fitTheilSen')

#------------------------------------------------------------------------------

def sigmaG(x):
    """Compute a robust estimator of the standard deviation

    See Eq. 3.36 (page 84) in Statistics, Data Mining, and Machine
    in Astronomy, by Ivezic, Connolly, VanderPlas & Gray

    Parameters
    ----------
    x : 1d numpy array, float
        Array of input values which standard deviation is requested.

    Returns
    -------
    sigmag : float
        Robust estimator of the standard deviation
    """

    q25, q75 = np.percentile(x,[25.0, 75.0])
    sigmag = 0.7413*(q75-q25)
    return sigmag

#------------------------------------------------------------------------------

def select_data_for_fit(wv_master, xpos_arc, solution):
    """Select information from valid arc lines to facilitate posterior fits.

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table (Angstroms).
    xpos_arc : 1d numpy array, float
        Location of arc lines (pixels).
    solution : ouput from previous call to arccalibration

    Returns
    -------
    nfit : int
        Number of valid points for posterior fits.
    ifit : list of int
        List of indices corresponding to the arc lines which coordinates are
        going to be employed in the posterior fits.
    xfit : 1d numpy aray
        X coordinate of points for posterior fits.
    yfit : 1d numpy array
        Y coordinate of points for posterior fits.
    wfit : 1d numpy array
        Cost function of points for posterior fits. The inverse of these values
        can be employed for weighted fits.
    """

    nlines_arc = len(solution)

    if nlines_arc != xpos_arc.size:
        raise ValueError('FATAL ERROR in select_data_for_fit: invalid nlines_arc')

    nfit = 0
    ifit = []
    xfit = np.array([])
    yfit = np.array([])
    wfit = np.array([])
    for i in range(nlines_arc):
        if solution[i]['lineok']:
            ifit.append(i)
            xfit = np.append(xfit, xpos_arc[i])
            yfit = np.append(yfit, wv_master[solution[i]['id']])
            wfit = np.append(wfit, solution[i]['funcost'])
            nfit += 1
    return nfit, ifit, xfit, yfit, wfit

#------------------------------------------------------------------------------

def fit_solution(wv_master,xpos_arc,solution,naxis1_arc,poly_degree,weighted,LDEBUG=False,LPLOT=False):
    """Fit polynomial to arc calibration solution.

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table (Angstroms).
    xpos_arc : 1d numpy array, float
        Location of arc lines (pixels).
    solution : list of dictionaries
        A list of size equal to the number of arc lines, which elements are
        dictionaries containing all the relevant information concerning the
        line identification.
    naxis1_arc : int
        NAXIS1 of arc spectrum.
    poly_degree : int
        Polynomial degree corresponding to the wavelength calibration function
        to be fitted.
    weighted : bool
        Determines whether the polynomial fit is weighted or not, using as
        weights the values of the cost function obtained in the line
        identification.
    LDEBUG : bool
        If True intermediate results are displayed.
    LPLOT : bool
        If True intermediate plots are displayed.

    Returns
    -------
    coeff : 1d numpy array, float
        Coefficients of the polynomial fit
    crval1_approx : float
        Approximate CRVAL1 value.
    cdetl1_approx : float
        Approximate CDELT1 value.
    """

    nlines_arc = len(solution)

    if nlines_arc != xpos_arc.size:
        raise ValueError('FATAL ERROR in fit_solution: invalid nlines_arc')

    # Select information from valid lines.
    nfit, ifit, xfit, yfit, wfit = select_data_for_fit(wv_master, xpos_arc, solution)

    # Select list of filtered out and unidentified lines
    list_R = []
    list_T = []
    list_P = []
    list_unidentified = []
    for i in range(nlines_arc):
        if not solution[i]['lineok']:
            if solution[i]['type'] is None:
                list_unidentified.append(i)
            elif solution[i]['type'] == 'R':
                list_R.append(i)
            elif solution[i]['type'] == 'T':
                list_T.append(i)
            elif solution[i]['type'] == 'P':
                list_P.append(i)
            else:
                raise ValueError('FATAL ERROR in fit_solution: unexpected "type"')

    # Obtain approximate linear fit using the robust Theil-Sen method.
    intercept, slope = fitTheilSen(xfit, yfit)
    cdelt1_approx = slope
    crval1_approx = intercept + cdelt1_approx

    # Polynomial fit.
    if weighted:
        coeff = polynomial.polyfit(xfit, yfit, poly_degree, w=1.0/wfit)
    else:
        coeff = polynomial.polyfit(xfit, yfit, poly_degree)

    xpol = np.linspace(1,naxis1_arc,naxis1_arc)
    poly = polynomial.Polynomial(coeff)
    ypol = poly(xpol)-(crval1_approx+(xpol-1)*cdelt1_approx)

    return coeff, crval1_approx, cdelt1_approx

#------------------------------------------------------------------------------

def gen_triplets_master(wv_master, LDEBUG=False, LPLOT=False):
    """Compute information associated to triplets in master table

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table (Angstroms).
    LDEBUG : bool
        If True intermediate results are displayed.
    LPLOT : bool
        If True intermediate plots are displayed.

    Returns
    -------
    ntriplets_master : int
        Number of triplets built from master table.
    ratios_master_sorted : 1d numpy array, float
        Array with values of the relative position of the central line of each
        triplet, sorted in ascending order.
    triplets_master_sorted_list : list of tuples
        List with tuples of three numbers, corresponding to the three line
        indices in the master table. The list is sorted to be in correspondence
        with `ratios_master_sorted`.
       
    """

    _logger = logging.getLogger('numina.array.wavecal')

    nlines_master = wv_master.size

    #---
    # Generate all the possible triplets with the numbers of the lines in the
    # master table. Each triplet is defined as a tuple of three numbers
    # corresponding to the three line indices in the master table. The
    # collection of tuples is stored in an ordinary python list.
    iter_comb_triplets = itertools.combinations(range(nlines_master), 3)
    triplets_master_list = []
    for val in iter_comb_triplets:
        triplets_master_list.append(val)

    # Verify that the number of triplets coincides with the expected value.
    ntriplets_master = len(triplets_master_list)
    if ntriplets_master == scipy.misc.comb(nlines_master, 3, exact=True):
        _logger.debug('Lines in master table %d', nlines_master)
        _logger.debug('Triplets in master table %d', ntriplets_master)
    else:
        raise ValueError('Invalid number of combinations')

    # For each triplet, compute the relative position of the central line.
    ratios_master = np.zeros(ntriplets_master)
    for i_tupla in range(ntriplets_master):
        i1, i2, i3 = triplets_master_list[i_tupla]
        delta1 = wv_master[i2]-wv_master[i1]
        delta2 = wv_master[i3]-wv_master[i1]
        ratios_master[i_tupla] = delta1/delta2

    # Compute the array of indices that index the above ratios in sorted order.
    isort_ratios_master = np.argsort(ratios_master)

    # Simultaneous sort of position ratios and triplets.
    ratios_master_sorted = ratios_master[isort_ratios_master]
    triplets_master_sorted_list = \
      [triplets_master_list[i] for i in isort_ratios_master]

    return ntriplets_master, ratios_master_sorted, triplets_master_sorted_list

#------------------------------------------------------------------------------

def gen_doublets_master(wv_master, LDEBUG=False):
    """Compute information associated to doublets in master table

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table (Angstroms).
    LDEBUG : bool
        If True intermediate results are displayed.

    Returns
    -------
    ndoublets_master : int
        Number of doublets built from master table.
    doublets_master_list : list of tuples
        List with tuples of two numbers, corresponding to the two line
        indices in the master table.
       
    """

    nlines_master = wv_master.size

    #---
    # Generate all the possible doublets with the numbers of the lines in the
    # master table. Each doublet is defined as a tuple of two numbers
    # corresponding to the two line indices in the master table. The
    # collection of tuples is stored in an ordinary python list.
    iter_comb_doublets = itertools.combinations(range(nlines_master), 2)
    doublets_master_list = []
    for val in iter_comb_doublets:
        doublets_master_list.append(val)

    # Verify that the number of doublets coincides with the expected value.
    ndoublets_master = len(doublets_master_list)
    if ndoublets_master != scipy.misc.comb(nlines_master, 2, exact=True):

        raise ValueError('FATAL ERROR: invalid number of combinations')

    return ndoublets_master, doublets_master_list

#------------------------------------------------------------------------------

def arccalibration(wv_master, 
                   xpos_arc,
                   naxis1_arc,
                   wv_ini_search, wv_end_search,
                   error_xpos_arc,
                   times_sigma_r,
                   frac_triplets_for_sum,
                   times_sigma_TheilSen,
                   poly_degree,
                   times_sigma_polfilt,
                   times_sigma_inclusion,
                   LDEBUG=False,
                   LPLOT=False):
    """Performs line identification for arc calibration.

    This function is a wrapper of two functions, which are responsible of 
    computing all the relevant information concerning the triplets 
    generated from the master table and the actual identification procedure
    of the arc lines, respectively. The separation of those computations in two
    different functions helps to avoid the repetition of calls to the first
    function when calibrating several arcs using the same master table. 

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table (Angstroms).
    xpos_arc : 1d numpy array, float
        Location of arc lines (pixels).
    naxis1_arc : int
        NAXIS1 for arc spectrum.
    wv_ini_search : float
        Minimum valid wavelength.
    wv_end_search : float
        Maximum valid wavelength.
    error_xpos_arc : float
        Error in arc line position (pixels).
    times_sigma_r : float
        Times sigma to search for valid line position ratios.
    frac_triplets_for_sum : float
        Fraction of distances to different triplets to sum when computing the
        cost function.
    times_sigma_TheilSen : float
        Number of times the (robust) standard deviation around the linear fit
        (using the Theil-Sen method) to reject points.
    poly_degree : int
        Polynomial degree for fit.
    times_sigma_polfilt : float
        Number of times the (robust) standard deviation around the polynomial
        fit to reject points.
    times_sigma_inclusion : float
        Number of times the (robust) standard deviation around the polynomial
        fit to include a new line in the set of identified lines.
    LDEBUG : bool
        If True intermediate results are displayed.
    LPLOT : bool
        If True intermediate plots are displayed.


    Returns
    -------
    solution : list of dictionaries
        A list of size equal to the number of arc lines, which elements are
        dictionaries containing all the relevant information concerning the
        line identification.

    """

    ntriplets_master, ratios_master_sorted, triplets_master_sorted_list = \
      gen_triplets_master(wv_master, LDEBUG)

    solution = arccalibration_direct(wv_master, 
                                     ntriplets_master,
                                     ratios_master_sorted,
                                     triplets_master_sorted_list,
                                     xpos_arc,
                                     naxis1_arc,
                                     wv_ini_search, wv_end_search,
                                     error_xpos_arc,
                                     times_sigma_r,
                                     frac_triplets_for_sum,
                                     times_sigma_TheilSen,
                                     poly_degree,
                                     times_sigma_polfilt,
                                     times_sigma_inclusion,
                                     LDEBUG,
                                     LPLOT)
    return solution

#------------------------------------------------------------------------------

def arccalibration_direct_doublets(wv_master, 
                                   ndoublets_master,
                                   doublets_master_list,
                                   xpos_arc,
                                   naxis1_arc,
                                   wv_ini_search, 
                                   wv_end_search,
                                   error_xpos_arc,
                                   frac_doublets_for_sum,
                                   LDEBUG=False,
                                   LPLOT=False):
    """Performs line identification for arc calibration using line doublets.

    This function assumes that a previous call to the function responsible for
    the computation of information related to the doublets derived from the
    master table has been previously executed.

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table (Angstroms).
    ndoublets_master : int
        Number of doublets built from master table.
    doublets_master_list : list of tuples
        List with tuples of two numbers, corresponding to the two line
        indices in the master table.
    xpos_arc : 1d numpy array, float
        Location of arc lines (pixels).
    naxis1_arc : int
        NAXIS1 for arc spectrum.
    wv_ini_search : float
        Minimum valid wavelength.
    wv_end_search : float
        Maximum valid wavelength.
    error_xpos_arc : float
        Error in arc line position (pixels).
    frac_doublets_for_sum : float
        Fraction of distances to different doublets to sum when computing the
        cost function.
    LDEBUG : bool
        If True intermediate results are displayed.
    LPLOT : bool
        If True intermediate plots are displayed.


    Returns
    -------
    None

    """

    nlines_master = wv_master.size

    nlines_arc = xpos_arc.size
    if nlines_arc < 5:
        raise ValueError('insufficient arc lines')

    #---
    # Generate doublets with consecutive arc lines. For each doublet,
    # all the doublets from the master are examined. Each doublet
    # from the master table provides an estimate for CRVAL1 and CDELT1.
    # As an additional constraint, the only valid solutions are those for which
    # the initial and the final wavelengths for the arc are restricted to a
    # predefined wavelength interval.
    crval1_search = np.array([])
    cdelt1_search = np.array([])
    error_crval1_search = np.array([])
    error_cdelt1_search = np.array([])
    idoublet_search = np.array([])
    clabel_search = np.array([])
    clabel_search2 = []
    deltaw_search = np.array([])  # special array for doublets

    ndoublets_arc = nlines_arc - 1

    for i in range(ndoublets_arc):
        i1, i2 = i, i+1
        x1 = xpos_arc[i1]
        x2 = xpos_arc[i2]
        xmean = (x1+x2)/2
        dist12 = x2-x1
        for j_loc in range(ndoublets_master):
            j1, j2 = doublets_master_list[j_loc]
            w1 = wv_master[j1]
            w2 = wv_master[j2]
            wmean = (w1+w2)/2
            cdelt1_temp = (w2-w1)/dist12
            crval1_temp = wmean - (xmean-1)*cdelt1_temp
            crvaln_temp = crval1_temp + float(naxis1_arc-1)*cdelt1_temp
            if wv_ini_search <= crval1_temp <= wv_end_search:
                if wv_ini_search <= crvaln_temp <= wv_end_search:
                    # Compute errors
                    error_crval1_temp = cdelt1_temp*error_xpos_arc* \
                      np.sqrt((x1-1)**2+(x2-1)**2)/dist12
                    error_cdelt1_temp = np.sqrt(2)*cdelt1_temp* \
                      error_xpos_arc/dist12
                    # Store values and errors
                    crval1_search = np.append(crval1_search, crval1_temp)
                    cdelt1_search = np.append(cdelt1_search, cdelt1_temp)
                    error_crval1_search = np.append(error_crval1_search, 
                                                    error_crval1_temp)
                    error_cdelt1_search = np.append(error_cdelt1_search, 
                                                    error_cdelt1_temp)
                    # Store additional information about the doublets
                    idoublet_search = np.append(idoublet_search, i)
                    clabel_search.append((j1, j2))
                    #clabel_search = np.append(clabel_search, (j1, j2))
                    # Store absolute difference in wavelength
                    deltaw_search = np.append(deltaw_search, w2-w1)

    # Normalize the values of CDELT1 and CRVAL1 to the interval [0,1] in each
    # case.
    cdelt1_max = (wv_end_search-wv_ini_search)/float(naxis1_arc-1)
    cdelt1_search_norm = cdelt1_search/cdelt1_max
    error_cdelt1_search_norm = error_cdelt1_search/cdelt1_max
    #
    crval1_search_norm = (crval1_search-wv_ini_search)
    crval1_search_norm /= (wv_end_search-wv_ini_search)
    error_crval1_search_norm = error_crval1_search/(wv_end_search-wv_ini_search)

    #---
    # Segregate the different solutions (normalized to [0,1]) by doublet. In
    # this way the solutions are saved in different layers (a layer for each
    # doublet). The solutions will be stored as python lists of numpy arrays.
    ndoublets_layered_list = []
    cdelt1_layered_list = []
    error_cdelt1_layered_list = []
    crval1_layered_list = []
    error_crval1_layered_list = []
    idoublet_layered_list = []
    clabel_layered_list = []
    for i in range(ndoublets_arc):
        ldum = (idoublet_search == i)
        ndoublets_layered_list.append(ldum.sum())
        #
        cdelt1_dum = cdelt1_search_norm[ldum]
        cdelt1_layered_list.append(cdelt1_dum)
        error_cdelt1_dum = error_cdelt1_search_norm[ldum]
        error_cdelt1_layered_list.append(error_cdelt1_dum)
        #
        crval1_dum = crval1_search_norm[ldum]
        crval1_layered_list.append(crval1_dum)
        error_crval1_dum = error_crval1_search_norm[ldum]
        error_crval1_layered_list.append(error_crval1_dum)
        #
        idoublet_dum = idoublet_search[ldum]
        idoublet_layered_list.append(idoublet_dum)
        #
        clabel_dum = [i for (i, v) in zip(clabel_search, ldum) if v]
        clabel_layered_list.append(clabel_dum)
    


    #---
    # Computation of the cost function.
    #
    # For each solution, corresponding to a particular doublet, find the
    # nearest solutions in the remaining ndoublets_arc-1 layers. Compute the
    # distance (in normalized coordinates) to those closests solutions, 
    # and obtain the sum of distances considering only a fraction of them
    # (after sorting them in ascending order).
    ndoublets_for_sum = \
      max(1, int(my_round(frac_doublets_for_sum*float(ndoublets_arc))))
    funcost_search = np.zeros(len(idoublet_search))
    for k in range(len(idoublet_search)):
        idoublet_local = idoublet_search[k]
        x0 = cdelt1_search_norm[k]
        ex0 = error_cdelt1_search_norm[k]
        y0 = crval1_search_norm[k]
        ey0 = error_crval1_search_norm[k]
        dist_to_layers = np.array([])
        for i in range(ndoublets_arc):
            if i != idoublet_local:
                if ndoublets_layered_list[i] > 0:
                    x1 = cdelt1_layered_list[i]
                    ex1 = error_cdelt1_layered_list[i]
                    y1 = crval1_layered_list[i]
                    ey1 = error_crval1_layered_list[i]
                    dist2 = (x0-x1)**2 + (y0-y1)**2

                    dist_to_layers = np.append(dist_to_layers, dist2.min())
                else:
                    dist_to_layers = np.append(dist_to_layers, np.inf)
        dist_to_layers.sort() # in-place sort
        funcost_search[k] = dist_to_layers[list(six.moves.range(ndoublets_for_sum))].sum()

    # Normalize the cost function
    funcost_min = funcost_search.min()
    funcost_search /= funcost_min

    # Segregate the cost function by arc doublet.
    funcost_layered_list = []
    for i in range(ndoublets_arc):
        ldum = (idoublet_search == i)
        funcost_dum = funcost_search[ldum]
        funcost_layered_list.append(funcost_dum)

    return None

#------------------------------------------------------------------------------

def arccalibration_direct(wv_master, 
                          ntriplets_master,
                          ratios_master_sorted,
                          triplets_master_sorted_list,
                          xpos_arc,
                          naxis1_arc,
                          wv_ini_search, 
                          wv_end_search,
                          error_xpos_arc,
                          times_sigma_r,
                          frac_triplets_for_sum,
                          times_sigma_TheilSen,
                          poly_degree,
                          times_sigma_polfilt,
                          times_sigma_inclusion,
                          LDEBUG=False,
                          LPLOT=False):
    """Performs line identification for arc calibration using line triplets.

    This function assumes that a previous call to the function responsible for
    the computation of information related to the triplets derived from the
    master table has been previously executed.

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table (Angstroms).
    ntriplets_master : int
        Number of triplets built from master table.
    ratios_master_sorted : 1d numpy array, float
        Array with values of the relative position of the central line of each
        triplet, sorted in ascending order.
    triplets_master_sorted_list : list of tuples
        List with tuples of three numbers, corresponding to the three line
        indices in the master table. The list is sorted to be in correspondence
        with `ratios_master_sorted`.
    xpos_arc : 1d numpy array, float
        Location of arc lines (pixels).
    naxis1_arc : int
        NAXIS1 for arc spectrum.
    wv_ini_search : float
        Minimum valid wavelength.
    wv_end_search : float
        Maximum valid wavelength.
    error_xpos_arc : float
        Error in arc line position (pixels).
    times_sigma_r : float
        Times sigma to search for valid line position ratios.
    frac_triplets_for_sum : float
        Fraction of distances to different triplets to sum when computing the
        cost function.
    times_sigma_TheilSen : float
        Number of times the (robust) standard deviation around the linear fit
        (using the Theil-Sen method) to reject points.
    poly_degree : int
        Polynomial degree for fit.
    times_sigma_polfilt : float
        Number of times the (robust) standard deviation around the polynomial
        fit to reject points.
    times_sigma_inclusion : float
        Number of times the (robust) standard deviation around the polynomial
        fit to include a new line in the set of identified lines.
    LDEBUG : bool
        If True intermediate results are displayed.
    LPLOT : bool
        If True intermediate plots are displayed.


    Returns
    -------
    solution : list of dictionaries
        A list of size equal to the number of arc lines, which elements are
        dictionaries containing all the relevant information concerning the
        line identification.

    """

    nlines_master = wv_master.size

    nlines_arc = xpos_arc.size
    if nlines_arc < 5:
        raise ValueError('insufficient arc lines')

    #---
    # Generate triplets with consecutive arc lines. For each triplet,
    # compatible triplets from the master table are sought. Each compatible
    # triplet from the master table provides an estimate for CRVAL1 and CDELT1.
    # As an additional constraint, the only valid solutions are those for which
    # the initial and the final wavelengths for the arc are restricted to a
    # predefined wavelength interval.
    crval1_search = np.array([])
    cdelt1_search = np.array([])
    error_crval1_search = np.array([])
    error_cdelt1_search = np.array([])
    itriplet_search = np.array([])
    clabel_search = []

    ntriplets_arc = nlines_arc - 2

    for i in range(ntriplets_arc):
        i1, i2, i3 = i, i+1, i+2

        dist12 = xpos_arc[i2]-xpos_arc[i1]
        dist13 = xpos_arc[i3]-xpos_arc[i1]
        ratio_arc = dist12/dist13

        pol_r = ratio_arc*(ratio_arc-1)+1
        error_ratio_arc = np.sqrt(2)*error_xpos_arc/dist13*np.sqrt(pol_r)

        ratio_arc_min = max(0.0, ratio_arc-times_sigma_r*error_ratio_arc)
        ratio_arc_max = min(1.0, ratio_arc+times_sigma_r*error_ratio_arc)

        j_loc_min = np.searchsorted(ratios_master_sorted, ratio_arc_min)-1
        j_loc_max = np.searchsorted(ratios_master_sorted, ratio_arc_max)+1

        if j_loc_min < 0:
            j_loc_min = 0
        if j_loc_max > ntriplets_master:
            j_loc_max = ntriplets_master


        for j_loc in range(j_loc_min, j_loc_max):
            j1, j2, j3 = triplets_master_sorted_list[j_loc]
            cdelt1_temp = (wv_master[j3]-wv_master[j1])/dist13
            crval1_temp = wv_master[j2]-(xpos_arc[i2]-1)*cdelt1_temp
            crvaln_temp = crval1_temp + float(naxis1_arc-1)*cdelt1_temp
            if wv_ini_search <= crval1_temp <= wv_end_search:
                if wv_ini_search <= crvaln_temp <= wv_end_search:
                    # Compute errors
                    error_crval1_temp = cdelt1_temp*error_xpos_arc* \
                      np.sqrt(1+2*((xpos_arc[i2]-1)**2)/(dist13**2))
                    error_cdelt1_temp = np.sqrt(2)*cdelt1_temp* \
                      error_xpos_arc/dist13
                    # Store values and errors
                    crval1_search = np.append(crval1_search, crval1_temp)
                    cdelt1_search = np.append(cdelt1_search, cdelt1_temp)
                    error_crval1_search = np.append(error_crval1_search, 
                                                    error_crval1_temp)
                    error_cdelt1_search = np.append(error_cdelt1_search, 
                                                    error_cdelt1_temp)
                    # Store additional information about the triplets
                    itriplet_search = np.append(itriplet_search, i)
                    clabel_search.append((j1, j2, j3))


    #assert 1 == 0
    # Normalize the values of CDELT1 and CRVAL1 to the interval [0,1] in each
    # case.
    cdelt1_max = (wv_end_search-wv_ini_search)/float(naxis1_arc-1)
    cdelt1_search_norm = cdelt1_search/cdelt1_max
    error_cdelt1_search_norm = error_cdelt1_search/cdelt1_max
    #
    crval1_search_norm = (crval1_search-wv_ini_search)
    crval1_search_norm /= (wv_end_search-wv_ini_search)
    error_crval1_search_norm = error_crval1_search/(wv_ini_search-wv_end_search)



    #---
    # Segregate the different solutions (normalized to [0,1]) by triplet. In
    # this way the solutions are saved in different layers (a layer for each
    # triplet). The solutions will be stored as python lists of numpy arrays.
    ntriplets_layered_list = []
    cdelt1_layered_list = []
    error_cdelt1_layered_list = []
    crval1_layered_list = []
    error_crval1_layered_list = []
    itriplet_layered_list = []
    clabel_layered_list = []
    for i in range(ntriplets_arc):
        ldum = (itriplet_search == i)
        ntriplets_layered_list.append(ldum.sum())
        #
        cdelt1_dum = cdelt1_search_norm[ldum]
        cdelt1_layered_list.append(cdelt1_dum)
        error_cdelt1_dum = error_cdelt1_search_norm[ldum]
        error_cdelt1_layered_list.append(error_cdelt1_dum)
        #
        crval1_dum = crval1_search_norm[ldum]
        crval1_layered_list.append(crval1_dum)
        error_crval1_dum = error_crval1_search_norm[ldum]
        error_crval1_layered_list.append(error_crval1_dum)
        #
        itriplet_dum = itriplet_search[ldum]
        itriplet_layered_list.append(itriplet_dum)
        #
        # This can be done also
        # with itertools.compress
        # A list with only those where ldum is True
        clabel_dum = [i for (i, v) in zip(clabel_search, ldum) if v]
        clabel_layered_list.append(clabel_dum)


    #---
    # Computation of the cost function.
    #
    # For each solution, corresponding to a particular triplet, find the
    # nearest solutions in the remaining ntriplets_arc-1 layers. Compute the
    # distance (in normalized coordinates) to those closests solutions, 
    # and obtain the sum of distances considering only a fraction of them
    # (after sorting them in ascending order).
    ntriplets_for_sum = \
      max(1, int(my_round(frac_triplets_for_sum*float(ntriplets_arc))))

    funcost_search = np.zeros(len(itriplet_search))
    for k in range(len(itriplet_search)):
        itriplet_local = itriplet_search[k]
        x0 = cdelt1_search_norm[k]
        ex0 = error_cdelt1_search_norm[k]
        y0 = crval1_search_norm[k]
        ey0 = error_crval1_search_norm[k]
        dist_to_layers = np.array([])
        for i in range(ntriplets_arc):
            if i != itriplet_local:
                if ntriplets_layered_list[i] > 0:
                    x1 = cdelt1_layered_list[i]
                    ex1 = error_cdelt1_layered_list[i]
                    y1 = crval1_layered_list[i]
                    ey1 = error_crval1_layered_list[i]
                    dist2 = (x0-x1)**2 + (y0-y1)**2

                    dist_to_layers = np.append(dist_to_layers, dist2.min())
                else:
                    dist_to_layers = np.append(dist_to_layers, np.inf)
        dist_to_layers.sort() # in-place sort
        funcost_search[k] = dist_to_layers[list(six.moves.range(ntriplets_for_sum))].sum()


    # Normalize the cost function
    funcost_min = funcost_search.min()

    funcost_search /= funcost_min

    # Segregate the cost function by arc triplet.
    funcost_layered_list = []
    for i in range(ntriplets_arc):
        ldum = (itriplet_search == i)
        funcost_dum = funcost_search[ldum]
        funcost_layered_list.append(funcost_dum)

    #---
    # Line identification: several scenarios are considered.
    #
    # * Lines with three identifications:
    #   - Type A: the three identifications are identical. Keep the lowest
    #     value of the three cost functions.
    #   - Type B: two identifications are identical and one is different. Keep
    #     the line with two identifications and the lowest of the corresponding
    #     two cost functions.
    #   - Type C: the three identifications are different. Keep the one which
    #     closest to a previosly identified type B line. Use the corresponding
    #     cost function.
    #
    # * Lines with two identifications (second and penultimate lines).
    #   - Type D: the two identifications are identical. Keep the lowest cost
    #     function.
    #
    # * Lines with only one identification (first and last lines).
    #   - Type E: the two lines next (or previous) to the considered line have
    #     been identified. Keep its cost function.
    #

    # We store the identifications of each line in a python list of lists call
    # diagonal_ids (which grows as the different triplets are considered). A
    # similar list of lists is also employed to store the corresponding cost
    # functions.
    for i in range(ntriplets_arc):
        jdum = funcost_layered_list[i].argmin()
        k1, k2, k3 = clabel_layered_list[i][jdum]
        funcost_dum = funcost_layered_list[i][jdum]
        if i == 0:
            diagonal_ids = [[k1],[k2],[k3]]
            diagonal_funcost = [[funcost_dum],[funcost_dum], [funcost_dum]]
        else:
            diagonal_ids[i].append(k1)
            diagonal_ids[i+1].append(k2)
            diagonal_ids.append([k3])
            diagonal_funcost[i].append(funcost_dum)
            diagonal_funcost[i+1].append(funcost_dum)
            diagonal_funcost.append([funcost_dum])
    

    # The solutions are stored in a list of dictionaries. The dictionaries
    # contain the following elements:
    # - lineok: bool, indicates whether the line has been properly identified
    # - type: 'A','B','C','D','E',...
    # - id: index of the line in the master table
    # - funcost: cost function associated the the line identification
    solution = []
    for i in range(nlines_arc):
        solution.append({'lineok': False, 
                         'type': None,
                         'id': None, 
                         'funcost': None})

    # Type A lines.
    for i in range(2,nlines_arc-2):
        j1,j2,j3 = diagonal_ids[i]
        if j1 == j2 == j3:
            solution[i]['lineok'] = True
            solution[i]['type'] = 'A'
            solution[i]['id'] = j1
            solution[i]['funcost'] = min(diagonal_funcost[i])
    

    # Type B lines.
    for i in range(2,nlines_arc-2):
        if solution[i]['lineok'] == False:
            j1,j2,j3 = diagonal_ids[i]
            f1,f2,f3 = diagonal_funcost[i]
            if j1 == j2:
                if max(f1,f2) < f3:
                    solution[i]['lineok'] = True
                    solution[i]['type'] = 'B'
                    solution[i]['id'] = j1
                    solution[i]['funcost'] = min(f1,f2)
            elif j1 == j3:
                if max(f1,f3) < f2:
                    solution[i]['lineok'] = True
                    solution[i]['type'] = 'B'
                    solution[i]['id'] = j1
                    solution[i]['funcost'] = min(f1,f3)
            elif j2 == j3:
                if max(f2,f3) < f1:
                    solution[i]['lineok'] = True
                    solution[i]['type'] = 'B'
                    solution[i]['id'] = j2
                    solution[i]['funcost'] = min(f2,f3)


    # Type C lines.
    for i in range(2,nlines_arc-2):
        if solution[i]['lineok'] == False:
            j1,j2,j3 = diagonal_ids[i]
            f1,f2,f3 = diagonal_funcost[i]
            if solution[i-1]['type'] == 'B':
                if min(f2,f3) > f1:
                    solution[i]['lineok'] = True
                    solution[i]['type'] = 'C'
                    solution[i]['id'] = j1
                    solution[i]['funcost'] = f1
            elif solution[i+1]['type'] == 'B':
                if min(f1,f2) > f3:
                    solution[i]['lineok'] = True
                    solution[i]['type'] = 'C'
                    solution[i]['id'] = j3
                    solution[i]['funcost'] = f3
    

    # Type D lines.
    for i in [1,nlines_arc-2]:
        j1,j2 = diagonal_ids[i]
        if j1 == j2:
            f1,f2 = diagonal_funcost[i]
            solution[i]['lineok'] = True
            solution[i]['type'] = 'D'
            solution[i]['id'] = j1
            solution[i]['funcost'] = min(f1,f2)


    # Type E lines.
    i = 0
    if solution[i+1]['lineok'] and solution[i+2]['lineok']:
        solution[i]['lineok'] = True
        solution[i]['type'] = 'E'
        solution[i]['id'] = diagonal_ids[i][0]
        solution[i]['funcost'] = diagonal_funcost[i][0]
    i = nlines_arc-1
    if solution[i-2]['lineok'] and solution[i-1]['lineok']:
        solution[i]['lineok'] = True
        solution[i]['type'] = 'E'
        solution[i]['id'] = diagonal_ids[i][0]
        solution[i]['funcost'] = diagonal_funcost[i][0]
    


    #---
    # Check that the solutions do not contain duplicated values. If they are
    # present (probably due to the influence of an unknown line that
    # unfortunately falls too close to a real line in the master table), we
    # keep the solution with the lowest cost function. The removed lines are
    # labelled as type='R'. The procedure is repeated several times in case
    # a line appears more than twice.
    lduplicated = True
    while lduplicated:
        lduplicated = False
        for i1 in range(nlines_arc):
            if solution[i1]['lineok']:
                j1 = solution[i1]['id']
                for i2 in range(i1+1,nlines_arc):
                    if solution[i2]['lineok']:
                        j2 = solution[i2]['id']
                        if j1 == j2:
                            lduplicated = True
                            f1 = solution[i1]['funcost']
                            f2 = solution[i2]['funcost']
                            if f1 < f2:
                                solution[i2]['lineok'] = False
                                solution[i2]['type'] = 'R'
                            else:
                                solution[i1]['lineok'] = False
                                solution[i1]['type'] = 'R'
    

    #---
    # Filter out points with a large deviation from a robust linear fit. The
    # filtered lines are labelled as type='T'.

    nfit, ifit, xfit, yfit, wfit = \
      select_data_for_fit(wv_master, xpos_arc, solution)
    intercept, slope = fitTheilSen(xfit, yfit)
    rfit = abs(yfit - (intercept + slope*xfit))

    sigma_rfit = sigmaG(rfit)

    for i in range(nfit):
        if rfit[i] > times_sigma_TheilSen*sigma_rfit:
            solution[ifit[i]]['lineok'] = False
            solution[ifit[i]]['type'] = 'T'

    #---
    # Filter out points that deviates from a polynomial fit. The filtered lines
    # are labelled as type='P'.

    nfit, ifit, xfit, yfit, wfit = \
      select_data_for_fit(wv_master, xpos_arc, solution)
    coeff_fit = polynomial.polyfit(xfit, yfit, poly_degree, w=1/wfit)
    poly = polynomial.Polynomial(coeff_fit)
    rfit = abs(yfit - poly(xfit))

    sigma_rfit = sigmaG(rfit)

    for i in range(nfit):
        if rfit[i] > times_sigma_polfilt*sigma_rfit:
            solution[ifit[i]]['lineok'] = False
            solution[ifit[i]]['type'] = 'P'

    #---
    # Include unidentified lines by using the prediction of the polynomial fit
    # to the current set of identified lines. The included lines are labelled
    # as type='I'.

    nfit, ifit, xfit, yfit, wfit = \
      select_data_for_fit(wv_master, xpos_arc, solution)
    coeff_fit = polynomial.polyfit(xfit, yfit, poly_degree, w=1/wfit)
    poly = polynomial.Polynomial(coeff_fit)
    rfit = abs(yfit - poly(xfit))

    sigma_rfit = sigmaG(rfit)


    list_id_already_found = []
    list_funcost_already_found = []
    for i in range(nlines_arc):
        if solution[i]['lineok']:
            list_id_already_found.append(solution[i]['id'])
            list_funcost_already_found.append(solution[i]['funcost'])

    for i in range(nlines_arc):
        if not solution[i]['lineok']:
            zfit = poly(xpos_arc[i]) # predicted wavelength
            isort = np.searchsorted(wv_master, zfit)
            if isort == 0:
                ifound = 0
                dlambda = wv_master[ifound]-zfit
            elif isort == nlines_master:
                ifound = isort - 1
                dlambda = zfit - wv_master[ifound]
            else:
                dlambda1 = zfit-wv_master[isort-1]
                dlambda2 = wv_master[isort]-zfit
                if dlambda1 < dlambda2:
                    ifound = isort - 1
                    dlambda = dlambda1
                else:
                    ifound = isort
                    dlambda = dlambda2

            if ifound not in list_id_already_found:  # unused line
                if dlambda < times_sigma_inclusion*sigma_rfit:
                    list_id_already_found.append(ifound)
                    solution[i]['lineok'] = True
                    solution[i]['type'] = 'I'
                    solution[i]['id'] = ifound
                    solution[i]['funcost'] = max(list_funcost_already_found)

    return solution
