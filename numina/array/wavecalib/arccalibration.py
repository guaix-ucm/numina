#
# Copyright 2015-2016 Universidad Complutense de Madrid
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

"""Automatic identification of lines and wavelength calibration"""

from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
from numpy.polynomial import Polynomial
import scipy.misc

from ..display.pause_debugplot import pause_debugplot
from ..robustfit import fit_theil_sen
from ..display.polfit_residuals import polfit_residuals_with_cook_rejection
from ..stats import robust_std
from .solutionarc import CrLinear, WavecalFeature,SolutionArcCalibration


def select_data_for_fit(list_of_wvfeatures):
    """Select information from valid arc lines to facilitate posterior fits.

    Parameters
    ----------
    list_of_wvfeatures : list (of WavecalFeature instances)
        A list of size equal to the number of identified lines, which
        elements are instances of the class WavecalFeature, containing
        all the relevant information concerning the line
        identification.

    Returns
    -------
    nfit : int
        Number of valid points for posterior fits.
    ifit : list of int
        List of indices corresponding to the arc lines which
        coordinates are going to be employed in the posterior fits.
    xfit : 1d numpy aray
        X coordinate of points for posterior fits.
    yfit : 1d numpy array
        Y coordinate of points for posterior fits.
    wfit : 1d numpy array
        Cost function of points for posterior fits. The inverse of
        these values can be employed for weighted fits.

    """

    nlines_arc = len(list_of_wvfeatures)

    nfit = 0
    ifit = []
    xfit = np.array([])
    yfit = np.array([])
    wfit = np.array([])
    for i in range(nlines_arc):
        if list_of_wvfeatures[i].line_ok:
            ifit.append(i)
            xfit = np.append(xfit, [list_of_wvfeatures[i].xpos])
            yfit = np.append(yfit, [list_of_wvfeatures[i].reference])
            wfit = np.append(wfit, [list_of_wvfeatures[i].funcost])
            nfit += 1

    return nfit, ifit, xfit, yfit, wfit


def fit_list_of_wvfeatures(list_of_wvfeatures,
                           naxis1_arc,
                           crpix1,
                           poly_degree_wfit,
                           weighted=False,
                           debugplot=0,
                           plot_title=None):
    """Fit polynomial to arc calibration list_of_wvfeatures.

    Parameters
    ----------
    list_of_wvfeatures : list (of WavecalFeature instances)
        A list of size equal to the number of identified lines, which
        elements are instances of the class WavecalFeature, containing
        all the relevant information concerning the line
        identification.
    naxis1_arc : int
        NAXIS1 of arc spectrum.
    crpix1 : float
        CRPIX1 value to be employed in the wavelength calibration.
    poly_degree_wfit : int
        Polynomial degree corresponding to the wavelength calibration
        function to be fitted.
    weighted : bool
        Determines whether the polynomial fit is weighted or not,
        using as weights the values of the cost function obtained in
        the line identification. Since the weights can be very
        different, typically weighted fits are not good because, in
        practice, they totally ignore the points with the smallest
        weights (which, in the other hand, are useful when handling
        the borders of the wavelength calibration range).
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses
    plot_title : string or None
        Title for residuals plot.

    Returns
    -------
    solution_wv : SolutionArcCalibration instance
        Instance of class SolutionArcCalibration, containing the
        information concerning the arc lines that have been properly
        identified. The information about all the lines (including
        those initially found but at the end discarded) is stored in
        the list of WavecalFeature instances 'list_of_wvfeatures'.

    """

    nlines_arc = len(list_of_wvfeatures)

    # select information from valid lines.
    nfit, ifit, xfit, yfit, wfit = select_data_for_fit(list_of_wvfeatures)

    # select list of filtered out and unidentified lines
    list_r = []
    list_t = []
    list_p = []
    list_k = []
    list_unidentified = []
    for i in range(nlines_arc):
        if not list_of_wvfeatures[i].line_ok:
            if list_of_wvfeatures[i].category == 'X':
                list_unidentified.append(i)
            elif list_of_wvfeatures[i].category == 'R':
                list_r.append(i)
            elif list_of_wvfeatures[i].category == 'T':
                list_t.append(i)
            elif list_of_wvfeatures[i].category == 'P':
                list_p.append(i)
            elif list_of_wvfeatures[i].category == 'K':
                list_k.append(i)
            else:
                raise ValueError('Unexpected "category"')

    # polynomial fit
    if weighted:
        weights = 1.0 / wfit
    else:
        weights = np.zeros_like(wfit) + 1.0

    if xfit.size <= poly_degree_wfit:
        raise ValueError("Insufficient number of points for fit.")
    poly, stats_list = Polynomial.fit(
        x=xfit, y=yfit, deg=poly_degree_wfit, full=True, w=weights
    )
    poly = Polynomial.cast(poly)
    coeff = poly.coef
    residual_std = np.sqrt(stats_list[0]/(len(xfit)-2))[0]

    if debugplot >= 10:
        print('>>> Fitted coefficients:\n', coeff)
        print('>>> Residual std.......:', residual_std)

    # obtain CRVAL1 and CDELT1 for a linear wavelength scale from the
    # last polynomial fit
    crval1_linear = poly(crpix1)
    crmin1_linear = poly(1)
    crmax1_linear = poly(naxis1_arc)
    cdelt1_linear = (crmax1_linear - crval1_linear) / (naxis1_arc - crpix1)
    if debugplot >= 10:
        print('>>> CRVAL1 linear scale:', crval1_linear)
        print('>>> CDELT1 linear scale:', cdelt1_linear)

    # generate solution (note that the class SolutionArcCalibration
    # only sotres the information in list_of_wvfeatures corresponding
    # to lines that have been properly identified
    cr_linear = CrLinear(
        crpix1,
        crval1_linear,
        crmin1_linear,
        crmax1_linear,
        cdelt1_linear
    )

    solution_wv = SolutionArcCalibration(
        features=list_of_wvfeatures,
        coeff=coeff,
        residual_std=residual_std,
        cr_linear=cr_linear
    )

    if debugplot % 10 != 0:
        # polynomial fit
        xpol = np.linspace(1, naxis1_arc, naxis1_arc)
        ypol = poly(xpol) - (crval1_linear + (xpol - crpix1) * cdelt1_linear)
        # identified lines
        xp = np.copy(xfit)
        yp = yfit - (crval1_linear + (xp - crpix1) * cdelt1_linear)
        yres = yfit - poly(xp)  # residuals
        # include residuals plot with identified lines
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_xlim([1 - 0.05 * naxis1_arc, naxis1_arc + 0.05 * naxis1_arc])
        ax2.set_xlabel('pixel position in arc spectrum [from 1 to NAXIS1]')
        ax2.set_ylabel('residuals (Angstrom)')
        ax2.plot(xp, yres, 'go')
        ax2.axhline(y=0.0, color="black", linestyle="dashed")
        # residuals with R, T, P and K lines
        for val in zip(["R", "T", "P", "K"],
                       [list_r, list_t, list_p, list_k],
                       ['red', 'blue', 'magenta', 'orange']):
            list_x = val[1]
            if len(list_x) > 0:
                xxp = np.array([])
                yyp = np.array([])
                for i in list_x:
                    xxp = np.append(xxp, [list_of_wvfeatures[i].xpos])
                    yyp = np.append(yyp, [list_of_wvfeatures[i].reference])
                yyres = yyp - poly(xxp)
                ax2.plot(xxp, yyres, marker='x', markersize=15, c=val[2],
                         linewidth=0)

        # plot with differences between linear fit and fitted
        # polynomial
        ax = fig.add_subplot(2, 1, 1, sharex=ax2)
        ax.set_xlim([1 - 0.05 * naxis1_arc, naxis1_arc + 0.05 * naxis1_arc])
        ax.set_ylabel('differences with\nlinear solution (Angstrom)')
        ax.plot(xp, yp, 'go', label="identified")
        for i in range(nfit):
            ax.text(xp[i], yp[i], list_of_wvfeatures[ifit[i]].category,
                    fontsize=15)
        # polynomial fit
        ax.plot(xpol, ypol, 'c-', label="fit")
        # unidentified lines
        if len(list_unidentified) > 0:
            ymin = np.concatenate((yp, ypol)).min()
            ymax = np.concatenate((yp, ypol)).max()
            for i in list_unidentified:
                xxp = np.array([list_of_wvfeatures[i].xpos,
                                list_of_wvfeatures[i].xpos])
                yyp = np.array([ymin, ymax])
                if i == list_unidentified[0]:
                    ax.plot(xxp, yyp, 'r--', label='unidentified')
                else:
                    ax.plot(xxp, yyp, 'r--')
        # R, T, P and K lines
        for val in zip(["R", "T", "P", "K"],
                       [list_r, list_t, list_p, list_k],
                       ['red', 'blue', 'magenta', 'orange']):
            list_x = val[1]
            if len(list_x) > 0:
                xxp = np.array([])
                yyp = np.array([])
                for i in list_x:
                    xxp = np.append(xxp, [list_of_wvfeatures[i].xpos])
                    yyp = np.append(yyp, [list_of_wvfeatures[i].reference])
                yyp -= crval1_linear + (xxp - crpix1) * cdelt1_linear
                ax.plot(xxp, yyp, marker='x', markersize=15, c=val[2],
                        linewidth=0, label='removed')
                for k in range(len(xxp)):
                    ax.text(xxp[k], yyp[k], val[0], fontsize=15)

        # shrink current axis and put a legend
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0,
                          box.width, box.height * 0.92])
        delta_ybox = box.height * 0.15
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 - delta_ybox,
                         box.width, box.height * 0.92])
        ax.legend(loc=3, bbox_to_anchor=(0.0, 1.1, 1., 0.07),
                  mode="expand", borderaxespad=0., ncol=4,
                  numpoints=1)

        # title
        if plot_title is None:
            plt.title("Wavelength calibration\n\n\n")
        else:
            plt.title(plot_title + "\n\n\n")

        # include important parameters in plot
        ax.text(0.98, 0.25, "poldeg: " + str(poly_degree_wfit),
                fontsize=12,
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="bottom")
        ax.text(0.98, 0.15, "CRVAL1: " + str(round(crval1_linear, 4)),
                fontsize=12,
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="bottom")
        ax.text(0.98, 0.05, "CDELT1: " + str(round(cdelt1_linear, 4)),
                fontsize=12,
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="bottom")
        ax2.text(0.98, 0.05, "r.m.s.: " + str(round(residual_std, 4)),
                 fontsize=12,
                 transform=ax2.transAxes,
                 horizontalalignment="right",
                 verticalalignment="bottom")

        plt.show(block=False)
        plt.pause(0.001)
        pause_debugplot(debugplot)

    return solution_wv


def gen_triplets_master(wv_master, debugplot=0):
    """Compute information associated to triplets in master table.

    Determine all the possible triplets that can be generated from the
    array `wv_master`. In addition, the relative position of the
    central line of each triplet is also computed.

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table
        (Angstroms).
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    Returns
    -------
    ntriplets_master : int
        Number of triplets built from master table.
    ratios_master_sorted : 1d numpy array, float
        Array with values of the relative position of the central line
        of each triplet, sorted in ascending order.
    triplets_master_sorted_list : list of tuples
        List with tuples of three numbers, corresponding to the three
        line indices in the master table. The list is sorted to be in
        correspondence with `ratios_master_sorted`.

    """

    nlines_master = wv_master.size

    # Generate all the possible triplets with the numbers of the lines
    # in the master table. Each triplet is defined as a tuple of three
    # numbers corresponding to the three line indices in the master
    # table. The collection of tuples is stored in an ordinary python
    # list.
    iter_comb_triplets = itertools.combinations(range(nlines_master), 3)
    triplets_master_list = [val for val in iter_comb_triplets]

    # Verify that the number of triplets coincides with the expected
    # value.
    ntriplets_master = len(triplets_master_list)
    if ntriplets_master == scipy.misc.comb(nlines_master, 3, exact=True):
        if debugplot >= 10:
            print('>>> Total number of lines in master table:', 
                  nlines_master)
            print('>>> Number of triplets in master table...:', 
                  ntriplets_master)
    else:
        raise ValueError('Invalid number of combinations')

    # For each triplet, compute the relative position of the central
    # line.
    ratios_master = np.zeros(ntriplets_master)
    for index, value in enumerate(triplets_master_list):
        i1, i2, i3 = value
        delta1 = wv_master[i2] - wv_master[i1]
        delta2 = wv_master[i3] - wv_master[i1]
        ratios_master[index] = delta1 / delta2

    # Compute the array of indices that index the above ratios in
    # sorted order.
    isort_ratios_master = np.argsort(ratios_master)

    # Simultaneous sort of position ratios and triplets.
    ratios_master_sorted = ratios_master[isort_ratios_master]
    triplets_master_sorted_list = [triplets_master_list[i]
                                   for i in isort_ratios_master]

    if debugplot in [21, 22]:
        # compute and plot histogram with position ratios
        bins_in = np.linspace(0.0, 1.0, 41)
        hist, bins_out = np.histogram(ratios_master, bins=bins_in)
        #
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width_hist = 0.8*(bins_out[1]-bins_out[0])
        center = (bins_out[:-1]+bins_out[1:])/2
        ax.bar(center, hist, align='center', width=width_hist)
        ax.set_xlabel('distance ratio in each triplet')
        ax.set_ylabel('Number of triplets')
        plt.title("Number of lines/triplets: " +
                  str(nlines_master) + "/" + str(ntriplets_master))
        plt.show(block=False)
        plt.pause(0.001)
        pause_debugplot(debugplot)

    return ntriplets_master, ratios_master_sorted, triplets_master_sorted_list


def arccalibration(wv_master,
                   xpos_arc,
                   naxis1_arc,
                   crpix1,
                   wv_ini_search, wv_end_search,
                   error_xpos_arc,
                   times_sigma_r,
                   frac_triplets_for_sum,
                   times_sigma_theil_sen,
                   poly_degree_wfit,
                   times_sigma_polfilt,
                   times_sigma_cook,
                   times_sigma_inclusion,
                   debugplot=0):
    """Performs arc line identification for arc calibration.

    This function is a wrapper of two functions, which are responsible
    of computing all the relevant information concerning the triplets
    generated from the master table and the actual identification
    procedure of the arc lines, respectively.

    The separation of those computations in two different functions
    helps to avoid the repetition of calls to the first function when
    calibrating several arcs using the same master table.

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table
        (Angstroms).
    xpos_arc : 1d numpy array, float
        Location of arc lines (pixels).
    naxis1_arc : int
        NAXIS1 for arc spectrum.
    crpix1 : float
        CRPIX1 value to be employed in the wavelength calibration.
    wv_ini_search : float
        Minimum valid wavelength.
    wv_end_search : float
        Maximum valid wavelength.
    error_xpos_arc : float
        Error in arc line position (pixels).
    times_sigma_r : float
        Times sigma to search for valid line position ratios.
    frac_triplets_for_sum : float
        Fraction of distances to different triplets to sum when
        computing the cost function.
    times_sigma_theil_sen : float
        Number of times the (robust) standard deviation around the
        linear fit (using the Theil-Sen method) to reject points.
    poly_degree_wfit : int
        Degree for polynomial fit to wavelength calibration.
    times_sigma_polfilt : float
        Number of times the (robust) standard deviation around the
        polynomial fit to reject points.
    times_sigma_cook : float
        Number of times the standard deviation of Cook's distances
        to detect outliers. If zero, this method of outlier detection
        is ignored.
    times_sigma_inclusion : float
        Number of times the (robust) standard deviation around the
        polynomial fit to include a new line in the set of identified
        lines.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses

    Returns
    -------
    list_of_wvfeatures : list (of WavecalFeature instances)
        A list of size equal to the number of identified lines, which
        elements are instances of the class WavecalFeature, containing
        all the relevant information concerning the line
        identification.

    """

    ntriplets_master, ratios_master_sorted, triplets_master_sorted_list = \
      gen_triplets_master(wv_master=wv_master, debugplot=debugplot)

    list_of_wvfeatures = arccalibration_direct(
        wv_master=wv_master,
        ntriplets_master=ntriplets_master,
        ratios_master_sorted=ratios_master_sorted,
        triplets_master_sorted_list=triplets_master_sorted_list,
        xpos_arc=xpos_arc,
        naxis1_arc=naxis1_arc,
        crpix1=crpix1,
        wv_ini_search=wv_ini_search,
        wv_end_search=wv_end_search,
        error_xpos_arc=error_xpos_arc,
        times_sigma_r=times_sigma_r,
        frac_triplets_for_sum=frac_triplets_for_sum,
        times_sigma_theil_sen=times_sigma_theil_sen,
        poly_degree_wfit=poly_degree_wfit,
        times_sigma_polfilt=times_sigma_polfilt,
        times_sigma_cook=times_sigma_cook,
        times_sigma_inclusion=times_sigma_inclusion,
        debugplot=debugplot)

    return list_of_wvfeatures


def arccalibration_direct(wv_master,
                          ntriplets_master,
                          ratios_master_sorted,
                          triplets_master_sorted_list,
                          xpos_arc,
                          naxis1_arc,
                          crpix1,
                          wv_ini_search, 
                          wv_end_search,
                          error_xpos_arc=1.0,
                          times_sigma_r=3.0,
                          frac_triplets_for_sum=0.50,
                          times_sigma_theil_sen=10.0,
                          poly_degree_wfit=3,
                          times_sigma_polfilt=10.0,
                          times_sigma_cook=10.0,
                          times_sigma_inclusion=5.0,
                          debugplot=0):
    """Performs line identification for arc calibration using line triplets.

    This function assumes that a previous call to the function
    responsible for the computation of information related to the
    triplets derived from the master table has been previously
    executed.

    Parameters
    ----------
    wv_master : 1d numpy array, float
        Array with wavelengths corresponding to the master table
        (Angstroms).
    ntriplets_master : int
        Number of triplets built from master table.
    ratios_master_sorted : 1d numpy array, float
        Array with values of the relative position of the central line
        of each triplet, sorted in ascending order.
    triplets_master_sorted_list : list of tuples
        List with tuples of three numbers, corresponding to the three
        line indices in the master table. The list is sorted to be in
        correspondence with `ratios_master_sorted`.
    xpos_arc : 1d numpy array, float
        Location of arc lines (pixels).
    naxis1_arc : int
        NAXIS1 for arc spectrum.
    crpix1 : float
        CRPIX1 value to be employed in the wavelength calibration.
    wv_ini_search : float
        Minimum valid wavelength.
    wv_end_search : float
        Maximum valid wavelength.
    error_xpos_arc : float
        Error in arc line position (pixels).
    times_sigma_r : float
        Times sigma to search for valid line position ratios.
    frac_triplets_for_sum : float
        Fraction of distances to different triplets to sum when
        computing the cost function.
    times_sigma_theil_sen : float
        Number of times the (robust) standard deviation around the
        linear fit (using the Theil-Sen method) to reject points.
    poly_degree_wfit : int
        Degree for polynomial fit to wavelength calibration.
    times_sigma_polfilt : float
        Number of times the (robust) standard deviation around the
        polynomial fit to reject points.
    times_sigma_cook : float
        Number of times the standard deviation of Cook's distances
        to detect outliers. If zero, this method of outlier detection
        is ignored.
    times_sigma_inclusion : float
        Number of times the (robust) standard deviation around the
        polynomial fit to include a new line in the set of identified
        lines.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed:
        00 : no debug, no plots
        01 : no debug, plots without pauses
        02 : no debug, plots with pauses
        10 : debug, no plots
        11 : debug, plots without pauses
        12 : debug, plots with pauses
        21 : debug, many plots without pauses
        22 : debug, many plots with pauses

    Returns
    -------
    list_of_wvfeatures : list (of WavecalFeature instances)
        A list of size equal to the number of identified lines, which
        elements are instances of the class WavecalFeature, containing
        all the relevant information concerning the line
        identification.

    """

    nlines_master = wv_master.size

    if wv_ini_search is None:
        wv_ini_search = wv_master.min()
    if wv_end_search is None:
        wv_end_search = wv_master.max()

    nlines_arc = xpos_arc.size
    if nlines_arc < 5:
        raise ValueError('Insufficient arc lines=' + str(nlines_arc))

    # ---
    # Generate triplets with consecutive arc lines. For each triplet,
    # compatible triplets from the master table are sought. Each
    # compatible triplet from the master table provides an estimate for
    # CRVAL1 and CDELT1. As an additional constraint, the only valid
    # solutions are those for which the initial and the final
    # wavelengths for the arc are restricted to a predefined wavelength
    # interval.
    crval1_search = np.array([])
    cdelt1_search = np.array([])
    error_crval1_search = np.array([])
    error_cdelt1_search = np.array([])
    itriplet_search = np.array([], dtype=int)
    clabel_search = []

    ntriplets_arc = nlines_arc - 2
    if debugplot >= 10:
        print('>>> Total number of arc lines............:', nlines_arc)
        print('>>> Total number of arc triplets.........:', ntriplets_arc)

    # Loop in all the arc line triplets. Note that only triplets built
    # from consecutive arc lines are considered.
    for i in range(ntriplets_arc):
        i1, i2, i3 = i, i+1, i+2

        dist12 = xpos_arc[i2] - xpos_arc[i1]
        dist13 = xpos_arc[i3] - xpos_arc[i1]
        ratio_arc = dist12 / dist13

        pol_r = ratio_arc * (ratio_arc - 1) + 1
        error_ratio_arc = np.sqrt(2) * error_xpos_arc/dist13 * np.sqrt(pol_r)

        ratio_arc_min = max(0.0, ratio_arc-times_sigma_r*error_ratio_arc)
        ratio_arc_max = min(1.0, ratio_arc+times_sigma_r*error_ratio_arc)

        # determine compatible triplets from the master list
        j_loc_min = np.searchsorted(ratios_master_sorted, ratio_arc_min)-1
        j_loc_max = np.searchsorted(ratios_master_sorted, ratio_arc_max)+1

        if j_loc_min < 0:
            j_loc_min = 0
        if j_loc_max > ntriplets_master:
            j_loc_max = ntriplets_master

        if debugplot >= 10:
            print(i, ratio_arc_min, ratio_arc, ratio_arc_max, 
                  j_loc_min, j_loc_max)

        # each triplet from the master list provides a potential
        # solution for CRVAL1 and CDELT1
        for j_loc in range(j_loc_min, j_loc_max):
            j1, j2, j3 = triplets_master_sorted_list[j_loc]
            # initial solutions for CDELT1, CRVAL1 and CRMAX1
            cdelt1_temp = (wv_master[j3]-wv_master[j1])/dist13
            crval1_temp = wv_master[j2]-(xpos_arc[i2]-crpix1)*cdelt1_temp
            crmin1_temp = crval1_temp + float(1-crpix1)*cdelt1_temp
            crmax1_temp = crval1_temp + float(naxis1_arc-crpix1)*cdelt1_temp
            # check that CRMIN1 and CRMAX1 are within the valid limits
            if wv_ini_search <= crmin1_temp <= wv_end_search \
                    and wv_ini_search <= crmax1_temp <= wv_end_search:
                # Compute errors
                error_crval1_temp = \
                    cdelt1_temp*error_xpos_arc * \
                    np.sqrt(1+2*((xpos_arc[i2]-crpix1)**2)/(dist13**2))
                error_cdelt1_temp = np.sqrt(2)*cdelt1_temp * \
                                    error_xpos_arc/dist13
                # Store values and errors
                crval1_search = np.append(crval1_search, [crval1_temp])
                cdelt1_search = np.append(cdelt1_search, [cdelt1_temp])
                error_crval1_search = np.append(error_crval1_search,
                                                [error_crval1_temp])
                error_cdelt1_search = np.append(error_cdelt1_search,
                                                [error_cdelt1_temp])
                # Store additional information about the triplets
                itriplet_search = np.append(itriplet_search, [i])
                clabel_search.append((j1, j2, j3))

    # maximum allowed value for CDELT1
    cdelt1_max = (wv_end_search-wv_ini_search)/float(naxis1_arc-1)

    # normalize the values of CDELT1 and CRVAL1 to the interval [0,1]
    # in each case
    cdelt1_search_norm = cdelt1_search/cdelt1_max
    error_cdelt1_search_norm = error_cdelt1_search/cdelt1_max
    #
    crval1_search_norm = (crval1_search-wv_ini_search)
    crval1_search_norm /= (wv_end_search-wv_ini_search)
    error_crval1_search_norm = error_crval1_search
    error_crval1_search_norm /= (wv_end_search-wv_ini_search)

    # intermediate plots
    if debugplot in [21, 22]:
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt

        # CDELT1 vs CRVAL1 diagram (original coordinates)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('cdelt1 (Angstroms/pixel)')
        ax.set_ylabel('crval1 (Angstroms)')
        ax.scatter(cdelt1_search, crval1_search, s=200, alpha=0.1)
        xmin = 0.0
        xmax = cdelt1_max
        dx = xmax-xmin
        xmin -= dx/20
        xmax += dx/20
        ax.set_xlim([xmin, xmax])
        ymin = wv_ini_search
        ymax = wv_end_search
        dy = ymax-ymin
        ymin -= dy/20
        ymax += dy/20
        ax.set_ylim([ymin, ymax])
        xp_limits = np.array([0., cdelt1_max])
        yp_limits = wv_end_search-float(naxis1_arc-1)*xp_limits
        xp_limits = np.concatenate((xp_limits, [xp_limits[0], xp_limits[0]]))
        yp_limits = np.concatenate((yp_limits, [yp_limits[1], yp_limits[0]]))
        ax.plot(xp_limits, yp_limits, linestyle='-', color='magenta')
        plt.title("Potential solutions within the valid parameter space")
        plt.show(block=False)
        plt.pause(0.001)
        print('Number of points in last plot:', len(cdelt1_search))
        pause_debugplot(debugplot)

        # CDELT1 vs CRVAL1 diagram (normalized coordinates)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('normalized cdelt1')
        ax.set_ylabel('normalized crval1')
        ax.scatter(cdelt1_search_norm, crval1_search_norm, s=200, alpha=0.1)
        xmin = -0.05
        xmax = 1.05
        ymin = -0.05
        ymax = 1.05
        xp_limits = np.array([0., 1., 0., 0.])
        yp_limits = np.array([1., 0., 0., 1.])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.plot(xp_limits, yp_limits, linestyle='-', color='magenta')
        plt.title("Potential solutions within the valid parameter space")
        plt.show(block=False)
        plt.pause(0.001)
        print('Number of points in last plot:', len(cdelt1_search_norm))
        pause_debugplot(debugplot)

        # CDELT1 vs CRVAL1 diagram (normalized coordinates)
        # with different color for each arc triplet and overplotting 
        # the arc triplet number
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('normalized cdelt1')
        ax.set_ylabel('normalized crval1')
        ax.scatter(cdelt1_search_norm, crval1_search_norm, s=200, alpha=0.1,
                   c=itriplet_search)
        for i in range(len(itriplet_search)):
            ax.text(cdelt1_search_norm[i], crval1_search_norm[i], 
                    str(int(itriplet_search[i])), fontsize=6)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.plot(xp_limits, yp_limits, linestyle='-', color='magenta')
        plt.title("Potential solutions: arc line triplet number")
        plt.show(block=False)
        plt.pause(0.001)
        print('Number of points in last plot:', len(cdelt1_search_norm))
        pause_debugplot(debugplot)

        # CDELT1 vs CRVAL1 diagram (normalized coordinates)
        # including triplet numbers
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('normalized cdelt1')
        ax.set_ylabel('normalized crval1')
        ax.scatter(cdelt1_search_norm, crval1_search_norm, s=200, alpha=0.1,
                   c=itriplet_search)
        for i in range(len(clabel_search)):
            ax.text(cdelt1_search_norm[i], crval1_search_norm[i], 
                    clabel_search[i], fontsize=6)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.plot(xp_limits, yp_limits, linestyle='-', color='magenta')
        plt.title("Potential solutions: master line triplets")
        plt.show(block=False)
        plt.pause(0.001)
        print('Number of points in last plot:', len(cdelt1_search_norm))
        pause_debugplot(debugplot)

        # CDELT1 vs CRVAL1 diagram (normalized coordinates)
        # with error bars (note that errors in this plot are highly
        # correlated)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('normalized cdelt1')
        ax.set_ylabel('normalized crval1')
        ax.errorbar(cdelt1_search_norm, crval1_search_norm, 
                    xerr=error_cdelt1_search_norm,
                    yerr=error_crval1_search_norm,
                    fmt='none')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.plot(xp_limits, yp_limits, linestyle='-', color='magenta')
        plt.title("Potential solutions within the valid parameter space")
        plt.show(block=False)
        plt.pause(0.001)
        print('Number of points in last plot:', len(cdelt1_search_norm))
        pause_debugplot(debugplot)

    # ---
    # Segregate the different solutions (normalized to [0,1]) by
    # triplet. In this way the solutions are saved in different layers
    # (a layer for each triplet). The solutions will be stored as python
    # lists of numpy arrays.
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
        clabel_dum = [k for (k, v) in zip(clabel_search, ldum) if v]
        clabel_layered_list.append(clabel_dum)
    
    if debugplot >= 10:
        print('>>> Total number of potential solutions: ' +
              str(sum(ntriplets_layered_list)) + " (double check ==) " +
              str(len(itriplet_search)))
        print('>>> List with no. of solutions/triplet.:\n' +
              str(ntriplets_layered_list))
        pause_debugplot(debugplot)

    # ---
    # Computation of the cost function.
    #
    # For each solution, corresponding to a particular triplet, find
    # the nearest solution in each of the remaining ntriplets_arc-1
    # layers. Compute the distance (in normalized coordinates) to those
    # closest solutions, and obtain the sum of distances considering
    # only a fraction of them (after sorting them in ascending order).
    ntriplets_for_sum = max(
        1, int(round(frac_triplets_for_sum*float(ntriplets_arc)))
    )
    funcost_search = np.zeros(len(itriplet_search))
    for k in range(len(itriplet_search)):
        itriplet_local = itriplet_search[k]
        x0 = cdelt1_search_norm[k]
        y0 = crval1_search_norm[k]
        dist_to_layers = np.array([])
        for i in range(ntriplets_arc):
            if i != itriplet_local:
                if ntriplets_layered_list[i] > 0:
                    x1 = cdelt1_layered_list[i]
                    y1 = crval1_layered_list[i]
                    dist2 = (x0-x1)**2 + (y0-y1)**2
                    dist_to_layers = np.append(dist_to_layers, [min(dist2)])
                else:
                    dist_to_layers = np.append(dist_to_layers, [np.inf])
        dist_to_layers.sort()  # in-place sort
        funcost_search[k] = dist_to_layers[range(ntriplets_for_sum)].sum()

    # normalize the cost function
    funcost_min = min(funcost_search)
    if debugplot >= 10:
        print('funcost_min:', funcost_min)
    funcost_search /= funcost_min

    # segregate the cost function by arc triplet.
    funcost_layered_list = []
    for i in range(ntriplets_arc):
        ldum = (itriplet_search == i)
        funcost_dum = funcost_search[ldum]
        funcost_layered_list.append(funcost_dum)
    if debugplot >= 10:
        for i in range(ntriplets_arc):
            if ntriplets_layered_list[i] > 0:
                jdum = funcost_layered_list[i].argmin()
                print('>>>', i, funcost_layered_list[i][jdum],
                      clabel_layered_list[i][jdum],
                      cdelt1_layered_list[i][jdum],
                      crval1_layered_list[i][jdum])
            else:
                print('>>>', i, None, "(None, None, None)", None, None)
        pause_debugplot(debugplot)

    # intermediate plots
    if debugplot in [21, 22]:
        import matplotlib
        matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt

        # CDELT1 vs CRVAL1 diagram (normalized coordinates) with symbol
        # size proportional to the inverse of the cost function
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('normalized cdelt1')
        ax.set_ylabel('normalized crval1')
        ax.scatter(cdelt1_search_norm, crval1_search_norm, 
                   s=2000/funcost_search, c=itriplet_search, alpha=0.2)
        xmin = -0.05
        xmax = 1.05
        ymin = -0.05
        ymax = 1.05
        xp_limits = np.array([0., 1., 0., 0.])
        yp_limits = np.array([1., 0., 0., 1.])
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
        plt.title("Potential solutions within the valid parameter space\n" +
                  "[symbol size proportional to 1/(cost function)]")
        plt.show(block=False)
        plt.pause(0.001)
        print('Number of points in last plot:', len(cdelt1_search_norm))
        pause_debugplot(debugplot)

        # CDELT1 vs CRVAL1 diagram (normalized coordinates)
        # with symbol size proportional to the inverse of the cost
        # function and over-plotting triplet number
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('normalized cdelt1')
        ax.set_ylabel('normalized crval1')
        ax.scatter(cdelt1_search_norm, crval1_search_norm, 
                   s=2000/funcost_search, c=itriplet_search, alpha=0.2)
        for i in range(len(itriplet_search)):
            ax.text(cdelt1_search_norm[i], crval1_search_norm[i], 
                    str(int(itriplet_search[i])), fontsize=6)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
        plt.title("Potential solutions: arc line triplet number\n" +
                  "[symbol size proportional to 1/(cost function)]")
        plt.show(block=False)
        plt.pause(0.001)
        print('Number of points in last plot:', len(cdelt1_search))
        pause_debugplot(debugplot)

        # CDELT1 vs CRVAL1 diagram (normalized coordinates)
        # for i in range(ntriplets_arc):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.set_xlabel('normalized cdelt1')
        #     ax.set_ylabel('normalized crval1')
        #     xdum = cdelt1_layered_list[i]
        #     ydum = crval1_layered_list[i]
        #     sdum = 2000/funcost_layered_list[i]
        #     ax.scatter(xdum, ydum, s=sdum, alpha=0.8)
        #     ax.set_xlim([xmin, xmax])
        #     ax.set_ylim([ymin, ymax])
        #     ax.plot(xp_limits, yp_limits, linestyle='-', color='red')
        #     plt.title("Potential solutions: arc line triplet " + str(i) +
        #               " (from 0 to " + str(ntriplets_arc-1) + ")\n" +
        #               "[symbol size proportional to 1/(cost function)]")
        #     plt.show(block=False)
        #     plt.pause(0.001)
        #     print('Number of points in last plot:', xdum.size)
        #     pause_debugplot(debugplot)

    # ---
    # Line identification: several scenarios are considered.
    #
    # * Lines with three identifications:
    #   - Category A: the three identifications are identical. Keep the
    #     lowest value of the three cost functions.
    #   - Category B: two identifications are identical and one is
    #     different. Keep the line with two identifications and the
    #     lowest of the corresponding two cost functions.
    #   - Category C: the three identifications are different. Keep the
    #     one which is closest to a previously identified category B
    #     line. Use the corresponding cost function.
    #
    # * Lines with two identifications (second and penultimate lines).
    #   - Category D: the two identifications are identical. Keep the
    #     lowest cost function value.
    #
    # * Lines with only one identification (first and last lines).
    #   - Category E: the two lines next (or previous) to the considered
    #     line have been identified. Keep its cost function.
    #

    # We store the identifications of each line in a python list of
    # lists named diagonal_ids (which grows as the different triplets
    # are considered). A similar list of lists is also employed to
    # store the corresponding cost functions.
    # It is important to set the identification of the lines to None
    # when no valid master triplet has been associated to a given
    # arc line triplet.
    for i in range(ntriplets_arc):
        if ntriplets_layered_list[i] > 0:
            jdum = funcost_layered_list[i].argmin()
            k1, k2, k3 = clabel_layered_list[i][jdum]
            funcost_dum = funcost_layered_list[i][jdum]
        else:
            k1, k2, k3 = None, None, None
            funcost_dum = np.inf
        if i == 0:
            diagonal_ids = [[k1], [k2], [k3]]
            diagonal_funcost = [[funcost_dum], [funcost_dum], [funcost_dum]]
        else:
            diagonal_ids[i].append(k1)
            diagonal_ids[i+1].append(k2)
            diagonal_ids.append([k3])
            diagonal_funcost[i].append(funcost_dum)
            diagonal_funcost[i+1].append(funcost_dum)
            diagonal_funcost.append([funcost_dum])

    if debugplot >= 10:
        for i in range(nlines_arc):
            print(i, diagonal_ids[i], diagonal_funcost[i])
        pause_debugplot(debugplot)

    # The solutions are stored in a list of WavecalFeature instances.
    # Each WavecalFeature contains the following elements:
    # - line_ok: bool, indicates whether the line has been properly
    #   identified
    # - category: 'A','B','C','D','E',..., 'X'. Note that 'X' indicates
    #   that the line is still undefined.
    # - id: index of the line in the master table
    # - funcost: cost function associated the the line identification

    # initialize list_of_wvfeatures
    list_of_wvfeatures = []
    for i in range(nlines_arc):
        tmp_feature = WavecalFeature(line_ok=False,
                                     category='X',
                                     lineid=-1,
                                     funcost=np.inf,
                                     xpos=xpos_arc[i],
                                     ypos=0.0,
                                     peak=0.0,
                                     fwhm=0.0,
                                     reference=0.0)
        list_of_wvfeatures.append(tmp_feature)

    # Category A lines
    for i in range(2, nlines_arc - 2):
        j1, j2, j3 = diagonal_ids[i]
        if j1 == j2 == j3 and j1 is not None:
            list_of_wvfeatures[i].line_ok = True
            list_of_wvfeatures[i].category = 'A'
            list_of_wvfeatures[i].lineid = j1
            list_of_wvfeatures[i].funcost = min(diagonal_funcost[i])
            list_of_wvfeatures[i].reference = wv_master[j1]
    
    if debugplot >= 10:
        print('\n* Including category A lines:')
        for i in range(nlines_arc):
            print(i, list_of_wvfeatures[i])
        pause_debugplot(debugplot)

    # Category B lines
    for i in range(2, nlines_arc - 2):
        if not list_of_wvfeatures[i].line_ok:
            j1, j2, j3 = diagonal_ids[i]
            f1, f2, f3 = diagonal_funcost[i]
            if j1 == j2 and j1 is not None:
                if max(f1, f2) < f3:
                    list_of_wvfeatures[i].line_ok = True
                    list_of_wvfeatures[i].category = 'B'
                    list_of_wvfeatures[i].lineid = j1
                    list_of_wvfeatures[i].funcost = min(f1, f2)
                    list_of_wvfeatures[i].reference = wv_master[j1]
            elif j1 == j3 and j1 is not None:
                if max(f1, f3) < f2:
                    list_of_wvfeatures[i].line_ok = True
                    list_of_wvfeatures[i].category = 'B'
                    list_of_wvfeatures[i].lineid = j1
                    list_of_wvfeatures[i].funcost = min(f1, f3)
                    list_of_wvfeatures[i].reference = wv_master[j1]
            elif j2 == j3 and j2 is not None:
                if max(f2, f3) < f1:
                    list_of_wvfeatures[i].line_ok = True
                    list_of_wvfeatures[i].category = 'B'
                    list_of_wvfeatures[i].lineid = j2
                    list_of_wvfeatures[i].funcost = min(f2, f3)
                    list_of_wvfeatures[i].reference = wv_master[j2]

    if debugplot >= 10:
        print('\n* Including category B lines:')
        for i in range(nlines_arc):
            print(i, list_of_wvfeatures[i])
        pause_debugplot(debugplot)

    # Category C lines
    for i in range(2, nlines_arc - 2):
        if not list_of_wvfeatures[i].line_ok:
            j1, j2, j3 = diagonal_ids[i]
            f1, f2, f3 = diagonal_funcost[i]
            if list_of_wvfeatures[i-1].category == 'B':
                if min(f2, f3) > f1:
                    list_of_wvfeatures[i].line_ok = True
                    list_of_wvfeatures[i].category = 'C'
                    list_of_wvfeatures[i].lineid = j1
                    list_of_wvfeatures[i].funcost = f1
                    list_of_wvfeatures[i].reference = wv_master[j1]
            elif list_of_wvfeatures[i+1].category == 'B':
                if min(f1, f2) > f3:
                    list_of_wvfeatures[i].line_ok = True
                    list_of_wvfeatures[i].category = 'C'
                    list_of_wvfeatures[i].lineid = j3
                    list_of_wvfeatures[i].funcost = f3
                    list_of_wvfeatures[i].reference = wv_master[j3]

    if debugplot >= 10:
        print('\n* Including category C lines:')
        for i in range(nlines_arc):
            print(i, list_of_wvfeatures[i])
        pause_debugplot(debugplot)

    # Category D lines
    for i in [1, nlines_arc - 2]:
        j1, j2 = diagonal_ids[i]
        if j1 == j2 and j1 is not None:
            f1, f2 = diagonal_funcost[i]
            list_of_wvfeatures[i].line_ok = True
            list_of_wvfeatures[i].category = 'D'
            list_of_wvfeatures[i].lineid = j1
            list_of_wvfeatures[i].funcost = min(f1, f2)
            list_of_wvfeatures[i].reference = wv_master[j1]

    if debugplot >= 10:
        print('\n* Including category D lines:')
        for i in range(nlines_arc):
            print(i, list_of_wvfeatures[i])
        pause_debugplot(debugplot)

    # Category E lines
    i = 0
    if list_of_wvfeatures[i+1].line_ok and list_of_wvfeatures[i+2].line_ok:
        j1 = diagonal_ids[i][0]
        if j1 is not None:
            list_of_wvfeatures[i].line_ok = True
            list_of_wvfeatures[i].category = 'E'
            list_of_wvfeatures[i].lineid = diagonal_ids[i][0]
            list_of_wvfeatures[i].funcost = diagonal_funcost[i][0]
            list_of_wvfeatures[i].reference = wv_master[j1]
    i = nlines_arc-1
    if list_of_wvfeatures[i-2].line_ok and list_of_wvfeatures[i-1].line_ok:
        j1 = diagonal_ids[i][0]
        if j1 is not None:
            list_of_wvfeatures[i].line_ok = True
            list_of_wvfeatures[i].category = 'E'
            list_of_wvfeatures[i].lineid = diagonal_ids[i][0]
            list_of_wvfeatures[i].funcost = diagonal_funcost[i][0]
            list_of_wvfeatures[i].reference = wv_master[j1]

    if debugplot >= 10:
        print('\n* Including category E lines:')
        for i in range(nlines_arc):
            print(i, list_of_wvfeatures[i])
        pause_debugplot(debugplot)
        fit_list_of_wvfeatures(list_of_wvfeatures, naxis1_arc, crpix1,
                               poly_degree_wfit,
                               weighted=False, debugplot=debugplot)

    # ---
    # Check that the solutions do not contain duplicated values. If
    # they are present (probably due to the influence of an unknown
    # line that unfortunately falls too close to a real line in the
    # master table), we keep the solution with the lowest cost
    # function. The removed lines are labelled as category='R'. The
    # procedure is repeated several times in case a line appears more
    # than twice.
    lduplicated = True
    nduplicated = 0
    while lduplicated:
        lduplicated = False
        for i1 in range(nlines_arc):
            if list_of_wvfeatures[i1].line_ok:
                j1 = list_of_wvfeatures[i1].lineid
                for i2 in range(i1+1, nlines_arc):
                    if list_of_wvfeatures[i2].line_ok:
                        j2 = list_of_wvfeatures[i2].lineid
                        if j1 == j2:
                            lduplicated = True
                            nduplicated += 1
                            f1 = list_of_wvfeatures[i1].funcost
                            f2 = list_of_wvfeatures[i2].funcost
                            if f1 < f2:
                                list_of_wvfeatures[i2].line_ok = False
                                list_of_wvfeatures[i2].category = 'R'
                                # do not uncomment the next line:
                                # list_of_wvfeatures[i2].reference = None
                            else:
                                list_of_wvfeatures[i1].line_ok = False
                                list_of_wvfeatures[i1].category = 'R'
                                # do not uncomment the next line:
                                # list_of_wvfeatures[i1].reference = None

    if debugplot >= 10:
        if nduplicated > 0:
            print('\n* Removing category R lines:')
            for i in range(nlines_arc):
                print(i, list_of_wvfeatures[i])
            fit_list_of_wvfeatures(list_of_wvfeatures, naxis1_arc, crpix1,
                                   poly_degree_wfit,
                                   weighted=False, debugplot=debugplot)
        else:
            print('\n* No duplicated category R lines have been found')

    # ---
    # Filter out points with a large deviation from a robust linear
    # fit. The filtered lines are labelled as category='T'.
    if debugplot >= 10:
        print('\n>>> Theil-Sen filtering...')
    nfit, ifit, xfit, yfit, wfit = select_data_for_fit(list_of_wvfeatures)
    if nfit < 5:
        nremoved = 0
        if debugplot >= 10:
            print("nfit=", nfit)
            print("=> Skipping Theil-Sen filtering!")
    else:
        intercept, slope = fit_theil_sen(xfit, yfit)
        if debugplot >= 10:
            cdelt1_approx = slope
            crval1_approx = intercept + slope * crpix1
            print('>>> Theil-Sen CRVAL1: ', crval1_approx)
            print('>>> Theil-Sen CDELT1: ', cdelt1_approx)
        rfit = yfit - (intercept + slope*xfit)
        if debugplot >= 10:
            print('rfit:\n', rfit)
        sigma_rfit = robust_std(rfit)
        if debugplot >= 10:
            print('robust std:', sigma_rfit)
            print('normal std:', np.std(rfit))
        nremoved = 0
        for i in range(nfit):
            if abs(rfit[i]) > times_sigma_theil_sen * sigma_rfit:
                list_of_wvfeatures[ifit[i]].line_ok = False
                list_of_wvfeatures[ifit[i]].category = 'T'
                # do not uncomment the next line:
                # list_of_wvfeatures[ifit[i]].reference = None
                nremoved += 1
    
    if debugplot >= 10:
        if nremoved > 0:
            print('\n* Removing category T lines:')
            for i in range(nlines_arc):
                print(i, list_of_wvfeatures[i])
            fit_list_of_wvfeatures(list_of_wvfeatures, naxis1_arc, crpix1,
                                   poly_degree_wfit,
                                   weighted=False, debugplot=debugplot)
        else:
            print('\nNo category T lines have been found and removed')

    # ---
    # Filter out points that deviates from a polynomial fit. The
    # filtered lines are labelled as category='P'.
    if times_sigma_polfilt > 0:
        if debugplot >= 10:
            print('\n>>> Polynomial filtering...')
        nfit, ifit, xfit, yfit, wfit = select_data_for_fit(list_of_wvfeatures)
        if nfit <= poly_degree_wfit:
            print("nfit=", nfit)
            raise ValueError("Insufficient number of points for fit.")
        # Note: do not use weighted fit because the weights can be very
        # different and the fit is, in practice, forced to pass through
        # some points while ignoring other points. Sometimes this leads to
        # the rejection of valid points (especially at the borders).
        poly = Polynomial.fit(x=xfit, y=yfit, deg=poly_degree_wfit)
        poly = Polynomial.cast(poly)
        rfit = yfit - poly(xfit)
        if debugplot >= 10:
            print('rfit:', rfit)
        sigma_rfit = robust_std(rfit)
        if debugplot >= 10:
            print('robust std:', sigma_rfit)
            print('normal std:', np.std(rfit))
        nremoved = 0
        for i in range(nfit):
            if abs(rfit[i]) > times_sigma_polfilt * sigma_rfit:
                list_of_wvfeatures[ifit[i]].line_ok = False
                list_of_wvfeatures[ifit[i]].category = 'P'
                # do not uncomment the next line:
                # list_of_wvfeatures[ifit[i]].reference = None
                nremoved += 1

        if debugplot >= 10:
            if nremoved > 0:
                print('\n* Removing category P lines:')
                for i in range(nlines_arc):
                    print(i, list_of_wvfeatures[i])
                fit_list_of_wvfeatures(list_of_wvfeatures, naxis1_arc, crpix1,
                                       poly_degree_wfit,
                                       weighted=False, debugplot=debugplot)
            else:
                print('\nNo category P lines have been found and removed')
    else:
        if debugplot >= 10:
            print('\n=> Skipping polynomial filtering!')

    # ---
    # Remove outliers using the Cook distance. The filtered lines are
    # labelled as category='K'.
    if times_sigma_cook > 0:
        if debugplot >= 10:
            print('\n>>> Removing outliers using Cook distance...')
        nfit, ifit, xfit, yfit, wfit = select_data_for_fit(list_of_wvfeatures)
        # There must be enough points to compute reasonable Cook distances
        if nfit <= poly_degree_wfit + 3:
            nremoved = 0
            print("nfit=", nfit)
            print("=> Skipping outliers detection using Cook distance!")
        else:
            poly, yres, reject = polfit_residuals_with_cook_rejection(
                x=xfit, y=yfit, deg=poly_degree_wfit,
                times_sigma_cook=times_sigma_cook,
                debugplot=debugplot)
            nremoved = 0
            for i in range(nfit):
                if abs(reject[i]):
                    list_of_wvfeatures[ifit[i]].line_ok = False
                    list_of_wvfeatures[ifit[i]].category = 'K'
                    # do not uncomment the next line:
                    # list_of_wvfeatures[ifit[i]].reference = None
                    nremoved += 1

        if debugplot >= 10:
            if nremoved > 0:
                print('\n* Removing category K lines:')
                for i in range(nlines_arc):
                    print(i, list_of_wvfeatures[i])
                fit_list_of_wvfeatures(list_of_wvfeatures, naxis1_arc, crpix1,
                                       poly_degree_wfit,
                                       weighted=False, debugplot=debugplot)
            else:
                print('\nNo category K lines have been found and removed')
    else:
        if debugplot >= 10:
            print('\n=> Skipping outlier detection using Cook distance!')

    # ---
    # If all the arc lines have been identified, compute the final
    # fit and exit
    line_ok = np.array([wvfeature.line_ok for wvfeature in list_of_wvfeatures])
    if np.all(line_ok):
        return list_of_wvfeatures

    # ---
    # Include unidentified lines by using the prediction of the
    # polynomial fit to the current set of identified lines. The
    # included lines are labelled as category='I'.
    loop_include_new_lines = True
    new_lines_included = False
    while loop_include_new_lines:
        if debugplot >= 10:
            print('\n>>> Polynomial prediction of unknown lines...')
        nfit, ifit, xfit, yfit, wfit = select_data_for_fit(list_of_wvfeatures)
        if nfit <= poly_degree_wfit:
            raise ValueError("Insufficient number of points for fit.")
        poly = Polynomial.fit(x=xfit, y=yfit, deg=poly_degree_wfit)
        poly = Polynomial.cast(poly)
        rfit = yfit - poly(xfit)
        if debugplot >= 10:
            print('rfit:\n', rfit)
        sigma_rfit = robust_std(rfit)
        if debugplot >= 10:
            print('robust std:', sigma_rfit)
            print('normal std:', np.std(rfit))

        intercept, slope = fit_theil_sen(xfit, yfit)
        if debugplot >= 10:
            print('crval1, cdelt1 (linear fit):', intercept, slope)

        list_id_already_found = []
        list_funcost_already_found = []
        for i in range(nlines_arc):
            if list_of_wvfeatures[i].line_ok:
                list_id_already_found.append(list_of_wvfeatures[i].lineid)
                list_funcost_already_found.append(
                    list_of_wvfeatures[i].funcost)

        nnewlines = 0
        for i in range(nlines_arc):
            if not list_of_wvfeatures[i].line_ok:
                zfit = poly(xpos_arc[i])  # predicted wavelength
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
                if debugplot >= 10:
                    print(i, ifound, wv_master[ifound], zfit, dlambda)
                if ifound not in list_id_already_found:  # unused line
                    condition1 = dlambda < times_sigma_inclusion * sigma_rfit
                    condition2 = dlambda/slope < error_xpos_arc
                    if condition1 or condition2:
                        list_id_already_found.append(ifound)
                        list_of_wvfeatures[i].line_ok = True
                        list_of_wvfeatures[i].category = 'I'
                        list_of_wvfeatures[i].lineid = ifound
                        # assign the worse cost function value
                        list_of_wvfeatures[i].funcost = max(
                            list_funcost_already_found
                        )
                        list_of_wvfeatures[i].reference = wv_master[ifound]
                        nnewlines += 1

        if debugplot >= 10:
            if nnewlines > 0:
                new_lines_included = True
                print('\n* Including category I lines:')
                for i in range(nlines_arc):
                    print(i, list_of_wvfeatures[i])
                fit_list_of_wvfeatures(list_of_wvfeatures, naxis1_arc, crpix1,
                                       poly_degree_wfit,
                                       weighted=False, debugplot=debugplot)
            else:
                if new_lines_included:
                    print("\nNo additional category I lines have been found " +
                          "and added")
                else:
                    print('\nNo category I lines have been found and added')

        if nnewlines == 0:
            loop_include_new_lines = False

    return list_of_wvfeatures
