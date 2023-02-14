#
# Copyright 2015-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

import pytest
import numpy as np
from numpy.polynomial import polynomial

from ..arccalibration import gen_triplets_master
from ..arccalibration import arccalibration_direct
from ..arccalibration import fit_list_of_wvfeatures

try:
    import matplotlib

    HAVE_PLOTS = True
except ImportError:
    HAVE_PLOTS = False


# -----------------------------------------------------------------------------


def simulate_master_table(my_seed, wv_ini_master, wv_end_master, nlines_master,
                          ldebug=False):
    """Generates a simulated master table of wavelengths.

    The location of the lines follows a random uniform distribution
    between `wv_ini_master` and `wv_end_master`. The total number of generated
    lines is `nlines_master`. The seed for random number generation can be
    re-initialized with `my_seed`.

    Parameters
    ----------
    my_seed : int
        Seed to re-initialize random number generation.
    wv_ini_master : float
        Minimum wavelength in master table.
    wv_end_master : float
        Maximum wavelength in master table.
    nlines_master : int
        Total number of lines in master table.
    ldebug : bool
        If True intermediate results are displayed.

    Returns
    -------
    wv_master : numpy array
        Array with wavelengths corresponding to the master table (Angstroms).
    """

    if my_seed is not None:
        np.random.seed(my_seed)

    if wv_end_master < wv_ini_master:
        raise ValueError('wv_ini_master=' + str(wv_ini_master) +
                         ' must be <= wv_end_master=' + str(wv_end_master))

    wv_master = np.random.uniform(low=wv_ini_master,
                                  high=wv_end_master,
                                  size=nlines_master)
    wv_master.sort()  # in-place sort

    if ldebug:
        print('>>> Master table:')
        for val in zip(range(nlines_master), wv_master):
            print(val)

    return wv_master


# -----------------------------------------------------------------------------


def simulate_arc(wv_ini_master, wv_end_master, wv_master,
                 wv_ini_arc, wv_end_arc, naxis1_arc,
                 prob_line_master_in_arc,
                 delta_xpos_min_arc, delta_lambda, error_xpos_arc,
                 poly_degree, fraction_unknown_lines,
                 ldebug=False, lplot=False):
    """Generates simulated input for arc calibration.

    Parameters
    ----------
    wv_ini_master : float
        Minimum wavelength in master table.
    wv_end_master : float
        Maximum wavelength in master table.
    wv_master : numpy array
        Array with wavelengths corresponding to the master table (Angstroms).
    wv_ini_arc : float
        Minimum wavelength in arc spectrum.
    wv_end_arc : float
        Maximum wavelength in arc spectrum.
    naxis1_arc : int
        NAXIS1 of arc spectrum.
    prob_line_master_in_arc : float
        Probability that a master table line appears in the arc spectrum.
    delta_xpos_min_arc : float
        Minimum distance (pixels) between lines in arc spectrum.
    delta_lambda : float
        Maximum deviation (Angstroms) from linearity in arc calibration
        polynomial.
    error_xpos_arc : float
        Standard deviation (pixels) employed to introduce noise in the arc
        spectrum lines. The initial lines are shifted from their original
        location following a Normal distribution with mean iqual to zero and
        sigma equal to this parameter.
    poly_degree : int
        Polynomial degree corresponding to the original wavelength calibration
        function.
    fraction_unknown_lines : float
        Fraction of lines that on average will be unknown (i.e., lines that
        appear in the arc spectrum that are not present in the master table).
    ldebug : bool
        If True intermediate results are displayed.
    lplot : bool
        If True intermediate plots are displayed.

    Returns
    -------
    nlines_arc : int
        Number of arc lines
    xpos_arc : numpy array
        Location of arc lines (pixels).
    crval1_arc : float
        CRVAL1 for arc spectrum (linear approximation).
    cdelt1_arc : float
        CDELT1 for arc spectrum (linear approximation).
    c0_arc, c1_arc, c2_arc : floats
        Coefficients of the second order polynomial.
    ipos_wv_arc : numpy array
        Number of line in master table corresponding to each arc line. Unknown
        lines (i.e. those that are not present in the master table) are
        assigned to -1.
    coeff_original : numpy array
        Polynomial coefficients ordered from low to high, corresponding to the
        fit to the arc lines before the inclusion of unknown lines.

    """

    if (wv_ini_arc < wv_ini_master) or (wv_ini_arc > wv_end_master):
        print('wv_ini_master:', wv_ini_master)
        print('wv_end_master:', wv_end_master)
        print('wv_ini_arc...:', wv_ini_arc)
        raise ValueError('wavelength_ini_arc outside valid range')

    if (wv_end_arc < wv_ini_master) or (wv_end_arc > wv_end_master):
        print('wv_ini_master:', wv_ini_master)
        print('wv_end_master:', wv_end_master)
        print('wv_end_arc...:', wv_end_arc)
        raise ValueError('wavelength_ini_arc outside valid range')

    if wv_end_arc < wv_ini_arc:
        raise ValueError('wv_ini_arc=' + str(wv_ini_arc) +
                         ' must be <= wv_end_arc=' + str(wv_end_arc))

    # ---

    nlines_master = wv_master.size

    crval1_arc = wv_ini_arc
    cdelt1_arc = (wv_end_arc - wv_ini_arc) / float(naxis1_arc - 1)
    crpix1_arc = 1.0
    if ldebug:
        print('>>> CRVAL1, CDELT1, CRPIX1....:', crval1_arc, cdelt1_arc,
              crpix1_arc)

    # ---
    # The arc lines constitute a subset of the master lines in the considered
    # wavelength range.
    i1_master = np.searchsorted(wv_master, wv_ini_arc)
    i2_master = np.searchsorted(wv_master, wv_end_arc)
    nlines_temp = i2_master - i1_master
    nlines_arc_ini = int(round(nlines_temp * prob_line_master_in_arc))
    ipos_wv_arc_ini = np.random.choice(range(i1_master, i2_master),
                                       size=nlines_arc_ini,
                                       replace=False)
    ipos_wv_arc_ini.sort()  # in-place sort
    wv_arc_ini = wv_master[ipos_wv_arc_ini]
    if ldebug:
        print('>>> Number of master lines in arc region.:', nlines_temp)
        print('>>> Initial number of arc lines..........:', nlines_arc_ini)
        print('>>> Initial selection of master list lines for arc:')
        print(ipos_wv_arc_ini)
    # Remove too close lines.
    ipos_wv_arc = np.copy(ipos_wv_arc_ini[0:1])
    wv_arc = np.copy(wv_arc_ini[0:1])
    i_last = 0
    for i in range(1, nlines_arc_ini):
        delta_xpos = (wv_arc_ini[i] - wv_arc_ini[i_last]) / cdelt1_arc
        if delta_xpos > delta_xpos_min_arc:
            ipos_wv_arc = np.append(ipos_wv_arc, ipos_wv_arc_ini[i])
            wv_arc = np.append(wv_arc, wv_arc_ini[i])
            i_last = i
        else:
            if ldebug:
                print('--> skipping line #', i, '. Too close to line #',
                      i_last)
    nlines_arc = len(wv_arc)
    if ldebug:
        print('>>> Intermediate number of arc lines.....:', nlines_arc)
        print('>>> Intermediate selection of master list lines for arc:')
        print(ipos_wv_arc)

    # Generate pixel location of the arc lines.
    if delta_lambda == 0.0:
        # linear solution
        xpos_arc = (wv_arc - crval1_arc) / cdelt1_arc + 1.0
        c0_arc = wv_ini_arc
        c1_arc = cdelt1_arc
        c2_arc = 0.0
    else:
        # polynomial solution
        c0_arc = wv_ini_arc
        c1_arc = (wv_end_arc - wv_ini_arc - 4 * delta_lambda) / float(
            naxis1_arc - 1)
        c2_arc = 4 * delta_lambda / float(naxis1_arc - 1) ** 2
        xpos_arc = (
        -c1_arc + np.sqrt(c1_arc ** 2 - 4 * c2_arc * (c0_arc - wv_arc)))
        xpos_arc /= 2 * c2_arc
        xpos_arc += 1  # convert from 0,...,(NAXIS1-1) to 1,...,NAXIS1

    # Introduce noise in arc line positions.
    if error_xpos_arc > 0:
        xpos_arc += np.random.normal(loc=0.0,
                                     scale=error_xpos_arc,
                                     size=nlines_arc)
    # Check that the order of the lines has not been modified.
    xpos_arc_copy = np.copy(xpos_arc)
    xpos_arc_copy.sort()  # in-place sort
    if sum(xpos_arc == xpos_arc_copy) != len(xpos_arc):
        raise ValueError(
            'FATAL ERROR: arc line switch after introducing noise')

    if lplot and HAVE_PLOTS:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([1, naxis1_arc])
        ax.set_ylim([wv_ini_arc, wv_end_arc])
        ax.plot(xpos_arc, wv_arc, 'ro')
        xp = np.array([1, naxis1_arc])
        yp = np.array([wv_ini_arc, wv_end_arc])
        ax.plot(xp, yp, 'b-')
        xp = np.arange(1, naxis1_arc + 1)
        yp = c0_arc + c1_arc * (xp - 1) + c2_arc * (xp - 1) ** 2
        ax.plot(xp, yp, 'g:')
        ax.set_xlabel('pixel position in arc spectrum')
        ax.set_ylabel('wavelength (Angstrom)')
        plt.show(block=False)

    # Unweighted polynomial fit.
    coeff_original = polynomial.polyfit(xpos_arc, wv_arc, poly_degree)
    poly_original = polynomial.Polynomial(coeff_original)

    if lplot and HAVE_PLOTS:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim([1, naxis1_arc])
        if delta_lambda == 0.0:
            if error_xpos_arc > 0:
                ax.set_ylim([-4 * error_xpos_arc * cdelt1_arc,
                             4 * error_xpos_arc * cdelt1_arc])
            else:
                ax.set_ylim([-1.1, 1.1])
        else:
            ax.set_ylim([-delta_lambda * 1.5, delta_lambda * 1.5])
        yp = wv_arc - (crval1_arc + (xpos_arc - 1) * cdelt1_arc)
        ax.plot(xpos_arc, yp, 'ro')
        xp = np.array([1, naxis1_arc])
        yp = np.array([0, 0])
        ax.plot(xp, yp, 'b-')
        xp = np.arange(1, naxis1_arc + 1)
        yp = c0_arc + c1_arc * (xp - 1) + c2_arc * (xp - 1) ** 2
        yp -= crval1_arc + cdelt1_arc * (xp - 1)
        ax.plot(xp, yp, 'g:')
        yp = poly_original(xp)
        yp -= crval1_arc + cdelt1_arc * (xp - 1)
        ax.plot(xp, yp, 'm-')
        ax.set_xlabel('pixel position in arc spectrum')
        ax.set_ylabel('residuals (Angstrom)')
        plt.show(block=False)

    # ---
    # Include unknown lines (lines that do not appear in the master table).
    nunknown_lines = int(round(fraction_unknown_lines * float(nlines_arc)))
    if ldebug:
        print('>>> Number of unknown arc lines..........:', nunknown_lines)
    for i in range(nunknown_lines):
        iiter = 0
        while True:
            iiter += 1
            if iiter > 1000:
                raise ValueError('iiter > 1000 while adding unknown lines')
            xpos_dum = np.random.uniform(low=1.0,
                                         high=float(naxis1_arc),
                                         size=1)
            isort = np.searchsorted(xpos_arc, xpos_dum)
            newlineok = False
            if isort == 0:
                dxpos1 = abs(xpos_arc[isort] - xpos_dum)
                if dxpos1 > delta_xpos_min_arc:
                    newlineok = True
            elif isort == nlines_arc:
                dxpos2 = abs(xpos_arc[isort - 1] - xpos_dum)
                if dxpos2 > delta_xpos_min_arc:
                    newlineok = True
            else:
                dxpos1 = abs(xpos_arc[isort] - xpos_dum)
                dxpos2 = abs(xpos_arc[isort - 1] - xpos_dum)
                if (dxpos1 > delta_xpos_min_arc) and \
                        (dxpos2 > delta_xpos_min_arc):
                    newlineok = True
            if newlineok:
                xpos_arc = np.insert(xpos_arc, isort, xpos_dum)
                ipos_wv_arc = np.insert(ipos_wv_arc, isort, -1)
                nlines_arc += 1
                if ldebug:
                    print('--> adding unknown line at pixel:', xpos_dum)
                break
    if ldebug:
        print('>>> Final number of arc lines............:', nlines_arc)
        for val in zip(range(nlines_arc), ipos_wv_arc, xpos_arc):
            print(val)

    if lplot and HAVE_PLOTS:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim([0.0, 3.0])
        ax.vlines(wv_master, 0.0, 1.0)
        ax.vlines(wv_arc, 1.0, 2.0, colors='r', linestyle=':')
        ax.vlines(wv_ini_arc, 0.0, 3.0, colors='m', linewidth=3.0)
        ax.vlines(wv_end_arc, 0.0, 3.0, colors='m', linewidth=3.0)
        ax.set_xlabel('wavelength')
        axbis = ax.twiny()
        axbis.vlines(xpos_arc, 2.0, 3.0, colors='g')
        xmin_xpos_master = (wv_ini_master - crval1_arc) / cdelt1_arc + 1.0
        xmax_xpos_master = (wv_end_master - crval1_arc) / cdelt1_arc + 1.0
        axbis.set_xlim([xmin_xpos_master, xmax_xpos_master])
        axbis.set_xlabel('pixel position in arc spectrum')
        plt.show(block=False)

    return nlines_arc, xpos_arc, crval1_arc, cdelt1_arc, \
           c0_arc, c1_arc, c2_arc, \
           ipos_wv_arc, coeff_original


# -----------------------------------------------------------------------------

def execute_arccalibration(my_seed=432, wv_ini_master=3000, wv_end_master=7000,
                           nlines_master=120,
                           wv_ini_arc=4000, wv_end_arc=5000,
                           naxis1_arc=1024, crpix1=1.0,
                           prob_line_master_in_arc=0.80,
                           delta_xpos_min_arc=4.0,
                           delta_lambda=5.0, error_xpos_arc=0.3,
                           poly_degree=2, fraction_unknown_lines=0.20,
                           wv_ini_search=None, wv_end_search=None,
                           times_sigma_r=3.0, frac_triplets_for_sum=0.50,
                           times_sigma_theil_sen=10.0, poly_degree_wfit=2,
                           times_sigma_polfilt=10.0,
                           times_sigma_cook=10.0,
                           times_sigma_inclusion=5.0,
                           ldebug=False, lplot=False):
    """Execute a particular arc calibration simulation.

    This function simulates a master list, generates a simulated arc, and
    carry out its wavelength calibration.

    Parameters
    ----------
    my_seed : int
        Seed to re-initialize random number generation.
    wv_ini_master : float
        Minimum wavelength in master table.
    wv_end_master : float
        Maximum wavelength in master table.
    nlines_master : int
        Total number of lines in master table.
    ldebug : bool
        If True intermediate results are displayed.

    Returns
    -------
    coeff : numpy array
        Coefficients of the polynomial fit.
    crval1_approx : float
        Approximate CRVAL1 value.
    cdetl1_approx : float
        Approximate CDELT1 value.
    """

    wv_master = simulate_master_table(my_seed, wv_ini_master, wv_end_master,
                                      nlines_master,
                                      ldebug=ldebug)
    ntriplets_master, ratios_master_sorted, triplets_master_sorted_list = \
        gen_triplets_master(wv_master)

    nlines_arc, xpos_arc, crval1_arc, cdelt1_arc, \
    c0_arc, c1_arc, c2_arc, ipos_wv_arc, coeff_original = \
        simulate_arc(wv_ini_master, wv_end_master, wv_master,
                     wv_ini_arc, wv_end_arc, naxis1_arc,
                     prob_line_master_in_arc,
                     delta_xpos_min_arc, delta_lambda, error_xpos_arc,
                     poly_degree, fraction_unknown_lines,
                     ldebug=ldebug, lplot=lplot)

    if wv_ini_search is None:
        wv_ini_search = wv_ini_master - 0.1 * (wv_end_master - wv_ini_master)
    if wv_end_search is None:
        wv_end_search = wv_end_master + 0.1 * (wv_end_master - wv_ini_master)

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
        wvmin_useful=None,
        wvmax_useful=None,
        error_xpos_arc=error_xpos_arc,
        times_sigma_r=times_sigma_r,
        frac_triplets_for_sum=frac_triplets_for_sum,
        times_sigma_theil_sen=times_sigma_theil_sen,
        poly_degree_wfit=poly_degree_wfit,
        times_sigma_polfilt=times_sigma_polfilt,
        times_sigma_cook=times_sigma_cook,
        times_sigma_inclusion=times_sigma_inclusion)

    solution_wv = \
        fit_list_of_wvfeatures(
            list_of_wvfeatures=list_of_wvfeatures,
            naxis1_arc=naxis1_arc,
            crpix1=crpix1,
            poly_degree_wfit=poly_degree_wfit
        )

    return solution_wv


# -----------------------------------------------------------------------------


def test__execute_notebook_example(ldebug=False, lplot=False):
    """Test the explanation of the ipython notebook example."""
    solution_wv = execute_arccalibration(ldebug=ldebug, lplot=lplot)

    coeff_expected = np.array([3.99875794e+03, 9.59950578e-01, 1.72739867e-05])
    assert np.allclose(solution_wv.coeff, coeff_expected)
    assert np.allclose(solution_wv.cr_linear.crval, 3999.7179085283897)  # 3996.42717772)
    assert np.allclose(solution_wv.cr_linear.crmin, 3999.7179085283897)
    assert np.allclose(solution_wv.cr_linear.crmax, 4999.8604201544294)
    assert np.allclose(solution_wv.cr_linear.cdelt, 0.97765641410170068)  # 0.978303317095)

    print("TEST: test__execute_notebook_example... OK")


# @pytest.mark.xfail
def test__execute_simple_case(ldebug=False, lplot=False):
    """Test the explanation of the ipython notebook example."""
    solution_wv = execute_arccalibration(nlines_master=15,
                                         error_xpos_arc=0.3,
                                         wv_ini_arc=3000, wv_end_arc=7000,
                                         prob_line_master_in_arc=1.0,
                                         fraction_unknown_lines=0.0,
                                         frac_triplets_for_sum=0.5,
                                         ldebug=ldebug, lplot=lplot)

    coeff_expected = np.array([2.99467778e+03, 3.89781863e+00, 1.22960881e-05])
    assert np.allclose(solution_wv.coeff, coeff_expected)
    assert np.allclose(solution_wv.cr_linear.crval, 2998.5756138701254)  # 2995.4384155)
    assert np.allclose(solution_wv.cr_linear.crmin, 2998.5756138701254)
    assert np.allclose(solution_wv.cr_linear.crmax, 6998.9374406492443)
    assert np.allclose(solution_wv.cr_linear.cdelt, 3.9104221180636549)  # 3.91231531392)

    print("TEST: test__execute_simple_case... OK")


# -----------------------------------------------------------------------------


if __name__ == '__main__':
    # test__execute_notebook_example(ldebug=True, lplot=True)
    test__execute_simple_case(ldebug=True, lplot=False)
