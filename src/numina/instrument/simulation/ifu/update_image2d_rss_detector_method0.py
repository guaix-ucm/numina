#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.units import Unit
import numpy as np

from numina.array.distortion import fmap


def update_image2d_rss_detector_method0(
        islice,
        simulated_x_ifu_all,
        simulated_y_ifu_all,
        simulated_wave_all,
        naxis1_ifu,
        bins_x_ifu,
        bins_wave,
        bins_x_detector,
        bins_y_detector,
        wv_cdelt1,
        extra_degradation_spectral_direction,
        dict_ifu2detector,
        image2d_rss_method0,
        image2d_detector_method0
):
    """Update the two 2D images: RSS and detector.

    The function updates the following 2D arrays:
    - image2d_rss_method0,
    - image2d_detector_method0
    with the photons observed through the slice 'islice'.

    Note that both arrays are generated simultaneously in
    order to make use of the same value of
    'extra_degradation_spectral_direction'.

    This function can be executed in parallel.

    Parameters
    ----------
    islice : int
        Slice number.
    simulated_x_ifu_all : `~astropy.units.Quantity`
        Simulated X coordinates of the photons in the IFU.
    simulated_y_ifu_all : `~astropy.units.Quantity`
        Simulated Y coordinates of the photons in the IFU.
    simulated_wave_all : `~astropy.units.Quantity`
        Simulated wavelengths of the photons in the IFU.
    naxis1_ifu : `~astropy.units.Quantity`
        IFU NAXIS1 (along the slice)
    bins_x_ifu : `~numpy.ndarray`
        Bin edges in the naxis1_ifu direction
        (along the slice).
    bins_wave : `~numpy.ndarray`
        Bin edges in the wavelength direction.
    bins_x_detector : `~numpy.ndarray`
        Bin edges in the naxis1_detector direction
        (spectral direction).
    bins_y_detector : `~numpy.ndarray`
        Bin edges in the naxis2_detector direction
        (slices direction).
    wv_cdelt1 : `~astropy.units.Quantity`
        CDELT1 value along the spectral direction.
    extra_degradation_spectral_direction : `~astropy.units.Quantity`
        Additional degradation in the spectral direction, in units of
        the detector pixels, for each simulated photon.
    dict_ifu2detector : dict
        A Python dictionary containing the 2D polynomials that allow
        to transform (X, Y, wavelength) coordinates in the IFU focal
        plane to (X, Y) coordinates in the detector.
    image2d_rss_method0 : `~numpy.ndarray`
        2D array containing the RSS image. This array is
        updated by this function.
    image2d_detector_method0 : `~numpy.ndarray`
        2D array containing the detector image. This array is
        updated by this function.

    """

    # determine photons that pass through the considered slice
    y_ifu_expected = 1.5 + 2 * islice
    condition = np.abs(simulated_y_ifu_all.value - y_ifu_expected) < 1
    iok = np.where(condition)[0]
    nphotons_slice = len(iok)

    if nphotons_slice > 0:
        # -------------------------------------------------
        # 1) spectroscopic 2D image with continguous slices
        # -------------------------------------------------
        h, xedges, yedges = np.histogram2d(
            x=simulated_x_ifu_all.value[iok],
            y=simulated_wave_all.value[iok] +
              (simulated_y_ifu_all.value[iok] - y_ifu_expected) * wv_cdelt1.value +
              extra_degradation_spectral_direction.value[iok] * wv_cdelt1.value,
            bins=(bins_x_ifu.value, bins_wave.value)
        )
        j1 = islice * naxis1_ifu.value
        j2 = j1 + naxis1_ifu.value
        image2d_rss_method0[j1:j2, :] += h

        # -----------------------------------------
        # 2) spectroscopic 2D image in the detector
        # -----------------------------------------
        # use models to predict location in Hawaii detector
        # important: reverse here X <-> Y
        wavelength_unit = Unit(dict_ifu2detector['wavelength-unit'])
        dumdict = dict_ifu2detector['contents'][islice]
        order = dumdict['order']
        aij = np.array(dumdict['aij'])
        bij = np.array(dumdict['bij'])
        y_hawaii, x_hawaii = fmap(
            order=order,
            aij=aij,
            bij=bij,
            x=simulated_x_ifu_all.value[iok],
            # important: use the wavelength unit employed to determine
            # the polynomial transformation
            y=simulated_wave_all.to(wavelength_unit).value[iok]
        )
        # disperse photons along the spectral direction according to their
        # location within the slice in the vertical direction
        x_hawaii += simulated_y_ifu_all[iok].value - y_ifu_expected
        # include additional degradation in spectral resolution
        x_hawaii += extra_degradation_spectral_direction.value[iok]
        # compute 2D histogram
        # important: reverse X <-> Y
        h, xedges, yedges = np.histogram2d(
            x=y_hawaii,
            y=x_hawaii,
            bins=(bins_y_detector, bins_x_detector)
        )
        image2d_detector_method0 += h
