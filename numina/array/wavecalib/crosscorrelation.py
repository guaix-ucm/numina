#
# Copyright 2015-2016 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from __future__ import division
from __future__ import print_function

from mpl_toolkits.axes_grid.inset_locator import inset_axes
import numpy as np
from numpy.polynomial import Polynomial

from numina.array.display.ximplotxy import ximplotxy
from numina.modeling.gaussbox import gauss_box_model


def filtmask(sp, fmin=0.02, fmax=0.15, debugplot=0):
    """Filter spectrum in Fourier space and apply cosine bell.

    Parameters
    ----------
    sp : numpy array
        Spectrum to be filtered and masked.
    fmin : float
        Minimum frequency to be employed.
    fmax : float
        Maximum frequency to be employed.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    sp_filtmask : numpy array
        Filtered and masked spectrum

    """

    # Fourier filtering
    xf = np.fft.fftfreq(sp.size)
    yf = np.fft.fft(sp)
    if abs(debugplot) in (21, 22):
        ximplotxy(xf, yf.real, xlim=(0, 0.51),
                  plottype='semilog', debugplot=debugplot)

    cut = (np.abs(xf) > fmax)
    yf[cut] = 0.0
    cut = (np.abs(xf) < fmin)
    yf[cut] = 0.0
    if abs(debugplot) in (21, 22):
        ximplotxy(xf, yf.real, xlim=(0, 0.51),
                  plottype='semilog', debugplot=debugplot)

    sp_filt = np.fft.ifft(yf).real
    if abs(debugplot) in (21, 22):
        xdum = np.arange(1, sp_filt.size + 1)
        ximplotxy(xdum, sp_filt, title="filtered median spectrum",
                  debugplot=debugplot)

    sp_filtmask = sp_filt * cosinebell(sp_filt.size, 0.1)
    if abs(debugplot) in (21, 22):
        xdum = np.arange(1, sp_filt.size + 1)
        ximplotxy(xdum, sp_filtmask,
                  title="filtered and masked median spectrum",
                  debugplot=debugplot)

    return sp_filtmask


def cosinebell(n, fraction):
    """Return a cosine bell spanning n pixels, masking a fraction of pixels

    Parameters
    ----------
    n : int
        Number of pixels.
    fraction : float
        Length fraction over which the data will be masked.

    """

    mask = np.ones(n)
    nmasked = int(fraction * n)
    for i in range(nmasked):
        yval = 0.5 * (1 - np.cos(np.pi * float(i) / float(nmasked)))
        mask[i] = yval
        mask[n - i - 1] = yval

    return mask


def convolve_comb_lines(lines_wave, lines_flux, sigma,
                        crpix1, crval1, cdelt1, naxis1):
    """Convolve a set of lines of known wavelengths and flux.

    Parameters
    ----------
    lines_wave : array like
        Input array with wavelengths
    lines_flux : array like
        Input array with fluxes
    sigma : float
        Sigma of the broadening gaussian to be applied.
    crpix1 : float
        CRPIX1 of the desired wavelength calibration.
    crval1 : float
        CRVAL1 of the desired wavelength calibration.
    cdelt1 : float
        CDELT1 of the desired wavelength calibration.
    naxis1 : integer
        NAXIS1 of the output spectrum.

    Returns
    -------
    xwave : array like
        Array with wavelengths for the output spectrum.
    spectrum : array like
        Array with the expected fluxes at each pixel.

    """

    # generate wavelengths for output spectrum
    xwave = crval1 + (np.arange(naxis1) + 1 - crpix1) * cdelt1

    # initialize output spectrum
    spectrum = np.zeros(naxis1)

    # convolve each line
    for wave, flux in zip(lines_wave, lines_flux):
        sp_tmp = gauss_box_model(x=xwave, amplitude=flux, mean=wave,
                                 stddev=sigma)
        spectrum += sp_tmp

    return xwave, spectrum


def periodic_corr1d(sp_reference, sp_offset,
                    fminmax=None,
                    debugplot=0):
    """Periodic correlation between two spectra, implemented using FFT.

    Parameters
    ----------
    sp_reference : numpy array
        Reference spectrum.
    sp_offset : numpy array
        Spectrum which offset is going to be measured relative to the
        reference spectrum.
    fminmax : tuple of floats or None
        Minimum and maximum frequencies to be used. If None, no
        frequency filtering is employed.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------
    offset : float
        Offset between the two input spectra.
    fpeak : float
        Maximum of the cross-correlation function.

    """

    # protections
    if sp_reference.ndim != 1 or sp_offset.ndim != 1:
        raise ValueError("Invalid array dimensions")
    if sp_reference.shape != sp_offset.shape:
        raise ValueError("x and y shapes are different")

    naxis1 = len(sp_reference)

    xcorr = np.arange(naxis1)
    naxis1_half = int(naxis1 / 2)
    for i in range(naxis1_half):
        xcorr[i + naxis1_half] -= naxis1
    isort = xcorr.argsort()
    xcorr = xcorr[isort]

    if fminmax is not None:
        fmin, fmax = fminmax
        sp_reference_filtmask = filtmask(sp_reference, fmin=fmin, fmax=fmax)
        sp_offset_filtmask = filtmask(sp_offset, fmin=fmin, fmax=fmax)
        if abs(debugplot) % 10 != 0:
            from numina.array.display.matplotlib_qt import plt
            xdum = np.arange(naxis1) + 1
            # reference spectrum
            ax = ximplotxy(xdum, sp_reference, show=False,
                           title='reference spectrum',
                           label='original spectrum')
            ax.plot(xdum, sp_reference_filtmask,
                    label='filtered and masked spectrum')
            ax.legend()
            plt.show()
            # offset spectrum
            ax = ximplotxy(xdum, sp_offset, show=False,
                           title='offset spectrum',
                           label='original spectrum')
            ax.plot(xdum, sp_offset_filtmask,
                    label='filtered and masked spectrum')
            ax.legend()
            plt.show()
    else:
        sp_reference_filtmask = np.copy(sp_reference)
        sp_offset_filtmask = np.copy(sp_offset)

    if abs(debugplot) % 10 != 0:
        from numina.array.display.matplotlib_qt import plt
        xdum = np.arange(naxis1) + 1
        ax = ximplotxy(xdum, sp_reference_filtmask, show=False,
                       label='reference spectrum')
        ax.plot(xdum, sp_offset_filtmask, label='offset spectrum')
        ax.legend()
        plt.show()

    corr = np.fft.ifft(np.fft.fft(sp_offset_filtmask) * \
                       np.fft.fft(sp_reference_filtmask).conj()).real
    corr = corr[isort]
    ixpeak = corr.argmax()

    # fit correlation peak with 2nd order polynomial
    nfit = 7
    nmed = nfit // 2
    imin = ixpeak - nmed
    imax = ixpeak + nmed
    lpeak_ok = True
    if imin < 0 or imax > len(corr):
        x_refined_peak = 0
        y_refined_peak = 0
        lpeak_ok = False
        poly_peak = Polynomial([0.0])
    else:
        x_fit = np.arange(-nmed, nmed + 1, dtype=np.float)
        y_fit = corr[imin:(imax+1)]
        poly_peak = Polynomial.fit(x_fit, y_fit, 2)
        poly_peak = Polynomial.cast(poly_peak)
        coef = poly_peak.coef
        if coef[2] != 0:
            x_refined_peak = -coef[1] / (2.0 * coef[2])
        else:
            x_refined_peak = 0.0
        y_refined_peak = poly_peak(x_refined_peak)
        x_refined_peak += ixpeak

    offset = x_refined_peak - naxis1_half
    fpeak = y_refined_peak

    if abs(debugplot) % 10 != 0:
        title="periodic correlation (offset={0:6.2f} pixels)".format(offset)
        from numina.array.display.matplotlib_qt import plt
        ax = ximplotxy(xcorr, corr,
                       xlabel='offset (pixels)',
                       ylabel='cross-correlation function',
                       title=title,
                       xlim=(-naxis1/2, naxis1/2), show=False)
        ax.axvline(offset, color='grey', linestyle='dashed')
        # inset plot
        inset_ax = inset_axes(
            ax,
            width="40%",
            height="40%",
            loc=1
        )
        inset_ax.plot(xcorr, corr)
        inset_ax.set_xlim([-50,50])
        if lpeak_ok:
            xplot = np.arange(-nmed, nmed, 0.5)
            yplot = poly_peak(xplot)
            xplot += ixpeak - naxis1_half
            inset_ax.plot(xplot, yplot, '-')
            inset_ax.plot([x_refined_peak - naxis1_half],
                          [y_refined_peak], 'o')
        inset_ax.axvline(offset, color='grey', linestyle='dashed')
        plt.show()

    return offset, fpeak
