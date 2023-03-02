#
# Copyright 2015-2023 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0+
# License-Filename: LICENSE.txt
#

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter

from numina.array.display.pause_debugplot import pause_debugplot
from numina.array.display.ximplotxy import ximplotxy
from numina.modeling.gaussbox import gauss_box_model


def filtmask(sp, fmin, fmax,
             sp_label='spectrum',
             debugplot=0):
    """Filter spectrum in Fourier space.

    Parameters
    ----------
    sp : numpy array
        Spectrum to be filtered and masked.
    fmin : float
        Minimum frequency to be employed.
    fmax : float
        Maximum frequency to be employed.
    sp_label : str
        Additional string to be attached to labels in legends.
    debugplot : int
        Debugging level for messages and plots. For details see
        'numina.array.display.pause_debugplot.py'.

    Returns
    -------
    sp_filt : numpy array
        Filtered spectrum

    """

    # Fourier filtering
    xf = np.fft.fftfreq(sp.size)
    yf = np.fft.fft(sp)
    if abs(debugplot) in (21, 22):
        iok = np.where(xf >= 0)
        ximplotxy(xf[iok], yf[iok].real,
                  plottype='semilogy',
                  xlabel='frequency', ylabel='power',
                  title=sp_label+' (before masking)', debugplot=debugplot)

    cut = (np.abs(xf) > fmax)
    yf[cut] = 0.0
    cut = (np.abs(xf) < fmin)
    yf[cut] = 0.0
    if abs(debugplot) in (21, 22):
        iok = np.where(xf >= 0)
        ximplotxy(xf[iok], yf[iok].real,
                  plottype='semilogy',
                  xlabel='frequency', ylabel='power',
                  title=sp_label+' (after masking)', debugplot=debugplot)

    sp_filt = np.fft.ifft(yf).real
    if abs(debugplot) in (21, 22):
        from numina.array.display.matplotlib_qt import plt
        xdum = np.arange(1, sp_filt.size + 1)
        ax = ximplotxy(xdum, sp, title="filtering " + sp_label,
                       label='original', show=False)
        ax.plot(xdum, sp_filt, label='filtered')
        ax.legend()
        plt.show()

    return sp_filt


def cosinebell(n, frac_cosbell):
    """Return a cosine bell spanning n pixels, masking a fraction of pixels

    Parameters
    ----------
    n : int
        Number of pixels.
    frac_cosbell : float
        Length fraction over which the data will be masked.

    """

    mask = np.ones(n)
    nmasked = int(frac_cosbell * n)
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
                    remove_mean=False,
                    frac_cosbell=None,
                    zero_padding=None,
                    fminmax=None,
                    naround_zero=None,
                    norm_spectra=False,
                    nfit_peak=7,
                    sp_label='spectrum',
                    plottitle=None,
                    pdf=None,
                    debugplot=0):
    """Periodic correlation between two spectra, implemented using FFT.

    Parameters
    ----------
    sp_reference : numpy array
        Reference spectrum.
    sp_offset : numpy array
        Spectrum which offset is going to be measured relative to the
        reference spectrum.
    remove_mean : bool
        If True, remove mean from input spectra.
    frac_cosbell : float or None
        Fraction of spectrum where the cosine bell falls to zero.
    zero_padding : int or None
        Number of extended pixels set to zero.
    fminmax : tuple of floats or None
        Minimum and maximum frequencies to be used. If None, no
        frequency filtering is employed.
    naround_zero : int
        Half width of the window (around zero offset) to look for
        the correlation peak. If None, the whole correlation
        spectrum is employed. Otherwise, the peak will be sought
        in the interval [-naround_zero, +naround_zero].
    norm_spectra : bool
        If True, the filtered spectra are normalized before computing
        the correlation function. This can be important when comparing
        the peak value of this function using different spectra.
    nfit_peak : int
        Total number of points (must be odd!) around the peak of the
        crosscorrelation function to be employed to estimate the peak
        location (using a fit to a second order polynomial).
    sp_label : str
        Additional string to be attached to labels in legends.
    plottitle : str
        Optional plot title.
    pdf : PdfFile object or None
        If not None, output is sent to PDF file.
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
    if nfit_peak % 2 == 0:
        nfit_peak += 1

    if plottitle is None:
        plottitle = ' '

    # remove mean
    if remove_mean:
        sp_reference_m = sp_reference - np.mean(sp_reference)
        sp_offset_m = sp_offset - np.mean(sp_offset)
    else:
        sp_reference_m = sp_reference.copy()
        sp_offset_m = sp_offset.copy()

    # cosine bell
    if frac_cosbell is not None:
        if frac_cosbell < 0.0 or frac_cosbell > 0.5:
            raise ValueError(f'Invalid frac_cosbell: {frac_cosbell}')
        sp_reference_mb = sp_reference_m * cosinebell(
            sp_reference_m.size, frac_cosbell
        )
        sp_offset_mb = sp_offset_m * cosinebell(
            sp_offset_m.size, frac_cosbell
        )
    else:
        sp_reference_mb = sp_reference_m
        sp_offset_mb = sp_offset_m

    # zero padding
    if zero_padding is not None:
        if zero_padding < 0:
            raise ValueError(f'Invalid zero_padding: {zero_padding}')
        sp_reference_mbz = np.concatenate(
            (sp_reference_mb, np.zeros(zero_padding))
        )
        sp_offset_mbz = np.concatenate(
            (sp_offset_mb, np.zeros(zero_padding))
        )
    else:
        sp_reference_mbz = sp_reference_mb
        sp_offset_mbz = sp_offset_mb

    if fminmax is not None:
        fmin, fmax = fminmax
        sp_reference_filtmask = filtmask(
            sp_reference_mbz,
            fmin=fmin,
            fmax=fmax,
            sp_label='reference ' + sp_label,
            debugplot=debugplot)
        sp_offset_filtmask = filtmask(
            sp_offset_mbz,
            fmin=fmin,
            fmax=fmax,
            sp_label='offset ' + sp_label,
            debugplot=debugplot)
    else:
        sp_reference_filtmask = sp_reference_mbz
        sp_offset_filtmask = sp_offset_mbz

    naxis1 = len(sp_reference_filtmask)
    xcorr = np.arange(naxis1, dtype=int)
    naxis1_half = int(naxis1 / 2)
    naxis1_half_remainder = naxis1 % 2
    for i in range(naxis1_half):
        xcorr[i + naxis1_half + naxis1_half_remainder] -= naxis1
    isort = xcorr.argsort()
    xcorr = xcorr[isort]

    if abs(debugplot) in (21, 22):
        from numina.array.display.matplotlib_qt import plt
        xdum = np.arange(len(sp_reference)) + 1
        xdumf = np.arange(len(sp_reference_filtmask)) + 1
        # reference spectrum
        ax = ximplotxy(xdum, sp_reference, show=False,
                       title='reference ' + sp_label,
                       label='original ' + sp_label)
        if remove_mean or frac_cosbell or zero_padding or fminmax:
            ax.plot(xdumf, sp_reference_filtmask,
                    label='processed ' + sp_label)
        ax.legend()
        plt.show()
        # offset spectrum
        ax = ximplotxy(xdum, sp_offset, show=False,
                       title='offset ' + sp_label,
                       label='original ' + sp_label)
        if remove_mean or frac_cosbell or zero_padding or fminmax:
            ax.plot(xdumf, sp_offset_filtmask,
                    label='processed ' + sp_label)
        ax.legend()
        plt.show()

    if (abs(debugplot) in (21, 22)) or (pdf is not None):
        xdum = np.arange(naxis1) + 1
        ax = ximplotxy(xdum, sp_reference_filtmask, show=False,
                       title=plottitle,
                       label='reference ' + sp_label)
        ax.plot(xdum, sp_offset_filtmask, label='offset ' + sp_label)
        ax.legend()
        if pdf is not None:
            pdf.savefig()
        else:
            pause_debugplot(debugplot=debugplot, pltshow=True)

    # normalize spectra if required
    if norm_spectra:
        sp_reference_norm = np.copy(sp_reference_filtmask)
        sp_offset_norm = np.copy(sp_offset_filtmask)
        sp_dum = np.concatenate((sp_reference_norm, sp_offset_norm))
        spmin = min(sp_dum)
        spmax = max(sp_dum)
        idum = np.where(sp_reference_norm > 0)
        sp_reference_norm[idum] /= spmax
        idum = np.where(sp_reference_norm < 0)
        sp_reference_norm[idum] /= -spmin
        idum = np.where(sp_offset_norm > 0)
        sp_offset_norm[idum] /= spmax
        idum = np.where(sp_offset_norm < 0)
        sp_offset_norm[idum] /= -spmin
        if (abs(debugplot) in (21, 22)) or (pdf is not None):
            xdum = np.arange(naxis1) + 1
            ax = ximplotxy(xdum, sp_reference_norm, show=False,
                           title=plottitle + ' [normalized]',
                           label='reference ' + sp_label)
            ax.plot(xdum, sp_offset_norm, label='offset ' + sp_label)
            ax.legend()
            if pdf is not None:
                pdf.savefig()
            else:
                pause_debugplot(debugplot=debugplot, pltshow=True)
    else:
        sp_reference_norm = sp_reference_filtmask
        sp_offset_norm = sp_offset_filtmask

    corr = np.fft.ifft(np.fft.fft(sp_offset_norm) *
                       np.fft.fft(sp_reference_norm).conj()).real
    corr = corr[isort]

    # determine correlation peak
    if naround_zero is None:
        iminpeak = 0
        imaxpeak = naxis1 - 1
    else:
        izero = np.where(xcorr == 0)[0][0]
        iminpeak = max(izero - naround_zero, 0)
        imaxpeak = min(izero + naround_zero, naxis1 - 1)
    ixpeak = corr[iminpeak:(imaxpeak + 1)].argmax() + iminpeak

    # fit correlation peak with 2nd order polynomial
    nmed = nfit_peak // 2
    imin = ixpeak - nmed
    imax = ixpeak + nmed
    lpeak_ok = True
    if imin < 0 or imax > len(corr):
        x_fit = np.zeros(nfit_peak)
        y_fit = np.zeros(nfit_peak)
        x_refined_peak = 0
        y_refined_peak = 0
        lpeak_ok = False
        poly_peak = Polynomial([0.0])
    else:
        x_fit = xcorr[imin:(imax+1)].astype(float)
        y_fit = corr[imin:(imax+1)]
        poly_peak = Polynomial.fit(x_fit, y_fit, 2)
        poly_peak = Polynomial.cast(poly_peak)
        coef = poly_peak.coef
        if coef[2] != 0:
            x_refined_peak = -coef[1] / (2.0 * coef[2])
        else:
            x_refined_peak = 0.0
        y_refined_peak = poly_peak(x_refined_peak)

    offset = x_refined_peak
    fpeak = y_refined_peak

    if (abs(debugplot) % 10 != 0) or (pdf is not None):
        ax = ximplotxy(xcorr, corr,
                       xlabel='offset (pixels)',
                       ylabel='cross-correlation function',
                       title=plottitle,
                       xlim=(-naxis1/2, naxis1/2), show=False)
        ax.axvline(offset, color='grey', linestyle='dashed')
        coffset = f"(offset:{offset:6.2f} pixels)"
        ax.text(0.01, 0.99, coffset,
                horizontalalignment='left',
                verticalalignment='top',
                transform=ax.transAxes)
        if naround_zero is not None:
            cwindow = f"(peak region: [{-naround_zero},{naround_zero}] pixels)"
            ax.text(0.01, 0.93, cwindow,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax.transAxes)
        # inset plot
        inset_ax = inset_axes(
            ax,
            width="40%",
            height="40%",
            loc=1
        )
        inset_ax.plot(xcorr, corr)
        if naround_zero is not None:
            inset_ax.set_xlim([-naround_zero, naround_zero])
        else:
            inset_ax.set_xlim([-50, 50])
        if lpeak_ok:
            xplot = np.linspace(xcorr[imin], xcorr[imax], num=50)
            yplot = poly_peak(xplot)
            inset_ax.plot(xplot, yplot, '-')
            inset_ax.plot(x_fit, y_fit, 'o')
            inset_ax.plot([x_refined_peak], [y_refined_peak], 'o')
        inset_ax.axvline(offset, color='grey', linestyle='dashed')
        if pdf is not None:
            pdf.savefig()
        else:
            pause_debugplot(debugplot=debugplot,
                            tight_layout=False, pltshow=True)

    return offset, fpeak


def compute_broadening(wv_obj, sp_obj, wv_ref, sp_ref,
                       sigmalist,
                       remove_mean=False,
                       frac_cosbell=None,
                       zero_padding=None,
                       fminmax=None,
                       naround_zero=None,
                       nfit_peak=None,
                       ax1=None, ax2=None,
                       debugplot=0):
    """Compute broadening to match 'sp_obj' with 'sp_ref'.

    The reference spectrum 'sp_ref' should have a better spectral
    resolution than the object spectrum 'sp_obj'.

    It is assumed that the wavelength arrays are provided in
    ascending order.

    The wavelength sampling 'wv_ref' and 'wv_ref' can be different.

    The comparison is performed in the common wavelength
    interval. Within this interval, both spectra are linearly
    resampled to the result of merging 'wv_obj' and 'wv_ref',
    using for this purpose a constant wavelength increment
    (selected from the minimum wavelength step present in any
    of the two input wavelength arrays).

    Parameters
    ----------
    wv_obj : numpy array
        Wavelength sampling of the object spectrum.
    sp_obj : numpy array
        Flux of the object spectrum.
    wv_ref : numpy array
        Wavelength sampling of the reference spectrum.
    sp_ref : numpy array
        Flux of the reference spectrum.
    sigmalist : numpy array or list
        Sigma broadening (in pixels).
    remove_mean : bool
        If True, remove mean from input spectra.
    frac_cosbell : float or None
        Fraction of spectrum where the cosine bell falls to zero.
    zero_padding : int or None
        Number of extended pixels set to zero.
    fminmax : tuple of floats or None
        Minimum and maximum frequencies to be used. If None, no
        frequency filtering is employed.
    naround_zero : int or None
        Half width of the window (around zero offset) to look for
        the correlation peak.
    nfit_peak : int
        Total number of points (must be odd!) around the peak of the
        crosscorrelation function to be employed to estimate the peak
        location (using a fit to a second order polynomial).
    ax1 : matplotlib Axes object
        If not none, this plot represents the offset and fpeak
        variation as a function of sigma.
    ax2 : matplotlib Axes object
        If not none, this plot represents the two input spectra and
        the broadened reference spectrum.
    debugplot : int
        Determines whether intermediate computations and/or plots
        are displayed. The valid codes are defined in
        numina.array.display.pause_debugplot.

    Returns
    -------
    offset_broad : float
        Offset (in pixels) between the two input spectra measured
        at the sigma value where fpeak is maximum.
    sigma_broad : float
        Sigma (in pixels) for which fpeak is maximum.
    sp_ref_broad : numpy array
        Reference spectrum broadened with sigma_broad (at the original
        wavelength sampling).

    """

    # determine wether there is an intersection between the two
    # wavelength ranges
    minmax_obj = [min(wv_obj), max(wv_obj)]
    minmax_ref = [min(wv_ref), max(wv_ref)]
    minimum_max = min(minmax_obj[1], minmax_ref[1])
    maximum_min = max(minmax_obj[0], minmax_ref[0])
    overlap = minimum_max - maximum_min
    if overlap <= 0:
        raise ValueError("Wavelength ranges of the two spectra don't overlap!")

    # merge wavelength ranges (removing duplicates)
    wv = np.unique(np.sort(np.concatenate((wv_obj, wv_ref))))
    # truncate to common interval
    wv = wv[(wv >= maximum_min) * (wv <= minimum_max)]
    # check whether the sampling is constant
    delta_wv = list(set(np.diff(wv)))
    if np.allclose(delta_wv, delta_wv[0], rtol=1e-05, atol=1e-08):
        # nothing to do
        pass
    else:
        # linear resampling using the smallest step in any of the input arrays
        min_delta_wv_obj = min(list(set(np.diff(wv_obj))))
        min_delta_wv_ref = min(list(set(np.diff(wv_ref))))
        min_delta_wv = min(min_delta_wv_obj, min_delta_wv_ref)
        wv_min = wv[0]
        wv_max = wv[-1]
        nsamples = int((wv_max - wv_min) / min_delta_wv + 0.5) + 1
        wv = np.linspace(wv_min, wv_max, num=nsamples)
        if abs(debugplot) >= 10:
            print('WARNING (compute_broadening): Interpolation required:')
            print(f'  wv_min..: {wv_min}')
            print(f'  wv_max..: {wv_max}')
            print(f'  step....: {min_delta_wv}')
            print(f'  nsamples: {nsamples}')

    # linear interpolation of input spectrum using the merged
    # wavelength sampling
    funinterp_obj = interp1d(wv_obj, sp_obj, kind='linear')
    funinterp_ref = interp1d(wv_ref, sp_ref, kind='linear')
    flux_obj = funinterp_obj(wv)
    flux_ref = funinterp_ref(wv)

    # normalize each spectrum dividing by its median
    flux_obj /= np.median(flux_obj)
    flux_ref /= np.median(flux_ref)

    # plot initial resampled spectra
    if abs(debugplot) in (21, 22):
        ax = ximplotxy(wv, flux_ref,
                       xlabel='wavelength (Angstrom)',
                       ylabel='flux (arbitrary units)',
                       label='flux_ref', show=False)
        ax.plot(wv, flux_obj, label='flux_obj')
        ax.legend()
        pause_debugplot(debugplot=debugplot, pltshow=True)

    nsigmas = len(list(sigmalist))
    offset = np.zeros(nsigmas)
    fpeak = np.zeros(nsigmas)
    for i, sigma in enumerate(sigmalist):
        # broaden reference spectrum
        flux_ref_broad = gaussian_filter(flux_ref, sigma)
        # plot the two spectra
        if abs(debugplot) in (21, 22):
            ax = ximplotxy(wv, flux_ref_broad,
                           xlabel='wavelength (Angstrom)',
                           ylabel='flux (arbitrary units)',
                           label='flux_ref', show=False)
            ax.plot(wv, flux_obj, label='flux_obj')
            ax.set_title('sigma: ' + str(sigma) + ' pixels')
            ax.legend()
            pause_debugplot(debugplot=debugplot, pltshow=True)
        # periodic correlation between the two spectra
        offset[i], fpeak[i] = periodic_corr1d(
            flux_ref_broad, flux_obj,
            remove_mean=remove_mean,
            frac_cosbell=frac_cosbell,
            zero_padding=zero_padding,
            fminmax=fminmax,
            naround_zero=naround_zero,
            nfit_peak=nfit_peak,
            norm_spectra=True,
            debugplot=debugplot
        )

    if ax1 is not None:
        ax1.plot(sigmalist, offset,
                 color='C0', marker='o', linestyle='', label='offset')
        ax1.set_xlabel('sigma (pixels)')
        ax1.set_ylabel('offset (pixels)', color='C0')
        ax1_ = ax1.twinx()
        ax1_.plot(sigmalist, fpeak,
                  color='C1', marker='o', linestyle='', label='fpeak')
        ax1_.set_ylabel('fpeak', color='C1')

    offset_broad = offset[np.argmax(fpeak)]
    sigma_broad = sigmalist[np.argmax(fpeak)]
    sp_ref_broad = gaussian_filter(sp_ref, sigma_broad)

    if ax2 is not None:
        ax2.plot(wv_obj, sp_obj, label='sp_obj')
        ax2.plot(wv_ref, sp_ref, color='#aaaaaa', label='sp_ref')
        ax2.plot(wv_ref, sp_ref_broad, label='sp_ref_broad')
        ax2.set_xlabel('wavelength (Angstrom)')
        ax2.set_ylabel('flux (arbitrary units)')
        ax2.legend()

    return offset_broad, sigma_broad, sp_ref_broad
