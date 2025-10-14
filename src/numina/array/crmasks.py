#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Combination of arrays avoiding coincident cosmic ray hits."""
import ast
import inspect
import logging
import os
import sys
import uuid

import argparse
from astropy.io import fits
from ccdproc import cosmicray_lacosmic
from datetime import datetime
import math
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
import numpy.ma as ma
from scipy import ndimage
from skimage.registration import phase_cross_correlation
import yaml

from numina.array.display.plot_hist_step import plot_hist_step
from numina.array.distortion import shift_image2d
from numina.array.numsplines import spline_positive_derivative
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
import teareduce as tea

VALID_CRMETHODS = ['simboundary', 'lacosmic', 'sb_lacosmic']
VALID_BOUNDARY_FITS = ['spline', 'piecewise']
VALID_COMBINATIONS = ['mediancr', 'meancrt', 'meancr']


def is_valid_number(x):
    """Check if x is a valid number (not NaN or Inf)."""
    return isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x)


def all_valid_numbers(seq):
    """Check if all elements in seq are valid numbers."""
    if not isinstance(seq, (list, tuple)):
        raise TypeError("Input must be a list or tuple.")
    return all(is_valid_number(x) for x in seq)


def remove_isolated_pixels(h):
    """Remove isolated pixels in a 2D array corresponding to a 2D histogram.

    Parameters
    ----------
    h : 2D numpy array
        The input 2D array (e.g., a histogram).

    Returns
    -------
    hclean : 2D numpy array
        The cleaned 2D array with isolated pixels removed.
    """
    if not isinstance(h, np.ndarray) or h.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    naxis2, naxis1 = h.shape
    flag = np.zeros_like(h, dtype=float)
    # Fill isolated holes
    for i in range(naxis2):
        for j in range(naxis1):
            if h[i, j] == 0:
                k, ktot = 0, 0
                fsum = 0.0
                for ii in [i-1, i, i+1]:
                    if 0 <= ii < naxis2:
                        for jj in [j-1, j, j+1]:
                            if 0 <= jj < naxis1:
                                ktot += 1
                                if h[ii, jj] != 0:
                                    k += 1
                                    fsum += h[ii, jj]
                if k == ktot - 1:
                    flag[i, j] = fsum / k
    hclean = h.copy()
    hclean[flag > 0] = flag[flag > 0]
    # Remove pixels with less than 4 neighbors
    flag = np.zeros_like(h, dtype=np.uint8)
    for i in range(naxis2):
        for j in range(naxis1):
            if h[i, j] != 0:
                k = 0
                for ii in [i-1, i, i+1]:
                    if 0 <= ii < naxis2:
                        for jj in [j-1, j, j+1]:
                            if 0 <= jj < naxis1:
                                if h[ii, jj] != 0:
                                    k += 1
                if k < 5:
                    flag[i, j] = 1
    hclean[flag == 1] = 0
    # Remove pixels with no neighbor on the left hand side
    # when moving from left to right in each row
    for i in range(naxis2):
        j = 0
        loop = True
        while loop:
            j += 1
            if j < naxis1:
                loop = h[i, j] != 0
            else:
                loop = False
        if j < naxis1:
            hclean[i, j:] = 0

    return hclean


def compute_flux_factor(image3d, median2d, _logger, interactive=False,
                        nxbins_half=50, nybins_half=50,
                        ymin=0.495, ymax=1.505):
    """Compute the flux factor for each image based on the median.

    Parameters
    ----------
    image3d : 3D numpy array
        The 3D array containing the images to compute the flux factor.
        Note that this array contains the original images after
        subtracting the bias, if any.
    median2d : 2D numpy array
        The median of the input arrays.
    _logger : logging.Logger
        The logger to use for logging.
    interactive : bool, optional
        If True, enable interactive mode for plots (default is False).
    nxbins_half : int, optional
        Half the number of bins in the x direction (default is 50).
    nybins_half : int, optional
        Half the number of bins in the y direction (default is 50).
    ymin : float, optional
        Minimum value for the y-axis (default is 0.495).
    ymax : float, optional
        Maximum value for the y-axis (default is 1.505).

    Returns
    -------
    flux_factor : 1D numpy array
        The flux factor for each image.
    """
    naxis3, naxis2, naxis1 = image3d.shape
    naxis2_, naxis1_ = median2d.shape
    if naxis2 != naxis2_ or naxis1 != naxis1_:
        raise ValueError("image3d and median2d must have the same shape in the last two dimensions.")

    xmin = np.min(median2d)
    xmax = np.max(median2d)
    dx = xmax - xmin
    xminh = xmin - dx / 20
    xmaxh = xmax + dx / 20
    bins0 = np.linspace(xmin, xmax, 100)
    h0, edges0 = np.histogram(median2d.flatten(), bins=bins0)
    hstep = (edges0[1] - edges0[0])
    fig, ax = plt.subplots()
    for i in range(naxis3):
        xmax_ = np.max(image3d[i])
        bins = np.arange(xmin, xmax_ + hstep, hstep)
        h, edges = np.histogram(image3d[i].flatten(), bins=bins)
        plot_hist_step(ax, edges, h, color=f'C{i}', label=f'Image {i+1}')
    plot_hist_step(ax, edges0, h0, color='black', label='Median')
    ax.set_xlim(xminh, xmaxh)
    ax.set_xlabel('pixel value')
    ax.set_ylabel('Number of pixels')
    ax.set_title('Before applying flux factor')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    plt.tight_layout()
    png_filename = 'histogram_before_flux_factor.png'
    _logger.info(f"saving {png_filename}")
    plt.savefig(png_filename, dpi=150)
    if interactive:
        _logger.info("Entering interactive mode (press 'q' to close plot)")
        plt.show()
    plt.close(fig)

    if naxis3 % 2 == 1:
        argsort = np.argsort(image3d, axis=0)
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12, 5.5))
        vmin, vmax = tea.zscale(median2d)
        tea.imshow(fig, ax1, median2d, ds9mode=True, vmin=vmin, vmax=vmax, aspect='auto')
        tea.imshow(fig, ax2, argsort[naxis3//2] + 1, ds9mode=True, vmin=1, vmax=naxis3, cmap='brg',
                   cblabel='Image number', aspect='auto')
        plt.tight_layout()
        png_filename = 'image_number_to_median.png'
        _logger.info(f"saving {png_filename}")
        plt.savefig(png_filename, dpi=150)
        if interactive:
            _logger.info("Entering interactive mode (press 'q' to close plot)")
            plt.show()
        plt.close(fig)

    xmin = np.min(median2d)
    xmax = np.max(median2d)
    nxbins = 2 * nxbins_half + 1
    nybins = 2 * nybins_half + 1
    xbin_edges = np.linspace(xmin, xmax, nxbins + 1)
    ybin_edges = np.linspace(ymin, ymax, nybins + 1)
    xbin = (xbin_edges[:-1] + xbin_edges[1:])/2
    ybin = (ybin_edges[:-1] + ybin_edges[1:])/2
    extent = [xbin_edges[0], xbin_edges[-1], ybin_edges[0], ybin_edges[-1]]

    cblabel = 'Number of pixels'
    flux_factor = []
    for idata, data in enumerate(image3d):
        ratio = np.divide(data, median2d, out=np.zeros_like(median2d, dtype=float), where=median2d != 0)
        h, edges = np.histogramdd(
            sample=(ratio.flatten(), median2d.flatten()),
            bins=(ybin_edges, xbin_edges)
        )
        vmin = np.min(h)
        if vmin == 0:
            vmin = 1
        vmax = np.max(h)
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5.5))
        tea.imshow(fig, ax1, h, norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent, aspect='auto', cblabel=cblabel)
        ax1.set_xlabel('pixel value')
        ax1.set_ylabel('ratio image/median')
        ax1.set_title(f'Image #{idata+1}')
        hclean = remove_isolated_pixels(h)
        tea.imshow(fig, ax2, hclean, norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent, aspect='auto', cblabel=cblabel)
        ax2.set_xlabel('pixel value')
        ax2.set_ylabel('ratio image/median')
        ax2.set_title(f'Image #{idata+1}')
        xmode = np.zeros(2)
        ymode = np.zeros(2)
        for side, imin, imax in zip((0, 1), (0, nybins_half+1), (nybins_half, nybins)):
            xfit = []
            yfit = []
            for i in range(imin, imax):
                fsum = np.sum(hclean[i, :])
                if fsum > 0:
                    pdensity = hclean[i, :] / fsum
                    perc = (1 - 1 / fsum)
                    p = np.interp(perc, np.cumsum(pdensity), np.arange(nxbins))
                    ax2.plot(xbin[int(p+0.5)], ybin[i], 'x', color=f'C{side}')
                    xfit.append(xbin[int(p+0.5)])
                    yfit.append(ybin[i])
            xfit = np.array(xfit)
            yfit = np.array(yfit)
            splfit = tea.AdaptiveLSQUnivariateSpline(yfit, xfit, t=2, adaptive=False)
            ax2.plot(splfit(yfit), yfit, f'C{side}-')
            knots = splfit.get_knots()
            ax2.plot(splfit(knots), knots, f'C{side}o', markersize=4)
            imax = np.argmax(splfit(yfit))
            ymode[side] = yfit[imax]
            xmode[side] = splfit(ymode[side])
            ax2.plot(xmode[side], ymode[side], f'C{side}o', markersize=8)
            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(ax1.get_ylim())
        if xmode[0] > xmode[1]:
            imode = 0
        else:
            imode = 1
        ax2.axhline(ymode[imode], color=f'C{imode}', linestyle=':')
        ax2.text(xbin[-5], ymode[imode]+(ybin[-1]-ybin[0])/40, f'{ymode[imode]:.3f}', color=f'C{imode}', ha='right')
        flux_factor.append(ymode[imode])
        plt.tight_layout()

        png_filename = f'flux_factor{idata+1}.png'
        _logger.info(f"saving {png_filename}")
        plt.savefig(png_filename, dpi=150)
        if interactive:
            _logger.info("Entering interactive mode (press 'q' to close plot)")
            plt.show()
        plt.close(fig)

    if len(flux_factor) != naxis3:
        raise ValueError(f"Expected {naxis3} flux factors, but got {len(flux_factor)}.")

    # round the flux factor to 6 decimal places to avoid
    # unnecessary precision when writting to the FITS header
    flux_factor = np.round(flux_factor, decimals=6)

    fig, ax = plt.subplots()
    for i in range(naxis3):
        xmax_ = np.max(image3d[i])
        bins = np.arange(xmin, xmax_ + hstep, hstep)
        h, edges = np.histogram(image3d[i].flatten() / flux_factor[i], bins=bins)
        plot_hist_step(ax, edges, h, color=f'C{i}', label=f'Image {i+1}')
    plot_hist_step(ax, edges0, h0, color='black', label='Median')
    ax.set_xlim(xminh, xmaxh)
    ax.set_xlabel('pixel value')
    ax.set_ylabel('Number of pixels')
    ax.set_title('After applying flux factor')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    plt.tight_layout()
    png_filename = 'histogram_after_flux_factor.png'
    _logger.info(f"saving {png_filename}")
    plt.savefig(png_filename, dpi=150)
    if interactive:
        _logger.info("Entering interactive mode (press 'q' to close plot)")
        plt.show()
    plt.close(fig)

    return flux_factor


def estimate_diagnostic_limits(rng, gain, rnoise, maxvalue, num_images, npixels):
    """Estimate the limits for the diagnostic plot.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    gain : float
        Gain value (in e/ADU) of the detector.
    rnoise : float
        Readout noise (in ADU) of the detector.
    maxvalue : float
        Maximum pixel value (in ADU) of the detector after
        subtracting the bias.
    num_images : int
        Number of different exposures.
    npixels : int
        Number of simulations to perform for each corner of the
        diagnostic plot.
    """

    if maxvalue < 0:
        maxvalue = 0.0
    xdiag_min = np.zeros(npixels, dtype=float)
    xdiag_max = np.zeros(npixels, dtype=float)
    ydiag_min = np.zeros(npixels, dtype=float)
    ydiag_max = np.zeros(npixels, dtype=float)
    for i in range(npixels):
        # lower limits
        data = rng.normal(loc=0, scale=rnoise, size=num_images)
        min1d = np.min(data)
        median1d = np.median(data)
        xdiag_min[i] = median1d
        ydiag_min[i] = median1d - min1d
        # upper limits
        lam = np.array([maxvalue] * num_images)
        data = rng.poisson(lam=lam * gain).astype(float) / gain
        if rnoise > 0:
            data += rng.normal(loc=0, scale=rnoise, size=num_images)
        min1d = np.min(data)
        median1d = np.median(data)
        xdiag_max[i] = median1d
        ydiag_max[i] = median1d - min1d
    xdiag_min = np.min(xdiag_min)
    ydiag_min = np.min(ydiag_min)
    xdiag_max = np.max(xdiag_max)
    ydiag_max = np.max(ydiag_max)
    return xdiag_min, xdiag_max, ydiag_min, ydiag_max


def define_piecewise_linear_function(xarray, yarray):
    """Define a piecewise linear function.

    Parameters
    ----------
    xarray : 1D numpy array
        The x coordinates of the points.
    yarray : 1D numpy array
        The y coordinates of the points.

    """
    isort = np.argsort(xarray)
    xfit = xarray[isort]
    yfit = yarray[isort]

    def function(x):
        y = np.interp(x, xfit, yfit)
        return y

    return function


def segregate_cr_flags(naxis1, naxis2, flag_only_la, flag_only_sb, flag_both,
                       enum_la_global, enum_sb_global, enum_both_global,
                       within_xy_diagram):
    """Segregate the cosmic ray flags into three categories:
    - detected only by the lacosmic method
    - detected only by the simboundary method
    - detected by both methods

    Parameters
    ----------
    naxis1 : int
        The size of the first dimension of the input arrays.
    naxis2 : int
        The size of the second dimension of the input arrays.
    flag_only_la : 1D numpy array
        A boolean array indicating which pixels are detected
        only by the lacosmic method.
    flag_only_sb : 1D numpy array
        A boolean array indicating which pixels are detected
        only by the simboundary method.
    flag_both : 1D numpy array
        A boolean array indicating which pixels are detected
        by both methods.
    enum_la_global : 1D numpy array
        An integer array with the enumeration of the pixels
        detected only by the lacosmic method.
    enum_sb_global : 1D numpy array
        An integer array with the enumeration of the pixels
        detected only by the simboundary method.
    enum_both_global : 1D numpy array
        An integer array with the enumeration of the pixels
        detected by both methods.
    within_xy_diagram : 1D numpy array
        A boolean array indicating which pixels are within the XY diagram.

    Returns
    -------
    flag_only_la_within_xy : 1D numpy array
        A boolean array indicating which pixels are detected
        only by the lacosmic method within the XY diagram.
    flag_only_sb_within_xy : 1D numpy array
        A boolean array indicating which pixels are detected
        only by the simboundary method within the XY diagram.
    flag_both_within_xy : 1D numpy array
        A boolean array indicating which pixels are detected
        by both methods within the XY diagram.
    (num_only_la, xcr_only_la, ycr_only_la) : tuple
        Number of pixels detected only by the lacosmic method,
        and their x and y coordinates (FITS convention; first pixel is (1, 1)).
        If no pixels are detected, xcr_only_la and ycr_only_la are None.
    (num_only_sb, xcr_only_sb, ycr_only_sb) : tuple
        Number of pixels detected only by the simboundary method,
        and their x and y coordinates (FITS convention; first pixel is (1, 1)).
        If no pixels are detected, xcr_only_sb and ycr_only_sb are None.
    (num_both, xcr_both, ycr_both) : tuple
        Number of pixels detected by both methods,
        and their x and y coordinates (FITS convention; first pixel is (1, 1)).
        If no pixels are detected, xcr_both and ycr_both are None.
    """

    # Segregate the cosmic rays within the XY diagnostic diagram
    flag_only_la_within_xy = flag_only_la & within_xy_diagram
    flag_only_sb_within_xy = flag_only_sb & within_xy_diagram
    flag_both_within_xy = flag_both & within_xy_diagram

    num_only_la_within_xy = np.sum(flag_only_la_within_xy)
    if num_only_la_within_xy > 0:
        pixels_detected = np.argwhere(flag_only_la_within_xy.reshape(naxis2, naxis1))
        xcr_only_la_within_xy = pixels_detected[:, 1] + 1  # FITS convention: first pixel is (1, 1)
        ycr_only_la_within_xy = pixels_detected[:, 0] + 1  # FITS convention: first pixel is (1, 1)
        ncr_only_la_within_xy = enum_la_global[flag_only_la_within_xy]
    else:
        xcr_only_la_within_xy, ycr_only_la_within_xy, ncr_only_la_within_xy = None, None, None

    num_only_sb_within_xy = np.sum(flag_only_sb_within_xy)
    if num_only_sb_within_xy > 0:
        pixels_detected = np.argwhere(flag_only_sb_within_xy.reshape(naxis2, naxis1))
        xcr_only_sb_within_xy = pixels_detected[:, 1] + 1  # FITS convention: first pixel is (1, 1)
        ycr_only_sb_within_xy = pixels_detected[:, 0] + 1  # FITS convention: first pixel is (1, 1)
        ncr_only_sb_within_xy = enum_sb_global[flag_only_sb_within_xy]
    else:
        xcr_only_sb_within_xy, ycr_only_sb_within_xy, ncr_only_sb_within_xy = None, None, None

    num_both_within_xy = np.sum(flag_both_within_xy)
    if num_both_within_xy > 0:
        pixels_detected = np.argwhere(flag_both_within_xy.reshape(naxis2, naxis1))
        xcr_both_within_xy = pixels_detected[:, 1] + 1  # FITS convention: first pixel is (1, 1)
        ycr_both_within_xy = pixels_detected[:, 0] + 1  # FITS convention: first pixel is (1, 1)
        ncr_both_within_xy = enum_both_global[flag_both_within_xy]
    else:
        xcr_both_within_xy, ycr_both_within_xy, ncr_both_within_xy = None, None, None

    return \
        flag_only_la_within_xy, flag_only_sb_within_xy, flag_both_within_xy, \
        (num_only_la_within_xy, xcr_only_la_within_xy, ycr_only_la_within_xy, ncr_only_la_within_xy), \
        (num_only_sb_within_xy, xcr_only_sb_within_xy, ycr_only_sb_within_xy, ncr_only_sb_within_xy), \
        (num_both_within_xy, xcr_both_within_xy, ycr_both_within_xy, ncr_both_within_xy)


def update_marks(naxis1, naxis2, flag_only_la, flag_only_sb, flag_both,
                 enum_la_global, enum_sb_global, enum_both_global,
                 xplot, yplot,
                 ax1, ax2, ax3, display_ncr=True):
    if flag_only_la.shape != (naxis2 * naxis1,):
        raise ValueError(f"{flag_only_la.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if flag_only_la.shape != flag_only_sb.shape:
        raise ValueError(f"{flag_only_sb.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if flag_only_la.shape != flag_both.shape:
        raise ValueError(f"{flag_both.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if flag_only_la.shape != xplot.shape or flag_only_la.shape != yplot.shape:
        raise ValueError(f"{xplot.shape=} and {yplot.shape=} must have {flag_only_la.shape=}.")

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    within_xy_diagram = (xlim[0] <= xplot) & (xplot <= xlim[1]) & (ylim[0] <= yplot) & (yplot <= ylim[1])

    flag_only_la_within_xy, flag_only_sb_within_xy, flag_both_within_xy, tuple_la, tuple_sb, tuple_both = \
        segregate_cr_flags(naxis1, naxis2, flag_only_la, flag_only_sb, flag_both,
                           enum_la_global, enum_sb_global, enum_both_global,
                           within_xy_diagram)
    num_only_la_within_xy, xcr_only_la_within_xy, ycr_only_la_within_xy, ncr_only_la_within_xy = tuple_la
    num_only_sb_within_xy, xcr_only_sb_within_xy, ycr_only_sb_within_xy, ncr_only_sb_within_xy = tuple_sb
    num_both_within_xy, xcr_both_within_xy, ycr_both_within_xy, ncr_both_within_xy = tuple_both

    for ax in [ax2, ax3]:
        for num, xcr, ycr, ncr, flag_only, color, marker in zip(
                [num_only_la_within_xy, num_only_sb_within_xy, num_both_within_xy],
                [xcr_only_la_within_xy, xcr_only_sb_within_xy, xcr_both_within_xy],
                [ycr_only_la_within_xy, ycr_only_sb_within_xy, ycr_both_within_xy],
                [ncr_only_la_within_xy, ncr_only_sb_within_xy, ncr_both_within_xy],
                [flag_only_la_within_xy, flag_only_sb_within_xy, flag_both_within_xy],
                ['r', 'b', 'y'],
                ['x', '+', 'o']):
            if num > 0:
                if ax == ax2:
                    for ix, iy, ncr in zip(xplot[flag_only], yplot[flag_only], ncr):
                        ax.text(ix, iy, str(ncr), color=color, fontsize=8, clip_on=True,
                                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
                else:
                    if display_ncr:
                        for ix, iy, ncr in zip(xcr, ycr, ncr):
                            ax.text(ix, iy, str(ncr), color=color, fontsize=8, clip_on=True,
                                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
                    else:
                        if marker == 'o':
                            ax.scatter(xcr, ycr, edgecolors=color, marker=marker, facecolors='none')
                        else:
                            ax.scatter(xcr, ycr, c=color, marker=marker)


def diagnostic_plot(xplot, yplot, xplot_boundary, yplot_boundary, flag_la, flag_sb,
                    sb_threshold, ylabel, interactive, target2d, target2d_name,
                    min2d, mean2d, image3d,
                    _logger=None, png_filename=None):
    """Diagnostic plot for the mediancr function.
    """
    if png_filename is None:
        raise ValueError("png_filename must be provided for diagnostic plots.")

    # Set up relevant parameters
    naxis3, naxis2, naxis1 = image3d.shape
    if target2d.shape != (naxis2, naxis1):
        raise ValueError("target2d must have shape (naxis2, naxis1).")
    if min2d.shape != (naxis2, naxis1):
        raise ValueError("min2d must have shape (naxis2, naxis1).")
    if mean2d.shape != (naxis2, naxis1):
        raise ValueError("mean2d must have shape (naxis2, naxis1).")
    if flag_la.shape != (naxis2 * naxis1,):
        raise ValueError(f"{flag_la.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if flag_la.shape != flag_sb.shape:
        raise ValueError(f"{flag_sb.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if xplot.shape != (naxis2 * naxis1,):
        raise ValueError(f"{xplot.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if yplot.shape != (naxis2 * naxis1,):
        raise ValueError(f"{yplot.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")

    display_ncr = False   # display the number of cosmic rays in the plot instead of symbols
    aspect_imshow = 'auto'  # 'equal' or 'auto'
    i_comparison_image = 0  # 0 for mean2d, 1, 2,... for image3d[comparison_image-1]

    if interactive:
        fig = plt.figure(figsize=(12, 8))
        x0_plot = 0.07
        y0_plot = 0.07
        width_plot = 0.4
        height_plot = 0.4
        vspace_plot = 0.09
        # top left, top right, bottom left, bottom right
        ax1 = fig.add_axes([x0_plot, y0_plot + height_plot + vspace_plot, width_plot, height_plot])
        ax2 = fig.add_axes([0.55, y0_plot + height_plot + vspace_plot, width_plot, height_plot],
                           sharex=ax1, sharey=ax1)
        ax3 = fig.add_axes([x0_plot, y0_plot, width_plot, height_plot])
        ax4 = fig.add_axes([0.55, y0_plot, width_plot, height_plot], sharex=ax3, sharey=ax3)
        dx_text = 0.07
        ax_vmin = fig.add_axes([x0_plot, y0_plot + height_plot + 0.005, dx_text, 0.03])
        ax_vmax = fig.add_axes([x0_plot + width_plot - dx_text, y0_plot + height_plot + 0.005, dx_text, 0.03])
    else:
        fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        ax1, ax2, ax3, ax4 = axarr.flatten()
        ax_vmin, ax_vmax = None, None

    # Segregate the cosmic rays detected by the different methods
    flag_only_la = flag_la & ~flag_sb
    flag_only_sb = flag_sb & ~flag_la
    flag_both = flag_la & flag_sb
    num_only_la = np.sum(flag_only_la)
    num_only_sb = np.sum(flag_only_sb)
    num_both = np.sum(flag_both)

    # Enumerate the cosmic rays detected by the different methods
    enum_la_global = np.zeros_like(flag_la, dtype=int)
    enum_la_global[flag_only_la] = np.arange(1, np.sum(flag_only_la) + 1, dtype=int)
    enum_sb_global = np.zeros_like(flag_sb, dtype=int)
    enum_sb_global[flag_only_sb] = np.arange(1, np.sum(flag_only_sb) + 1, dtype=int)
    enum_both_global = np.zeros_like(flag_la, dtype=int)
    enum_both_global[flag_both] = np.arange(1, np.sum(flag_both) + 1, dtype=int)

    ax1.plot(xplot, yplot, 'C0,')
    ax1.scatter(xplot[flag_only_la], yplot[flag_only_la],
                c='r', marker='x', label=f'Suspected pixels: {num_only_la} (lacosmic)')
    ax1.scatter(xplot[flag_only_sb], yplot[flag_only_sb],
                c='b', marker='+', label=f'Suspected pixels: {num_only_sb} (simboundary)')
    ax1.scatter(xplot[flag_both], yplot[flag_both],
                edgecolor='y', marker='o', facecolors='none', label=f'Suspected pixels: {num_both} (both methods)')
    if xplot_boundary is not None and yplot_boundary is not None:
        ax1.plot(xplot_boundary, yplot_boundary, 'C1-', label='Detection boundary')
    if sb_threshold is not None:
        ax1.axhline(sb_threshold, color='gray', linestyle=':', label=f'sb_threshold ({sb_threshold:.2f})')
    ax1.set_xlabel(r'min2d $-$ bias')  # the bias was subtracted from the input arrays
    ax1.set_ylabel(ylabel)
    ax1.set_title('Median-Mean Diagnostic Diagram')
    ax1.legend(loc='upper right', fontsize=8)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlabel(ax1.get_xlabel())
    ax2.set_ylabel(ax1.get_ylabel())
    ax2.set_title(ax1.get_title())

    vmin, vmax = tea.zscale(target2d)
    ax3_title = target2d_name
    img_ax3, _, _ = tea.imshow(fig, ax3, target2d, ds9mode=True, aspect=aspect_imshow, vmin=vmin, vmax=vmax,
                               title=ax3_title, cmap='viridis', colorbar=False)
    if i_comparison_image == 0:
        comparison_image = mean2d
        ax4_title = 'mean2d'
    else:
        comparison_image = image3d[i_comparison_image - 1]
        ax4_title = f'single exposure #{i_comparison_image}]'
    img_ax4, _, _ = tea.imshow(fig, ax4, comparison_image, ds9mode=True, aspect=aspect_imshow, vmin=vmin, vmax=vmax,
                               title=ax4_title, cmap='viridis', colorbar=False)
    ax3.set_title(ax3_title)
    ax4.set_title(ax4_title)
    update_marks(naxis1, naxis2, flag_only_la, flag_only_sb, flag_both,
                 enum_la_global, enum_sb_global, enum_both_global,
                 xplot, yplot,
                 ax1, ax2, ax3, display_ncr)

    updating = {'plot_limits': False}

    def sync_zoom_x(event_ax):
        if updating['plot_limits']:
            return
        try:
            updating['plot_limits'] = True
            if event_ax is ax1:
                xlim = ax1.get_xlim()
                ax2.set_xlim(xlim)
            elif event_ax is ax2:
                pass
            elif event_ax is ax3:
                pass
            elif event_ax is ax4:
                pass
        finally:
            updating['plot_limits'] = False

    def sync_zoom_y(event_ax):
        nonlocal img_ax3, img_ax4
        nonlocal display_ncr
        if updating['plot_limits']:
            return
        try:
            updating['plot_limits'] = True
            if event_ax is ax1:
                ylim = ax1.get_ylim()
                ax2.set_ylim(ylim)
                xlim = ax3.get_xlim()
                ylim = ax3.get_ylim()
                ax3.cla()
                img_ax3, _, _ = tea.imshow(fig, ax3, target2d, ds9mode=True, aspect=aspect_imshow,
                                           vmin=vmin, vmax=vmax,
                                           title=ax3_title, cmap='viridis', colorbar=False)
                ax3.set_xlim(xlim)
                ax3.set_ylim(ylim)
                xlim = ax4.get_xlim()
                ylim = ax4.get_ylim()
                ax4.cla()
                img_ax4, _, _ = tea.imshow(fig, ax4, comparison_image, ds9mode=True, aspect=aspect_imshow,
                                           vmin=vmin, vmax=vmax,
                                           title=ax4_title, cmap='viridis', colorbar=False)
                ax4.set_xlim(xlim)
                ax4.set_ylim(ylim)
                update_marks(naxis1, naxis2, flag_only_la, flag_only_sb, flag_both,
                             enum_la_global, enum_sb_global, enum_both_global,
                             xplot, yplot,
                             ax1, ax2, ax3, display_ncr)
                ax2.figure.canvas.draw_idle()
                ax3.figure.canvas.draw_idle()
                ax4.figure.canvas.draw_idle()
            elif event_ax is ax2:
                pass
            elif event_ax is ax3:
                pass
            elif event_ax is ax4:
                pass
        finally:
            updating['plot_limits'] = False

    ax1.callbacks.connect('xlim_changed', sync_zoom_x)
    ax1.callbacks.connect('ylim_changed', sync_zoom_y)

    if not interactive:
        plt.tight_layout()
    if png_filename is not None:
        _logger.info(f"saving {png_filename}")
        plt.savefig(png_filename, dpi=150)
    if interactive:
        init_limits = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in [ax1, ax2, ax3, ax4]}

        mouse_info = {'ax': None, 'x': None, 'y': None}

        def on_mouse_move(event):
            if event.inaxes:
                mouse_info['ax'] = event.inaxes
                mouse_info['x'] = event.xdata
                mouse_info['y'] = event.ydata

        fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

        def on_key(event):
            nonlocal vmin, vmax
            nonlocal img_ax3, img_ax4
            nonlocal display_ncr
            nonlocal aspect_imshow
            nonlocal i_comparison_image, comparison_image
            ax_mouse = mouse_info['ax']
            x_mouse = mouse_info['x']
            y_mouse = mouse_info['y']

            if event.key in ("h", "H", "r", "R"):
                for ax in [ax1, ax2, ax3, ax4]:
                    init_xlim, init_ylim = init_limits[ax]
                    ax.set_xlim(init_xlim)
                    ax.set_ylim(init_ylim)
                if i_comparison_image == 0:
                    comparison_image = mean2d
                    ax4_title = 'mean2d'
                else:
                    comparison_image = image3d[i_comparison_image - 1]
                    ax4_title = f'single exposure #{i_comparison_image}'
                ax4.set_title(ax4_title)
            elif event.key == '?':
                print("-" * 79)
                print("Keyboard shortcuts:")
                print("'h' or 'r': reset zoom to initial limits")
                print("'p': pan mode")
                print("'o': zoom to rectangle")
                print("'f': toggle full screen mode")
                print("'s': save the figure to a PNG file")
                print("." * 79)
                print("'?': show this help message")
                print("'i': print pixel info at mouse position (ax3 and ax4 only)")
                print("'n': toggle display of number of cosmic rays (ax3 and ax4 only)")
                print("'a': toggle imshow aspect='equal' / aspect='auto' (ax3 and ax4 only)")
                print("'t': toggle mean2d -> individual exposures in ax4")
                print("'0': switch to mean2d in ax4")
                print("'1', '2', ...: switch to individual exposure #1, #2, ... in ax4")
                print("',': set vmin and vmax to min and max of the zoomed region (ax3 and ax4 only)")
                print("'/': set vmin and vmax using zscale of the zoomed region (ax3 and ax4 only)")
                print("'q': close the plot and continue the program execution")
                print("-" * 79)
            elif event.key in ("i", "I"):
                if ax_mouse in [ax1, ax2]:
                    print(f'x_mouse = {x_mouse:.3f}, y_mouse = {y_mouse:.3f}')
                elif ax_mouse in [ax3, ax4]:
                    ix = int(round(x_mouse))
                    iy = int(round(y_mouse))
                    if 1 <= ix <= naxis1 and 1 <= iy <= naxis2:
                        print('-' * 79)
                        print(f'Pixel coordinates (FITS criterium): ix = {ix}, iy = {iy}')
                        print(f'target2d - min2d = {target2d[iy-1, ix-1] - min2d[iy-1, ix-1]:.3f}')
                        print(f'min2d - bias     = {min2d[iy-1, ix-1]:.3f}')
                        print('.' * 79)
                        for inum in range(image3d.shape[0]):
                            print(f'(image {inum+1} - bias) * flux_factor = {image3d[inum, iy-1, ix-1]:.3f}')
                        print('.' * 79)
                        for flag, crmethod in zip([flag_la, flag_sb, flag_both], ['lacosmic', 'simboundary']):
                            # Python convention: first pixel is (0, 0) but iy and ix are in FITS convention
                            # where the first pixel is (1, 1)
                            if flag.reshape((naxis2, naxis1))[iy-1, ix-1]:
                                print(f'Pixel found by {crmethod}')
                            else:
                                print(f'Pixel not found by {crmethod}')
            elif event.key == 'n':
                display_ncr = not display_ncr
                sync_zoom_y(ax1)
            elif event.key == 'a':
                if aspect_imshow == 'equal':
                    aspect_imshow = 'auto'
                else:
                    aspect_imshow = 'equal'
                sync_zoom_y(ax1)
            elif event.key in ['t', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                if ax_mouse == ax4:
                    i_comparison_image_previous = i_comparison_image
                    if event.key == 't':
                        i_comparison_image += 1
                        if i_comparison_image > naxis3:
                            i_comparison_image = 0
                    elif event.key == '0':
                        i_comparison_image = 0
                    else:
                        i_comparison_image = int(event.key)
                        if i_comparison_image > naxis3:
                            i_comparison_image = i_comparison_image_previous
                    if i_comparison_image != i_comparison_image_previous:
                        if i_comparison_image == 0:
                            comparison_image = mean2d
                            ax4_title = 'mean2d'
                        else:
                            comparison_image = image3d[i_comparison_image - 1]
                            ax4_title = f'single exposure #{i_comparison_image}'
                        print(f'Switching to {ax4_title} in ax4')
                        vmin, vmax = img_ax4.get_clim()
                        img_ax4.set_data(comparison_image)
                        img_ax4.set_clim(vmin=vmin, vmax=vmax)
                        ax4.set_title(ax4_title)
                        ax4.figure.canvas.draw_idle()
            elif event.key in [',', '/']:
                if ax_mouse in [ax3, ax4]:
                    xmin, xmax = ax_mouse.get_xlim()
                    ymin, ymax = ax_mouse.get_ylim()
                    ixmin = int(round(xmin))
                    ixmax = int(round(xmax))
                    iymin = int(round(ymin))
                    iymax = int(round(ymax))
                    if ixmin > ixmax:
                        ixmin, ixmax = ixmax, ixmin
                    if iymin > iymax:
                        iymin, iymax = iymax, iymin
                    if ixmin < 1:
                        ixmin = 1
                    if ixmax > naxis1:
                        ixmax = naxis1
                    if iymin < 1:
                        iymin = 1
                    if iymax > naxis2:
                        iymax = naxis2
                    region2d = tea.SliceRegion2D(f'[{ixmin}:{ixmax},{iymin}:{iymax}]', mode='fits').python
                    if event.key == ',':
                        if ax_mouse == ax3:
                            vmin, vmax = np.min(target2d[region2d]), np.max(target2d[region2d])
                        elif ax_mouse == ax4:
                            vmin, vmax = np.min(comparison_image[region2d]), np.max(comparison_image[region2d])
                    elif event.key == '/':
                        if ax_mouse == ax3:
                            vmin, vmax = tea.zscale(target2d[region2d])
                        elif ax_mouse == ax4:
                            vmin, vmax = tea.zscale(comparison_image[region2d])
                    text_box_vmin.set_val(f'{int(np.round(vmin, 0))}')
                    text_box_vmax.set_val(f'{int(np.round(vmax, 0))}')

                    img_ax3.set_clim(vmin=vmin, vmax=vmax)
                    img_ax4.set_clim(vmin=vmin, vmax=vmax)

                    ax3.figure.canvas.draw_idle()
                    ax4.figure.canvas.draw_idle()

        fig.canvas.mpl_connect("key_press_event", on_key)

        def submit_vmin(text):
            nonlocal vmin, vmax
            text = text.strip()
            if text:
                try:
                    vmin_ = float(text)
                    vmin = vmin_
                    img_ax3.set_clim(vmin=vmin, vmax=vmax)
                    img_ax4.set_clim(vmin=vmin, vmax=vmax)
                    ax3.figure.canvas.draw_idle()
                    ax4.figure.canvas.draw_idle()
                except ValueError:
                    print(f'Invalid input: {text}')

        def submit_vmax(text):
            nonlocal vmin, vmax
            text = text.strip()
            if text:
                try:
                    vmax_ = float(text)
                    vmax = vmax_
                    img_ax3.set_clim(vmin=vmin, vmax=vmax)
                    img_ax4.set_clim(vmin=vmin, vmax=vmax)
                    ax3.figure.canvas.draw_idle()
                    ax4.figure.canvas.draw_idle()
                except ValueError:
                    print(f'Invalid input: {text}')

        text_box_vmin = TextBox(ax_vmin, 'vmin:', initial=f'{int(np.round(vmin, 0))}', textalignment='right')
        text_box_vmin.on_submit(submit_vmin)
        text_box_vmax = TextBox(ax_vmax, 'vmax:', initial=f'{int(np.round(vmax, 0))}', textalignment='right')
        text_box_vmax.on_submit(submit_vmax)

        _logger.info("Entering interactive mode (press 'q' to close plot)")
        # plt.tight_layout()
        plt.show()
    plt.close(fig)


def gausskernel2d_elliptical(fwhm_x, fwhm_y, kernsize):
    """Calculate elliptical 2D Gaussian kernel.

    This kernel can be used as the psfk parameter of
    ccdproc.cosmicray_lacosmic.

    Parameters
    ----------
    fwhm_x : float
        Full width at half maximum in the x direction.
    fwhm_y : float
        Full width at half maximum in the y direction.
    kernsize : int
        Size of the kernel (must be odd).

    Returns
    -------
    kernel : 2D numpy array
        The elliptical Gaussian kernel. It is normalized
        so that the sum of all its elements is 1. It
        is returned as a float32 array (required by
        ccdproc.cosmicray_lacosmic).
    """

    if kernsize % 2 == 0 or kernsize < 3:
        raise ValueError("kernsize must be an odd integer >= 3.")
    if fwhm_x <= 0 or fwhm_y <= 0:
        raise ValueError("fwhm_x and fwhm_y must be positive numbers.")

    sigma_x = fwhm_x / (2 * math.sqrt(2 * math.log(2)))
    sigma_y = fwhm_y / (2 * math.sqrt(2 * math.log(2)))
    halfsize = kernsize // 2
    y, x = np.mgrid[-halfsize:halfsize+1, -halfsize:halfsize+1]
    kernel = np.exp(-0.5 * ((x / sigma_x) ** 2 + (y / sigma_y) ** 2))
    kernel /= np.sum(kernel)

    # reverse the psf kernel as that is what it is used in the convolution
    kernel = kernel[::-1, ::-1]
    return kernel.astype(np.float32)


def update_flag_with_user_masks(flag, pixels_to_be_masked, pixels_to_be_excluded, _logger):
    """Update the flag array with user-defined masks.

    Parameters
    ----------
    flag : 2D numpy array
        The input flag array to be updated.
    pixels_to_be_masked : list of (x, y) tuples, or None
        List of pixel coordinates to be included in the masks
        (FITS criterium; first pixel is (1, 1)).
    pixels_to_be_excluded : list of (x, y) tuples, or None
        List of pixel coordinates to be excluded from the masks
        (FITS criterium; first pixel is (1, 1)).
    _logger : logging.Logger
        The logger to use for logging.

    Returns
    -------
    None
    """
    # Include pixels to be forced to be masked
    if pixels_to_be_masked is not None:
        ix_pixels_to_be_masked = np.array([p[0] for p in pixels_to_be_masked], dtype=int) - 1
        iy_pixels_to_be_masked = np.array([p[1] for p in pixels_to_be_masked], dtype=int) - 1
        if np.any(ix_pixels_to_be_masked < 0) or np.any(ix_pixels_to_be_masked >= flag.shape[1]):
            raise ValueError("Some x coordinates in pixels_to_be_masked are out of bounds.")
        if np.any(iy_pixels_to_be_masked < 0) or np.any(iy_pixels_to_be_masked >= flag.shape[0]):
            raise ValueError("Some y coordinates in pixels_to_be_masked are out of bounds.")
        neff = 0
        for iy, ix in zip(iy_pixels_to_be_masked, ix_pixels_to_be_masked):
            if not flag[iy, ix]:
                flag[iy, ix] = True
                neff += 1
            else:
                _logger.warning("Pixel (%d, %d) to be masked was already masked.", ix + 1, iy + 1)
        _logger.info("Added %d/%d user-defined pixels to be masked.", neff, len(pixels_to_be_masked))

    # Exclude pixels to be excluded from the mask
    if pixels_to_be_excluded is not None:
        ix_pixels_to_be_excluded = np.array([p[0] for p in pixels_to_be_excluded], dtype=int) - 1
        iy_pixels_to_be_excluded = np.array([p[1] for p in pixels_to_be_excluded], dtype=int) - 1
        if np.any(ix_pixels_to_be_excluded < 0) or np.any(ix_pixels_to_be_excluded >= flag.shape[1]):
            raise ValueError("Some x coordinates in pixels_to_be_excluded are out of bounds.")
        if np.any(iy_pixels_to_be_excluded < 0) or np.any(iy_pixels_to_be_excluded >= flag.shape[0]):
            raise ValueError("Some y coordinates in pixels_to_be_excluded are out of bounds.")
        neff = 0
        for iy, ix in zip(iy_pixels_to_be_excluded, ix_pixels_to_be_excluded):
            if flag[iy, ix]:
                flag[iy, ix] = False
                neff += 1
            else:
                _logger.warning("Pixel (%d, %d) to be unmasked was not masked.", ix + 1, iy + 1)
        _logger.info("Removed %d/%d user-defined pixels from the mask.", neff, len(pixels_to_be_excluded))


def compute_crmasks(
        list_arrays,
        gain=None,
        rnoise=None,
        bias=None,
        crmethod='sb_lacosmic',
        flux_factor=None,
        interactive=True,
        dilation=1,
        pixels_to_be_masked=None,
        pixels_to_be_excluded=None,
        dtype=np.float32,
        verify_cr=False,
        semiwindow=15,
        color_scale='minmax',
        maxplots=-1,
        la_sigclip=None,
        la_psffwhm_x=None,
        la_psffwhm_y=None,
        la_fsmode=None,
        la_psfmodel=None,
        la_psfsize=None,
        sb_crosscorr_region=None,
        sb_boundary_fit=None,
        sb_knots_splfit=3,
        sb_fixed_points_in_boundary=None,
        sb_nsimulations=10,
        sb_niter_boundary_extension=3,
        sb_weight_boundary_extension=10.0,
        sb_threshold=0.0,
        sb_minimum_max2d_rnoise=5.0,
        sb_seed=None
        ):
    """
    Computation of cosmic rays masks using several equivalent exposures.

    This function computes cosmic ray masks from a list of 2D numpy arrays.
    Two different methods are implemented:
    1. Cosmic ray detection using the Laplacian edge detection algorithm
       (van Dokkum 2001), as implemented in ccdproc.cosmicray_lacosmic.
    2. Cosmic ray detection using a numerically derived boundary in the
       median combined image. The cosmic ray detection is based on a boundary
       that is derived numerically making use of the provided gain and readout
       noise values. The function also supports generating diagnostic plots to
       visualize the cosmic ray detection process.

    Parameters
    ----------
    list_arrays : list of 2D arrays
        The input arrays to be combined. The arrays are assumed to be
        provided in ADU.
    gain : 2D array, float or None
        The gain value (in e/ADU) of the detector.
        If None, it is assumed to be 1.0.
    rnoise : 2D array, float or None
        The readout noise (in ADU) of the detector.
        If None, it is assumed to be 0.0.
    bias : 2D array, float or None
        The bias value (in ADU) of the detector.
        If None, it is assumed to be 0.0.
    crmethod : str
        The method to use for cosmic ray detection. Valid options are:
        - 'lacosmic': use the cosmic-ray rejection by Laplacian edge
        detection (van Dokkum 2001), as implemented in ccdproc.
        - 'simboundary': use the numerically derived boundary to
        detect cosmic rays in the median combined image.
    flux_factor : str, list, float or None, optional
        The flux scaling factor for each exposure (default is None).
        If 'auto', the flux factor is determined automatically.
        If None or 'none', it is set to 1.0 for all images.
        If a float is provided, it is used as the flux factor for all images.
        If a list is provided, it should contain a value
        for each single image in `list_arrays`.
    interactive : bool, optional
        If True, enable interactive mode for plots.
    dilation : int, optional
        The dilation factor for the coincident cosmic ray mask.
    pixels_to_be_masked : str, list of (x, y) tuples, or None, optional
        List of pixel coordinates to be included in the masks
        (FITS criterium; first pixel is (1, 1)).
    pixels_to_be_excluded : str, list of (x, y) tuples, or None, optional
        List of pixel coordinates to be excluded from the masks
        (FITS criterium; first pixel is (1, 1)).
    dtype : data-type, optional
        The desired data type to build the 3D stack (default is np.float32).
    verify_cr : bool, optional
        If True, verify the cosmic ray detection by comparing the
        detected positions with the original images (default is True).
    semiwindow : int, optional
        The semiwindow size to plot the coincident cosmic rays (default is 15).
    color_scale : str, optional
        The color scale to use for the plots (default is 'minmax').
        Valid options are 'minmax' and 'zscale'.
    maxplots : int, optional
        The maximum number of coincident cosmic rays to plot (default is -1).
        If negative, all detected cosmic rays will be plotted.
    la_sigclip : float
        The sigma clipping threshold. Employed when crmethod='lacosmic'.
    la_psffwhm_x : float
        The full width at half maximum (FWHM, in pixels) of the PSF in
        the x direction. Employed when crmethod='lacosmic'.
    la_psffwhm_y : float
        The full width at half maximum (FWHM, in pixels) of the PSF
        in the y direction. Employed when crmethod='lacosmic'.
    la_fsmode : str
        The mode to use for the fine structure image. Valid options are:
        'median' or 'convolve'. Employed when crmethod='lacosmic'.
    la_psfmodel : str
        The model to use for the PSF if la_fsmode='convolve'.
        Valid options are:
        - circular kernels: 'gauss' or 'moffat'
        - Gaussian in the x and y directions: 'gaussx' and 'gaussy'
        - elliptical Gaussian: 'gaussxy' (this kernel is not available
          in ccdproc.cosmicray_lacosmic, so it is implemented here)
        Employed when crmethod='lacosmic'.
    la_psfsize : int
        The kernel size to use for the PSF. It must be an odd integer >= 3.
        Employed when crmethod='lacosmic'.
    sb_crosscorr_region : str, or None
        The region to use for the 2D cross-correlation to determine
        the offsets between the individual images and the median image.
        If None, no offsets are computed and it is assumed that
        the images are already aligned. The format of the region
        must follow the FITS convention '[xmin:xmax,ymin:ymax]',
        where the indices start from 1 to NAXIS[12].
    sb_boundary_fit : str, or None
        The method to use for the boundary fitting. Valid options are:
        - 'spline': use a spline fit to the boundary.
        - 'piecewise': use a piecewise linear fit to the boundary.
    sb_knots_splfit : int, optional
        The number of knots for the spline fit to the boundary.
    sb_fixed_points_in_boundary : str, or list or None
        The fixed points to use for the boundary fitting.
    sb_nsimulations : int, optional
        The number of simulations of each set of input images to compute
        the detection boundary.
    sb_niter_boundary_extension : int, optional
        The number of iterations for the boundary extension.
    sb_weight_boundary_extension : float, optional
        The weight for the boundary extension.
        In each iteration, the boundary is extended by applying an
        extra weight to the points above the previous boundary. This
        extra weight is computed as `sb_weight_boundary_extension**iter`,
        where `iter` is the current iteration number (starting from 1).
    sb_threshold: float, optional
        Minimum threshold for median2d - min2d to consider a pixel as a
        cosmic ray (default is None). If None, the threshold is computed
        automatically from the minimum boundary value in the numerical
        simulations.
    sb_minimum_max2d_rnoise : float, optional
        Minimum value for max2d in readout noise units to flag a pixel
        as a coincident cosmic ray.
    sb_seed : int or None, optional
        The random seed for reproducibility.

    Returns
    -------
    hdul_masks : hdulist
        The HDUList containing the mask arrays for cosmic ray
        removal using different methods. The primary HDU only contains
        information about the parameters used to determine the
        suspected pixels. The extensions are:
        - 'MEDIANCR': Mask for coincident cosmic rays detected using the
        median combination.
        - 'MEANCRT': Mask for cosmic rays detected when adding all the
        individual arrays. That summed image contains all the cosmic rays.
        of all the images.
        - 'CRMASK1', 'CRMASK2', ...: Masks for cosmic rays detected
        in each individual array.
    """

    _logger = logging.getLogger(__name__)

    # Check that the input is a list
    if not isinstance(list_arrays, list):
        raise TypeError("Input must be a list of arrays.")

    # Check that the list contains numpy 2D arrays
    if not all(isinstance(array, np.ndarray) and array.ndim == 2 for array in list_arrays):
        raise ValueError("All elements in the list must be 2D numpy arrays.")

    # Check that the list contains at least 3 arrays
    num_images = len(list_arrays)
    if num_images < 3:
        raise ValueError("At least 3 images are required for useful mediancr combination.")

    # Check that all arrays have the same shape
    for i, array in enumerate(list_arrays):
        if array.shape != list_arrays[0].shape:
            raise ValueError(f"Array {i} has a different shape than the first array.")
    naxis2, naxis1 = list_arrays[0].shape

    # Log the number of input arrays and their shapes
    _logger.info("number of input arrays: %d", len(list_arrays))
    for i, array in enumerate(list_arrays):
        _logger.info("array %d shape: %s, dtype: %s", i, array.shape, array.dtype)

    # Define the gain
    gain_scalar = None  # store gain as a scalar value (if applicable)
    if gain is None:
        gain = np.ones((naxis2, naxis1), dtype=float)
        _logger.info("gain not defined, assuming gain=1.0 for all pixels.")
        gain_scalar = 1.0
    elif isinstance(gain, (float, int)):
        gain = np.full((naxis2, naxis1), gain, dtype=float)
        _logger.info("gain defined as a constant value: %f", gain[0, 0])
        gain_scalar = float(gain[0, 0])
    elif isinstance(gain, np.ndarray):
        if gain.shape != (naxis2, naxis1):
            raise ValueError(f"gain must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("gain defined as a 2D array with shape: %s", gain.shape)
        if np.all(gain == gain[0, 0]):
            gain_scalar = float(gain[0, 0])
    else:
        raise TypeError(f"Invalid type for gain: {type(gain)}. Must be float, int, or numpy array.")

    # Define the readout noise
    rnoise_scalar = None  # store rnoise as a scalar value (if applicable)
    if rnoise is None:
        rnoise = np.zeros((naxis2, naxis1), dtype=float)
        _logger.info("readout noise not defined, assuming readout noise=0.0 for all pixels.")
        rnoise_scalar = 0.0
    elif isinstance(rnoise, (float, int)):
        rnoise = np.full((naxis2, naxis1), rnoise, dtype=float)
        _logger.info("readout noise defined as a constant value: %f", rnoise[0, 0])
        rnoise_scalar = float(rnoise[0, 0])
    elif isinstance(rnoise, np.ndarray):
        if rnoise.shape != (naxis2, naxis1):
            raise ValueError(f"rnoise must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("readout noise defined as a 2D array with shape: %s", rnoise.shape)
        if np.all(rnoise == rnoise[0, 0]):
            rnoise_scalar = float(rnoise[0, 0])
    else:
        raise TypeError(f"Invalid type for rnoise: {type(rnoise)}. Must be float, int, or numpy array.")

    # Define the bias
    if bias is None:
        bias = np.zeros((naxis2, naxis1), dtype=float)
        _logger.info("bias not defined, assuming bias=0.0 for all pixels.")
    elif isinstance(bias, (float, int)):
        bias = np.full((naxis2, naxis1), bias, dtype=float)
        _logger.info("bias defined as a constant value: %f", bias[0, 0])
    elif isinstance(bias, np.ndarray):
        if bias.shape != (naxis2, naxis1):
            raise ValueError(f"bias must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("bias defined as a 2D array with shape: %s", bias.shape)
    else:
        raise TypeError(f"Invalid type for bias: {type(bias)}. Must be float, int, or numpy array.")

    # Convert the list of arrays to a 3D numpy array
    image3d = np.zeros((num_images, naxis2, naxis1), dtype=dtype)
    for i, array in enumerate(list_arrays):
        image3d[i] = array.astype(dtype)

    # Subtract the bias from the input arrays
    _logger.info("subtracting bias from the input arrays")
    image3d -= bias

    # Check crmethod
    if crmethod not in VALID_CRMETHODS:
        raise ValueError(f"Invalid crmethod: {crmethod}. Valid options are {VALID_CRMETHODS}.")

    # Check flux_factor
    if flux_factor is None:
        flux_factor = np.ones(num_images, dtype=float)
    elif isinstance(flux_factor, str):
        if flux_factor.lower() == 'auto':
            _logger.info("flux_factor set to 'auto', computing values...")
            median2d = np.median(image3d, axis=0)
            flux_factor = compute_flux_factor(image3d, median2d, _logger, interactive)
            _logger.info("flux_factor set to %s", str(flux_factor))
        elif flux_factor.lower() == 'none':
            flux_factor = np.ones(num_images, dtype=float)
        elif isinstance(ast.literal_eval(flux_factor), list):
            flux_factor = ast.literal_eval(flux_factor)
            if len(flux_factor) != num_images:
                raise ValueError(f"flux_factor must have the same length as the number of images ({num_images}).")
            if not all_valid_numbers(flux_factor):
                raise ValueError(f"All elements in flux_factor={flux_factor} must be valid numbers.")
            flux_factor = np.array(flux_factor, dtype=float)
        elif isinstance(ast.literal_eval(flux_factor), (float, int)):
            flux_factor = np.full(num_images, ast.literal_eval(flux_factor), dtype=float)
        else:
            raise ValueError(f"Invalid flux_factor string: {flux_factor}. Use 'auto' or 'none'.")
    elif isinstance(flux_factor, list):
        if len(flux_factor) != num_images:
            raise ValueError(f"flux_factor must have the same length as the number of images ({num_images}).")
        if not all_valid_numbers(flux_factor):
            raise ValueError(f"All elements in flux_factor={flux_factor} must be valid numbers.")
        flux_factor = np.array(flux_factor, dtype=float)
    else:
        raise ValueError(f"Invalid flux_factor value: {flux_factor}.")
    _logger.info("flux_factor: %s", str(flux_factor))

    # Apply the flux factor to the input arrays
    for i in range(num_images):
        image3d[i] /= flux_factor[i]

    # Compute minimum, maximum, median and mean along the first axis
    min2d = np.min(image3d, axis=0)
    max2d = np.max(image3d, axis=0)
    median2d = np.median(image3d, axis=0)
    mean2d = np.mean(image3d, axis=0)

    # Compute points for diagnostic diagram of simboundary method
    xplot = min2d.flatten()  # bias was already subtracted above
    yplot = median2d.flatten() - min2d.flatten()

    # Check that color_scale is valid
    if color_scale not in ['minmax', 'zscale']:
        raise ValueError(f"Invalid color_scale: {color_scale}. Valid options are 'minmax' and 'zscale'.")

    # Define the pixels to be forced to be masked
    if pixels_to_be_masked is None:
        pass
    elif pixels_to_be_masked.lower() == 'none':
        pixels_to_be_masked = None
    else:
        pixels_to_be_masked = list(eval(str(pixels_to_be_masked)))
    _logger.info("pixels to be initially forced to be masked: %s",
                 "None" if pixels_to_be_masked is None else str(pixels_to_be_masked))

    # Define the pixels to be excluded from the masks
    if pixels_to_be_excluded is None:
        pass
    elif pixels_to_be_excluded.lower() == 'none':
        pixels_to_be_excluded = None
    else:
        pixels_to_be_excluded = list(eval(str(pixels_to_be_excluded)))
    _logger.info("pixels to be initially excluded from the masks: %s",
                 "None" if pixels_to_be_excluded is None else str(pixels_to_be_excluded))

    # Log the input parameters
    _logger.info("crmethod: %s", crmethod)
    if crmethod in ['simboundary', 'sb_lacosmic']:
        _logger.info("sb_crosscorr_region: %s", sb_crosscorr_region if sb_crosscorr_region is not None else "None")
        _logger.info("sb_boundary_fit: %s", sb_boundary_fit if sb_boundary_fit is not None else "None")
        _logger.info("knots for spline fit to the boundary: %d", sb_knots_splfit)
        _logger.info("fixed points in the boundary: %s",
                     str(sb_fixed_points_in_boundary) if sb_fixed_points_in_boundary is not None else "None")
        _logger.info("number of simulations to compute the detection boundary: %d", sb_nsimulations)
        _logger.info("sb_threshold for coincident cosmic ray detection: %s",
                     sb_threshold if sb_threshold is not None else "None")
        _logger.info("minimum max2d in rnoise units for coincident cosmic ray detection: %f", sb_minimum_max2d_rnoise)
        _logger.info("niter for boundary extension: %d", sb_niter_boundary_extension)
        _logger.info("random seed for reproducibility: %s", str(sb_seed))
        _logger.info("weight for boundary extension: %f", sb_weight_boundary_extension)

    if crmethod in ['lacosmic', 'sb_lacosmic']:
        if la_sigclip is None:
            _logger.info("la_sigclip for lacosmic not defined, assuming la_sigclip=5.0")
            la_sigclip = 5.0
        else:
            _logger.info("la_sigclip for lacosmic: %f", la_sigclip)
        if la_fsmode not in ['median', 'convolve']:
            raise ValueError("la_fsmode must be 'median' or 'convolve'.")
        else:
            _logger.info("la_fsmode for lacosmic: %s", la_fsmode)
        if la_psfmodel not in ['gauss', 'moffat', 'gaussx', 'gaussy', 'gaussxy']:
            raise ValueError("la_psfmodel must be 'gauss', 'moffat', 'gaussx', 'gaussy', or 'gaussxy'.")
        else:
            _logger.info("la_psfmodel for lacosmic: %s", la_psfmodel)
        if la_fsmode == 'convolve':
            if la_psffwhm_x is None or la_psffwhm_y is None or la_psfsize is None:
                raise ValueError("For la_fsmode='convolve', "
                                 "la_psffwhm_x, la_psffwhm_y, and la_psfsize must be provided.")
            else:
                _logger.info("la_psffwhm_x for lacosmic: %f", la_psffwhm_x)
                _logger.info("la_psffwhm_y for lacosmic: %f", la_psffwhm_y)
                _logger.info("la_psfsize for lacosmic: %d", la_psfsize)
            if la_psfsize % 2 == 0 or la_psfsize < 3:
                raise ValueError("la_psfsize must be an odd integer >= 3.")

    _logger.info("dtype for output arrays: %s", dtype)
    _logger.info("dilation factor: %d", dilation)
    _logger.info("verify cosmic ray detection: %s", verify_cr)
    _logger.info("semiwindow size for plotting coincident cosmic rays: %d", semiwindow)
    _logger.info("maximum number of coincident cosmic rays to plot: %d", maxplots)
    _logger.info("color scale for plots: %s", color_scale)

    if crmethod in ['lacosmic', 'sb_lacosmic']:
        # ---------------------------------------------------------------------
        # Detect residual cosmic rays in the median2d image using the
        # Laplacian edge detection method from ccdproc. This only works if gain and
        # rnoise are constant values (scalars).
        # ---------------------------------------------------------------------
        _logger.info("detecting cosmic rays in the median2d image using lacosmic...")
        if gain_scalar is None or rnoise_scalar is None:
            raise ValueError("gain and rnoise must be constant values (scalars) when using crmethod='lacosmic'.")
        if la_fsmode == 'median':
            median2d_lacosmic, flag_la = cosmicray_lacosmic(
                ccd=median2d,
                gain=gain_scalar,
                readnoise=rnoise_scalar,
                sigclip=la_sigclip,
                fsmode='median'
            )
        elif la_fsmode == 'convolve':
            if la_psfmodel != 'gaussxy':
                median2d_lacosmic, flag_la = cosmicray_lacosmic(
                    ccd=median2d,
                    gain=gain_scalar,
                    readnoise=rnoise_scalar,
                    sigclip=la_sigclip,
                    fsmode='convolve',
                    psfk=None,
                    psfmodel=la_psfmodel,
                )
            else:
                median2d_lacosmic, flag_la = cosmicray_lacosmic(
                    ccd=median2d,
                    gain=gain_scalar,
                    readnoise=rnoise_scalar,
                    sigclip=la_sigclip,
                    fsmode='convolve',
                    psfk=gausskernel2d_elliptical(fwhm_x=la_psffwhm_x, fwhm_y=la_psffwhm_y, kernsize=la_psfsize)
                )
        else:
            raise ValueError("la_fsmode must be 'median' or 'convolve'.")
        _logger.info("number of pixels flagged as cosmic rays by lacosmic: %d", np.sum(flag_la))
        update_flag_with_user_masks(flag_la, pixels_to_be_masked, pixels_to_be_excluded, _logger)
        flag_la = flag_la.flatten()
        if crmethod == 'lacosmic':
            xplot_boundary = None
            yplot_boundary = None
            sb_threshold = None
            flag_sb = np.zeros_like(flag_la, dtype=bool)

    if crmethod in ['simboundary', 'sb_lacosmic']:
        # ---------------------------------------------------------------------
        # Detect cosmic rays in the median2d image using the numerically
        # derived boundary.
        # ---------------------------------------------------------------------
        # Define sb_fixed_points_in_boundary
        if sb_fixed_points_in_boundary is None:
            pass
        elif sb_fixed_points_in_boundary.lower() == 'none':
            sb_fixed_points_in_boundary = None
        else:
            sb_fixed_points_in_boundary = list(eval(str(sb_fixed_points_in_boundary)))
            x_sb_fixed_points_in_boundary = []
            y_sb_fixed_points_in_boundary = []
            w_sb_fixed_points_in_boundary = []
            for item in sb_fixed_points_in_boundary:
                if not (isinstance(item, (list, tuple)) and len(item) in [2, 3]):
                    raise ValueError("Each item in sb_fixed_points_in_boundary must be a list or tuple of "
                                     "2 or 3 elements: (x, y) or (x, y, weight).")
                if not all_valid_numbers(item):
                    raise ValueError(f"All elements in sb_fixed_points_in_boundary={sb_fixed_points_in_boundary} "
                                     "must be valid numbers.")
                if len(item) == 2:
                    x_sb_fixed_points_in_boundary.append(float(item[0]))
                    y_sb_fixed_points_in_boundary.append(float(item[1]))
                    w_sb_fixed_points_in_boundary.append(10000)
                else:
                    x_sb_fixed_points_in_boundary.append(float(item[0]))
                    y_sb_fixed_points_in_boundary.append(float(item[1]))
                    w_sb_fixed_points_in_boundary.append(float(item[2]))
            x_sb_fixed_points_in_boundary = np.array(x_sb_fixed_points_in_boundary, dtype=float)
            y_sb_fixed_points_in_boundary = np.array(y_sb_fixed_points_in_boundary, dtype=float)
            w_sb_fixed_points_in_boundary = np.array(w_sb_fixed_points_in_boundary, dtype=float)

        if sb_boundary_fit is None:
            raise ValueError(f"sb_boundary_fit is None and must be one of {VALID_BOUNDARY_FITS}.")
        elif sb_boundary_fit not in VALID_BOUNDARY_FITS:
            raise ValueError(f"Invalid sb_boundary_fit: {sb_boundary_fit}. Valid options are {VALID_BOUNDARY_FITS}.")
        if sb_boundary_fit == 'piecewise':
            if sb_fixed_points_in_boundary is None:
                raise ValueError("For sb_boundary_fit='piecewise', "
                                 "sb_fixed_points_in_boundary must be provided.")
            elif len(x_sb_fixed_points_in_boundary) < 2:
                raise ValueError("For sb_boundary_fit='piecewise', "
                                 "at least two fixed points must be provided in sb_fixed_points_in_boundary.")

        # Compute offsets between each single exposure and the median image
        if sb_crosscorr_region is None:
            crossregion = None
        else:
            crossregion = tea.SliceRegion2D(sb_crosscorr_region, mode='fits')
        list_yx_offsets = []
        for i in range(num_images):
            if crossregion is None:
                list_yx_offsets.append((0.0, 0.0))
            else:
                reference_image = median2d[crossregion.python]
                moving_image = image3d[i][crossregion.python]
                yx_offsets, _, _ = phase_cross_correlation(
                    reference_image=reference_image,
                    moving_image=moving_image,
                    upsample_factor=100,
                    normalization=None  # use None to avoid artifacts with images with many cosmic rays
                )
                _logger.info("offsets for image %d: y=%+f, x=%+f", i+1, yx_offsets[0], yx_offsets[1])
                fig, axarr = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6.4*1.5, 4.8*1.5))
                axarr = axarr.flatten()
                vmin = np.min(reference_image)
                vmax = np.max(reference_image)
                tea.imshow(fig, axarr[0], reference_image, ds9mode=True, vmin=vmin, vmax=vmax,
                           aspect='auto', title='Median')
                tea.imshow(fig, axarr[1], moving_image, ds9mode=True, vmin=vmin, vmax=vmax,
                           aspect='auto', title=f'Image {i+1}')
                shifted_image2d = shift_image2d(
                    moving_image,
                    xoffset=yx_offsets[1],
                    yoffset=yx_offsets[0],
                    resampling=2
                )
                dumdiff1 = reference_image - moving_image
                dumdiff2 = reference_image - shifted_image2d
                vmin = np.percentile(dumdiff1, 5)
                vmax = np.percentile(dumdiff2, 95)
                tea.imshow(fig, axarr[2], dumdiff1, ds9mode=True, vmin=vmin, vmax=vmax,
                           aspect='auto', title=f'Median - Image {i+1}')
                tea.imshow(fig, axarr[3], dumdiff2, ds9mode=True, vmin=vmin, vmax=vmax,
                           aspect='auto', title=f'Median - Shifted Image {i+1}')
                # plt.tight_layout()
                png_filename = f'xyoffset_crosscorr_{i+1}.png'
                _logger.info(f"saving {png_filename}")
                plt.savefig(png_filename, dpi=150)
                if interactive:
                    _logger.info("Entering interactive mode (press 'q' to close plot)")
                    plt.show()
                plt.close(fig)
                list_yx_offsets.append(yx_offsets)

        # Estimate limits for the diagnostic plot
        rng = np.random.default_rng(sb_seed)  # Random number generator for reproducibility
        xdiag_min, xdiag_max, ydiag_min, ydiag_max = estimate_diagnostic_limits(
            rng=rng,
            gain=np.median(gain),  # Use median value to simplify the computation
            rnoise=np.median(rnoise),  # Use median value to simplify the computation
            maxvalue=np.max(min2d),
            num_images=num_images,
            npixels=10000
        )
        if np.min(xplot) < xdiag_min:
            xdiag_min = np.min(xplot)
        if np.max(xplot) > xdiag_max:
            xdiag_max = np.max(xplot)
        if sb_fixed_points_in_boundary is not None:
            if np.max(y_sb_fixed_points_in_boundary) > ydiag_max:
                ydiag_max = np.max(y_sb_fixed_points_in_boundary)
        ydiag_max *= 1.20  # Add 20% margin to the maximum y limit
        _logger.info("xdiag_min=%f", xdiag_min)
        _logger.info("ydiag_min=%f", ydiag_min)
        _logger.info("xdiag_max=%f", xdiag_max)
        _logger.info("ydiag_max=%f", ydiag_max)

        # Define binning for the diagnostic plot
        nbins_xdiag = 100
        nbins_ydiag = 100
        bins_xdiag = np.linspace(xdiag_min, xdiag_max, nbins_xdiag + 1)
        bins_ydiag = np.linspace(0, ydiag_max, nbins_ydiag + 1)
        xcbins = (bins_xdiag[:-1] + bins_xdiag[1:]) / 2
        ycbins = (bins_ydiag[:-1] + bins_ydiag[1:]) / 2

        # Create a 2D histogram for the diagnostic plot, using
        # integers to avoid rounding errors
        hist2d_accummulated = np.zeros((nbins_ydiag, nbins_xdiag), dtype=int)
        lam = median2d.copy()
        lam[lam < 0] = 0  # Avoid negative values
        lam3d = np.zeros((num_images, naxis2, naxis1))
        if sb_crosscorr_region is None:
            for i in range(num_images):
                lam3d[i] = lam
        else:
            _logger.info("xy-shifting median2d to speed up simulations...")
            for i in range(num_images):
                _logger.info("shifted image %d/%d -> delta_y=%+f, delta_x=%+f",
                             i + 1, num_images, -list_yx_offsets[i][0], -list_yx_offsets[i][1])
                # apply negative offsets to the median image to simulate the
                # expected individual exposures
                lam3d[i] = shift_image2d(lam,
                                         xoffset=-list_yx_offsets[i][1],
                                         yoffset=-list_yx_offsets[i][0],
                                         resampling=2)
        _logger.info("computing simulated 2D histogram...")
        for k in range(sb_nsimulations):
            time_ini = datetime.now()
            image3d_simul = np.zeros((num_images, naxis2, naxis1))
            for i in range(num_images):
                # convert from ADU to electrons to apply Poisson noise, and then back to ADU
                image3d_simul[i] = rng.poisson(lam=lam3d[i] * gain).astype(float) / gain
                # add readout noise in ADU
                image3d_simul[i] += rng.normal(loc=0, scale=rnoise)
            min2d_simul = np.min(image3d_simul, axis=0)
            median2d_simul = np.median(image3d_simul, axis=0)
            xplot_simul = min2d_simul.flatten()
            yplot_simul = median2d_simul.flatten() - min2d_simul.flatten()
            hist2d, edges = np.histogramdd(
                sample=(yplot_simul, xplot_simul),
                bins=(bins_ydiag, bins_xdiag)
            )
            hist2d_accummulated += hist2d.astype(int)
            time_end = datetime.now()
            _logger.info("simulation %d/%d, time elapsed: %s", k + 1, sb_nsimulations, time_end - time_ini)
        # Average the histogram over the number of simulations
        hist2d_accummulated = hist2d_accummulated.astype(float) / sb_nsimulations
        vmin = np.min(hist2d_accummulated[hist2d_accummulated > 0])
        if vmin == 0:
            vmin = 1
        vmax = np.max(hist2d_accummulated)
        cmap1 = plt.get_cmap('cividis_r')
        cmap2 = plt.get_cmap('viridis')
        n_colors = 256
        n_colors2 = int((np.log10(vmax) - np.log10(1.0)) / (np.log10(vmax) - np.log10(vmin)) * n_colors)
        n_colors2 += 1
        if n_colors2 > n_colors:
            n_colors2 = n_colors
        if n_colors2 < n_colors:
            n_colors1 = n_colors - n_colors2
        else:
            n_colors1 = 0
        colors1 = cmap1(np.linspace(0, 1, n_colors1))
        colors2 = cmap2(np.linspace(0, 1, n_colors2))
        combined_colors = np.vstack((colors1, colors2))
        combined_cmap = LinearSegmentedColormap.from_list('combined_cmap', combined_colors)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.1, 5.5))
        # Display 2D histogram of the simulated data
        extent = [bins_xdiag[0], bins_xdiag[-1], bins_ydiag[0], bins_ydiag[-1]]
        tea.imshow(fig, ax1, hist2d_accummulated, norm=norm, extent=extent,
                   aspect='auto', cblabel='Number of pixels', cmap=combined_cmap)
        # Display 2D histogram of the original data
        hist2d_original, edges = np.histogramdd(
            sample=(yplot, xplot),
            bins=(bins_ydiag, bins_xdiag)
        )
        tea.imshow(fig, ax2, hist2d_original, norm=norm, extent=extent,
                   aspect='auto', cblabel='Number of pixels', cmap=combined_cmap)

        # Determine the detection boundary for coincident cosmic ray detection
        _logger.info("computing numerical boundary for coincident cosmic ray detection...")
        xboundary = []
        yboundary = []
        for i in range(nbins_xdiag):
            fsum = np.sum(hist2d_accummulated[:, i])
            if fsum > 0:
                pdensity = hist2d_accummulated[:, i] / fsum
                perc = (1 - (1 / sb_nsimulations) / fsum)
                p = np.interp(perc, np.cumsum(pdensity), np.arange(nbins_ydiag))
                xboundary.append(xcbins[i])
                yboundary.append(ycbins[int(p + 0.5)])
        xboundary = np.array(xboundary)
        yboundary = np.array(yboundary)
        ax1.plot(xboundary, yboundary, 'r+')
        boundaryfit = None  # avoid flake8 warning
        if sb_boundary_fit == 'spline':
            for iterboundary in range(sb_niter_boundary_extension + 1):
                wboundary = np.ones_like(xboundary, dtype=float)
                if iterboundary == 0:
                    label = 'initial spline fit'
                else:
                    wboundary[yboundary > boundaryfit(xboundary)] = sb_weight_boundary_extension**iterboundary
                    label = f'Iteration {iterboundary}'
                if sb_fixed_points_in_boundary is None:
                    xboundary_fit = xboundary
                    yboundary_fit = yboundary
                    wboundary_fit = wboundary
                else:
                    wboundary_max = np.max(wboundary)
                    xboundary_fit = np.concatenate((xboundary, x_sb_fixed_points_in_boundary))
                    yboundary_fit = np.concatenate((yboundary, y_sb_fixed_points_in_boundary))
                    wboundary_fit = np.concatenate((wboundary, w_sb_fixed_points_in_boundary * wboundary_max))
                isort = np.argsort(xboundary_fit)
                boundaryfit, knots = spline_positive_derivative(
                    x=xboundary_fit[isort],
                    y=yboundary_fit[isort],
                    w=wboundary_fit[isort],
                    n_total_knots=sb_knots_splfit,
                )
                ydum = boundaryfit(xcbins)
                ydum[xcbins < knots[0]] = boundaryfit(knots[0])
                ydum[xcbins > knots[-1]] = boundaryfit(knots[-1])
                ax1.plot(xcbins, ydum, '-', color=f'C{iterboundary}', label=label)
                ax1.plot(knots, boundaryfit(knots), 'o', color=f'C{iterboundary}', markersize=4)
        elif sb_boundary_fit == 'piecewise':
            boundaryfit = define_piecewise_linear_function(
                xarray=x_sb_fixed_points_in_boundary,
                yarray=y_sb_fixed_points_in_boundary
            )
            ax1.plot(xcbins, boundaryfit(xcbins), 'r-', label='Piecewise linear fit')
        else:
            raise ValueError(f"Invalid sb_boundary_fit: {sb_boundary_fit}. Valid options are {VALID_BOUNDARY_FITS}.")
        if sb_fixed_points_in_boundary is not None:
            ax1.plot(x_sb_fixed_points_in_boundary, y_sb_fixed_points_in_boundary, 'ms', markersize=6, alpha=0.5,
                     label='Fixed points')
        ax1.set_xlabel(r'min2d $-$ bias')
        ax1.set_ylabel(r'median2d $-$ min2d')
        ax1.set_title(f'Simulated data (sb_nsimulations = {sb_nsimulations})')
        if sb_niter_boundary_extension > 1:
            ax1.legend(loc=4)
        xplot_boundary = np.linspace(xdiag_min, xdiag_max, 100)
        yplot_boundary = boundaryfit(xplot_boundary)
        if sb_boundary_fit == 'spline':
            # For spline fit, force the boundary to be constant outside the knots
            yplot_boundary[xplot_boundary < knots[0]] = boundaryfit(knots[0])
            yplot_boundary[xplot_boundary > knots[-1]] = boundaryfit(knots[-1])
        ax2.plot(xplot_boundary, yplot_boundary, 'r-', label='Detection boundary')
        if sb_fixed_points_in_boundary is not None:
            ax2.plot(x_sb_fixed_points_in_boundary, y_sb_fixed_points_in_boundary, 'ms', markersize=6, alpha=0.5,
                     label='Fixed points')
        ax2.set_xlim(xdiag_min, xdiag_max)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel(ax1.get_xlabel())
        ax2.set_ylabel(ax1.get_ylabel())
        ax2.set_title('Original data')
        ax2.legend(loc=4)
        plt.tight_layout()
        png_filename = 'diagnostic_histogram2d.png'
        _logger.info(f"saving {png_filename}")
        plt.savefig(png_filename, dpi=150)
        if interactive:
            _logger.info("Entering interactive mode (press 'q' to close plot)")
            plt.show()
        plt.close(fig)

        if sb_threshold is None:
            # Use the minimum value of the boundary as the sb_threshold
            sb_threshold = np.min(yplot_boundary)
            _logger.info("updated sb_threshold for cosmic ray detection: %f", sb_threshold)

        # Apply the criterium to detect coincident cosmic rays
        flag1 = yplot > boundaryfit(xplot)
        flag2 = yplot > sb_threshold
        flag_sb = np.logical_and(flag1, flag2)
        flag3 = max2d.flatten() > sb_minimum_max2d_rnoise * rnoise.flatten()
        flag_sb = np.logical_and(flag_sb, flag3)
        _logger.info("number of pixels flagged as cosmic rays by simboundary: %d", np.sum(flag_sb))
        if crmethod == 'simboundary':
            flag_la = np.zeros_like(flag_sb, dtype=bool)

    # Define the final cosmic ray flag
    if flag_la is None and flag_sb is None:
        raise RuntimeError("Both flag_la and flag_sb are None. This should never happen.")
    elif flag_la is None:
        flag = flag_sb
    elif flag_sb is None:
        flag = flag_la
    else:
        # Combine the flags from lacosmic and simboundary
        flag = np.logical_or(flag_la, flag_sb)
        _logger.info("number of pixels flagged as cosmic rays by lacosmic+simboundary: %d", np.sum(flag))
    flag = flag.reshape((naxis2, naxis1))

    # Show diagnostic plot for the cosmic ray detection
    _logger.info("generating diagnostic plot for MEDIANCR...")
    ylabel = r'median2d $-$ min2d'
    diagnostic_plot(xplot, yplot, xplot_boundary, yplot_boundary, flag_la, flag_sb,
                    sb_threshold, ylabel, interactive,
                    target2d=median2d, target2d_name='median2d',
                    min2d=min2d, mean2d=mean2d, image3d=image3d,
                    _logger=_logger, png_filename='diagnostic_mediancr.png')

    # Check if any cosmic ray was detected
    if not np.any(flag):
        _logger.info("no coincident cosmic rays detected.")
        mask_mediancr = np.zeros_like(median2d, dtype=bool)
    else:
        _logger.info("coincident cosmic rays detected...")
        # Convert the flag to an integer array for dilation
        flag_integer = flag.astype(np.uint8)
        if dilation > 0:
            _logger.info("before dilation: %d pixels flagged as coincident cosmic rays", np.sum(flag_integer))
            structure = ndimage.generate_binary_structure(2, 2)
            flag_integer_dilated = ndimage.binary_dilation(
                flag_integer,
                structure=structure,
                iterations=dilation
            ).astype(np.uint8)
            _logger.info("after dilation: %d pixels flagged as coincident cosmic rays", np.sum(flag_integer_dilated))
        else:
            flag_integer_dilated = flag_integer
            _logger.info("no dilation applied: %d pixels flagged as coincident cosmic rays", np.sum(flag_integer))
        # Set to 2 the pixels that were originally flagged as cosmic rays
        # (this is to distinguish them from the pixels that were dilated,
        # which will be set to 1)
        flag_integer_dilated[flag] = 2
        # Compute mask
        mask_mediancr = flag_integer_dilated > 0
        # Fix the median2d array by replacing the flagged pixels with the minimum value
        # of the corresponding pixel in the input arrays
        median2d_corrected = median2d.copy()
        median2d_corrected[mask_mediancr] = min2d[mask_mediancr]
        # Label the connected pixels as individual cosmic rays
        labels_cr, number_cr = ndimage.label(flag_integer_dilated > 0)
        _logger.info("number of coincident cosmic rays (connected pixels) detected: %d", number_cr)
        # Sort the cosmic rays x coordinate
        _logger.info("sorting cosmic rays by x coordinate...")
        xsort_cr = np.zeros(number_cr, dtype=float)
        for i in range(1, number_cr + 1):
            ijloc = np.argwhere(labels_cr == i)
            xsort_cr[i - 1] = np.mean(ijloc[:, 1])
        isort_cr = np.argsort(xsort_cr)
        num_plot_max = num_images
        # Determine the number of rows and columns for the plot,
        # considering that we want to plot also 3 additional images:
        # the median2d, the mask and the median2d_corrected
        if num_images == 3:
            nrows, ncols = 2, 3
            figsize = (10, 6)
        elif num_images in [4, 5]:
            nrows, ncols = 2, 4
            figsize = (13, 6)
        elif num_images == 6:
            nrows, ncols = 3, 3
            figsize = (13, 3)
        elif num_images in [7, 8, 9]:
            nrows, ncols = 3, 4
            figsize = (13, 9)
        else:
            _logger.warning("only the first 9 images will be plotted")
            nrows, ncols = 3, 4
            figsize = (13, 9)
            num_plot_max = 9
        pdf = PdfPages('mediancr_identified_cr.pdf')

        maxplots_eff = maxplots
        if maxplots_eff < 0 or verify_cr:
            maxplots_eff = number_cr
        _logger.info(f"generating {maxplots_eff} plots of coincident cosmic rays...")
        for idum in range(min(number_cr, maxplots_eff)):
            i = isort_cr[idum]
            ijloc = np.argwhere(labels_cr == i + 1)
            ic = int(np.mean(ijloc[:, 0]) + 0.5)
            jc = int(np.mean(ijloc[:, 1]) + 0.5)
            i1 = ic - semiwindow
            if i1 < 0:
                i1 = 0
            i2 = ic + semiwindow
            if i2 >= naxis2:
                i2 = naxis2 - 1
            j1 = jc - semiwindow
            if j1 < 0:
                j1 = 0
            j2 = jc + semiwindow
            if j2 >= naxis1:
                j2 = naxis1 - 1
            fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            axarr = axarr.flatten()
            # Important: use interpolation=None instead of interpolation='None' to avoid
            # having blurred images when opening the PDF file with macos Preview
            cmap = 'viridis'
            cblabel = 'Number of counts'
            if color_scale == 'zscale':
                vmin, vmax = tea.zscale(median2d[i1:(i2+1), j1:(j2+1)])
            else:
                vmin = np.min(median2d[i1:(i2+1), j1:(j2+1)])
                vmax = np.max(median2d[i1:(i2+1), j1:(j2+1)])
            for k in range(num_plot_max):
                ax = axarr[k]
                title = title = f'image#{k+1}/{num_images}'
                tea.imshow(fig, ax, image3d[k][i1:(i2+1), j1:(j2+1)], vmin=vmin, vmax=vmax,
                           extent=[j1-0.5, j2+0.5, i1-0.5, i2+0.5],
                           title=title, cmap=cmap, cblabel=cblabel, interpolation=None)
            for k in range(3):
                ax = axarr[k + num_plot_max]
                cmap = 'viridis'
                if k == 0:
                    image2d = median2d
                    title = 'median'
                elif k == 1:
                    image2d = flag_integer_dilated
                    title = 'flag_integer_dilated'
                    cmap = 'plasma'
                    cblabel = 'flag'
                elif k == 2:
                    image2d = median2d_corrected
                    title = 'median corrected'
                else:
                    raise ValueError(f'Unexpected {k=}')
                if k in [0, 2]:
                    if color_scale == 'zscale':
                        vmin_, vmax_ = tea.zscale(image2d[i1:(i2+1), j1:(j2+1)])
                    else:
                        vmin_ = np.min(image2d[i1:(i2+1), j1:(j2+1)])
                        vmax_ = np.max(image2d[i1:(i2+1), j1:(j2+1)])
                else:
                    vmin_, vmax_ = 0, 2
                tea.imshow(fig, ax, image2d[i1:(i2+1), j1:(j2+1)], vmin=vmin_, vmax=vmax_,
                           extent=[j1-0.5, j2+0.5, i1-0.5, i2+0.5],
                           title=title, cmap=cmap, cblabel=cblabel, interpolation=None)
            nplot_missing = nrows * ncols - num_plot_max - 3
            if nplot_missing > 0:
                for k in range(nplot_missing):
                    ax = axarr[-k-1]
                    ax.axis('off')
            fig.suptitle(f'CR#{idum+1}/{number_cr}')
            plt.tight_layout()
            plt.show(block=False)
            if verify_cr:
                accept_cr = input(f"Accept this cosmic ray detection #{idum+1} ([y]/n)? ")
                if accept_cr.lower() == 'n':
                    _logger.info("removing cosmic ray detection #%d from the mask\n", idum + 1)
                    mask_mediancr[labels_cr == i + 1] = False
                else:
                    _logger.info("keeping cosmic ray detection #%d in the mask", idum + 1)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        pdf.close()
        _logger.info("plot generation complete")
        _logger.info("saving mediancr_identified_cr.pdf")

    # Generate list of HDUs with masks
    hdu_mediancr = fits.ImageHDU(mask_mediancr.astype(np.uint8), name='MEDIANCR')
    list_hdu_masks = [hdu_mediancr]

    # Apply the same algorithm but now with mean2d and with each individual array
    for i, target2d in enumerate([mean2d] + list_arrays):
        if i == 0:
            _logger.info("detecting cosmic rays in the mean2d image...")
            target2d_name = 'mean2d'
        else:
            _logger.info(f"detecting cosmic rays in array {i}...")
            target2d_name = f'single exposure #{i}'
        if crmethod in ['lacosmic', 'sb_lacosmic']:
            if la_fsmode == 'median':
                array_lacosmic, flag_la = cosmicray_lacosmic(
                    ccd=target2d,
                    gain=gain_scalar,
                    readnoise=rnoise_scalar,
                    sigclip=la_sigclip,
                    fsmode='median'
                )
            elif la_fsmode == 'convolve':
                if la_psfmodel != 'gaussxy':
                    array_lacosmic, flag_la = cosmicray_lacosmic(
                        ccd=target2d,
                        gain=gain_scalar,
                        readnoise=rnoise_scalar,
                        sigclip=la_sigclip,
                        fsmode='convolve',
                        psfk=None,
                        psfmodel=la_psfmodel,
                    )
                else:
                    array_lacosmic, flag_la = cosmicray_lacosmic(
                        ccd=target2d,
                        gain=gain_scalar,
                        readnoise=rnoise_scalar,
                        sigclip=la_sigclip,
                        fsmode='convolve',
                        psfk=gausskernel2d_elliptical(fwhm_x=la_psffwhm_x, fwhm_y=la_psffwhm_y, kernsize=la_psfsize)
                    )
            else:
                raise ValueError("la_fsmode must be 'median' or 'convolve'.")
            # For the mean2d array, update the flag with the user masks
            if i == 0:
                update_flag_with_user_masks(flag, pixels_to_be_masked, pixels_to_be_excluded, _logger)
            flag_la = flag_la.flatten()
            if crmethod == 'lacosmic':
                xplot_boundary = None
                yplot_boundary = None
                sb_threshold = None
                flag_sb = np.zeros_like(flag_la, dtype=bool)
        if crmethod in ['simboundary', 'sb_lacosmic']:
            xplot = min2d.flatten()
            yplot = target2d.flatten() - min2d.flatten()
            flag1 = yplot > boundaryfit(xplot)
            flag2 = yplot > sb_threshold
            flag_sb = np.logical_and(flag1, flag2)
            flag3 = max2d.flatten() > sb_minimum_max2d_rnoise * rnoise.flatten()
            flag_sb = np.logical_and(flag_sb, flag3)
            if crmethod == 'simboundary':
                flag_la = np.zeros_like(flag_sb, dtype=bool)
        # For the mean2d mask, force the flag to be True if the pixel
        # was flagged as a coincident cosmic ray when using the median2d array
        # (this is to ensure that all pixels flagged in MEDIANCR are also
        # flagged in MEANCRT)
        if i == 0:
            flag_la = np.logical_or(flag_la, list_hdu_masks[0].data.astype(bool).flatten())
            flag_sb = np.logical_or(flag_sb, list_hdu_masks[0].data.astype(bool).flatten())

        # For the individual array masks, force the flag to be True if the pixel
        # is flagged both in the individual exposure and in the mean2d array
        if i > 0:
            flag_la = np.logical_and(flag_la, list_hdu_masks[1].data.astype(bool).flatten())
            flag_sb = np.logical_and(flag_sb, list_hdu_masks[1].data.astype(bool).flatten())
        _logger.info("number of pixels flagged as cosmic rays (lacosmic)...: %d", np.sum(flag_la))
        _logger.info("number of pixels flagged as cosmic rays (simboundary): %d", np.sum(flag_sb))
        if i == 0:
            _logger.info("generating diagnostic plot for MEANCRT...")
            png_filename = 'diagnostic_meancr.png'
            ylabel = r'mean2d $-$ min2d'
        else:
            _logger.info(f"generating diagnostic plot for CRMASK{i}...")
            png_filename = f'diagnostic_crmask{i}.png'
            ylabel = f'array{i}' + r' $-$ min2d'
        diagnostic_plot(xplot, yplot, xplot_boundary, yplot_boundary, flag_la, flag_sb,
                        sb_threshold, ylabel, interactive,
                        target2d=target2d, target2d_name=target2d_name,
                        min2d=min2d, mean2d=mean2d, image3d=image3d,
                        _logger=_logger, png_filename=png_filename)
        flag = np.logical_or(flag_la, flag_sb)
        flag = flag.reshape((naxis2, naxis1))
        flag_integer = flag.astype(np.uint8)
        if dilation > 0:
            _logger.info("before dilation: %d pixels flagged as cosmic rays", np.sum(flag_integer))
            structure = ndimage.generate_binary_structure(2, 2)
            flag_integer_dilated = ndimage.binary_dilation(
                flag_integer,
                structure=structure,
                iterations=dilation
            ).astype(np.uint8)
            _logger.info("after dilation: %d pixels flagged as cosmic rays", np.sum(flag_integer_dilated))
        else:
            flag_integer_dilated = flag_integer
            _logger.info("no dilation applied: %d pixels flagged as cosmic rays", np.sum(flag_integer))
        flag_integer_dilated[flag] = 2
        # Compute mask
        mask = flag_integer_dilated > 0
        if i == 0:
            name = 'MEANCRT'
        else:
            name = f'CRMASK{i}'
        hdu_mask = fits.ImageHDU(mask.astype(np.uint8), name=name)
        list_hdu_masks.append(hdu_mask)

    # Find pixels masked in all individual CRMASKi
    mask_all = np.ones((naxis2, naxis1), dtype=bool)
    for hdu in list_hdu_masks[2:]:
        mask_all = np.logical_and(mask_all, hdu.data.astype(bool))
    problematic_pixels = np.argwhere(mask_all)
    _logger.info("number of pixels masked in all individual CRMASKi: %d", len(problematic_pixels))
    if len(problematic_pixels) > 0:
        # Label the connected problematic pixels as individual problematic cosmic rays
        labels_cr, number_cr = ndimage.label(mask_all)
        _logger.info("number of connected problematic pixel regions: %d", number_cr)
        # Sort the problematic regions by x coordinate
        _logger.info("sorting problematic regions by x coordinate...")
        xsort_cr = np.zeros(number_cr, dtype=float)
        for i in range(1, number_cr + 1):
            ijloc = np.argwhere(labels_cr == i)
            xsort_cr[i - 1] = np.mean(ijloc[:, 1])
        isort_cr = np.argsort(xsort_cr)
        # print the coordinates of the problematic pixels
        for idum in range(number_cr):
            i = isort_cr[idum]
            ijloc = np.argwhere(labels_cr == i + 1)
            _logger.info("reg. #%d: no. of pixels = %d, (x, y) FITS-pixel = (%.1f, %.1f)",
                         idum + 1, len(ijloc), np.mean(ijloc[:, 1]) + 1, np.mean(ijloc[:, 0]) + 1)

    # Generate output HDUList with masks
    args = inspect.signature(compute_crmasks).parameters
    if crmethod == 'lacosmic':
        prefix_of_excluded_args = 'sb_'
    elif crmethod == 'simboundary':
        prefix_of_excluded_args = 'la_'
    else:
        prefix_of_excluded_args = 'xxx'
    filtered_args = {k: v for k, v in locals().items() if
                     k in args and
                     k not in ['list_arrays'] and
                     k[:3] != prefix_of_excluded_args}
    hdu_primary = fits.PrimaryHDU()
    hdu_primary.header['UUID'] = str(uuid.uuid4())
    for i, fluxf in enumerate(flux_factor):
        hdu_primary.header[f'FLUXF{i+1}'] = fluxf
    hdu_primary.header.add_history(f"CRMasks generated by {__name__}")
    hdu_primary.header.add_history(f"at {datetime.now().isoformat()}")
    for key, value in filtered_args.items():
        if isinstance(value, np.ndarray):
            if np.unique(value).size == 1:
                value = value.flatten()[0]
            elif value.ndim == 1 and len(value) == num_images:
                value = str(value.tolist())
            else:
                value = f'array_shape: {value.shape}'
        elif isinstance(value, list):
            value = str(value)
        hdu_primary.header.add_history(f"- {key} = {value}")

    hdul_masks = fits.HDUList([hdu_primary] + list_hdu_masks)
    return hdul_masks


def apply_crmasks(list_arrays, hdul_masks=None, combination=None,
                  dtype=np.float32, apply_flux_factor=True, bias=None):
    """
    Correct cosmic rays applying previously computed masks.

    The input arrays are bias subtracted, and optionally re-scaled by
    a flux factor read from the header of `hdul_masks`. Then, they are
    combined using the specified combination method.

    Parameters
    ----------
    list_arrays : list of 2D arrays
        The input arrays to be combined.
    hdul_masks : HDUList
        The HDUList containing the mask arrays for cosmic ray removal.
        The mask for the mediancr combination should be in
        the 'MEDIANCR' extension.
    combination : str
        The type of combination to apply. There are three options:
        - 'mediancr', the median combination is applied, and masked pixels
        (those equal to 1 in extension 'MEDIANCR' of `hdul_masks`) are
        replaced by the minimum value of the corresponding pixel in the
        input arrays.
        - 'meancrt', the mean combination is applied, and masked pixels
        (those equal to 1 in extension 'MEANCRT' of `hdul_masks`) are
        replaced by the mediancr value.
        - 'meancr', the mean combination is applied making use of
        the individual mask of each image (extensions 'CRMASK1', 'CRMASK2',
        etc. in `hdul_masks`). Those pixels that are masked in all the individual
        images are replaced by the minimum value of the corresponding pixel
        in the input arrays.
    apply_flux_factor : bool, optional
        If True, the flux factor is applied to the input arrays before
        combining them. The flux factor is read from the header of the
        `hdul_masks` (keywords 'FLUXF1', 'FLUXF2', etc.). Default is True.
    bias : float or 2D array, optional
        The bias level to be subtracted from the input arrays. If a float is
        provided, it is assumed to be constant for all pixels. If a 2D array
        is provided, it must have the same shape as the input arrays. If None,
        the bias is assumed to be zero for all pixels. Default is None.
    dtype : data-type, optional
        The desired data type for the output arrays (default is np.float32).

    Returns
    -------
    combined2d: 2D array
        The combined array with masked pixels replaced accordingly
        depending on the combination method.
    variance2d : 2D array
        The variance of the input arrays along the first axis.
    map2d : 2D array
        The number of input pixels used to compute the median at each pixel.
    """

    _logger = logging.getLogger(__name__)

    # Check that the input is a list
    if not isinstance(list_arrays, list):
        raise TypeError("Input must be a list of arrays.")

    # Check that the combination method is valid
    if combination not in VALID_COMBINATIONS:
        raise ValueError(f"Combination: {combination} must be one of {VALID_COMBINATIONS}.")

    # Check that the list contains numpy 2D arrays
    if not all(isinstance(array, np.ndarray) and array.ndim == 2 for array in list_arrays):
        raise ValueError("All elements in the list must be 2D numpy arrays.")

    # Check that the list contains at least 3 arrays
    num_images = len(list_arrays)
    if num_images < 3:
        raise ValueError("At least 3 images are required for a useful combination.")

    # Check that all arrays have the same shape
    for i, array in enumerate(list_arrays):
        if array.shape != list_arrays[0].shape:
            raise ValueError(f"Array {i} has a different shape than the first array.")
    naxis2, naxis1 = list_arrays[0].shape

    # Log the number of input arrays and their shapes
    _logger.info("number of input arrays: %d", len(list_arrays))
    for i, array in enumerate(list_arrays):
        _logger.info("array %d shape: %s, dtype: %s", i, array.shape, array.dtype)

    # Define the bias
    if bias is None:
        bias = np.zeros((naxis2, naxis1), dtype=float)
        _logger.info("bias not defined, assuming bias=0.0 for all pixels.")
    elif isinstance(bias, (float, int)):
        bias = np.full((naxis2, naxis1), bias, dtype=float)
        _logger.info("bias defined as a constant value: %f", bias[0, 0])
    elif isinstance(bias, np.ndarray):
        if bias.shape != (naxis2, naxis1):
            raise ValueError(f"bias must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("bias defined as a 2D array with shape: %s", bias.shape)
    else:
        raise TypeError(f"Invalid type for bias: {type(bias)}. Must be float, int, or numpy array.")

    # Read the flux factor from the masks
    _logger.info("apply_flux_factor: %s", apply_flux_factor)
    if apply_flux_factor:
        flux_factor = []
        for i in range(num_images):
            flux_factor.append(hdul_masks[0].header[f'FLUXF{i+1}'])
        flux_factor = np.array(flux_factor, dtype=float)
        _logger.info("flux factor values: %s", str(flux_factor))
    else:
        flux_factor = np.ones(num_images, dtype=float)

    # Convert the list of arrays to a 3D numpy array
    shape3d = (num_images, naxis2, naxis1)
    image3d = np.zeros(shape3d, dtype=dtype)
    _logger.info("applying bias and flux factors to input arrays...")
    for i, array in enumerate(list_arrays):
        image3d[i] = (array.astype(dtype) - bias) / flux_factor[i]

    # Compute minimum and median along the first axis of image3d
    min2d_rescaled = np.min(image3d, axis=0)
    median2d = np.median(image3d, axis=0)

    # Apply the requested combination method
    _logger.info("applying combination method: %s", combination)
    _logger.info("using crmasks in %s", hdul_masks[0].header['UUID'])
    if combination in ['mediancr', 'meancrt']:
        # Define the mask_mediancr
        mask_mediancr = hdul_masks['MEDIANCR'].data.astype(bool)
        _logger.info("applying mask MEDIANCR: %d masked pixels", np.sum(mask_mediancr))
        # Replace the masked pixels with the minimum value
        # of the corresponding pixel in the input arrays
        median2d_corrected = median2d.copy()
        median2d_corrected[mask_mediancr] = min2d_rescaled[mask_mediancr]

    if combination == 'mediancr':
        combined2d = median2d_corrected
        # Define the variance and map arrays
        variance2d = np.var(image3d, axis=0, ddof=1)
        variance2d[mask_mediancr] = 0.0  # Set variance to 0 for the masked pixels
        map2d = np.ones((naxis2, naxis1), dtype=int) * num_images
        map2d[mask_mediancr] = 1  # Set the map to 1 for the masked pixels
    elif combination == 'meancrt':
        # Define the mask_meancr
        mask_meancrt = hdul_masks['MEANCRT'].data.astype(bool)
        _logger.info("applying mask MEANCRT: %d masked pixels", np.sum(mask_meancrt))
        mean2d = np.mean(image3d, axis=0)
        # Replace the masked pixels in mean2d with the median2d_corrected value
        mean2d_corrected = mean2d.copy()
        mean2d_corrected[mask_meancrt] = median2d_corrected[mask_meancrt]
        combined2d = mean2d_corrected
        # Define the variance and map arrays
        variance2d = np.var(image3d, axis=0, ddof=1)
        variance2d[mask_meancrt] = 0.0  # Set variance to 0 for the masked pixels
        map2d = np.ones((naxis2, naxis1), dtype=int) * num_images
        map2d[mask_meancrt] = 1  # Set the map to 1 for the masked pixels
    elif combination == 'meancr':
        image3d_masked = ma.array(
            np.zeros(shape3d, dtype=dtype),
            mask=np.full(shape3d, fill_value=True, dtype=bool)
        )
        # Loop through each image and apply the corresponding mask
        total_mask = np.zeros((naxis2, naxis1), dtype=int)
        for i in range(num_images):
            image3d_masked[i, :, :] = list_arrays[i].astype(dtype) * flux_factor[i]
            mask = hdul_masks[f'CRMASK{i+1}'].data
            _logger.info("applying mask %s: %d masked pixels", f'CRMASK{i+1}', np.sum(mask))
            total_mask += mask.astype(int)
            image3d_masked[i, :, :].mask = mask.astype(bool)
        # Compute the mean of the masked 3D array
        combined2d = ma.mean(image3d_masked, axis=0).data
        # Replace pixels without data with the minimum value
        mask_nodata = total_mask == num_images
        if np.any(mask_nodata):
            _logger.info("replacing %d pixels without data by the minimum value", np.sum(mask_nodata))
            combined2d[mask_nodata] = min2d_rescaled[mask_nodata]
        else:
            _logger.info("no pixels without data found, no replacement needed")
        # Define the variance and map arrays
        variance2d = ma.var(image3d_masked, axis=0, ddof=1).data
        map2d = np.ones((naxis2, naxis1), dtype=int) * num_images - total_mask
    else:
        raise ValueError(f"Invalid combination method: {combination}. "
                         f"Valid options are {VALID_COMBINATIONS}.")

    return combined2d.astype(dtype), variance2d.astype(dtype), map2d


def main(args=None):
    """
    Main function to compute and apply CR masks.
    """
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG, WARNING, ERROR, CRITICAL
        format='%(name)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting mediancr combination...")

    parser = argparse.ArgumentParser(
        description="Combine 2D arrays using mediancr, meancrt or meancr methods."
    )

    parser.add_argument("inputyaml",
                        help="Input YAML file.",
                        type=str)
    parser.add_argument("--verbose",
                        help="Increase output verbosity",
                        action="store_true")
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    # Read parameters from YAML file
    with open(args.inputyaml, 'rt') as fstream:
        input_params = yaml.safe_load(fstream)
    if args.verbose:
        logger.info(f'{input_params=}')

    # Check that mandatory parameters are present
    if 'images' not in input_params:
        raise ValueError("'images' must be provided in input YAML file.")
    else:
        list_of_fits_files = input_params['images']
        if not isinstance(list_of_fits_files, list) or len(list_of_fits_files) < 3:
            raise ValueError("'images' must be a list of at least 3 FITS files.")
        for file in list_of_fits_files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"File {file} not found.")
            else:
                logger.info("found input file: %s", file)
    for item in ['gain', 'rnoise', 'bias']:
        if item not in input_params:
            raise ValueError(f"'{item}' must be provided in input YAML file.")

    # Default values for missing parameters in input YAML file
    if 'extnum' in input_params:
        extnum = int(input_params['extnum'])
    else:
        extnum = 0
        logger.info("extnum not provided, assuming extnum=0")

    # Read the input list of files, which should contain paths to 2D FITS files,
    # and load the arrays from the specified extension number.
    list_arrays = [fits.getdata(file, extnum=extnum) for file in input_params['images']]

    # Check if the list is empty
    if not list_arrays:
        raise ValueError("The input list is empty. Please provide a valid list of 2D arrays.")

    # Check that the requirements are provided
    if 'requirements' not in input_params:
        raise ValueError("'requirements' must be provided in input YAML file.")
    requirements = input_params['requirements']
    if not isinstance(requirements, dict):
        raise ValueError("'requirements' must be a dictionary.")
    if not requirements:
        raise ValueError("'requirements' dictionary is empty.")

    # Define parameters for compute_crmasks
    crmasks_params = dict()
    for key in ['gain', 'rnoise', 'bias']:
        crmasks_params[key] = input_params[key]
    for item in input_params['requirements']:
        crmasks_params[item] = input_params['requirements'][item]

    # Compute the different cosmic ray masks
    hdul_masks = compute_crmasks(
        list_arrays=list_arrays,
        **crmasks_params
    )

    # Save the cosmic ray masks to a FITS file
    output_masks = 'crmasks.fits'
    logger.info("Saving cosmic ray masks to %s", output_masks)
    hdul_masks.writeto(output_masks, overwrite=True)
    logger.info("Cosmic ray masks saved")

    # Apply cosmic ray masks
    for combination in VALID_COMBINATIONS:
        logger.info("Applying cosmic ray masks using combination method: %s", combination)
        output_combined = f'combined_{combination}.fits'
        combined, variance, maparray = apply_crmasks(
            list_arrays=list_arrays,
            bias=input_params['bias'],
            hdul_masks=hdul_masks,
            combination=combination,
            dtype=np.float32
        )
        # Save the combined array, variance, and map to a FITS file
        logger.info("Saving combined (bias subtracted) array, variance, and map to %s", output_combined)
        hdu_combined = fits.PrimaryHDU(combined.astype(np.float32))
        add_script_info_to_fits_history(hdu_combined.header, args)
        hdu_combined.header.add_history('Contents of --inputlist:')
        for item in list_of_fits_files:
            hdu_combined.header.add_history(f'- {item}')
        hdu_combined.header.add_history(f"Masks UUID: {hdul_masks[0].header['UUID']}")
        hdu_variance = fits.ImageHDU(variance.astype(np.float32), name='VARIANCE')
        hdu_map = fits.ImageHDU(maparray.astype(np.int16), name='MAP')
        hdul = fits.HDUList([hdu_combined, hdu_variance, hdu_map])
        hdul.writeto(output_combined, overwrite=True)
        logger.info("Combined (bias subtracted) array, variance, and map saved")


if __name__ == "__main__":

    main()
