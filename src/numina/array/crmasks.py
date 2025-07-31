#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Median combination of arrays avoiding multiple cosmic rays in the same pixel."""
import ast
import inspect
import logging
import sys
import uuid

import argparse
from astropy.io import fits
from datetime import datetime
import math
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import ndimage

from numina.array.numsplines import spline_positive_derivative
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
import teareduce as tea

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
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))
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
                    ax2.plot(xbin[int(p+0.5)], ybin[i], '+', color=f'C{side}')
                    xfit.append(xbin[int(p+0.5)])
                    yfit.append(ybin[i])
            xfit = np.array(xfit)
            yfit = np.array(yfit)
            splfit = tea.AdaptiveLSQUnivariateSpline(yfit, xfit, t=2, adaptive=True)
            ax2.plot(splfit(yfit), yfit, f'C{side}-')
            imax = np.argmax(splfit(yfit)) + imin
            ymode[side] = ybin[imax]
            xmode[side] = splfit(ymode[side])
            ax2.plot(xmode[side], ymode[side], f'C{side}o')
        if xmode[0] > xmode[1]:
            imode = 0
        else:
            imode = 1
        ax2.axhline(ymode[imode], color=f'C{imode}')
        ax2.text(xbin[-5], ymode[imode]+(ybin[-1]-ybin[0])/40, f'{ymode[imode]:.3f}', color=f'C{imode}', ha='right')
        flux_factor.append(ymode[imode])
        plt.tight_layout()

        png_filename = f'flux_factor{idata+1}.png'
        plt.savefig(png_filename, dpi=150)
        if interactive:
            _logger.info("Entering interactive mode (press 'q' to close plot).")
            plt.show()
        plt.close(fig)

    if len(flux_factor) != naxis3:
        raise ValueError(f"Expected {naxis3} flux factors, but got {len(flux_factor)}.")

    # round the flux factor to 6 decimal places to avoid
    # unnecessary precision when writting to the FITS header
    flux_factor = np.round(flux_factor, decimals=6)
    return flux_factor


def estimate_diagnostic_limits(rng, gain, rnoise, maxvalue, num_images, flux_factor, nsimulations):
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
    flux_factor : numpy.ndarray
        Flux factor for the images.
    nsimulations : int
        Number of simulations to perform for each corner of the
        diagnostic plot.
    """
    if len(flux_factor) != num_images:
        raise ValueError(f"flux_factor must have the same length as num_images ({num_images}).")

    xdiag_min = np.zeros(nsimulations, dtype=float)
    xdiag_max = np.zeros(nsimulations, dtype=float)
    ydiag_min = np.zeros(nsimulations, dtype=float)
    ydiag_max = np.zeros(nsimulations, dtype=float)
    for i in range(nsimulations):
        # lower limits
        data = rng.normal(loc=0, scale=rnoise, size=num_images)
        min1d = np.min(data)
        median1d = np.median(data)
        xdiag_min[i] = median1d
        ydiag_min[i] = median1d - min1d
        # upper limits
        lam = np.array([maxvalue] * num_images)
        data = rng.poisson(lam=lam * gain).astype(float) / gain * flux_factor
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


def diagnostic_plot(xplot, yplot, xplot_boundary, yplot_boundary, flag,
                    threshold, ylabel, interactive, _logger, png_filename):
    """Diagnostic plot for the mediancr function.
    """
    if png_filename is None:
        raise ValueError("png_filename must be provided for diagnostic plots.")
    fig, ax = plt.subplots()
    ax.plot(xplot, yplot, 'C0,')
    xmin, xmax = ax.get_xlim()
    ax.plot(xplot_boundary, yplot_boundary, 'C1-', label='Exclusion boundary')
    ax.axhline(threshold, color='gray', linestyle=':', label=f'Threshold ({threshold:.2f})')
    ax.plot(xplot[flag], yplot[flag], 'rx', label=f'Suspected pixels ({np.sum(flag)})')
    ax.set_xlim(xmin, xmax)
    ymin = 0
    ymax = np.max(yplot_boundary)
    dy = ymax - ymin
    ymin -= dy * 0.05
    ymax += dy * 0.05
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'min2d $-$ bias')  # the bias was subtracted from the input arrays
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2)
    plt.tight_layout()
    _logger.info(f"saving {png_filename}.")
    plt.savefig(png_filename, dpi=150)
    if interactive:
        _logger.info("Entering interactive mode (press 'q' to close plot).")
        plt.show()
    plt.close(fig)


def compute_crmasks(
        list_arrays,
        gain=None,
        rnoise=None,
        bias=None,
        flux_factor=None,
        knots_splfit=3,
        nsimulations=10,
        niter_boundary_extension=3,
        weight_boundary_extension=10.0,
        threshold=None,
        minimum_max2d_rnoise=5.0,
        interactive=True,
        dilation=1,
        dtype=np.float32,
        seed=None,
        plots=False,
        semiwindow=15,
        color_scale='minmax',
        maxplots=10
        ):
    """
    Computation of cosmic rays masks using several equivalent exposures.

    This function computes cosmic ray masks from a list of 2D numpy arrays.
    The cosmic ray detection is based on a boundary that is derived numerically
    making use of the provided gain and readout noise values. The function
    also supports generating diagnostic plots to visualize the cosmic ray
    detection process.

    Parameters
    ----------
    list_arrays : list of 2D arrays
        The input arrays to be combined.
    gain : 2D array, float or None
        The gain value (in e/ADU) of the detector.
        If None, it is assumed to be 1.0.
    rnoise : 2D array, float or None
        The readout noise (in ADU) of the detector.
        If None, it is assumed to be 0.0.
    bias : 2D array, float or None
        The bias value (in ADU) of the detector.
        If None, it is assumed to be 0.0.
    flux_factor : str, list or None, optional
        The flux scaling factor for each exposure (default is None).
        If 'auto', the flux factor is determined automatically.
        If a list is provided, it should contain a value
        for each single image in `list_arrays`.
    knots_splfit : int, optional
        The number of knots for the spline fit to the boundary.
    nsimulations : int, optional
        The number of simulations of the each image to compute
        the exclusion boundary.
    niter_boundary_extension : int, optional
        The number of iterations for the boundary extension.
    weight_boundary_extension : float, optional
        The weight for the boundary extension.
        In each iteration, the boundary is extended by applying an
        extra weight to the points above the previous boundary. This
        extra weight is computed as `weight_boundary_extension**iter`,
        where `iter` is the current iteration number (starting from 1).
    threshold: float, optional
        Minimum threshold for median2d - min2d to consider a pixel as a
        cosmic ray (default is None). If None, the threshold is computed
        automatically from the minimum boundary value in the numerical
        simulations.
    minimum_max2d_rnoise : float, optional
        Minimum value for max2d in readout noise units to flag a pixel
        as a double cosmic ray.
    interactive : bool, optional
        If True, enable interactive mode for plots.
    dilation : int, optional
        The dilation factor for the double cosmic ray mask.
    dtype : data-type, optional
        The desired data type to build the 3D stack (default is np.float32).
    seed : int or None, optional
        The random seed for reproducibility.
    plots : bool, optional
        If True, generate plots with detected double cosmic rays
        (default is False).
    semiwindow : int, optional
        The semiwindow size to plot the double cosmic rays (default is 15).
        Only used if `plots` is True.
    color_scale : str, optional
        The color scale to use for the plots (default is 'minmax').
        Valid options are 'minmax' and 'zscale'.
    maxplots : int, optional
        The maximum number of double cosmic rays to plot (default is 10).
        If negative, all detected cosmic rays will be plotted.
        Only used if `plots` is True.

    Returns
    -------
    hdul_masks : hdulist
        The HDUList containing the mask arrays for cosmic ray
        removal using different methods. The primary HDU only contains
        information about the parameters used to determine the
        suspected pixels. The extensions are:
        - 'MEDIANCR': Mask for double cosmic rays detected using the
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
    if gain is None:
        gain = np.ones((naxis2, naxis1), dtype=float)
        _logger.info("gain not defined, assuming gain=1.0 for all pixels.")
    elif isinstance(gain, (float, int)):
        gain = np.full((naxis2, naxis1), gain, dtype=float)
        _logger.info("gain defined as a constant value: %f", gain[0, 0])
    elif isinstance(gain, np.ndarray):
        if gain.shape != (naxis2, naxis1):
            raise ValueError(f"gain must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("gain defined as a 2D array with shape: %s", gain.shape)
    else:
        raise TypeError(f"Invalid type for gain: {type(gain)}. Must be float, int, or numpy array.")

    # Define the readout noise
    if rnoise is None:
        rnoise = np.zeros((naxis2, naxis1), dtype=float)
        _logger.info("readout noise not defined, assuming readout noise=0.0 for all pixels.")
    elif isinstance(rnoise, (float, int)):
        rnoise = np.full((naxis2, naxis1), rnoise, dtype=float)
        _logger.info("readout noise defined as a constant value: %f", rnoise[0, 0])
    elif isinstance(rnoise, np.ndarray):
        if rnoise.shape != (naxis2, naxis1):
            raise ValueError(f"rnoise must have the same shape as the input arrays ({naxis2=}, {naxis1=}).")
        _logger.info("readout noise defined as a 2D array with shape: %s", rnoise.shape)
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

    # Check flux_factor
    if flux_factor is None:
        flux_factor = np.ones(num_images, dtype=float)
    if isinstance(flux_factor, str):
        if flux_factor.lower() == 'auto':
            pass  # flux_factor will be set later
        elif flux_factor.lower() == 'none':
            flux_factor = np.ones(num_images, dtype=float)
        elif isinstance(ast.literal_eval(flux_factor), list):
            flux_factor = ast.literal_eval(flux_factor)
            if len(flux_factor) != num_images:
                raise ValueError(f"flux_factor must have the same length as the number of images ({num_images}).")
            if not all_valid_numbers(flux_factor):
                raise ValueError(f"All elements in flux_factor={flux_factor} must be valid numbers.")
            flux_factor = np.array(flux_factor, dtype=float)
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

    # Check that color_scale is valid
    if color_scale not in ['minmax', 'zscale']:
        raise ValueError(f"Invalid color_scale: {color_scale}. Valid options are 'minmax' and 'zscale'.")

    # Log the input parameters
    _logger.info("flux_factor: %s", str(flux_factor))
    _logger.info("knots for spline fit to the boundary: %d", knots_splfit)
    _logger.info("number of simulations to compute the exclusion boundary: %d", nsimulations)
    _logger.info("threshold for double cosmic ray detection: %s", threshold if threshold is not None else "None")
    _logger.info("minimum max2d in rnoise units for double cosmic ray detection: %f", minimum_max2d_rnoise)
    _logger.info("niter for boundary extension: %d", niter_boundary_extension)
    _logger.info("weight for boundary extension: %f", weight_boundary_extension)
    _logger.info("dtype for output arrays: %s", dtype)
    _logger.info("random seed for reproducibility: %s", str(seed))
    _logger.info("dilation factor: %d", dilation)
    if plots:
        _logger.info("semiwindow size for plotting double cosmic rays: %d", semiwindow)
        _logger.info("maximum number of double cosmic rays to plot: %d", maxplots)
        _logger.info("color scale for plots: %s", color_scale)

    # Convert the list of arrays to a 3D numpy array
    image3d = np.zeros((num_images, naxis2, naxis1), dtype=dtype)
    for i, array in enumerate(list_arrays):
        image3d[i] = array.astype(dtype)

    # Subtract the bias from the input arrays
    _logger.info("subtracting bias from the input arrays")
    image3d -= bias

    # Compute minimum, maximum, median, mean and variance along the first axis
    min2d = np.min(image3d, axis=0)
    max2d = np.max(image3d, axis=0)
    median2d = np.median(image3d, axis=0)
    mean2d = np.mean(image3d, axis=0)
    xplot = min2d.flatten()
    yplot = median2d.flatten() - min2d.flatten()

    # If flux_factor is 'auto', compute the corresponding values
    if isinstance(flux_factor, str) and flux_factor.lower() == 'auto':
        _logger.info("flux_factor set to 'auto', computing values...")
        flux_factor = compute_flux_factor(image3d, median2d, _logger, interactive)
        _logger.info("flux_factor set to %s", str(flux_factor))

    # Estimate limits for the diagnostic plot
    rng = np.random.default_rng(seed)  # Random number generator for reproducibility
    xdiag_min, xdiag_max, ydiag_min, ydiag_max = estimate_diagnostic_limits(
        rng=rng,
        gain=np.median(gain),  # Use median value to simplify the computation
        rnoise=np.median(rnoise),  # Use median value to simplify the computation
        maxvalue=np.max(min2d),
        num_images=num_images,
        flux_factor=flux_factor,
        nsimulations=10000
    )
    if np.min(xplot) < xdiag_min:
        xdiag_min = np.min(xplot)
    if np.max(xplot) > xdiag_max:
        xdiag_max = np.max(xplot)
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
    _logger.info("computing simulated 2D histogram...")
    for k in range(nsimulations):
        time_ini = datetime.now()
        image3d_simul = np.zeros((num_images, naxis2, naxis1))
        for i in range(num_images):
            image3d_simul[i] = rng.poisson(lam=lam * flux_factor[i] * gain).astype(float) / gain
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
        _logger.info("simulation %d/%d, time elapsed: %s", k + 1, nsimulations, time_end - time_ini)
    # Average the histogram over the number of simulations
    hist2d_accummulated = hist2d_accummulated.astype(float) / nsimulations
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
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10.1, 4.6))
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

    # Determine the exclusion boundary for double cosmic ray detection
    _logger.info("computing numerical boundary for double cosmic ray detection...")
    xboundary = []
    yboundary = []
    for i in range(nbins_xdiag):
        fsum = np.sum(hist2d_accummulated[:, i])
        if fsum > 0:
            pdensity = hist2d_accummulated[:, i] / fsum
            perc = (1 - (1 / nsimulations) / fsum)
            p = np.interp(perc, np.cumsum(pdensity), np.arange(nbins_ydiag))
            xboundary.append(xcbins[i])
            yboundary.append(ycbins[int(p + 0.5)])
    ax1.plot(xboundary, yboundary, 'r+')
    splfit = None  # avoid flake8 warning
    for iterboundary in range(niter_boundary_extension + 1):
        wboundary = np.ones_like(xboundary, dtype=float)
        if iterboundary == 0:
            label = 'initial fit'
        else:
            wboundary[yboundary > splfit(xboundary)] = weight_boundary_extension**iterboundary
            label = f'iteration {iterboundary}'
        splfit, knots = spline_positive_derivative(
            x=np.array(xboundary),
            y=np.array(yboundary),
            w=wboundary,
            n_total_knots=knots_splfit,
        )
        ydum = splfit(xcbins)
        ydum[xcbins < knots[0]] = splfit(knots[0])
        ydum[xcbins > knots[-1]] = splfit(knots[-1])
        ax1.plot(xcbins, ydum, '-', color=f'C{iterboundary}', label=label)
        ax1.plot(knots, splfit(knots), 'o', color=f'C{iterboundary}', markersize=4)
    ax1.set_xlabel(r'min2d $-$ bias')
    ax1.set_ylabel(r'median2d $-$ min2d')
    ax1.set_title(f'Simulated data (nsimulations = {nsimulations})')
    if niter_boundary_extension > 1:
        ax1.legend()
    xplot_boundary = np.linspace(xdiag_min, xdiag_max, 100)
    yplot_boundary = splfit(xplot_boundary)
    yplot_boundary[xplot_boundary < knots[0]] = splfit(knots[0])
    yplot_boundary[xplot_boundary > knots[-1]] = splfit(knots[-1])
    ax2.plot(xplot_boundary, yplot_boundary, 'r-')
    ax2.set_xlim(xdiag_min, xdiag_max)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlabel(ax1.get_xlabel())
    ax2.set_ylabel(ax1.get_ylabel())
    ax2.set_title('Original data')
    plt.tight_layout()
    plt.savefig('diagnostic_histogram2d.png', dpi=150)
    if interactive:
        _logger.info("Entering interactive mode (press 'q' to close plot).")
        plt.show()

    plt.close(fig)

    if threshold is None:
        # Use the minimum value of the boundary as the threshold
        threshold = np.min(yplot_boundary)
        _logger.info("updated threshold for cosmic ray detection: %f", threshold)

    # Apply the criterium to detect double cosmic rays
    flag1 = yplot > splfit(xplot)
    flag2 = yplot > threshold
    flag = np.logical_and(flag1, flag2)
    flag3 = max2d.flatten() > minimum_max2d_rnoise * rnoise.flatten()
    flag = np.logical_and(flag, flag3)
    _logger.info("number of pixels flagged as double cosmic rays: %d", np.sum(flag))
    _logger.info("generating diagnostic plot for MEDIANCR...")
    ylabel = r'median2d $-$ min2d'
    diagnostic_plot(xplot, yplot, xplot_boundary, yplot_boundary, flag,
                    threshold, ylabel, interactive, _logger,
                    png_filename='diagnostic_mediancr.png')

    flag = flag.reshape((naxis2, naxis1))
    if not np.any(flag):
        _logger.info("no double cosmic rays detected.")
        mask_mediancr = np.zeros_like(median2d, dtype=bool)
    else:
        _logger.info("double cosmic rays detected, applying correction...")
        # Convert the flag to an integer array for dilation
        flag_integer = flag.astype(np.uint8)
        if dilation > 0:
            _logger.info("before dilation: %d pixels flagged as double cosmic rays.", np.sum(flag_integer))
            structure = ndimage.generate_binary_structure(2, 2)
            flag_integer_dilated = ndimage.binary_dilation(
                flag_integer,
                structure=structure,
                iterations=dilation
            ).astype(np.uint8)
            _logger.info("after dilation: %d pixels flagged as double cosmic rays.", np.sum(flag_integer_dilated))
        else:
            flag_integer_dilated = flag_integer
            _logger.info("no dilation applied: %d pixels flagged as double cosmic rays.", np.sum(flag_integer))
        # Set to 2 the pixels that were originally flagged as cosmic rays
        # (this is to distinguish them from the pixels that were dilated,
        # which will be set to 1)
        flag_integer_dilated[flag] = 2
        # Compute mask
        mask_mediancr = flag_integer_dilated > 0
        # Plot the cosmic rays if requested
        if plots:
            # Fix the median2d array by replacing the flagged pixels with the minimum value
            # of the corresponding pixel in the input arrays
            median2d_corrected = median2d.copy()
            median2d_corrected[mask_mediancr] = min2d[mask_mediancr]
            # Label the connected pixels as individual cosmic rays
            labels_cr, number_cr = ndimage.label(flag_integer_dilated > 0)
            _logger.info("number of double cosmic rays (connected pixels) detected: %d", number_cr)
            # Sort the cosmic rays by global detection criterium
            _logger.info("sorting cosmic rays by detection criterium...")
            detection_value = np.zeros(number_cr, dtype=float)
            imax_cr = np.zeros(number_cr, dtype=int)
            jmax_cr = np.zeros(number_cr, dtype=int)
            for i in range(1, number_cr + 1):
                ijloc = np.argwhere(labels_cr == i)
                detection_value[i-1] = 0
                imax_cr[i-1] = -1
                jmax_cr[i-1] = -1
                for k in ijloc:
                    ic, jc = k
                    if flag_integer_dilated[ic, jc] == 2:
                        detection_value_ = min(
                            median2d[ic, jc] - min2d[ic, jc] - np.interp(min2d[ic, jc], xplot_boundary, yplot_boundary),
                            median2d[ic, jc] - min2d[ic, jc] - threshold
                        )
                        if detection_value_ > detection_value[i-1]:
                            detection_value[i-1] = detection_value_
                            imax_cr[i-1] = ic
                            jmax_cr[i-1] = jc
            isort_cr = np.argsort(detection_value)[::-1]
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

            if maxplots < 0:
                maxplots = number_cr
            _logger.info(f"generating {maxplots} plots for double cosmic rays ranked by detection criterium...")
            for idum in range(min(number_cr, maxplots)):
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
                for k in range(num_plot_max):
                    ax = axarr[k]
                    title = title = f'image#{k+1}/{num_images}'
                    if color_scale == 'zscale':
                        vmin, vmax = tea.zscale(image3d[k, i1:(i2+1), j1:(j2+1)])
                    else:
                        vmin = np.min(image3d[k][i1:(i2+1), j1:(j2+1)])
                        vmax = np.max(image3d[k][i1:(i2+1), j1:(j2+1)])
                    tea.imshow(fig, ax, image3d[k][i1:(i2+1), j1:(j2+1)], vmin=vmin, vmax=vmax,
                               extent=[j1-0.5, j2+0.5, i1-0.5, i2+0.5],
                               title=title, cmap=cmap, cblabel=cblabel, interpolation=None)
                for k in range(3):
                    ax = axarr[k + num_plot_max]
                    cmap = 'viridis'
                    if k == 0:
                        image2d = median2d
                        title = 'median'
                        if color_scale == 'zscale':
                            vmin, vmax = tea.zscale(median2d[i1:(i2+1), j1:(j2+1)])
                        else:
                            vmin = np.min(median2d[i1:(i2+1), j1:(j2+1)])
                            vmax = np.max(median2d[i1:(i2+1), j1:(j2+1)])
                    elif k == 1:
                        image2d = flag_integer_dilated
                        title = 'flag_integer_dilated'
                        vmin, vmax = 0, 2
                        cmap = 'plasma'
                        cblabel = 'flag'
                    elif k == 2:
                        image2d = median2d_corrected
                        title = 'median corrected'
                        if color_scale == 'zscale':
                            vmin, vmax = tea.zscale(median2d_corrected[i1:(i2+1), j1:(j2+1)])
                        else:
                            vmin = np.min(median2d_corrected[i1:(i2+1), j1:(j2+1)])
                            vmax = np.max(median2d_corrected[i1:(i2+1), j1:(j2+1)])
                    else:
                        raise ValueError(f'Unexpected {k=}')
                    tea.imshow(fig, ax, image2d[i1:(i2+1), j1:(j2+1)], vmin=vmin, vmax=vmax,
                               extent=[j1-0.5, j2+0.5, i1-0.5, i2+0.5],
                               title=title, cmap=cmap, cblabel=cblabel, interpolation=None)
                nplot_missing = nrows * ncols - num_plot_max - 3
                if nplot_missing > 0:
                    for k in range(nplot_missing):
                        ax = axarr[-k-1]
                        ax.axis('off')
                fig.suptitle(f'CR#{idum+1}/{number_cr}\nMaximum detection parameter: {detection_value[i]:.2f}')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

            pdf.close()
            _logger.info("plot generation complete.")

    # Generate list of HDUs with masks
    hdu_mediancr = fits.ImageHDU(mask_mediancr.astype(np.uint8), name='MEDIANCR')
    list_hdu_masks = [hdu_mediancr]

    # Apply the same algorithm but now with mean2d and with each individual array
    for i, array in enumerate([mean2d] + list_arrays):
        xplot = min2d.flatten()
        if i == 0:
            yplot = array.flatten() - min2d.flatten()
        else:
            # For the individual arrays, apply the flux factor
            yplot = array.flatten()/flux_factor[i-1] - min2d.flatten()
        flag1 = yplot > splfit(xplot)
        flag2 = yplot > threshold
        flag = np.logical_and(flag1, flag2)
        flag3 = max2d.flatten() > minimum_max2d_rnoise * rnoise.flatten()
        flag = np.logical_and(flag, flag3)
        # for the individual arrays, force the flag to be True if the pixel
        # was flagged as a double cosmic ray when using the mean2d array
        if i > 0:
            flag = np.logical_and(flag, list_hdu_masks[1].data.astype(bool).flatten())
        _logger.info("number of pixels flagged as cosmic rays: %d", np.sum(flag))
        if i == 0:
            _logger.info("generating diagnostic plot for MEANCRT...")
            png_filename = 'diagnostic_meancr.png'
            ylabel = r'mean2d $-$ min2d'

        else:
            _logger.info(f"generating diagnostic plot for CRMASK{i}...")
            png_filename = f'diagnostic_crmask{i}.png'
            ylabel = f'array{i}' + r' $-$ min2d'
        diagnostic_plot(xplot, yplot, xplot_boundary, yplot_boundary, flag,
                        threshold, ylabel, interactive, _logger,
                        png_filename=png_filename)
        flag = flag.reshape((naxis2, naxis1))
        flag_integer = flag.astype(np.uint8)
        if dilation > 0:
            _logger.info("before dilation: %d pixels flagged as cosmic rays.", np.sum(flag_integer))
            structure = ndimage.generate_binary_structure(2, 2)
            flag_integer_dilated = ndimage.binary_dilation(
                flag_integer,
                structure=structure,
                iterations=dilation
            ).astype(np.uint8)
            _logger.info("after dilation: %d pixels flagged as cosmic rays.", np.sum(flag_integer_dilated))
        else:
            flag_integer_dilated = flag_integer
            _logger.info("no dilation applied: %d pixels flagged as cosmic rays.", np.sum(flag_integer))
        flag_integer_dilated[flag] = 2
        # Compute mask
        mask = flag_integer_dilated > 0
        if i == 0:
            name = 'MEANCRT'
        else:
            name = f'CRMASK{i}'
        hdu_mask = fits.ImageHDU(mask.astype(np.uint8), name=name)
        list_hdu_masks.append(hdu_mask)

    # Generate output HDUList with masks
    args = inspect.signature(compute_crmasks).parameters
    filtered_args = {k: v for k, v in locals().items() if k in args and k not in ['list_arrays']}
    hdu_primary = fits.PrimaryHDU()
    hdu_primary.header['UUID'] = str(uuid.uuid4())
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


def apply_crmasks(list_arrays, hdul_masks, combination=None, dtype=np.float32):
    """
    Correct cosmic rays applying previously computed masks.

    The pixels composing each cosmic ray can be surrounded by a dilation factor,
    which expands the mask around the detected cosmic ray pixels. Each masked pixel
    is replaced by the minimum value of the corresponding pixel in the input arrays.

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

    # Convert the list of arrays to a 3D numpy array
    shape3d = (num_images, naxis2, naxis1)
    image3d = np.zeros(shape3d, dtype=dtype)
    for i, array in enumerate(list_arrays):
        image3d[i] = array.astype(dtype)

    # Compute minimum and median along the first axis of image3d
    min2d = np.min(image3d, axis=0)
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
        median2d_corrected[mask_mediancr] = min2d[mask_mediancr]

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
            image3d_masked[i, :, :] = list_arrays[i].astype(dtype)
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
            combined2d[mask_nodata] = min2d[mask_nodata]
        else:
            _logger.info("no pixels without data found, no replacement needed")
        # Define the variance and map arrays
        variance2d = ma.var(image3d_masked, axis=0, ddof=1).data
        map2d = np.ones((naxis2, naxis1), dtype=int) * num_images - total_mask
    else:
        raise ValueError(f"Invalid combination method: {combination}. "
                         f"Valid options are {VALID_COMBINATIONS}.")

    return combined2d, variance2d, map2d


def main(args=None):
    """
    Main function to compute and apply CR masks.

    Since this main function is intended to be run for two different
    purposes, we are using argparse with subparsers to split the
    arguments for the two functionalities:
    - `compute`: to compute the cosmic ray masks.
    - `apply`: to apply the cosmic ray masks to a list of 2D arrays.

    The main function will parse the command line arguments and call
    the appropriate function based on the subcommand used.
    """
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG, WARNING, ERROR, CRITICAL
        format='%(name)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting mediancr combination...")

    parser = argparse.ArgumentParser(
        description="Combine 2D arrays using mediancr or meancr methods."
    )

    # Global arguments
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    # Subparsers for different functionalities
    subparsers = parser.add_subparsers(
        dest='command',
        help="Choose a command to execute.",
        required=False
    )

    # Subparser for computing cosmic ray masks
    parser_compute = subparsers.add_parser(
        'compute',
        help="Compute cosmic ray masks from a list of 2D arrays."
    )
    parser_compute.add_argument("--inputlist",
                                help="Input text file with list of 2D arrays.",
                                type=str,
                                required=True)
    parser_compute.add_argument("--gain",
                                help="Detector gain (ADU)",
                                type=float)
    parser_compute.add_argument("--rnoise",
                                help="Readout noise (ADU)",
                                type=float)
    parser_compute.add_argument("--bias",
                                help="Detector bias (ADU, default: 0.0)",
                                type=float, default=0.0)
    parser_compute.add_argument("--flux_factor",
                                help="Flux factor to be applied to each image",
                                type=str, default='none')
    parser_compute.add_argument("--knots_splfit",
                                help="Number of inner knots for the spline fit to the boundary (default: 3)",
                                type=int, default=3)
    parser_compute.add_argument("--nsimulations",
                                help="Number of simulations to compute exclusion boundary (default: 10)",
                                type=int, default=10)
    parser_compute.add_argument("--niter_boundary_extension",
                                help="Number of iterations for the boundary extension (default: 3)",
                                type=int, default=3)
    parser_compute.add_argument("--weight_boundary_extension",
                                help="Weight for the boundary extension (default: 10)",
                                type=float, default=10.0)
    parser_compute.add_argument("--threshold",
                                help="Minimum threshold for median2d - min2d to flag a pixel (default: None)",
                                type=float, default=None)
    parser_compute.add_argument("--minimum_max2d_rnoise",
                                help="Minimum value for max2d in rnoise units to flag a pixel (default: 5.0)",
                                type=float, default=5.0)
    parser_compute.add_argument("--interactive",
                                help="Interactive mode for diagnostic plot (program will stop after the plot)",
                                action="store_true")
    parser_compute.add_argument("--dilation",
                                help="Dilation factor for cosmic ray mask",
                                type=int, default=1)
    parser_compute.add_argument("--output_masks",
                                help="Output FITS file for the cosmic ray masks",
                                type=str, default='masks.fits')
    parser_compute.add_argument("--plots",
                                help="Generate plots with detected double cosmic rays",
                                action="store_true")
    parser_compute.add_argument("--semiwindow",
                                help="Semiwindow size for plotting double cosmic rays",
                                type=int, default=15)
    parser_compute.add_argument("--color_scale",
                                help="Color scale for the plots (default: 'minmax')",
                                type=str, choices=['minmax', 'zscale'], default='minmax')
    parser_compute.add_argument("--maxplots",
                                help="Maximum number of double cosmic rays to plot (-1 for all)",
                                type=int, default=10)
    parser_compute.add_argument("--extname",
                                help="Extension name in the input arrays (default: 'PRIMARY')",
                                type=str, default='PRIMARY')

    # Subparser for applying cosmic ray masks
    parser_apply = subparsers.add_parser(
        'apply',
        help="Apply cosmic ray masks to a list of 2D arrays."
    )
    parser_apply.add_argument("--inputlist",
                              help="Input text file with list of 2D arrays.",
                              type=str,
                              required=True)
    parser_apply.add_argument("--input_masks",
                              help="Input FITS file with the cosmic ray masks",
                              type=str, required=True)
    parser_apply.add_argument("--output_combined",
                              help="Output FITS file for the combined array and mask",
                              type=str, required=True)
    parser_apply.add_argument("--combination",
                              help=f"Combination method: {', '.join(VALID_COMBINATIONS)}",
                              type=str, choices=VALID_COMBINATIONS,
                              default='mediancr')
    parser_apply.add_argument("--extname",
                              help="Extension name in the input arrays (default: 'PRIMARY')",
                              type=str, default='PRIMARY')

    args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    # Read the input list of files, which should contain paths to 2D FITS files,
    # and load the arrays from the specified extension name.
    with open(args.inputlist, 'rt', encoding='utf-8') as f:
        list_of_fits_files = [line.strip() for line in f if line.strip()]
    list_arrays = [fits.getdata(file, extname=args.extname) for file in list_of_fits_files]

    # Check if the list is empty
    if not list_arrays:
        raise ValueError("The input list is empty. Please provide a valid list of 2D arrays.")

    # First task: compute cosmic ray masks
    if args.command == 'compute':
        # Check if gain and rnoise are provided
        if args.gain is None:
            raise ValueError("Gain must be provided for mediancr combination.")
        if args.rnoise is None:
            raise ValueError("Readout noise must be provided for mediancr combination.")

        # Compute the different cosmic ray masks
        hdul_masks = compute_crmasks(
            list_arrays=list_arrays,
            gain=args.gain,
            rnoise=args.rnoise,
            bias=args.bias,
            flux_factor=args.flux_factor,
            knots_splfit=args.knots_splfit,
            nsimulations=args.nsimulations,
            niter_boundary_extension=args.niter_boundary_extension,
            weight_boundary_extension=args.weight_boundary_extension,
            threshold=args.threshold,
            minimum_max2d_rnoise=args.minimum_max2d_rnoise,
            interactive=args.interactive,
            dilation=args.dilation,
            dtype=np.float32,
            plots=args.plots,
            semiwindow=args.semiwindow,
            color_scale=args.color_scale,
            maxplots=args.maxplots
        )

        # Save the masks to a FITS file if requested
        if args.output_masks:
            logger.info("Saving cosmic ray masks to %s", args.output_masks)
            hdul_masks.writeto(args.output_masks, overwrite=True)
            logger.info("Cosmic ray masks saved")

    # Second task: apply cosmic ray masks
    elif args.command == 'apply':
        with fits.open(args.input_masks) as hdul_masks:
            # Compute the combined array, variance, and map
            combined, variance, maparray = apply_crmasks(
                list_arrays=list_arrays,
                hdul_masks=hdul_masks,
                combination=args.combination,
                dtype=np.float32
            )
            # Save the combined array, variance, and map to a FITS file
            logger.info("Saving combined array, variance, and map to %s", args.output_combined)
            hdu_combined = fits.PrimaryHDU(combined.astype(np.float32))
            add_script_info_to_fits_history(hdu_combined.header, args)
            hdu_combined.header.add_history('Contents of --inputlist:')
            for item in list_of_fits_files:
                hdu_combined.header.add_history(f'- {item}')
            hdu_combined.header.add_history(f"Masks UUID: {hdul_masks[0].header['UUID']}")
            hdu_variance = fits.ImageHDU(variance.astype(np.float32), name='VARIANCE')
            hdu_map = fits.ImageHDU(maparray.astype(np.int16), name='MAP')
            hdul = fits.HDUList([hdu_combined, hdu_variance, hdu_map])
            hdul.writeto(args.output_combined, overwrite=True)
            logger.info("Combined array, variance, and map saved")
    else:
        raise ValueError(f"Unknown command: {args.command}.")


if __name__ == "__main__":

    main()
