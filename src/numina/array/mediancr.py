#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Median combination of arrays avoiding multiple cosmic rays in the same pixel."""
import logging
import sys

import argparse
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import teareduce as tea


def _mediancr(
        list_arrays,
        find_double_cr=True,
        gain=None,
        rnoise=None,
        bias=0.0,
        flatmin=1.0,
        flatmax=1.0,
        knots_splfit=2,
        nsimulations=10000,
        times_boundary_extension=1.0,
        threshold=None,
        minimum_max2d_rnoise=5.0,
        interactive=False,
        dilation=1,
        dtype=np.float32,
        plots=False,
        semiwindow=15,
        color_scale='minmax',
        maxplots=10
        ):
    """
    Median combination of arrays while avoiding multiple cosmic rays in the same pixel.

    This function combines a list of 2D numpy arrays using the median method,
    and applies a cosmic ray detection algorithm to avoid multiple cosmic rays
    affecting the same pixel. The cosmic ray detection is based on a boundary
    that is derived numerically making use of the provided gain and readout noise
    values. The function also supports generating diagnostic plots to visualize
    the cosmic ray detection process.

    The pixels composing each cosmic ray can be surrounded by a dilation factor,
    which expands the mask around the detected cosmic ray pixels. Each masked pixel
    is replaced by the minimum value of the corresponding pixel in the input arrays.

    Parameters
    ----------
    list_arrays : list of 2D arrays
        The input arrays to be combined.
    find_double_cr : bool, optional
        If True, apply the double cosmic ray detection algorithm
        (default is True). If False, the function will only compute
        the median without detecting double cosmic rays.
    gain : float
        The gain value (in e/ADU) of the detector.
    bias : float, optional
        The bias value (in ADU) of the detector (default is 0.0).
    rnoise : float
        The readout noise (in ADU) of the detector.
    flatmin : float, optional
        The minimum value for the flat field (default is 0.1).
    flatmax : float, optional
        The maximum value for the flat field (default is 3.0).
    knots_splfit : int, optional
        The number of knots for the spline fit to the boundary
        (default is 2).
    nsimulations : int, optional
        The number of simulations to perform for each point in the
        exclusion boundary (default is 10000).
    times_boundary_extension : float, optional
        The factor to extend the boundary computed at percentile 98, in
        units of the difference between the percentiles 98 and 50.
        This is used to compute the numerical boundary for double
        cosmic ray detection (default is 1.0).
    threshold: float, optional
        Minimum threshold for median2d - min2d to consider a pixel as a
        cosmic ray (default is None). If None, the threshold is computed
        automatically from the minimum boundary value in the numerical
        simulations.
    minimum_max2d_rnoise : float, optional
        Minimum value for max2d in readout noise units to flag a pixel
        as a double cosmic ray (default is 5.0).
    interactive : bool, optional
        If True, enable interactive mode for plots (default is False).
    dilation : int, optional
        The dilation factor for the double cosmic ray mask (default is 1).
    dtype : data-type, optional
        The desired data type for the output arrays (default is np.float32).
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
    median2d_corrected : 2D array
        The median-combined array with double cosmic rays corrected.
        If `find_double_cr` is False, this is simply the median
        of the input arrays.
    variance2d : 2D array
        The variance of the input arrays along the first axis.
    map2d : 2D array
        The number of input pixels used to compute the median at each pixel.
    mean2d : 2D array
        The simple mean-combined array of the input arrays.
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

    if find_double_cr:
        # Check that gain is defined
        if gain is None:
            raise ValueError("Gain must be defined for mediancr combination.")

        # Check that readout noise is defined
        if rnoise is None:
            raise ValueError("Readout noise must be defined for mediancr combination.")

        # Check that color_scale is valid
        if color_scale not in ['minmax', 'zscale']:
            raise ValueError(f"Invalid color_scale: {color_scale}. Valid options are 'minmax' and 'zscale'.")

        # Log the input parameters
        _logger.info("gain for double cosmic ray detection: %f", gain)
        _logger.info("readout noise for double cosmic ray detection: %f", rnoise)
        _logger.info("bias for double cosmic ray detection: %f", bias)
        _logger.info("flat field minimum: %f", flatmin)
        _logger.info("flat field maximum: %f", flatmax)
        _logger.info("knots for spline fit to the boundary: %d", knots_splfit)
        _logger.info("number of simulations for each point in the boundary: %d", nsimulations)
        _logger.info("threshold for double cosmic ray detection: %s", threshold if threshold is not None else "None")
        _logger.info("minimum max2d in rnoise units for double cosmic ray detection: %f", minimum_max2d_rnoise)
        _logger.info("times boundary extension for double cosmic ray detection: %f", times_boundary_extension)
        _logger.info("dtype for output arrays: %s", dtype)
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
    if bias != 0.0:
        _logger.info("subtracting bias from the input arrays: %f", bias)
        image3d -= bias

    # Compute minimum, median, maximum and variance along the first axis
    min2d = np.min(image3d, axis=0)
    max2d = np.max(image3d, axis=0)
    median2d = np.median(image3d, axis=0)
    mean2d = np.mean(image3d, axis=0)
    variance2d = np.var(image3d, axis=0, ddof=1)
    # Number of pixels used to compute the median at each pixel
    map2d = np.ones((naxis2, naxis1), dtype=int) * num_images
    if not find_double_cr:
        _logger.info("find_double_cr is set to False, skipping double cosmic ray detection.")
        return median2d, variance2d, map2d, mean2d

    # Numerical boundary for double cosmic ray detection
    _logger.info("computing numerical boundary for double cosmic ray detection...")
    seed = 1234
    ntest = 100  # number of points along the x-axis for the boundary
    # test values for the x-axis: enlarged by flatmax to ensure we cover a wide range
    xtest_array = 10**np.linspace(0, np.log10(np.max(min2d) * flatmax), ntest)
    xplot_boundary = np.zeros(ntest, dtype=float)  # x values for the boundary
    yplot_boundary_50 = np.zeros(ntest, dtype=float)  # y values for the boundary
    yplot_boundary_98 = np.zeros(ntest, dtype=float)  # y values for the boundary
    rng = np.random.default_rng(seed)  # Random number generator for reproducibility
    for i in range(ntest):
        xtest = xtest_array[i]
        min_rep = np.zeros(nsimulations, dtype=float)
        max_rep = np.zeros(nsimulations, dtype=float)
        median_rep = np.zeros(nsimulations, dtype=float)
        # Simulate the minimum, median and maximum of the data
        for k in range(nsimulations):
            data = np.ones(num_images, dtype=float) * xtest
            # Transform data from ADU to electrons, generate Poisson distribution
            # and transform back from electrons to ADU
            flatfield = rng.uniform(low=flatmin, high=flatmax, size=num_images)
            data_with_noise = flatfield * rng.poisson(lam=(data) * gain).astype(float) / gain
            # Add readout noise
            if rnoise > 0:
                data_with_noise += rng.normal(loc=0, scale=rnoise, size=num_images)
            min_rep[k] = np.min(data_with_noise)
            max_rep[k] = np.max(data_with_noise)
            median_rep[k] = np.median(data_with_noise)
        # Compute the boundary using the requested percentile in the y-axis
        xplot_boundary[i] = np.median(min_rep)
        yplot_boundary_50[i], yplot_boundary_98[i] = np.percentile(median_rep - min_rep, [50, 98])

    # Fit a spline to the boundary points
    _logger.info("fitting splines to the boundary points...")
    isort = np.argsort(xplot_boundary)
    xplot_boundary = xplot_boundary[isort]
    yplot_boundary_50 = yplot_boundary_50[isort]
    yplot_boundary_98 = yplot_boundary_98[isort]
    ifit = xplot_boundary <= np.max(min2d)
    xplot_boundary = xplot_boundary[ifit]
    yplot_boundary_50 = yplot_boundary_50[ifit]
    yplot_boundary_98 = yplot_boundary_98[ifit]
    spl50 = tea.AdaptiveLSQUnivariateSpline(xplot_boundary, yplot_boundary_50, t=knots_splfit)
    spl98 = tea.AdaptiveLSQUnivariateSpline(xplot_boundary, yplot_boundary_98, t=knots_splfit)
    yplot_boundary = spl98(xplot_boundary) + \
        (spl98(xplot_boundary) - spl50(xplot_boundary)) * times_boundary_extension

    if threshold is None:
        # Use the minimum value of the boundary as the threshold
        threshold = np.min(yplot_boundary)
        _logger.info("updated threshold for cosmic ray detection: %f", threshold)

    # Apply the criterium to detect double cosmic rays
    xplot = min2d.flatten()
    yplot = median2d.flatten() - min2d.flatten()
    # The advantage of using np.interp is that it is faster than
    # using spl50 and spl98 directly, but also because for xplot values
    # outside the range of xplot_boundary, it will return the value at the
    # closest boundary point, which is what we want for the exclusion boundary.
    flag1 = yplot > np.interp(xplot, xplot_boundary, yplot_boundary)
    flag2 = yplot > threshold
    flag = np.logical_and(flag1, flag2)
    flag3 = max2d.flatten() > minimum_max2d_rnoise * rnoise
    flag = np.logical_and(flag, flag3)
    _logger.info("number of pixels flagged as double cosmic rays: %d", np.sum(flag))
    fig, ax = plt.subplots()
    ax.plot(xplot, yplot, 'C0,')
    xmin, xmax = ax.get_xlim()
    ax.plot(xplot_boundary, yplot_boundary_50, 'C3.', label='percentile 50')
    ax.plot(xplot_boundary, spl50(xplot_boundary), 'C3--', label='spline fit 50')
    xknots = spl50.get_knots()
    yknots = spl50(xknots)
    ax.plot(xknots, yknots, 'C3o', label='knots 50')
    ax.plot(xplot_boundary, yplot_boundary_98, 'C4.', label='percentile 98')
    ax.plot(xplot_boundary, spl98(xplot_boundary), 'C4--', label='spline fit 98')
    xknots = spl98.get_knots()
    yknots = spl98(xknots)
    ax.plot(xknots, yknots, 'C4o', label='knots 98')
    ax.plot(xplot_boundary, yplot_boundary, 'C1-', label='Exclusion boundary')
    ax.axhline(threshold, color='gray', linestyle=':', label=f'Threshold ({threshold:.2f})')
    ax.plot(xplot[flag], yplot[flag], 'rx', label=f'Suspected pixels ({np.sum(flag)})')
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'min2d $-$ bias')  # the bias was subtracted from the input arrays
    ax.set_ylabel(r'median2d $-$ min2d')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
    plt.tight_layout()
    _logger.info("saving mediancr_diagnostic.png.")
    plt.savefig('mediancr_diagnostic.png', dpi=150)
    if interactive:
        _logger.info("Entering interactive mode (press 'q' to close plot).")
        plt.show()
        answer = input("Press Enter to continue or type 'exit' to quit: ")
        plt.close(fig)
        if answer.lower() == 'exit':
            _logger.info("Exiting program.")
            raise SystemExit()
    else:
        plt.close(fig)

    # Return if no pixels were flagged as double cosmic rays
    flag = flag.reshape((naxis2, naxis1))
    if not np.any(flag):
        _logger.info("no double cosmic rays detected.")
        return median2d, variance2d, map2d, mean2d

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

    # Fix the median2d array by replacing the flagged pixels with the minimum value
    # of the corresponding pixel in the input arrays
    median2d_corrected = median2d.copy()
    mask = flag_integer_dilated > 0
    median2d_corrected[mask] = min2d[mask]
    variance2d[mask] = 0.0  # Set variance to 0 for the flagged pixels
    map2d[mask] = 1  # Set the map to 1 for the flagged pixels

    # Plot the cosmic rays if requested
    if plots:
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

    return median2d_corrected, variance2d, map2d, mean2d


def main(args=None):
    """
    Main function to run the mediancr combination.
    """
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG, WARNING, ERROR, CRITICAL
        format='%(name)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting mediancr combination...")

    parser = argparse.ArgumentParser(
        description="Combine 2D arrays using mediancr method to avoid multiple cosmic rays in the same pixel."
    )

    parser.add_argument("inputlist",
                        help="Input text file with list of 2D arrays.",
                        type=str)
    parser.add_argument("--gain",
                        help="Detector gain (ADU)",
                        type=float)
    parser.add_argument("--rnoise",
                        help="Readout noise (ADU)",
                        type=float)
    parser.add_argument("--bias",
                        help="Detector bias (ADU, default: 0.0)",
                        type=float, default=0.0)
    parser.add_argument("--flatmin",
                        help="Minimum value for the flat field (default: 1.0)",
                        type=float, default=1.0)
    parser.add_argument("--flatmax",
                        help="Maximum value for the flat field (default: 1.0)",
                        type=float, default=1.0)
    parser.add_argument("--knots_splfit",
                        help="Number of inner knots for the spline fit to the boundary (default: 2)",
                        type=int, default=2)
    parser.add_argument("--nsimulations",
                        help="Number of simulations for each point in the boundary (default: 10000)",
                        type=int, default=10000)
    parser.add_argument("--times_boundary_extension",
                        help="Factor to extend the boundary computed at percentile 98 "
                             "for double cosmic ray detection (default: 1.0)",
                        type=float, default=1.0)
    parser.add_argument("--threshold",
                        help="Minimum threshold for median2d - min2d to flag a pixel (default: None)",
                        type=float, default=None)
    parser.add_argument("--minimum_max2d_rnoise",
                        help="Minimum value for max2d in rnoise units to flag a pixel (default: 3.0)",
                        type=float, default=3.0)
    parser.add_argument("--interactive",
                        help="Interactive mode for diagnostic plot (program will stop after the plot)",
                        action="store_true")
    parser.add_argument("--dilation",
                        help="Dilation factor for cosmic ray mask",
                        type=int, default=1)
    parser.add_argument("--output",
                        help="Output FITS file for the combined array and mask",
                        type=str)
    parser.add_argument("--plots",
                        help="Generate plots with detected double cosmic rays",
                        action="store_true")
    parser.add_argument("--semiwindow",
                        help="Semiwindow size for plotting double cosmic rays",
                        type=int, default=15)
    parser.add_argument("--color_scale",
                        help="Color scale for the plots (default: 'minmax')",
                        type=str, choices=['minmax', 'zscale'], default='minmax')
    parser.add_argument("--maxplots",
                        help="Maximum number of double cosmic rays to plot (-1 for all)",
                        type=int, default=10)
    parser.add_argument("--extname",
                        help="Extension name in the input arrays (default: 'PRIMARY')",
                        type=str, default='PRIMARY')
    parser.add_argument("--echo",
                        help="Display full command line",
                        action="store_true")

    args = parser.parse_args(args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.echo:
        print('\033[1m\033[31mExecuting: ' + ' '.join(sys.argv) + '\033[0m\n')

    # Read the input list of files, which should contain paths to 2D FITS files,
    # and load the arrays from the specified extension name.
    with open(args.inputlist, 'rt', encoding='utf-8') as f:
        list_arrays = [fits.getdata(line.strip(), extname=args.extname) for line in f if line.strip()]

    # Check if the list is empty
    if not list_arrays:
        raise ValueError("The input list is empty. Please provide a valid list of 2D arrays.")

    # Check if gain and rnoise are provided
    if args.gain is None:
        raise ValueError("Gain must be provided for mediancr combination.")
    if args.rnoise is None:
        raise ValueError("Readout noise must be provided for mediancr combination.")

    # Perform the mediancr combination
    median2d_corrected, variance, maparray, mean2d = _mediancr(
        list_arrays=list_arrays,
        gain=args.gain,
        rnoise=args.rnoise,
        bias=args.bias,
        flatmin=args.flatmin,
        flatmax=args.flatmax,
        knots_splfit=args.knots_splfit,
        nsimulations=args.nsimulations,
        times_boundary_extension=args.times_boundary_extension,
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

    # Save the combined array and mask to a FITS file
    if args.output:
        hdu_combined = fits.PrimaryHDU(median2d_corrected.astype(np.float32))
        hdu_variance = fits.ImageHDU(variance.astype(np.float32), name='VARIANCE')
        hdu_map = fits.ImageHDU(maparray.astype(np.int16), name='MAP')
        hdul = fits.HDUList([hdu_combined, hdu_variance, hdu_map])
        hdul.writeto(args.output, overwrite=True)
        logger.info("Combined array, variance, and map saved to %s", args.output)


if __name__ == "__main__":

    main()
