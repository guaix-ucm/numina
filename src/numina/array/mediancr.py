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
        threshold=None,
        boundary_slope=1.0,
        dilation=1,
        dtype=np.float32,
        plots=False,
        semiwindow=15,
        maxplots=10
        ):
    """
    Median combination of arrays while avoiding multiple cosmic rays in the same pixel.

    This function combines a list of 2D numpy arrays using the median method,
    and applies a cosmic ray detection algorithm to avoid multiple cosmic rays
    affecting the same pixel. The cosmic ray detection is based on a threshold
    and a boundary slope, which can be adjusted. The function also supports
    generating diagnostic plots to visualize the cosmic ray detection process.

    The pixels composing each cosmic ray can be surrounded by a dilation factor,
    which expands the mask around the detected cosmic ray pixels. Each masked pixel
    is replaced by the minimum value of the corresponding pixel in the input arrays.

    Parameters
    ----------
    list_arrays : list of 2D arrays
        The input arrays to be combined.
    threshold : float
        The threshold value for detecting doublecosmic rays.
    boundary_slope : float, optional
        The slope of the boundary for double cosmic ray detection (default is 1.0).
    dilation : int, optional
        The dilation factor for the double cosmic ray mask (default is 1).
    dtype : data-type, optional
        The desired data type for the output arrays (default is np.float32).
    plots : bool, optional
        If True, generate diagnostic plots (default is False).
    semiwindow : int, optional
        The semiwindow size to plot the double cosmic rays (default is 15).
        Only used if `plots` is True.
    maxplots : int, optional
        The maximum number of double cosmic rays to plot (default is 10).

    Returns
    -------
    median2d_corrected : 2D array
        The median-combined array with double cosmic rays corrected.
    variance2d : 2D array
        The variance of the input arrays along the first axis.
    map2d : 2D array
        The number of input pixels used to compute the median at each pixel.
    """

    _logger = logging.getLogger(__name__)

    # check that the input is a list
    if not isinstance(list_arrays, list):
        raise TypeError("Input must be a list of arrays.")

    # check that the list contains numpy 2D arrays
    if not all(isinstance(array, np.ndarray) and array.ndim == 2 for array in list_arrays):
        raise ValueError("All elements in the list must be 2D numpy arrays.")

    # check that the list contains at least 3 arrays
    num_images = len(list_arrays)
    if num_images < 3:
        raise ValueError("At least 3 images are required for useful mediancr combination.")

    # check that all arrays have the same shape
    for i, array in enumerate(list_arrays):
        if array.shape != list_arrays[0].shape:
            raise ValueError(f"Array {i} has a different shape than the first array.")
    naxis2, naxis1 = list_arrays[0].shape

    # Check that threshold is defined
    if threshold is None:
        raise ValueError("Threshold must be defined for mediancr combination.")

    # Log the input parameters
    _logger.info("number of input arrays: %d", len(list_arrays))
    for i, array in enumerate(list_arrays):
        _logger.info("array %d shape: %s, dtype: %s", i, array.shape, array.dtype)
    _logger.info("threshold for cosmic ray detection: %f", threshold)
    _logger.info("boundary slope: %f", boundary_slope)
    _logger.info("dilation factor: %d", dilation)
    if plots:
        _logger.info("diagnostic plots will be generated.")

    # Convert the list of arrays to a 3D numpy array
    image3d = np.zeros((num_images, naxis2, naxis1), dtype=dtype)
    for i, array in enumerate(list_arrays):
        image3d[i] = array.astype(dtype)

    # Compute minimum, median, maximum and variance along the first axis
    min2d = np.min(image3d, axis=0)
    median2d = np.median(image3d, axis=0)
    max2d = np.max(image3d, axis=0)
    variance2d = np.var(image3d, axis=0, ddof=1)
    # Number of pixels used to compute the median at each pixel
    map2d = np.ones((naxis2, naxis1), dtype=int) * num_images

    # Apply the different criteria to detect double cosmic rays
    xplot = min2d.flatten()
    yplot = median2d.flatten() - xplot
    flag1 = yplot > threshold
    flag2 = yplot > boundary_slope * xplot + threshold
    flag = np.logical_and(flag1, flag2)
    flag3 = max2d.flatten() > threshold
    flag = np.logical_and(flag, flag3)
    if plots:
        fig, axarr = plt.subplots(ncols=2, figsize=(10, 5))
        for iplot in range(2):
            ax = axarr[iplot]
            ax.plot(xplot, yplot, 'C0,')
            ax.plot(xplot[flag], yplot[flag], 'rx', label='Suspected CRs')
            if iplot == 1:
                ax.set_xlim(np.min(min2d), 10 * threshold)
                ax.set_ylim(-threshold / 10, 10 * threshold)
            ax.axhline(y=threshold, color='C1', linestyle='--', label='threshold')
            xmin, xmax = ax.get_xlim()
            ax.plot([xmin, xmax],
                    [boundary_slope * xmin + threshold, boundary_slope * xmax + threshold],
                    'C2-', label='exclusion line')
            ax.set_xlabel('min2d')
            ax.set_ylabel(r'median2d $-$ min2d')
            ax.legend(loc=1)
        plt.tight_layout()
        plt.savefig('mediancr_diagnostic.png', dpi=150)
        plt.close(fig)

    # Create a mask for the flagged pixels
    flag = flag.reshape((naxis2, naxis1))
    flag_integer = flag.astype(np.uint8)
    if not np.any(flag):
        _logger.info("no double cosmic rays detected.")
        return median2d, variance2d, map2d

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
                    detection_value_ = min(median2d[ic, jc] - min2d[ic, jc] - threshold,
                                           median2d[ic, jc] - (1 + boundary_slope) * min2d[ic, jc] - threshold)
                    if detection_value_ > detection_value[i-1]:
                        detection_value[i-1] = detection_value_
                        imax_cr[i-1] = ic
                        jmax_cr[i-1] = jc
        isort_cr = np.argsort(detection_value)[::-1]
        num_plot_max = num_images
        # determine the number of rows and columns for the plot,
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
            _logger.warning("Only the first 9 images will be plotted")
            nrows, ncols = 3, 4
            figsize = (13, 9)
            num_plot_max = 9
        pdf = PdfPages('mediancr_identified_cr.pdf')

        _logger.info("Generating plots for double cosmic rays ranked by detection criterium...")
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
            vmin_, vmax_ = tea.zscale(image3d[:, i1:(i2+1), j1:(j2+1)])
            fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            axarr = axarr.flatten()
            # important: use interpolation=None instead of interpolation='None' to avoid
            # having blurred images when opening the PDF file with macos Preview
            cmap = 'viridis'
            cblabel = 'Number of counts'
            for k in range(num_plot_max):
                ax = axarr[k]
                title = title = f'image#{k+1}/{num_images}'
                tea.imshow(fig, ax, image3d[k][i1:(i2+1), j1:(j2+1)], vmin=vmin_, vmax=vmax_,
                           extent=[j1-0.5, j2+0.5, i1-0.5, i2+0.5],
                           title=title, cmap=cmap, cblabel=cblabel, interpolation=None)
            for k in range(3):
                ax = axarr[k + num_plot_max]
                cmap = 'viridis'
                if k == 0:
                    image2d = median2d
                    title = 'median'
                    vmin, vmax = vmin_, vmax_
                elif k == 1:
                    image2d = flag_integer_dilated
                    title = 'flag_integer_dilated'
                    vmin, vmax = 0, 2
                    cmap = 'plasma'
                    cblabel = 'flag'
                elif k == 2:
                    image2d = median2d_corrected
                    title = 'median corrected'
                    vmin, vmax = vmin_, vmax_
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
        _logger.info("Plot generation complete.")

    return median2d_corrected, variance2d, map2d


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
    parser.add_argument("--threshold",
                        help="Threshold for cosmic ray detection",
                        type=float)
    parser.add_argument("--boundary-slope",
                        help="Slope of the boundary for cosmic ray detection",
                        type=float, default=1.0)
    parser.add_argument("--dilation",
                        help="Dilation factor for cosmic ray mask",
                        type=int, default=1)
    parser.add_argument("--output",
                        help="Output FITS file for the combined array and mask",
                        type=str)
    parser.add_argument("--plots",
                        help="Generate diagnostic plots",
                        action="store_true")
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

    combined_array, variance, map = _mediancr(
        list_arrays=list_arrays,
        threshold=args.threshold,
        boundary_slope=args.boundary_slope,
        dilation=args.dilation,
        plots=args.plots
    )

    # Save the combined array and mask to a FITS file
    if args.output:
        hdu_combined = fits.PrimaryHDU(combined_array)
        hdu_variance = fits.ImageHDU(variance.astype(np.float32), name='VARIANCE')
        hdu_map = fits.ImageHDU(map.astype(np.int16), name='MAP')
        hdul = fits.HDUList([hdu_combined, hdu_variance, hdu_map])
        hdul.writeto(args.output, overwrite=True)
        logger.info("Combined array, variance, and map saved to %s", args.output)


if __name__ == "__main__":

    main()
