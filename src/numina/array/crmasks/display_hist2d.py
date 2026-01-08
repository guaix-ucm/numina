#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Display 2D histogram of min2d-bias vs mean2d-bias values."""

from astropy.table import Table
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import sys

from numina.array.numsplines import spline_positive_derivative
from numina.tools.input_number import input_number
import teareduce as tea

from .define_piecewise_linear_function import define_piecewise_linear_function
from .valid_parameters import VALID_BOUNDARY_FITS


def xsort_and_show_fixed_points_in_boundary(
    _logger,
    num_fixed_points,
    x_mm_fixed_points_in_boundary,
    y_mm_fixed_points_in_boundary,
    w_mm_fixed_points_in_boundary,
):
    """Show fixed points in boundary.
    
    If there are fixed points, sort them by increasing x value."""
    if num_fixed_points > 0:
        isort = np.argsort(x_mm_fixed_points_in_boundary)
        if not np.all(isort == np.arange(num_fixed_points)):
            _logger.info("Sorting fixed points in boundary by increasing x value.")
        x_mm_fixed_points_in_boundary[:] = x_mm_fixed_points_in_boundary[isort]
        y_mm_fixed_points_in_boundary[:] = y_mm_fixed_points_in_boundary[isort]
        w_mm_fixed_points_in_boundary[:] = w_mm_fixed_points_in_boundary[isort]
        _logger.info("Current fixed points in boundary:")
        fixed_table = Table(names=("number", "X", "Y", "Weight"), dtype=(int, float, float, float))
        for idum in range(num_fixed_points):
            fixed_table.add_row(
                (
                    idum + 1,
                    x_mm_fixed_points_in_boundary[idum],
                    y_mm_fixed_points_in_boundary[idum],
                    w_mm_fixed_points_in_boundary[idum],
                )
            )
        _logger.info("%s", fixed_table)
    else:
        _logger.info("No fixed points in boundary.")


def display_hist2d(
    _logger,
    rlabel_mmcosmic,
    mm_hist2d_min_neighbors,
    hist2d_accummulated,
    mm_nsimulations,
    mm_synthetic,
    bins_xdiag,
    bins_ydiag,
    xplot,
    yplot,
    xdiag_min,
    xdiag_max,
    max2d,
    bool_to_be_cleaned,
    rnoise,
    mm_threshold,
    mm_boundary_fit,
    mm_knots_splfit,
    mm_minimum_max2d_rnoise,
    mm_dilation,
    mm_niter_boundary_extension,
    mm_weight_boundary_extension,
    mm_fixed_points_in_boundary,
    x_mm_fixed_points_in_boundary,
    y_mm_fixed_points_in_boundary,
    w_mm_fixed_points_in_boundary,
    interactive,
):
    """Display 2D histogram of min2d-bias vs mean2d-bias values."""
    # Remove bins that are surrounded by less than mm_hist2d_min_neighbors neighbors
    hist2d_accummulated_original = hist2d_accummulated.copy()
    loop = True
    while loop:
        hist2d_accummulated = hist2d_accummulated_original.copy()
        if mm_hist2d_min_neighbors > 0:
            kernel_neighbors = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
            neighbor_counts = ndimage.convolve(
                (hist2d_accummulated > 0).astype(int), kernel_neighbors, mode="constant", cval=0
            )
            surrounded_by_threshold = (hist2d_accummulated > 0) & (neighbor_counts < mm_hist2d_min_neighbors)
            num_zero_bins_before = np.sum(hist2d_accummulated == 0)
            hist2d_accummulated[surrounded_by_threshold] = 0
            num_zero_bins_after = np.sum(hist2d_accummulated == 0)
            _logger.info(
                "removed %d bins that were surrounded by less than %d neighbors",
                num_zero_bins_after - num_zero_bins_before,
                mm_hist2d_min_neighbors,
            )
        # Average the histogram over the number of simulations
        hist2d_accummulated = hist2d_accummulated.astype(float) / mm_nsimulations
        # Determine vmin and vmax for the color scale
        vmin = np.min(hist2d_accummulated[hist2d_accummulated > 0])
        if vmin == 0:
            vmin = 1
        vmax = np.max(hist2d_accummulated)
        cmap1 = plt.get_cmap("cividis_r")
        cmap2 = plt.get_cmap("viridis")
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
        combined_cmap = LinearSegmentedColormap.from_list("combined_cmap", combined_colors)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        def on_key_2dhist(event):
            nonlocal loop
            if event.key == "x":
                _logger.info("Exiting program as per user request ('x' key pressed).")
                plt.close(fig)
                sys.exit(0)
            elif event.key == "q":
                loop = False
            elif event.key == "r":
                plt.close(fig)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.1, 5.5))
        fig.canvas.mpl_connect("key_press_event", lambda event: on_key_2dhist(event))
        # Display 2D histogram of the simulated data
        extent = [bins_xdiag[0], bins_xdiag[-1], bins_ydiag[0], bins_ydiag[-1]]
        tea.imshow(
            fig,
            ax1,
            hist2d_accummulated,
            norm=norm,
            extent=extent,
            aspect="auto",
            cblabel="Number of pixels",
            cmap=combined_cmap,
        )
        # Display 2D histogram of the original data
        hist2d_original, edges = np.histogramdd(sample=(yplot, xplot), bins=(bins_ydiag, bins_xdiag))
        tea.imshow(
            fig,
            ax2,
            hist2d_original,
            norm=norm,
            extent=extent,
            aspect="auto",
            cblabel="Number of pixels",
            cmap=combined_cmap,
        )

        # Determine the detection boundary for coincident cosmic-ray detection
        _logger.info("computing numerical boundary for coincident cosmic-ray detection...")
        xboundary = []
        yboundary = []
        xcbins = (bins_xdiag[:-1] + bins_xdiag[1:]) / 2
        ycbins = (bins_ydiag[:-1] + bins_ydiag[1:]) / 2
        nbins_xdiag = len(xcbins)
        nbins_ydiag = len(ycbins)
        minimum_bin_value = 1.0 / mm_nsimulations
        for i in range(nbins_xdiag):
            fsum = np.sum(hist2d_accummulated[:, i])
            if fsum > 0:
                xboundary.append(xcbins[i])
                # Compute the probability density function for this x bin
                pdensity = hist2d_accummulated[:, i] / fsum
                # Find the y value where the cumulative distribution reaches the desired percentile
                # (note that 1 / mm_nsimulations is the minimum expected value different from zero in any bin)
                if fsum > minimum_bin_value:
                    perc = (fsum - minimum_bin_value) / fsum
                    # Interpolate to find the corresponding y value
                    p = np.interp(perc, np.cumsum(pdensity), np.arange(nbins_ydiag))
                    yboundary.append(ycbins[int(p + 0.5)])
                else:
                    yboundary.append(bins_ydiag[0])
        xboundary = np.array(xboundary)
        yboundary = np.array(yboundary)
        ax1.plot(xboundary, yboundary, "r+", label="boundary points")
        boundaryfit = None  # avoid flake8 warning
        if mm_boundary_fit == "spline":
            nmax_iterations_with_color = 6
            for iterboundary in range(mm_niter_boundary_extension + 1):
                wboundary = np.ones_like(xboundary, dtype=float)
                color = f"C{iterboundary}"
                alpha = 1.0
                if iterboundary == 0:
                    label = "initial spline fit"
                else:
                    wboundary[yboundary > boundaryfit(xboundary)] = mm_weight_boundary_extension**iterboundary
                    if iterboundary == mm_niter_boundary_extension:
                        label = f"final iteration {iterboundary}"
                        if mm_niter_boundary_extension > nmax_iterations_with_color:
                            color = f"C{nmax_iterations_with_color}"
                    else:
                        if mm_niter_boundary_extension > nmax_iterations_with_color:
                            if iterboundary < nmax_iterations_with_color:
                                label = f"Iteration {iterboundary}"
                            elif iterboundary == nmax_iterations_with_color:
                                if nmax_iterations_with_color == mm_niter_boundary_extension - 1:
                                    label = f"Iteration {iterboundary}"
                                else:
                                    label = f"Iterations {nmax_iterations_with_color} to {mm_niter_boundary_extension - 1}"
                                color = "gray"
                                alpha = 0.3
                            else:
                                label = None
                                color = "gray"
                                alpha = 0.3
                        else:
                            label = f"Iteration {iterboundary}"
                if mm_fixed_points_in_boundary is None:
                    xboundary_fit = xboundary
                    yboundary_fit = yboundary
                    wboundary_fit = wboundary
                else:
                    wboundary_max = np.max(wboundary)
                    xboundary_fit = np.concatenate((xboundary, x_mm_fixed_points_in_boundary))
                    yboundary_fit = np.concatenate((yboundary, y_mm_fixed_points_in_boundary))
                    wboundary_fit = np.concatenate((wboundary, w_mm_fixed_points_in_boundary * wboundary_max))
                isort = np.argsort(xboundary_fit)
                boundaryfit, knots = spline_positive_derivative(
                    x=xboundary_fit[isort],
                    y=yboundary_fit[isort],
                    w=wboundary_fit[isort],
                    n_total_knots=mm_knots_splfit,
                )
                ydum = boundaryfit(xcbins)
                ydum[xcbins < knots[0]] = boundaryfit(knots[0])
                ydum[xcbins > knots[-1]] = boundaryfit(knots[-1])
                ax1.plot(xcbins, ydum, "-", color=color, label=label)
                ax1.plot(knots, boundaryfit(knots), "o", color=color, alpha=alpha, markersize=4)
        elif mm_boundary_fit == "piecewise":
            boundaryfit = define_piecewise_linear_function(
                xarray=x_mm_fixed_points_in_boundary, yarray=y_mm_fixed_points_in_boundary
            )
            ax1.plot(xcbins, boundaryfit(xcbins), "r-", label="Piecewise linear fit")
        else:
            raise ValueError(f"Invalid mm_boundary_fit: {mm_boundary_fit}. Valid options are {VALID_BOUNDARY_FITS}.")

        if mm_threshold is None:
            # Use the minimum value of the boundary as the mm_threshold
            mm_threshold = np.min(yplot_boundary)
            _logger.info("updated mm_threshold for cosmic-ray detection: %f", mm_threshold)

        # Apply the criterium to detect coincident cosmic-ray pixels
        flag1 = yplot > boundaryfit(xplot)
        flag2 = yplot > mm_threshold
        flag_mm = np.logical_and(flag1, flag2)
        flag3 = max2d.flatten() > mm_minimum_max2d_rnoise * rnoise.flatten()
        flag_mm = np.logical_and(flag_mm, flag3)
        if mm_dilation > 0:
            _logger.info("applying binary dilation with size=%d to cosmic-ray mask", mm_dilation)
            num_pixels_before_dilation = np.sum(flag_mm)
            structure = ndimage.generate_binary_structure(2, 2)
            naxis2, naxis1 = max2d.shape
            flag_mm = ndimage.binary_dilation(
                flag_mm.reshape((naxis2, naxis1)), structure=structure, iterations=mm_dilation
            ).flatten()
            num_pixels_after_dilation = np.sum(flag_mm)
            ldum = len(str(num_pixels_after_dilation))
            _logger.info(f"number of pixels flagged before dilation : {num_pixels_before_dilation:{ldum}d}")
            _logger.info(f"number of pixels flagged after dilation  : {num_pixels_after_dilation:{ldum}d}")
        flag_mm = np.logical_and(flag_mm, bool_to_be_cleaned.flatten())
        _logger.info(
            "pixels flagged as cosmic rays by %s: %d (%08.4f%%)",
            rlabel_mmcosmic,
            np.sum(flag_mm),
            np.sum(flag_mm) / flag_mm.size * 100,
        )

        # Plot the results
        if mm_fixed_points_in_boundary is not None:
            ax1.plot(
                x_mm_fixed_points_in_boundary,
                y_mm_fixed_points_in_boundary,
                "ms",
                markersize=6,
                alpha=0.5,
                label="Fixed points",
            )
        ax1.set_xlabel(r"min2d $-$ bias")
        ax1.set_ylabel(r"median2d $-$ min2d")
        ax1.set_title(f"Simulated data\n(mm_nsimulations = {mm_nsimulations}, mm_synthetic={mm_synthetic})")
        if mm_niter_boundary_extension > 1:
            ax1.legend(loc=1)
        xplot_boundary = np.linspace(xdiag_min, xdiag_max, 100)
        yplot_boundary = boundaryfit(xplot_boundary)
        if mm_boundary_fit == "spline":
            # For spline fit, force the boundary to be constant outside the knots
            yplot_boundary[xplot_boundary < knots[0]] = boundaryfit(knots[0])
            yplot_boundary[xplot_boundary > knots[-1]] = boundaryfit(knots[-1])
        ax2.plot(xplot_boundary, yplot_boundary, "r-", label="Detection boundary")
        if mm_fixed_points_in_boundary is not None:
            ax2.plot(
                x_mm_fixed_points_in_boundary,
                y_mm_fixed_points_in_boundary,
                "ms",
                markersize=6,
                alpha=0.5,
                label="Fixed points",
            )
        ax2.set_xlim(xdiag_min, xdiag_max)
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_xlabel(ax1.get_xlabel())
        ax2.set_ylabel(ax1.get_ylabel())
        ax2.set_title("Original data")
        ax2.legend(loc=1)
        plt.tight_layout()
        png_filename = "diagnostic_histogram2d.png"
        _logger.info(f"saving {png_filename}")
        plt.savefig(png_filename, dpi=150)
        if interactive:
            _logger.info(
                "Entering interactive mode\n" "(press 'r' to repeat plot, 'q' to close figure, 'x' to quit program)"
            )
            plt.show()
        else:
            loop = False
        if loop:
            # Modify hist2d_min_neighbors
            mm_hist2d_min_neighbors = input_number(
                expected_type="int",
                prompt="Minimum number of neighbors to keep bins in the 2D histogram (0-8)",
                min_val=0,
                max_val=8,
                default=mm_hist2d_min_neighbors,
            )
            # Modify mm_boundary_fit
            new_mm_boundary_fit = ""
            while new_mm_boundary_fit not in ["piecewise", "spline"]:
                new_mm_boundary_fit = input(f"Type of boundary fit: piecewise or spline [{mm_boundary_fit}]: ").strip().lower()
                if new_mm_boundary_fit == "":
                    new_mm_boundary_fit = mm_boundary_fit
                if new_mm_boundary_fit not in ["piecewise", "spline"]:
                    _logger.info("Invalid value. Please type 'piecewise' or 'spline'.")
            mm_boundary_fit = new_mm_boundary_fit
            if mm_boundary_fit == "spline":
                # Modify number of knots
                mm_knots_splfit = input_number(
                    expected_type="int",
                    prompt="Number of knots for spline boundary fit (min. 2)",
                    min_val=2,
                    default=mm_knots_splfit,
                )
                # Modify number of iterations for boundary extension
                mm_niter_boundary_extension = input_number(
                    expected_type="int",
                    prompt="Number of iterations for boundary extension (min. 0)",
                    min_val=0,
                    default=mm_niter_boundary_extension,
                )
                mm_weight_boundary_extension = input_number(
                    expected_type="float",
                    prompt="Weight for boundary extension (greater than 1.0)",
                    min_val=1.0,
                    default=mm_weight_boundary_extension,
                )
            # Modify fixed points in the boundary
            if mm_fixed_points_in_boundary is None:
                num_fixed_points = 0
            else:
                num_fixed_points = len(x_mm_fixed_points_in_boundary)
            if mm_boundary_fit == "piecewise" and num_fixed_points < 2:
                _logger.info("At least two fixed points are needed for piecewise linear boundary fit.")
                modify_fixed = "y"
            else:
                modify_fixed = "?"
            while modify_fixed != "n":
                xsort_and_show_fixed_points_in_boundary(
                    _logger,
                    num_fixed_points,
                    x_mm_fixed_points_in_boundary,
                    y_mm_fixed_points_in_boundary,
                    w_mm_fixed_points_in_boundary,
                )
                while modify_fixed not in ["y", "n", ""]:
                    modify_fixed = (
                        input("Do you want to modify the fixed points in the boundary? (y/[n]): ").strip().lower()
                    )
                    if modify_fixed not in ["y", "n", ""]:
                        _logger.info("Invalid value. Please type 'y' or 'n' or press 'Enter'.")
                if modify_fixed != "y":
                    modify_fixed = "n"
                # Allow to delete individual fixed points or add new ones
                if modify_fixed == "y":
                    action = ""
                    while action != "n":
                        submenu = "Type:\n"
                        submenu += "- 'a' to add a fixed point\n"
                        valid_answers = "a/[n]"
                        if num_fixed_points > 0:
                            submenu += "- 'c' to clear all fixed points\n"
                            submenu += "- 'd' to delete an existing fixed point\n"
                            submenu += "- 'e' to edit an existing fixed point\n"
                            valid_answers = "a/c/d/e/[n]"
                        submenu += "- 'n' none (continue without additional changes)"
                        _logger.info(submenu)
                        action = input(f"Your choice ({valid_answers}): ").strip().lower()
                        if action == "d" and num_fixed_points > 0:
                            if num_fixed_points == 1:
                                _logger.info("Only one fixed point available, deleting it.")
                                index_to_delete = 1
                            else:
                                index_to_delete = input_number(
                                    expected_type="int",
                                    prompt=f"Index of fixed point to delete (1 to {num_fixed_points})",
                                    min_val=1,
                                    max_val=num_fixed_points,
                                )
                            x_mm_fixed_points_in_boundary = np.delete(
                                x_mm_fixed_points_in_boundary, index_to_delete - 1
                            )
                            y_mm_fixed_points_in_boundary = np.delete(
                                y_mm_fixed_points_in_boundary, index_to_delete - 1
                            )
                            w_mm_fixed_points_in_boundary = np.delete(
                                w_mm_fixed_points_in_boundary, index_to_delete - 1
                            )
                            num_fixed_points -= 1
                            if num_fixed_points == 0:
                                x_mm_fixed_points_in_boundary = None
                                y_mm_fixed_points_in_boundary = None
                                w_mm_fixed_points_in_boundary = None
                        elif action == "e" and num_fixed_points > 0:
                            if num_fixed_points == 1:
                                _logger.info("Only one fixed point available, editing it.")
                                index_to_edit = 1
                            else:
                                index_to_edit = input_number(
                                    expected_type="int",
                                    prompt=f"Index of fixed point to edit (1 to {num_fixed_points})",
                                    min_val=1,
                                    max_val=num_fixed_points,
                                )
                            x_new = input_number(
                                expected_type="float",
                                prompt="New x value of fixed point",
                                default=x_mm_fixed_points_in_boundary[index_to_edit - 1],
                            )
                            y_new = input_number(
                                expected_type="float",
                                prompt="New y value of fixed point",
                                default=y_mm_fixed_points_in_boundary[index_to_edit - 1],
                            )
                            w_new = input_number(
                                expected_type="float",
                                prompt="New weight of fixed point",
                                min_val=0.0,
                                default=w_mm_fixed_points_in_boundary[index_to_edit - 1],
                            )
                            x_mm_fixed_points_in_boundary[index_to_edit - 1] = x_new
                            y_mm_fixed_points_in_boundary[index_to_edit - 1] = y_new
                            w_mm_fixed_points_in_boundary[index_to_edit - 1] = w_new
                        elif action == "a":
                            x_new = input_number(
                                expected_type="float", prompt="x value of new fixed point", default=None
                            )
                            y_new = input_number(
                                expected_type="float", prompt="y value of new fixed point", default=None
                            )
                            w_new = input_number(
                                expected_type="float",
                                prompt="weight of new fixed point",
                                min_val=0.0,
                                default=1000.0,
                            )
                            if num_fixed_points == 0:
                                x_mm_fixed_points_in_boundary = np.array([x_new], dtype=float)
                                y_mm_fixed_points_in_boundary = np.array([y_new], dtype=float)
                                w_mm_fixed_points_in_boundary = np.array([w_new], dtype=float)
                            else:
                                x_mm_fixed_points_in_boundary = np.append(x_mm_fixed_points_in_boundary, x_new)
                                y_mm_fixed_points_in_boundary = np.append(y_mm_fixed_points_in_boundary, y_new)
                                w_mm_fixed_points_in_boundary = np.append(w_mm_fixed_points_in_boundary, w_new)
                            num_fixed_points += 1
                        elif action == "c" and num_fixed_points > 0:
                            num_fixed_points = 0
                            x_mm_fixed_points_in_boundary = None
                            y_mm_fixed_points_in_boundary = None
                            w_mm_fixed_points_in_boundary = None
                        elif action in ["n", ""]:
                            if action == "":
                                action = "n"
                            if mm_boundary_fit == "piecewise":
                                if num_fixed_points < 2:
                                    _logger.info(
                                        "At least two fixed points are needed for piecewise linear boundary fit."
                                    )
                                    input("Press Enter to continue...")
                                    action = "?"
                            if action == "n":
                                _logger.info("No changes made to fixed points in boundary.")
                                modify_fixed = "n"
                        else:
                            input("Invalid option. Press Enter to try again...")
                        if num_fixed_points > 0:
                            mm_fixed_points_in_boundary = True
                        else:
                            mm_fixed_points_in_boundary = None
                        xsort_and_show_fixed_points_in_boundary(
                            _logger,
                            num_fixed_points,
                            x_mm_fixed_points_in_boundary,
                            y_mm_fixed_points_in_boundary,
                            w_mm_fixed_points_in_boundary,
                        )
        else:
            plt.close(fig)

    return xplot_boundary, yplot_boundary, boundaryfit, flag_mm
