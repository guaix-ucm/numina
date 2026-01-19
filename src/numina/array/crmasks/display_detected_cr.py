#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Display the detected cosmic rays in the images."""

from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from numina.tools.progressbarlines import ProgressBarLines
import teareduce as tea


def display_detected_cr(
    num_images,
    image3d,
    median2d,
    median2d_corrected,
    flag_integer_dilated,
    acronym_aux,
    labels_cr,
    number_cr,
    mask_mediancr,
    list_mask_single_exposures=None,
    mask_all_singles=None,
    semiwindow=15,
    maxplots=-1,
    verify_cr=False,
    color_scale="zscale",
    xplot=None,
    yplot=None,
    boundaryfit=None,
    mm_threshold=None,
    _logger=None,
    output_dir=".",
):
    """Display the detected cosmic rays

    If list_mask_single_exposures is None, the plots correspond to the MEDIANCR
    computation. Otherwise, the plots correspond to problematic pixels (i.e.,
    cosmic-ray pixels flagged in all the single exposures).

    Note that this function can change 'mask_mediancr' if 'verify_cr' is True.
    """
    if list_mask_single_exposures is None:
        output_basename = "mediancr_identified"
        if mask_all_singles is not None:
            raise ValueError("mask_all_singles must be None when list_mask_single_exposures is None.")
    else:
        output_basename = "problematic_pixels"
        if len(list_mask_single_exposures) != num_images:
            raise ValueError("len(list_mask_single_exposures) must be equal to num_images.")
        if verify_cr:
            raise ValueError("verify_cr cannot be True when list_mask_single_exposures is provided.")
        if mask_all_singles is None:
            raise ValueError("mask_all_singles must be provided when list_mask_single_exposures is provided.")

    # Determine the number of rows and columns for the plot,
    # considering that we want to plot also 3 additional images:
    # the median2d, the mask and the median2d_corrected
    num_plot_max = num_images
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
    if list_mask_single_exposures is None:
        # plots for mediancr
        pdf_any4 = PdfPages(Path(output_dir) / f"{output_basename}_any4.pdf")
        pdf_only3 = PdfPages(Path(output_dir) / f"{output_basename}_only3.pdf")
        pdf_only2 = PdfPages(Path(output_dir) / f"{output_basename}_only2.pdf")
        pdf_other = PdfPages(Path(output_dir) / f"{output_basename}_other.pdf")
        num_any4 = 0
        num_only3 = 0
        num_only2 = 0
        num_other = 0
        cr_table_any4 = Table(names=("CR_number", "X_pixel", "Y_pixel", "Mask_value"), dtype=(int, int, int, int))
        cr_table_only3 = Table(names=("CR_number", "X_pixel", "Y_pixel", "Mask_value"), dtype=(int, int, int, int))
        cr_table_only2 = Table(names=("CR_number", "X_pixel", "Y_pixel", "Mask_value"), dtype=(int, int, int, int))
        cr_table_other = Table(names=("CR_number", "X_pixel", "Y_pixel", "Mask_value"), dtype=(int, int, int, int))
        cr_table_filename_any4 = f"{output_basename}_any4.csv"
        cr_table_filename_only3 = f"{output_basename}_only3.csv"
        cr_table_filename_only2 = f"{output_basename}_only2.csv"
        cr_table_filename_other = f"{output_basename}_other.csv"
    else:
        # plots for problematic pixels
        pdf = PdfPages(Path(output_dir) / f"{output_basename}.pdf")
        cr_table = Table(names=("CR_number", "X_pixel", "Y_pixel", "Mask_value"), dtype=(int, int, int, int))
        cr_table_filename = f"{output_basename}.csv"
    maxplots_eff = maxplots
    if verify_cr:
        # In verify_cr mode, we plot all the cosmic rays
        maxplots_eff = number_cr
    elif maxplots_eff < 0:
        if number_cr > 1000:
            maxplots_eff = 1000
            _logger.warning(f"limiting to {maxplots_eff} plots (out of {number_cr} CRs detected)")
        else:
            maxplots_eff = number_cr
    _logger.info(f"generating {maxplots_eff} plots...")
    xlabel = "X pixel (from 1 to NAXIS1)"
    ylabel = "Y pixel (from 1 to NAXIS2)"
    naxis2, naxis1 = median2d.shape
    total = min(number_cr, maxplots_eff)
    pbar = ProgressBarLines(total, _logger)
    for i in range(total):
        # Locate the cosmic ray pixels
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
        # Generate the plots
        fig, axarr = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axarr = axarr.flatten()
        # Important: use interpolation=None instead of interpolation='None' to avoid
        # having blurred images when opening the PDF file with macos Preview
        cmap = "viridis"
        cblabel = "Number of counts"
        if color_scale == "zscale":
            vmin, vmax = tea.zscale(median2d[i1 : (i2 + 1), j1 : (j2 + 1)])
        else:
            vmin = np.min(median2d[i1 : (i2 + 1), j1 : (j2 + 1)])
            vmax = np.max(median2d[i1 : (i2 + 1), j1 : (j2 + 1)])
        for k in range(num_plot_max):
            ax = axarr[k]
            title = title = f"image#{k+1}/{num_images}"
            tea.imshow(
                fig,
                ax,
                image3d[k][i1 : (i2 + 1), j1 : (j2 + 1)],
                vmin=vmin,
                vmax=vmax,
                extent=[j1 + 0.5, j2 + 1.5, i1 + 0.5, i2 + 1.5],
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                cmap=cmap,
                cblabel=cblabel,
                interpolation=None,
            )
        if list_mask_single_exposures is None:
            # Plots for mediancr
            for k in range(3):
                ax = axarr[k + num_plot_max]
                cmap = "viridis"
                if k == 0:
                    image2d = median2d
                    title = "median"
                    norm = None
                elif k == 1:
                    image2d = flag_integer_dilated
                    title = "flag_integer_dilated"
                    # cmap = 'plasma'
                    color_list = ["black", "grey", "blue", "red", "yellow"]
                    cmap = ListedColormap(color_list)
                    bounds = np.arange(-0.5, len(color_list), 1)  # integer limits: -0.5, 0.5, 1.5, ...
                    norm = BoundaryNorm(bounds, cmap.N)
                    cblabel = None
                elif k == 2:
                    image2d = median2d_corrected
                    title = "median corrected"
                    norm = None
                else:
                    raise ValueError(f"Unexpected {k=}")
                if k in [0, 2]:
                    if color_scale == "zscale":
                        vmin_, vmax_ = tea.zscale(image2d[i1 : (i2 + 1), j1 : (j2 + 1)])
                    else:
                        vmin_ = np.min(image2d[i1 : (i2 + 1), j1 : (j2 + 1)])
                        vmax_ = np.max(image2d[i1 : (i2 + 1), j1 : (j2 + 1)])
                else:
                    vmin_, vmax_ = None, None
                img, cax, cbar = tea.imshow(
                    fig,
                    ax,
                    image2d[i1 : (i2 + 1), j1 : (j2 + 1)],
                    vmin=vmin_,
                    vmax=vmax_,
                    extent=[j1 + 0.5, j2 + 1.5, i1 + 0.5, i2 + 1.5],
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=title,
                    cmap=cmap,
                    norm=norm,
                    cblabel=cblabel,
                    interpolation=None,
                )
                if k == 1:
                    cbar.set_ticks([0, 1, 2, 3, 4])
                    cbar.ax.yaxis.set_tick_params(length=0)
                    cbar.set_ticklabels(
                        ["0: no CR", "1: dilation", "2: mm", f"3: {acronym_aux}", f"4: mm & {acronym_aux}"]
                    )
                    # Overlay the grid of pixels
                    for idum in range(i1, i2 + 1):
                        ax.hlines(y=idum + 0.5, xmin=j1 + 0.5, xmax=j2 + 1.5, colors="w", lw=0.5, alpha=0.3)
                    for jdum in range(j1, j2 + 1):
                        ax.vlines(x=jdum + 0.5, ymin=i1 + 0.5, ymax=i2 + 1.5, colors="w", lw=0.5, alpha=0.3)
            nplot_missing = nrows * ncols - num_plot_max - 3
            if nplot_missing > 0:
                for k in range(nplot_missing):
                    ax = axarr[-k - 1]
                    ax.axis("off")
            fig.suptitle(f"CR#{i+1}/{number_cr}")
        else:
            # Plot the problematic pixels
            ijloc = np.argwhere(labels_cr == i + 1)
            xproblematic = ijloc[:, 1] + 1  # FITS criterium
            yproblematic = ijloc[:, 0] + 1  # FITS criter
            for k in range(num_plot_max):
                ax = axarr[k + num_plot_max]
                title = title = f"mask#{k+1}/{num_images}"
                image2d = list_mask_single_exposures[k].astype(int)
                tea.imshow(
                    fig,
                    ax,
                    image2d[i1 : (i2 + 1), j1 : (j2 + 1)],
                    vmin=0,
                    vmax=1,
                    extent=[j1 + 0.5, j2 + 1.5, i1 + 0.5, i2 + 1.5],
                    xlabel=xlabel,
                    ylabel=ylabel,
                    title=title,
                    cmap="plasma",
                    cblabel="flag",
                    interpolation=None,
                )
                for xdum, ydum in zip(xproblematic, yproblematic):
                    ax.plot([xdum - 0.5, xdum + 0.5], [ydum - 0.5, ydum + 0.5], "r-")
                    ax.plot([xdum - 0.5, xdum + 0.5], [ydum + 0.5, ydum - 0.5], "r-")
            fig.suptitle(f"Problematic pixels, region #{i+1}/{number_cr}")

        plt.tight_layout()

        if verify_cr:
            plt.show(block=False)
            print("-" * 50)
        different_flag_values = []
        for idum in range(i1, i2 + 1):
            for jdum in range(j1, j2 + 1):
                if labels_cr[idum, jdum] == i + 1:
                    different_flag_values.append(flag_integer_dilated[idum, jdum])
        different_flag_values = set(different_flag_values)
        for idum in range(i1, i2 + 1):
            for jdum in range(j1, j2 + 1):
                if labels_cr[idum, jdum] == i + 1:
                    if list_mask_single_exposures is None:
                        if 4 in different_flag_values:
                            cr_table_any4.add_row((i + 1, jdum + 1, idum + 1, flag_integer_dilated[idum, jdum]))
                        elif 3 in different_flag_values and 2 not in different_flag_values:
                            cr_table_only3.add_row((i + 1, jdum + 1, idum + 1, flag_integer_dilated[idum, jdum]))
                        elif 2 in different_flag_values and 3 not in different_flag_values:
                            cr_table_only2.add_row((i + 1, jdum + 1, idum + 1, flag_integer_dilated[idum, jdum]))
                        else:
                            cr_table_other.add_row((i + 1, jdum + 1, idum + 1, flag_integer_dilated[idum, jdum]))
                    else:
                        cr_table.add_row((i + 1, jdum + 1, idum + 1, flag_integer_dilated[idum, jdum]))
                    if verify_cr:
                        print(f"pixel (x,y) = ({jdum+1}, {idum+1}):  mask = {flag_integer_dilated[idum, jdum]}")
        if verify_cr:
            accept_cr = input(f"Accept this CR detection #{i+1}: " "[Y]es (default) | (n)o | (a)ll | (s)top)? ")
            if accept_cr.lower() == "n":
                _logger.info("removing cosmic ray detection #%d from the mask\n", i + 1)
                mask_mediancr[labels_cr == i + 1] = False
            elif accept_cr.lower() == "a":
                verify_cr = False
                _logger.info("accepting all remaining cosmic ray detections\n")
                _logger.info("generating remaining plots without user interaction...")
            elif accept_cr.lower() == "s":
                _logger.info("stopping the program execution")
                raise SystemExit(0)
            else:
                _logger.info("keeping cosmic ray detection #%d in the mask", i + 1)
        if list_mask_single_exposures is None:
            # plot for mediancr
            if 4 in different_flag_values:
                pdf_any4.savefig(fig, bbox_inches="tight")
                num_any4 += 1
            elif 3 in different_flag_values and 2 not in different_flag_values:
                pdf_only3.savefig(fig, bbox_inches="tight")
                num_only3 += 1
            elif 2 in different_flag_values and 3 not in different_flag_values:
                pdf_only2.savefig(fig, bbox_inches="tight")
                num_only2 += 1
            else:
                pdf_other.savefig(fig, bbox_inches="tight")
                num_other += 1
        else:
            # plot for problematic pixels
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if list_mask_single_exposures is None:
            # plot pixels in the M.M. diagnostic diagram
            fig, ax = plt.subplots(figsize=figsize)
            if boundaryfit is not None:
                xplot_boundary = np.linspace(np.min(xplot), np.max(xplot), 1000)
                yplot_boundary = boundaryfit(xplot_boundary)
            else:
                raise ValueError("boundaryfit must be provided for mediancr plots")
            ax.plot(xplot_boundary, yplot_boundary, color="C1", linestyle="-", label="detection boundary")
            if mm_threshold is not None:
                ax.axhline(y=mm_threshold, color="C0", linestyle=":", label="mm_threshold")
            legend4 = True
            legend3 = True
            legend2 = True
            legend0 = True
            for idum in range(i1, i2 + 1):
                for jdum in range(j1, j2 + 1):
                    if labels_cr[idum, jdum] == i + 1:
                        if flag_integer_dilated[idum, jdum] == 4:
                            symbol = "mo"
                            if legend4:
                                label = f"mm & {acronym_aux}"
                                legend4 = False
                            else:
                                label = None
                        elif flag_integer_dilated[idum, jdum] == 3:
                            symbol = "rx"
                            if legend3:
                                label = f"{acronym_aux}"
                                legend3 = False
                            else:
                                label = None
                        elif flag_integer_dilated[idum, jdum] == 2:
                            symbol = "b+"
                            if legend2:
                                label = "mm"
                                legend2 = False
                            else:
                                label = None
                        else:
                            symbol = "ko"
                            if legend0:
                                label = "dilation"
                                legend0 = False
                            else:
                                label = None
                        xval = xplot.reshape((naxis2, naxis1))[idum, jdum]
                        yval = yplot.reshape((naxis2, naxis1))[idum, jdum]
                        ax.plot(xval, yval, symbol, label=label)
                        ax.text(xval, yval, f"({jdum+1},{idum+1})", fontsize=6, color="gray")
            _, ymax = ax.get_ylim()
            ax.set_ylim(0.0, ymax)
            ax.set_xlabel(r"min2d $-$ bias")
            ax.set_ylabel(r"median2d $-$ min2d")
            ax.legend()
            fig.suptitle(f"CR#{i+1}/{number_cr}")
            plt.tight_layout()
            if 4 in different_flag_values:
                pdf_any4.savefig(fig, bbox_inches="tight")
            elif 3 in different_flag_values and 2 not in different_flag_values:
                pdf_only3.savefig(fig, bbox_inches="tight")
            elif 2 in different_flag_values and 3 not in different_flag_values:
                pdf_only2.savefig(fig, bbox_inches="tight")
            else:
                pdf_other.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pbar.update()

    _logger.info("plot generation complete")
    if list_mask_single_exposures is None:
        # plots for mediancr
        num_max = np.max([num_any4, num_only3, num_only2, num_other])
        snum_max = str(num_max)
        lmax = len(snum_max)
        pdf_any4.close()
        _logger.info(f"saving {num_any4:>{lmax}} CRs in {output_basename}_any4.pdf file")
        pdf_only3.close()
        _logger.info(f"saving {num_only3:>{lmax}} CRs in {output_basename}_only3.pdf file")
        pdf_only2.close()
        _logger.info(f"saving {num_only2:>{lmax}} CRs in {output_basename}_only2.pdf file")
        pdf_other.close()
        _logger.info(f"saving {num_other:>{lmax}} CRs in {output_basename}_other.pdf file")
        cr_table_any4.write(Path(output_dir) / cr_table_filename_any4, format="csv", overwrite=True)
        _logger.info(f"table of {num_any4} CRs in {output_basename}_any4.csv:\n{cr_table_any4}")
        _logger.info("table saved")
        cr_table_only3.write(Path(output_dir) / cr_table_filename_only3, format="csv", overwrite=True)
        _logger.info(f"table of {num_only3} CRs in {output_basename}_only3.csv:\n{cr_table_only3}")
        _logger.info("table saved")
        cr_table_only2.write(Path(output_dir) / cr_table_filename_only2, format="csv", overwrite=True)
        _logger.info(f"table of {num_only2} CRs in {output_basename}_only2.csv:\n{cr_table_only2}")
        _logger.info("table saved")
        cr_table_other.write(Path(output_dir) / cr_table_filename_other, format="csv", overwrite=True)
        _logger.info(f"table of {num_other} CRs in {output_basename}_other.csv:\n{cr_table_other}")
        _logger.info("table saved")
    else:
        # plots for problematic pixels
        pdf.close()
        _logger.info(f"saving {output_basename}.pdf file")
        cr_table.write(Path(output_dir) / cr_table_filename, format="csv", overwrite=True)
        _logger.info("\n%s", cr_table)
        _logger.info("table of identified cosmic rays saved to %s", cr_table_filename)
