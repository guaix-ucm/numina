#
# Copyright 2025 Universidad Complutense de Madrid
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

import teareduce as tea


def display_detected_cr(num_images, image3d, median2d, median2d_corrected,
                        flag_integer_dilated, labels_cr, number_cr, mask_mediancr,
                        list_mask_single_exposures=None, mask_all_singles=None,
                        semiwindow=15, maxplots=-1, verify_cr=False,
                        color_scale='zscale', _logger=None):
    """Display the detected cosmic rays

    If list_mask_single_exposures is None, the plots correspond to the MEDIANCR
    computation. Otherwise, the plots correspond to problematic pixels (i.e.,
    cosmic-ray pixels flagged in all the single exposures).
    """
    if list_mask_single_exposures is None:
        output_basename = 'mediancr_identified'
        if mask_all_singles is not None:
            raise ValueError("mask_all_singles must be None when list_mask_single_exposures is None.")
    else:
        output_basename = 'problematic_pixels'
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
    pdf = PdfPages(f'{output_basename}.pdf')
    cr_table = Table(names=('CR_number', 'X_pixel', 'Y_pixel', 'Mask_value'), dtype=(int, int, int, int))
    cr_table_filename = f'{output_basename}.csv'
    maxplots_eff = maxplots
    if verify_cr:
        # In verify_cr mode, we plot all the cosmic rays
        maxplots_eff = number_cr
    elif maxplots_eff < 0:
        if number_cr > 200:
            maxplots_eff = 200
            _logger.info(f"limiting to {maxplots_eff} plots (out of {number_cr} CRs detected)")
            input("Press Enter to continue...")
        else:
            maxplots_eff = number_cr
    _logger.info(f"generating {maxplots_eff} plots...")
    xlabel = 'X pixel (from 1 to NAXIS1)'
    ylabel = 'Y pixel (from 1 to NAXIS2)'
    naxis2, naxis1 = median2d.shape
    for i in range(min(number_cr, maxplots_eff)):
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
                       extent=[j1+0.5, j2+1.5, i1+0.5, i2+1.5],
                       xlabel=xlabel, ylabel=ylabel,
                       title=title, cmap=cmap, cblabel=cblabel, interpolation=None)
        if list_mask_single_exposures is None:
            # plots for mediancr
            for k in range(3):
                ax = axarr[k + num_plot_max]
                cmap = 'viridis'
                if k == 0:
                    image2d = median2d
                    title = 'median'
                    norm = None
                elif k == 1:
                    image2d = flag_integer_dilated
                    title = 'flag_integer_dilated'
                    # cmap = 'plasma'
                    color_list = ['black', 'grey', 'blue', 'red', 'yellow']
                    cmap = ListedColormap(color_list)
                    bounds = np.arange(-0.5, len(color_list), 1)   # integer limits: -0.5, 0.5, 1.5, ...
                    norm = BoundaryNorm(bounds, cmap.N)
                    cblabel = None
                elif k == 2:
                    image2d = median2d_corrected
                    title = 'median corrected'
                    norm = None
                else:
                    raise ValueError(f'Unexpected {k=}')
                if k in [0, 2]:
                    if color_scale == 'zscale':
                        vmin_, vmax_ = tea.zscale(image2d[i1:(i2+1), j1:(j2+1)])
                    else:
                        vmin_ = np.min(image2d[i1:(i2+1), j1:(j2+1)])
                        vmax_ = np.max(image2d[i1:(i2+1), j1:(j2+1)])
                else:
                    vmin_, vmax_ = None, None
                img, cax, cbar = tea.imshow(
                    fig, ax, image2d[i1:(i2+1), j1:(j2+1)], vmin=vmin_, vmax=vmax_,
                    extent=[j1+0.5, j2+1.5, i1+0.5, i2+1.5],
                    xlabel=xlabel, ylabel=ylabel,
                    title=title, cmap=cmap, norm=norm, cblabel=cblabel, interpolation=None
                )
                if k == 1:
                    cbar.set_ticks([0, 1, 2, 3, 4])
                    cbar.ax.yaxis.set_tick_params(length=0)
                    cbar.set_ticklabels(['0: no CR', '1: dilation', '2: mm', '3: la', '4: mm & la'])
                    # Overlay the grid of pixels
                    for idum in range(i1, i2 + 1):
                        ax.hlines(y=idum + 0.5, xmin=j1 + 0.5, xmax=j2 + 1.5, colors='w', lw=0.5, alpha=0.3)
                    for jdum in range(j1, j2 + 1):
                        ax.vlines(x=jdum + 0.5, ymin=i1 + 0.5, ymax=i2 + 1.5, colors='w', lw=0.5, alpha=0.3)
            nplot_missing = nrows * ncols - num_plot_max - 3
            if nplot_missing > 0:
                for k in range(nplot_missing):
                    ax = axarr[-k-1]
                    ax.axis('off')
            fig.suptitle(f'CR#{i+1}/{number_cr}')
        else:
            # print the coordinates of the problematic pixels
            ijloc = np.argwhere(labels_cr == i + 1)
            xproblematic = ijloc[:, 1] + 1  # FITS criterium
            yproblematic = ijloc[:, 0] + 1  # FITS criter
            # plots for problematic pixels
            for k in range(num_plot_max):
                ax = axarr[k + num_plot_max]
                title = title = f'mask#{k+1}/{num_images}'
                image2d = list_mask_single_exposures[k].astype(int)
                tea.imshow(fig, ax, image2d[i1:(i2+1), j1:(j2+1)], vmin=0, vmax=1,
                           extent=[j1+0.5, j2+1.5, i1+0.5, i2+1.5],
                           xlabel=xlabel, ylabel=ylabel,
                           title=title, cmap='plasma', cblabel='flag', interpolation=None)
                for xdum, ydum in zip(xproblematic, yproblematic):
                    ax.plot([xdum-0.5, xdum+0.5], [ydum-0.5, ydum+0.5], 'r-')
                    ax.plot([xdum-0.5, xdum+0.5], [ydum+0.5, ydum-0.5], 'r-')
            fig.suptitle(f'Problematic pixels, region #{i+1}/{number_cr}')

        plt.tight_layout()
        plt.show(block=False)

        if verify_cr:
            print('-' * 50)
        for idum in range(i1, i2 + 1):
            for jdum in range(j1, j2 + 1):
                if labels_cr[idum, jdum] == i + 1:
                    cr_table.add_row((i + 1, jdum + 1, idum + 1, flag_integer_dilated[idum, jdum]))
                    if verify_cr:
                        print(f'pixel (x,y) = ({jdum+1}, {idum+1}):  mask = {flag_integer_dilated[idum, jdum]}')
        if verify_cr:
            accept_cr = input(f"Accept this cosmic ray detection #{i+1} ([Y]es / [n]o / [a]ll / [s]top))? ")
            if accept_cr.lower() == 'n':
                _logger.info("removing cosmic ray detection #%d from the mask\n", i + 1)
                mask_mediancr[labels_cr == i + 1] = False
            elif accept_cr.lower() == 'a':
                verify_cr = False
                _logger.info("accepting all remaining cosmic ray detections\n")
                _logger.info("generating remaining plots without user interaction...")
            elif accept_cr.lower() == 's':
                _logger.info("stopping the program execution")
                raise SystemExit(0)
            else:
                _logger.info("keeping cosmic ray detection #%d in the mask", i + 1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    pdf.close()
    _logger.info("plot generation complete")
    _logger.info("saving mediancr_identified_cr.pdf")
    cr_table.write(cr_table_filename, format='csv', overwrite=True)
    _logger.info("\n%s", cr_table)
    _logger.info("table of identified cosmic rays saved to %s", cr_table_filename)
