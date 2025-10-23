#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Compute the flux factor for each image based on the median."""
import sys

from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import numpy as np

from numina.array.display.plot_hist_step import plot_hist_step
import teareduce as tea

from .remove_isolated_pixels import remove_isolated_pixels


def compute_flux_factor(image3d, median2d, ff_region, _logger, interactive=False, debug=False,
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
    ff_region : tea.SliceRegion2D instance
        The region of interest for the flux factor computation.
    _logger : logging.Logger
        The logger to use for logging.
    interactive : bool, optional
        If True, enable interactive mode for plots (default is False).
    debug : bool, optional
        If True, enable debug mode (default is False).
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
    if naxis3 > 7:
        _logger.warning("compute_flux_factor: naxis3 > 7, skipping flux factor computation.")
        return np.ones(naxis3, dtype=float)

    naxis2_, naxis1_ = median2d.shape
    if naxis2 != naxis2_ or naxis1 != naxis1_:
        raise ValueError("image3d and median2d must have the same shape in the last two dimensions.")

    if debug:
        # Histograms before applying flux factor
        xmin = np.min(median2d[ff_region.python])
        xmax = np.max(median2d[ff_region.python])
        dx = xmax - xmin
        xminh = xmin - dx / 20
        xmaxh = xmax + dx / 20
        bins0 = np.linspace(xmin, xmax, 100)
        h0, edges0 = np.histogram(median2d[ff_region.python].flatten(), bins=bins0)
        hstep = (edges0[1] - edges0[0])
        fig, ax = plt.subplots()
        for i in range(naxis3):
            xmax_ = np.max(image3d[i][ff_region.python])
            bins = np.arange(xmin, xmax_ + hstep, hstep)
            h, edges = np.histogram(image3d[i][ff_region.python].flatten(), bins=bins)
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
            _logger.info("Entering interactive mode (press 'q' to close figure)")
            plt.show()
        plt.close(fig)

    if naxis3 % 2 == 1:
        # Interactive plot showing the image number at the median position
        argsort = np.argsort(image3d, axis=0)

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(12, 5.5))
        aspect_imshow = 'auto'  # 'equal' or 'auto'

        i_comparison_image = 0   # 0 for median2d, 1, 2,... for image3d[comparison_image-1]

        def on_key(event):
            nonlocal img_ax1, img_ax2
            nonlocal i_comparison_image
            nonlocal aspect_imshow
            update_vmin_vmax = False
            # Determine the current region in the plot
            xmin, xmax = ax1.get_xlim()
            ymin, ymax = ax1.get_ylim()
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
            if event.key == 'a':
                if aspect_imshow == 'equal':
                    aspect_imshow = 'auto'
                else:
                    aspect_imshow = 'equal'
                ax1.set_aspect(aspect_imshow)
                ax2.set_aspect(aspect_imshow)
                ax1.figure.canvas.draw_idle()
                ax2.figure.canvas.draw_idle()
            elif event.key == ',':
                vmin, vmax = np.min(median2d[region2d]), np.max(median2d[region2d])
                update_vmin_vmax = True
            elif event.key == '/':
                vmin, vmax = tea.zscale(median2d[region2d])
                update_vmin_vmax = True
            elif event.key in ['t', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
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
                        comparison_image = median2d
                        ax1_title = 'median2d'
                    else:
                        comparison_image = image3d[i_comparison_image - 1]
                        ax1_title = f'single exposure #{i_comparison_image}'
                    _logger.info(f"Displaying {ax1_title} in left panel.")
                    img_ax1.set_data(comparison_image)
                    ax1.set_title(ax1_title)
                    ax1.figure.canvas.draw_idle()
            elif event.key == '?':
                _logger.info("-" * 79)
                _logger.info("Keyboard shortcuts:")
                _logger.info("'h' or 'r' : reset zoom to full image")
                _logger.info("'p' : pan mode")
                _logger.info("'o' : zoom to rectangle")
                _logger.info("'f' : toggle full screen mode")
                _logger.info("'s' : save figure to PNG file")
                _logger.info("." * 79)
                _logger.info("'a' : toggle aspect='equal' / 'aspect='auto' for imshow")
                _logger.info("',' : set vmin and vmax to min and max of the current region")
                _logger.info("'/' : set vmin and vmax using zscale of the current region")
                _logger.info("'t' : cycle through images (left panel)")
                _logger.info("'0-9' : display specific image number in left panel (0 for median)")
                _logger.info("'?' : display this help message")
                _logger.info("'q' : quit interactive mode")
                _logger.info("'x' : exit program")
                _logger.info("-" * 79)
            elif event.key == 'x':
                _logger.info("Exiting program as per user request ('x' key pressed).")
                plt.close(fig)
                sys.exit(0)

            if update_vmin_vmax:
                img_ax1.set_clim(vmin=vmin, vmax=vmax)
                ax1.figure.canvas.draw_idle()

        fig.canvas.mpl_connect('key_press_event', on_key)

        vmin, vmax = tea.zscale(median2d)
        img_ax1, _, _ = tea.imshow(fig, ax1, median2d, ds9mode=True, vmin=vmin, vmax=vmax,
                                   aspect=aspect_imshow, title='median2d')
        if naxis3 == 3:
            color_list = ['tab:red', 'tab:green', 'tab:blue']
        elif naxis3 == 5:
            color_list = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']
        elif naxis3 == 7:
            color_list = ['tab:red', 'tab:orange', 'tab:olive', 'tab:green', 'tab:cyan', 'tab:blue', 'tab:purple']
        else:
            raise ValueError("Cannot define color map for naxis3 odd and > 7.")
        cmap = ListedColormap(color_list)
        bounds = np.arange(0.5, len(color_list) + 1, 1)   # integer limits: -0.5, 0.5, 1.5, ...
        norm = BoundaryNorm(bounds, cmap.N)
        img_ax2, _, cbar2 = tea.imshow(fig, ax2, argsort[naxis3//2] + 1, ds9mode=True,
                                       cmap=cmap, norm=norm,
                                       title='Image number at median position',
                                       cblabel='Image number', aspect=aspect_imshow)
        cbar2.set_ticks(np.arange(1, naxis3 + 1))
        cbar2.ax.yaxis.set_tick_params(length=0)
        ax1.set_xlim(ff_region.fits[0].start - 0.5, ff_region.fits[0].stop + 0.5)
        ax1.set_ylim(ff_region.fits[1].start - 0.5, ff_region.fits[1].stop + 0.5)
        plt.tight_layout()
        png_filename = 'image_number_at_median_position.png'
        _logger.info(f"saving {png_filename}")
        plt.savefig(png_filename, dpi=150)
        if interactive:
            _logger.info("Entering interactive mode (press 'q' to close figure, 'x' to quit program)")
            plt.show()
        plt.close(fig)

    xmin = np.min(median2d[ff_region.python])
    xmax = np.max(median2d[ff_region.python])
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
        ratio = np.divide(data, median2d,
                          out=np.zeros_like(median2d, dtype=float),
                          where=median2d != 0)
        h, edges = np.histogramdd(
            sample=(ratio[ff_region.python].flatten(), median2d[ff_region.python].flatten()),
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
        good_splfits = True
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
            try:
                splfit = tea.AdaptiveLSQUnivariateSpline(yfit, xfit, t=2, adaptive=False)
                good_splfit = True
            except Exception as e:
                _logger.warning(f"Could not fit spline to flux factor data for image #{idata+1}, side {side}: {e}")
                good_splfit = False
            if good_splfit:
                ax2.plot(splfit(yfit), yfit, f'C{side}-')
                knots = splfit.get_knots()
                ax2.plot(splfit(knots), knots, f'C{side}o', markersize=4)
                imax = np.argmax(splfit(yfit))
                ymode[side] = yfit[imax]
                xmode[side] = splfit(ymode[side])
                ax2.plot(xmode[side], ymode[side], f'C{side}o', markersize=8)
            if not good_splfit:
                good_splfits = False
                _logger.warning(f"Skipping flux factor computation for image #{idata+1} due to bad spline fit.")
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(ax1.get_ylim())
        if good_splfits:
            if xmode[0] > xmode[1]:
                imode = 0
            else:
                imode = 1
            ax2.axhline(ymode[imode], color=f'C{imode}', linestyle=':')
            ax2.text(xbin[-5], ymode[imode]+(ybin[-1]-ybin[0])/40, f'{ymode[imode]:.3f}', color=f'C{imode}', ha='right')
            flux_factor.append(ymode[imode])
        else:
            ax2.axhline(1.0, color='k', linestyle=':')
            ax2.text(xbin[-5], 1.01, f'{1.0:.3f}', color='k', ha='right')
            flux_factor.append(1.0)
        plt.tight_layout()

        png_filename = f'flux_factor{idata+1}.png'
        _logger.info(f"saving {png_filename}")
        plt.savefig(png_filename, dpi=150)
        if interactive:
            _logger.info("Entering interactive mode (press 'q' to close figure)")
            plt.show()
        plt.close(fig)

    if len(flux_factor) != naxis3:
        raise ValueError(f"Expected {naxis3} flux factors, but got {len(flux_factor)}.")

    # round the flux factor to 6 decimal places to avoid
    # unnecessary precision when writting to the FITS header
    flux_factor = np.round(flux_factor, decimals=6)

    if debug:
        # Histograms after applying flux factor
        fig, ax = plt.subplots()
        for i in range(naxis3):
            xmax_ = np.max(image3d[i][ff_region.python])
            bins = np.arange(xmin, xmax_ + hstep, hstep)
            h, edges = np.histogram(image3d[i][ff_region.python].flatten() / flux_factor[i], bins=bins)
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
            _logger.info("Entering interactive mode (press 'q' to close figure)")
            plt.show()
        plt.close(fig)

    return flux_factor
