#
# Copyright 2025-2026 Universidad Complutense de Madrid
#
# This file is part of Numina
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
"""Diagnostic plot for cosmic ray mask computation."""

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
from pathlib import Path
import teareduce as tea

mpl.rcParams["keymap.quit"] = []  # disable 'q' key for quitting plots


def segregate_cr_flags(
    naxis1,
    naxis2,
    flag_only_aux,
    flag_only_mm,
    flag_both,
    enum_aux_global,
    enum_mm_global,
    enum_both_global,
    within_xy_diagram,
):
    """Segregate the cosmic ray flags into three categories:
    - detected only by the auxiliar method
    - detected only by the mmcosmic method
    - detected by both methods

    Parameters
    ----------
    naxis1 : int
        The size of the first dimension of the input arrays.
    naxis2 : int
        The size of the second dimension of the input arrays.
    flag_only_aux : 1D numpy array
        A boolean array indicating which pixels are detected
        only by the auxiliar method.
    flag_only_mm : 1D numpy array
        A boolean array indicating which pixels are detected
        only by the mmcosmic method.
    flag_both : 1D numpy array
        A boolean array indicating which pixels are detected
        by both methods.
    enum_aux_global : 1D numpy array
        An integer array with the enumeration of the pixels
        detected only by the auxiliar method.
    enum_mm_global : 1D numpy array
        An integer array with the enumeration of the pixels
        detected only by the mmcosmic method.
    enum_both_global : 1D numpy array
        An integer array with the enumeration of the pixels
        detected by both methods.
    within_xy_diagram : 1D numpy array
        A boolean array indicating which pixels are within the XY diagram.

    Returns
    -------
    flag_only_aux_within_xy : 1D numpy array
        A boolean array indicating which pixels are detected
        only by the auxiliar method within the XY diagram.
    flag_only_mm_within_xy : 1D numpy array
        A boolean array indicating which pixels are detected
        only by the mmcosmic method within the XY diagram.
    flag_both_within_xy : 1D numpy array
        A boolean array indicating which pixels are detected
        by both methods within the XY diagram.
    (num_only_aux, xcr_only_aux, ycr_only_aux) : tuple
        Number of pixels detected only by the auxiliar method,
        and their x and y coordinates (FITS convention; first pixel is (1, 1)).
        If no pixels are detected, xcr_only_aux and ycr_only_aux are None.
    (num_only_mm, xcr_only_mm, ycr_only_mm) : tuple
        Number of pixels detected only by the mmcosmic method,
        and their x and y coordinates (FITS convention; first pixel is (1, 1)).
        If no pixels are detected, xcr_only_mm and ycr_only_mm are None.
    (num_both, xcr_both, ycr_both) : tuple
        Number of pixels detected by both methods,
        and their x and y coordinates (FITS convention; first pixel is (1, 1)).
        If no pixels are detected, xcr_both and ycr_both are None.
    """

    # Segregate the cosmic rays within the XY diagnostic diagram
    flag_only_aux_within_xy = flag_only_aux & within_xy_diagram
    flag_only_mm_within_xy = flag_only_mm & within_xy_diagram
    flag_both_within_xy = flag_both & within_xy_diagram

    num_only_aux_within_xy = np.sum(flag_only_aux_within_xy)
    if num_only_aux_within_xy > 0:
        pixels_detected = np.argwhere(flag_only_aux_within_xy.reshape(naxis2, naxis1))
        xcr_only_aux_within_xy = pixels_detected[:, 1] + 1  # FITS convention: first pixel is (1, 1)
        ycr_only_aux_within_xy = pixels_detected[:, 0] + 1  # FITS convention: first pixel is (1, 1)
        ncr_only_aux_within_xy = enum_aux_global[flag_only_aux_within_xy]
    else:
        xcr_only_aux_within_xy, ycr_only_aux_within_xy, ncr_only_aux_within_xy = None, None, None

    num_only_mm_within_xy = np.sum(flag_only_mm_within_xy)
    if num_only_mm_within_xy > 0:
        pixels_detected = np.argwhere(flag_only_mm_within_xy.reshape(naxis2, naxis1))
        xcr_only_mm_within_xy = pixels_detected[:, 1] + 1  # FITS convention: first pixel is (1, 1)
        ycr_only_mm_within_xy = pixels_detected[:, 0] + 1  # FITS convention: first pixel is (1, 1)
        ncr_only_mm_within_xy = enum_mm_global[flag_only_mm_within_xy]
    else:
        xcr_only_mm_within_xy, ycr_only_mm_within_xy, ncr_only_mm_within_xy = None, None, None

    num_both_within_xy = np.sum(flag_both_within_xy)
    if num_both_within_xy > 0:
        pixels_detected = np.argwhere(flag_both_within_xy.reshape(naxis2, naxis1))
        xcr_both_within_xy = pixels_detected[:, 1] + 1  # FITS convention: first pixel is (1, 1)
        ycr_both_within_xy = pixels_detected[:, 0] + 1  # FITS convention: first pixel is (1, 1)
        ncr_both_within_xy = enum_both_global[flag_both_within_xy]
    else:
        xcr_both_within_xy, ycr_both_within_xy, ncr_both_within_xy = None, None, None

    return (
        flag_only_aux_within_xy,
        flag_only_mm_within_xy,
        flag_both_within_xy,
        (num_only_aux_within_xy, xcr_only_aux_within_xy, ycr_only_aux_within_xy, ncr_only_aux_within_xy),
        (num_only_mm_within_xy, xcr_only_mm_within_xy, ycr_only_mm_within_xy, ncr_only_mm_within_xy),
        (num_both_within_xy, xcr_both_within_xy, ycr_both_within_xy, ncr_both_within_xy),
    )


def update_marks(
    naxis1,
    naxis2,
    flag_only_aux,
    flag_only_mm,
    flag_both,
    enum_aux_global,
    enum_mm_global,
    enum_both_global,
    xplot,
    yplot,
    ax1,
    ax2,
    ax3,
    display_ncr=True,
):
    """Update the marks in the diagnostic plot.
    If ax2 and ax3 are None, only the segregation of the cosmic rays
    within the XY diagram is performed and the information of the
    suspected pixels is printed in the terminal.
    """
    if flag_only_aux.shape != (naxis2 * naxis1,):
        raise ValueError(f"{flag_only_aux.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if flag_only_aux.shape != flag_only_mm.shape:
        raise ValueError(f"{flag_only_mm.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if flag_only_aux.shape != flag_both.shape:
        raise ValueError(f"{flag_both.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if flag_only_aux.shape != xplot.shape or flag_only_aux.shape != yplot.shape:
        raise ValueError(f"{xplot.shape=} and {yplot.shape=} must have {flag_only_aux.shape=}.")

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    within_xy_diagram = (xlim[0] <= xplot) & (xplot <= xlim[1]) & (ylim[0] <= yplot) & (yplot <= ylim[1])

    flag_only_aux_within_xy, flag_only_mm_within_xy, flag_both_within_xy, tuple_aux, tuple_mm, tuple_both = (
        segregate_cr_flags(
            naxis1,
            naxis2,
            flag_only_aux,
            flag_only_mm,
            flag_both,
            enum_aux_global,
            enum_mm_global,
            enum_both_global,
            within_xy_diagram,
        )
    )
    num_only_aux_within_xy, xcr_only_aux_within_xy, ycr_only_aux_within_xy, ncr_only_aux_within_xy = tuple_aux
    num_only_mm_within_xy, xcr_only_mm_within_xy, ycr_only_mm_within_xy, ncr_only_mm_within_xy = tuple_mm
    num_both_within_xy, xcr_both_within_xy, ycr_both_within_xy, ncr_both_within_xy = tuple_both

    if ax2 is None and ax3 is None:
        ax_list = [None]
    else:
        ax_list = [ax2, ax3]
    for ax in ax_list:
        for num, method, xcr, ycr, ncr, flag_only, color, marker in zip(
            [num_only_aux_within_xy, num_only_mm_within_xy, num_both_within_xy],
            ["axiliar", "mmcosmic", "both"],
            [xcr_only_aux_within_xy, xcr_only_mm_within_xy, xcr_both_within_xy],
            [ycr_only_aux_within_xy, ycr_only_mm_within_xy, ycr_both_within_xy],
            [ncr_only_aux_within_xy, ncr_only_mm_within_xy, ncr_both_within_xy],
            [flag_only_aux_within_xy, flag_only_mm_within_xy, flag_both_within_xy],
            ["r", "b", "m"],
            ["x", "+", "o"],
        ):
            if num > 0:
                if ax is None:
                    print("-" * 78)
                    print(f"{num} cosmic rays detected with method {method}.")
                    for ix, iy, ncr in zip(xcr, ycr, ncr):
                        print(f"  Pixel (x, y) = ({ix}, {iy}), number {ncr}")
                elif ax == ax2:
                    for ix, iy, ncr in zip(xplot[flag_only], yplot[flag_only], ncr):
                        ax.text(
                            ix,
                            iy,
                            str(ncr),
                            color=color,
                            fontsize=8,
                            clip_on=True,
                            ha="center",
                            va="center",
                            bbox=dict(facecolor="white", alpha=0.5),
                        )
                elif ax == ax3:
                    if display_ncr:
                        for ix, iy, ncr in zip(xcr, ycr, ncr):
                            ax.text(
                                ix,
                                iy,
                                str(ncr),
                                color=color,
                                fontsize=8,
                                clip_on=True,
                                ha="center",
                                va="center",
                                bbox=dict(facecolor="white", alpha=0.5),
                            )
                    else:
                        if marker == "o":
                            ax.scatter(xcr, ycr, edgecolors=color, marker=marker, facecolors="none")
                        else:
                            ax.scatter(xcr, ycr, c=color, marker=marker)


def diagnostic_plot(
    xplot,
    yplot,
    boundaryfit,
    rlabel_aux_plain,
    flag_aux,
    flag_mm,
    mm_threshold,
    mm_dilation,
    ylabel,
    interactive,
    target2d,
    target2d_name,
    min2d,
    mean2d,
    image3d,
    _logger=None,
    png_filename=None,
    output_dir=".",
):
    """Diagnostic plot for the mediancr function."""
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
    if flag_aux is None:
        flag_aux_eff = np.zeros((naxis2 * naxis1,), dtype=bool)
    else:
        flag_aux_eff = flag_aux
    if flag_mm is None:
        flag_mm_eff = np.zeros((naxis2 * naxis1,), dtype=bool)
    else:
        flag_mm_eff = flag_mm
    if flag_aux_eff.shape != (naxis2 * naxis1,):
        raise ValueError(f"{flag_aux_eff.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if flag_aux_eff.shape != flag_mm_eff.shape:
        raise ValueError(f"{flag_mm_eff.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if xplot.shape != (naxis2 * naxis1,):
        raise ValueError(f"{xplot.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")
    if yplot.shape != (naxis2 * naxis1,):
        raise ValueError(f"{yplot.shape=} must have shape (naxis2*naxis1,)={naxis1*naxis2}.")

    display_ncr = False  # display the number of cosmic rays in the plot instead of symbols
    aspect_imshow = "auto"  # 'equal' or 'auto'
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
        ax2 = fig.add_axes([0.55, y0_plot + height_plot + vspace_plot, width_plot, height_plot], sharex=ax1, sharey=ax1)
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
    flag_only_aux = flag_aux_eff & ~flag_mm_eff
    flag_only_mm = flag_mm_eff & ~flag_aux_eff
    flag_both = flag_aux_eff & flag_mm_eff
    num_only_aux = np.sum(flag_only_aux)
    num_only_mm = np.sum(flag_only_mm)
    num_both = np.sum(flag_both)
    num_total = num_only_aux + num_only_mm + num_both

    # Enumerate the cosmic rays detected by the different methods
    enum_aux_global = np.zeros_like(flag_aux_eff, dtype=int)
    enum_aux_global[flag_only_aux] = np.arange(1, np.sum(flag_only_aux) + 1, dtype=int)
    enum_mm_global = np.zeros_like(flag_mm_eff, dtype=int)
    enum_mm_global[flag_only_mm] = np.arange(1, np.sum(flag_only_mm) + 1, dtype=int)
    enum_both_global = np.zeros_like(flag_aux_eff, dtype=int)
    enum_both_global[flag_both] = np.arange(1, np.sum(flag_both) + 1, dtype=int)

    ax1.plot(xplot, yplot, "C0,", label="Non-suspected pixels", zorder=1)
    ax1.scatter(
        xplot[flag_only_aux],
        yplot[flag_only_aux],
        c="r",
        marker="x",
        label=f"Suspected pixels: {num_only_aux} ({str(rlabel_aux_plain).rstrip()})",
        zorder=2,
    )
    ax1.scatter(
        xplot[flag_only_mm],
        yplot[flag_only_mm],
        c="b",
        marker="+",
        label=f"Suspected pixels: {num_only_mm} (mmcosmic, mm_dilation: {mm_dilation})",
        zorder=2,
    )
    ax1.scatter(
        xplot[flag_both],
        yplot[flag_both],
        edgecolor="m",
        marker="o",
        facecolors="none",
        label=f"Suspected pixels: {num_both} (both methods)",
        zorder=2,
    )
    if boundaryfit is not None:
        xplot_boundary = np.linspace(np.min(xplot), np.max(xplot), 1000)
        yplot_boundary = boundaryfit(xplot_boundary)
        ax1.plot(xplot_boundary, yplot_boundary, "C1-", label="Detection boundary", zorder=3)
    if mm_threshold is not None:
        ax1.axhline(mm_threshold, color="C0", linestyle=":", label=f"mm_threshold ({mm_threshold:.2f})", zorder=3)
    ax1.set_xlabel(r"min2d $-$ bias  [ADU]")  # the bias was subtracted from the input arrays
    ax1.set_ylabel(ylabel)
    ax1.set_title("Diagnostic Diagram")
    ax1.legend(loc="upper right", fontsize=8, title=f"Total: {num_total} suspected pixels")

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_xlabel(ax1.get_xlabel())
    ax2.set_ylabel(ax1.get_ylabel())
    ax2.set_title(ax1.get_title())

    vmin, vmax = tea.zscale(target2d)
    ax3_title = target2d_name
    img_ax3, _, _ = tea.imshow(
        fig,
        ax3,
        target2d,
        ds9mode=True,
        aspect=aspect_imshow,
        vmin=vmin,
        vmax=vmax,
        title=ax3_title,
        cmap="viridis",
        colorbar=False,
    )
    if i_comparison_image == 0:
        comparison_image = mean2d
        ax4_title = "mean2d"
    else:
        comparison_image = image3d[i_comparison_image - 1]
        ax4_title = f"single exposure #{i_comparison_image}]"
    img_ax4, _, _ = tea.imshow(
        fig,
        ax4,
        comparison_image,
        ds9mode=True,
        aspect=aspect_imshow,
        vmin=vmin,
        vmax=vmax,
        title=ax4_title,
        cmap="viridis",
        colorbar=False,
    )
    ax3.set_title(ax3_title)
    ax4.set_title(ax4_title)
    update_marks(
        naxis1,
        naxis2,
        flag_only_aux,
        flag_only_mm,
        flag_both,
        enum_aux_global,
        enum_mm_global,
        enum_both_global,
        xplot,
        yplot,
        ax1,
        ax2,
        ax3,
        display_ncr,
    )

    for ax, label, color in zip([ax1, ax2, ax3, ax4], ["(a)", "(b)", "(c)", "(d)"], ["k", "k", "w", "w"]):
        ax.text(
            0.97,
            0.03,
            label,
            transform=ax.transAxes,
            color=color,
            fontsize=15,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    updating = {"plot_limits": False}

    def sync_zoom_x(event_ax):
        if updating["plot_limits"]:
            return
        try:
            updating["plot_limits"] = True
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
            updating["plot_limits"] = False

    def sync_zoom_y(event_ax):
        nonlocal img_ax3, img_ax4
        nonlocal display_ncr
        if updating["plot_limits"]:
            return
        try:
            updating["plot_limits"] = True
            if event_ax is ax1:
                ylim = ax1.get_ylim()
                ax2.set_ylim(ylim)
                xlim = ax3.get_xlim()
                ylim = ax3.get_ylim()
                ax3.cla()
                img_ax3, _, _ = tea.imshow(
                    fig,
                    ax3,
                    target2d,
                    ds9mode=True,
                    aspect=aspect_imshow,
                    vmin=vmin,
                    vmax=vmax,
                    title=ax3_title,
                    cmap="viridis",
                    colorbar=False,
                )
                ax3.set_xlim(xlim)
                ax3.set_ylim(ylim)
                xlim = ax4.get_xlim()
                ylim = ax4.get_ylim()
                ax4.cla()
                img_ax4, _, _ = tea.imshow(
                    fig,
                    ax4,
                    comparison_image,
                    ds9mode=True,
                    aspect=aspect_imshow,
                    vmin=vmin,
                    vmax=vmax,
                    title=ax4_title,
                    cmap="viridis",
                    colorbar=False,
                )
                ax4.set_xlim(xlim)
                ax4.set_ylim(ylim)
                update_marks(
                    naxis1,
                    naxis2,
                    flag_only_aux,
                    flag_only_mm,
                    flag_both,
                    enum_aux_global,
                    enum_mm_global,
                    enum_both_global,
                    xplot,
                    yplot,
                    ax1,
                    ax2,
                    ax3,
                    display_ncr,
                )
                for ax, label, color in zip([ax3, ax4], ["(c)", "(d)"], ["w", "w"]):
                    ax.text(
                        0.97,
                        0.03,
                        label,
                        transform=ax.transAxes,
                        color=color,
                        fontsize=15,
                        fontweight="bold",
                        va="bottom",
                        ha="right",
                    )

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
            updating["plot_limits"] = False

    ax1.callbacks.connect("xlim_changed", sync_zoom_x)
    ax1.callbacks.connect("ylim_changed", sync_zoom_y)

    if not interactive:
        plt.tight_layout()
    if png_filename is not None:
        _logger.info(f"saving {png_filename}")
        plt.savefig(Path(output_dir) / png_filename, dpi=150)
    if interactive:
        init_limits = {ax: (ax.get_xlim(), ax.get_ylim()) for ax in [ax1, ax2, ax3, ax4]}

        mouse_info = {"ax": None, "x": None, "y": None}

        def on_mouse_move(event):
            if event.inaxes:
                mouse_info["ax"] = event.inaxes
                mouse_info["x"] = event.xdata
                mouse_info["y"] = event.ydata

        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

        def on_key(event):
            nonlocal vmin, vmax
            nonlocal img_ax3, img_ax4
            nonlocal display_ncr
            nonlocal aspect_imshow
            nonlocal i_comparison_image, comparison_image
            ax_mouse = mouse_info["ax"]
            x_mouse = mouse_info["x"]
            y_mouse = mouse_info["y"]

            if event.key in ("h", "H", "r", "R"):
                for ax in [ax1, ax2, ax3, ax4]:
                    init_xlim, init_ylim = init_limits[ax]
                    ax.set_xlim(init_xlim)
                    ax.set_ylim(init_ylim)
                if i_comparison_image == 0:
                    comparison_image = mean2d
                    ax4_title = "mean2d"
                else:
                    comparison_image = image3d[i_comparison_image - 1]
                    ax4_title = f"single exposure #{i_comparison_image}"
                ax4.set_title(ax4_title)
            elif event.key == "?":
                _logger.info("-" * 79)
                _logger.info("Keyboard shortcuts:")
                _logger.info("'h' or 'r': reset zoom to initial limits")
                _logger.info("'p': pan mode")
                _logger.info("'o': zoom to rectangle")
                _logger.info("'f': toggle full screen mode")
                _logger.info("'s': save the figure to a PNG file")
                _logger.info("." * 79)
                _logger.info("'?': show this help message")
                _logger.info("'i': print pixel info at mouse position (ax3 only)")
                _logger.info("'&': print CR pixels within the zoomed region (ax3 only)")
                _logger.info("'n': toggle display of number of cosmic rays (ax3 only)")
                _logger.info("'a': toggle imshow aspect='equal' / aspect='auto' (ax3 and ax4 only)")
                _logger.info("'t': toggle mean2d -> individual exposures in ax4")
                _logger.info("'0': switch to mean2d in ax4")
                _logger.info("'1', '2', ...: switch to individual exposure #1, #2, ... in ax4")
                _logger.info("',': set vmin and vmax to min and max of the zoomed region (ax3 and ax4 only)")
                _logger.info("'/': set vmin and vmax using zscale of the zoomed region (ax3 and ax4 only)")
                _logger.info("'c': close the plot and continue the program execution")
                _logger.info("'x': halt the program execution")
                _logger.info("-" * 79)
            elif event.key == "i":
                if ax_mouse in [ax1, ax2]:
                    print(f"x_mouse = {x_mouse:.3f}, y_mouse = {y_mouse:.3f}")
                elif ax_mouse in [ax3]:
                    ix = int(round(x_mouse))
                    iy = int(round(y_mouse))
                    if 1 <= ix <= naxis1 and 1 <= iy <= naxis2:
                        print("-" * 79)
                        print(f"Pixel coordinates (FITS criterium): ix = {ix}, iy = {iy}")
                        print(f"target2d - min2d = {target2d[iy-1, ix-1] - min2d[iy-1, ix-1]:.3f}")
                        print(f"min2d - bias     = {min2d[iy-1, ix-1]:.3f}")
                        print("." * 79)
                        for inum in range(image3d.shape[0]):
                            print(f"(image {inum+1} - bias) = {image3d[inum, iy-1, ix-1]:.3f}")
                        print("." * 79)
                        for flag, crmethod in zip(
                            [flag_aux_eff, flag_mm_eff, flag_both], [rlabel_aux_plain, "mmcosmic"]
                        ):
                            # Python convention: first pixel is (0, 0) but iy and ix are in FITS convention
                            # where the first pixel is (1, 1)
                            if flag.reshape((naxis2, naxis1))[iy - 1, ix - 1]:
                                print(f"Pixel found by {crmethod}")
                            else:
                                print(f"Pixel not found by {crmethod}")
            elif event.key == "&":
                if ax_mouse in [ax3]:
                    update_marks(
                        naxis1,
                        naxis2,
                        flag_only_aux,
                        flag_only_mm,
                        flag_both,
                        enum_aux_global,
                        enum_mm_global,
                        enum_both_global,
                        xplot,
                        yplot,
                        ax1,
                        None,
                        None,
                        None,
                    )
            elif event.key == "n":
                if ax_mouse in [ax3]:
                    display_ncr = not display_ncr
                    sync_zoom_y(ax1)
            elif event.key == "a":
                if aspect_imshow == "equal":
                    aspect_imshow = "auto"
                else:
                    aspect_imshow = "equal"
                sync_zoom_y(ax1)
            elif event.key in ["t", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                if ax_mouse == ax4:
                    i_comparison_image_previous = i_comparison_image
                    if event.key == "t":
                        i_comparison_image += 1
                        if i_comparison_image > naxis3:
                            i_comparison_image = 0
                    elif event.key == "0":
                        i_comparison_image = 0
                    else:
                        i_comparison_image = int(event.key)
                        if i_comparison_image > naxis3:
                            i_comparison_image = i_comparison_image_previous
                    if i_comparison_image != i_comparison_image_previous:
                        if i_comparison_image == 0:
                            comparison_image = mean2d
                            ax4_title = "mean2d"
                        else:
                            comparison_image = image3d[i_comparison_image - 1]
                            ax4_title = f"single exposure #{i_comparison_image}"
                        print(f"Switching to {ax4_title} in ax4")
                        vmin, vmax = img_ax4.get_clim()
                        img_ax4.set_data(comparison_image)
                        img_ax4.set_clim(vmin=vmin, vmax=vmax)
                        ax4.set_title(ax4_title)
                        ax4.figure.canvas.draw_idle()
            elif event.key in [",", "/"]:
                if ax_mouse in [ax3, ax4]:
                    # Determine the region in the image corresponding to the zoomed area
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
                    region2d = tea.SliceRegion2D(f"[{ixmin}:{ixmax},{iymin}:{iymax}]", mode="fits").python
                    if event.key == ",":
                        if ax_mouse == ax3:
                            vmin, vmax = np.min(target2d[region2d]), np.max(target2d[region2d])
                        elif ax_mouse == ax4:
                            vmin, vmax = np.min(comparison_image[region2d]), np.max(comparison_image[region2d])
                    elif event.key == "/":
                        if ax_mouse == ax3:
                            vmin, vmax = tea.zscale(target2d[region2d])
                        elif ax_mouse == ax4:
                            vmin, vmax = tea.zscale(comparison_image[region2d])
                    text_box_vmin.set_val(f"{int(np.round(vmin, 0))}")
                    text_box_vmax.set_val(f"{int(np.round(vmax, 0))}")

                    img_ax3.set_clim(vmin=vmin, vmax=vmax)
                    img_ax4.set_clim(vmin=vmin, vmax=vmax)

                    ax3.figure.canvas.draw_idle()
                    ax4.figure.canvas.draw_idle()
            elif event.key == "x":
                _logger.info("Exiting program as per user request ('x' key pressed).")
                plt.close(fig)
                sys.exit(0)
            elif event.key == "c":
                _logger.info("Continuing program execution as per user request ('c' key pressed).")
                plt.close(fig)

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
                    print(f"Invalid input: {text}")

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
                    print(f"Invalid input: {text}")

        text_box_vmin = TextBox(ax_vmin, "vmin:", initial=f"{int(np.round(vmin, 0))}", textalignment="right")
        text_box_vmin.on_submit(submit_vmin)
        text_box_vmax = TextBox(ax_vmax, "vmax:", initial=f"{int(np.round(vmax, 0))}", textalignment="right")
        text_box_vmax.on_submit(submit_vmax)

        _logger.info("Entering interactive mode\n(press '?' for help, 'c' to continue, 'x' to quit program)")
        # plt.tight_layout()
        plt.show()
    plt.close(fig)
